"""Self-built call logging: latency, token counts, estimated cost per call,
persisted to a local SQLite file. No external account, no vendor dashboard --
mirrors the observability approach in pr-review-agent. This is what lets the
comparison report and the MCP server's own /stats-equivalent answer "how much
is this actually costing, and how fast is it" without guessing.
"""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "calls.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    source        TEXT NOT NULL,      -- e.g. 'mcp_server', 'eval_harness'
    predictor     TEXT NOT NULL,      -- e.g. 'ollama:sql-specialist', 'frontier:claude-haiku-4-5'
    question      TEXT NOT NULL,
    sql           TEXT,
    latency_ms    REAL NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    error         TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_call(source: str, predictor: str, question: str, sql: str = None,
             latency_ms: float = 0.0, input_tokens: int = None,
             output_tokens: int = None, cost_usd: float = None, error: str = None) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT INTO calls
               (ts, source, predictor, question, sql, latency_ms, input_tokens, output_tokens, cost_usd, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), source, predictor, question, sql,
             latency_ms, input_tokens, output_tokens, cost_usd, error),
        )
    conn.close()


@contextmanager
def timed_call(source: str, predictor: str, question: str):
    """Usage:
        with timed_call("mcp_server", predictor.name, question) as record:
            pred = predictor.predict(question)
            record(sql=pred.sql, latency_ms=pred.latency_ms, ...)
    """
    fields = {}

    def record(**kwargs):
        fields.update(kwargs)

    yield record
    log_call(source=source, predictor=predictor, question=question, **fields)


def stats(predictor: str = None) -> dict:
    """Aggregate stats, optionally filtered to one predictor. Powers the
    'what has this actually cost so far' answer without an external dashboard."""
    conn = _connect()
    where = "WHERE predictor = ?" if predictor else ""
    params = (predictor,) if predictor else ()
    row = conn.execute(f"""
        SELECT COUNT(*), AVG(latency_ms),
               SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END),
               SUM(cost_usd)
        FROM calls {where}
    """, params).fetchone()
    conn.close()
    n, avg_latency, n_ok, total_cost = row
    return {
        "predictor": predictor or "all",
        "n_calls": n or 0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
        "success_rate": round(n_ok / n, 3) if n else None,
        "total_cost_usd": round(total_cost, 6) if total_cost else 0.0,
    }
