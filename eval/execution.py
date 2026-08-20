"""Safe, read-only SQL execution and result-set comparison for execution accuracy.

Execution accuracy (run predicted SQL, compare its *results* to gold's results)
is the standard rigorous metric for text-to-SQL -- it credits any query that is
semantically equivalent to gold, not just ones that are byte-for-byte identical
(e.g. different column aliases or join order but the same rows back).
"""
import re
import sqlite3
from pathlib import Path

_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|VACUUM|REPLACE)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


def guard_read_only(sql: str) -> None:
    """Reject anything that isn't a plain read query.

    This matters beyond the eval harness: the same guard runs in the MCP
    server, where an agent (not a curated eval set) supplies the question and
    therefore indirectly influences what SQL gets generated and executed.
    """
    if not _ALLOWED_START.match(sql):
        raise UnsafeSQLError(f"SQL must start with SELECT or WITH, got: {sql[:60]!r}")
    if _FORBIDDEN.search(sql):
        raise UnsafeSQLError(f"SQL contains a forbidden write/DDL keyword: {sql[:120]!r}")


def execute_readonly(db_path: Path, sql: str, timeout_s: float = 5.0) -> list:
    """Run `sql` against `db_path` in true read-only mode (SQLite URI mode=ro),
    so even a bug in guard_read_only can't result in a write."""
    guard_read_only(sql)
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_s * 1000)}")
        cur = conn.execute(sql)
        return cur.fetchall()
    finally:
        conn.close()


def results_match(gold_sql: str, gold_rows: list, pred_rows: list) -> bool:
    """Order-sensitive if gold query has ORDER BY (order is part of the spec,
    e.g. 'top N'), otherwise compare as multisets so column/row ordering from
    an equivalent-but-differently-written query still counts as correct."""
    ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))
    g = _norm_row_types(gold_rows)
    p = _norm_row_types(pred_rows)
    if ordered:
        return g == p
    return sorted(g, key=repr) == sorted(p, key=repr)


def _norm_row_types(rows: list) -> list:
    """Normalize numeric types (int vs float, e.g. 5 vs 5.0) and rounding so
    equivalent-but-differently-typed results aren't scored as mismatches."""
    norm = []
    for row in rows:
        norm_row = []
        for v in row:
            if isinstance(v, float):
                norm_row.append(round(v, 2))
            else:
                norm_row.append(v)
        norm.append(tuple(norm_row))
    return norm
