"""MCP server exposing the fine-tuned NL-to-SQL specialist as a real tool --
installable in Claude Desktop / Claude Code so any MCP client can call
nl_to_sql(question) and get back grounded, executed results, not just text.

By default this talks to a model served locally via Ollama (see
training/merge_and_quantize.py for how to get one running). Set
SQL_SPECIALIST_BACKEND=frontier to run against a prompted frontier model
instead (useful for demoing the tool before you've trained anything).

Run: python -m mcp_server.server
Claude Desktop config: point it at this file's absolute path with the venv's
python interpreter -- see README "MCP server" section.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer

from eval.execution import UnsafeSQLError, execute_readonly
from observability.logger import log_call

DB_PATH = Path(__file__).parent.parent / "schema" / "shopsphere.db"
MAX_ROWS_RETURNED = 200

mcp = MCPServer(
    "sql-specialist",
    instructions=(
        "Answers natural-language questions about the ShopSphere e-commerce "
        "database (customers, orders, products, reviews, support tickets) by "
        "generating SQL with a fine-tuned specialist model, executing it "
        "read-only, and returning the actual rows -- not just the query text."
    ),
)

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    backend = os.environ.get("SQL_SPECIALIST_BACKEND", "ollama")
    if backend == "ollama":
        from serving.ollama_predictor import OllamaPredictor
        _predictor = OllamaPredictor(model=os.environ.get("SQL_SPECIALIST_MODEL", "sql-specialist"))
    elif backend == "frontier":
        from eval.baseline_frontier import FrontierPredictor
        _predictor = FrontierPredictor(model=os.environ.get("SQL_SPECIALIST_MODEL", "claude-haiku-4-5"))
    else:
        raise ValueError(f"Unknown SQL_SPECIALIST_BACKEND: {backend!r} (expected 'ollama' or 'frontier')")
    return _predictor


@mcp.tool()
def nl_to_sql(question: str) -> dict:
    """Answer a natural-language question about the ShopSphere e-commerce
    database. Generates a SQL query, executes it read-only, and returns the
    result rows (capped at 200) along with the SQL used, so the answer is
    grounded in the actual data rather than the model's guess.

    The database covers customers, employees, products, orders, order_items,
    reviews, and support_tickets -- see schema/schema.sql for full column
    definitions. Only read queries are permitted; any write/DDL attempt is
    rejected before execution.
    """
    predictor = get_predictor()
    pred = predictor.predict(question)

    if pred.error:
        log_call(source="mcp_server", predictor=predictor.name, question=question,
                  latency_ms=pred.latency_ms, error=f"predictor error: {pred.error}")
        return {"error": f"predictor error: {pred.error}"}

    try:
        rows = execute_readonly(DB_PATH, pred.sql)
    except UnsafeSQLError as e:
        log_call(source="mcp_server", predictor=predictor.name, question=question, sql=pred.sql,
                  latency_ms=pred.latency_ms, input_tokens=pred.input_tokens,
                  output_tokens=pred.output_tokens, cost_usd=pred.cost_usd,
                  error=f"rejected unsafe SQL: {e}")
        return {"sql": pred.sql, "error": f"rejected unsafe SQL: {e}"}
    except Exception as e:
        log_call(source="mcp_server", predictor=predictor.name, question=question, sql=pred.sql,
                  latency_ms=pred.latency_ms, input_tokens=pred.input_tokens,
                  output_tokens=pred.output_tokens, cost_usd=pred.cost_usd,
                  error=f"execution error: {e}")
        return {"sql": pred.sql, "error": f"execution error: {e}"}

    log_call(source="mcp_server", predictor=predictor.name, question=question, sql=pred.sql,
              latency_ms=pred.latency_ms, input_tokens=pred.input_tokens,
              output_tokens=pred.output_tokens, cost_usd=pred.cost_usd)

    truncated = len(rows) > MAX_ROWS_RETURNED
    return {
        "sql": pred.sql,
        "row_count": len(rows),
        "rows": [list(r) for r in rows[:MAX_ROWS_RETURNED]],
        "truncated": truncated,
    }


@mcp.tool()
def sql_specialist_stats() -> dict:
    """Return observability stats for the currently active predictor:
    call count, average latency, success rate, and total cost so far."""
    from observability.logger import stats
    predictor = get_predictor()
    return stats(predictor.name)


if __name__ == "__main__":
    mcp.run()
