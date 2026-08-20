"""Single source of truth for the prompt shape used by training, serving, and
the MCP tool. Train/inference skew (fine-tuning on one prompt format, serving
with another) is a classic way to silently tank a fine-tuned model's accuracy
-- importing this module everywhere sidesteps that class of bug entirely.
"""
from pathlib import Path

SCHEMA_DDL = (Path(__file__).parent.parent / "schema" / "schema.sql").read_text()

SYSTEM_PROMPT = f"""You are a SQL generator for a SQLite database with this schema:

{SCHEMA_DDL}

Given a natural-language question, output ONLY the SQL query that answers it \
-- no explanation, no markdown code fences, no commentary. The query must be \
a single read-only SELECT or WITH statement."""


def build_chat_messages(question: str, sql: str = None) -> list:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql})
    return messages


def extract_sql(text: str) -> str:
    """Strip markdown code fences a model may wrap the SQL in, despite being
    told not to -- cheap to handle, expensive to debug as a silent 0% score."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
