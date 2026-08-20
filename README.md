# sql-specialist-mcp

A small, fine-tuned open-weight model that answers natural-language questions
about a SQLite database, served as an **MCP tool** any MCP client (Claude
Desktop, Claude Code, custom agents) can call directly — backed by an
execution-accuracy eval harness that benchmarks the specialist against
prompting a frontier model on accuracy, latency, and cost.

The point of this project isn't "build a text-to-SQL demo" — it's to show the
parts of LLM engineering that sit *below* prompting: taking a small model,
adapting it to one task via LoRA, serving it efficiently, and proving with a
real, execution-based eval that a cheap specialist is competitive with (or
better than) prompting a frontier model for this narrow job.

## Why this exists

Most "AI portfolio" text-to-SQL projects are LangChain quickstarts. Two
things here are meant to be different:

1. **The eval is rigorous, not vibes.** Every gold query is executed against
   the database at dataset-build time (139/139 validated), and scoring
   compares *result sets*, not query text — a semantically correct query with
   different column ordering still scores correct. A predictor that just
   echoes the gold SQL scores 100%; a predictor that always returns a
   trivially-wrong query scores 0%. Both are checked in as sanity tests
   (`tests/test_harness_oracle.py`) so the harness's own correctness isn't
   assumed.
2. **It ships as something usable, not just a demo repo.** The fine-tuned
   model is exposed as a real MCP tool (`nl_to_sql`) — point Claude Desktop
   or Claude Code at `mcp_server/server.py` and it can actually query the
   database as part of a conversation.

## What's real here (and what's a documented stand-in)

Being upfront about this matters more than it looks — it's the difference
between a project a recruiter can trust and one that reads like marketing.

**Genuinely solved, not a toy:**
- **The synthetic database and dataset are provably correct.** `shopsphere.db`
  is seeded deterministically (seed=42); every one of the 139 gold
  (question, SQL) pairs in `data/*.jsonl` is generated from parameterized
  templates and *executed against the real database at build time* — a
  template that produces invalid SQL fails the build, it doesn't silently
  ship a bad label.
- **The eval harness's correctness is itself tested**, not assumed —
  `tests/test_harness_oracle.py` asserts an oracle predictor (returns gold
  SQL verbatim) scores exactly 100% and a deliberately-wrong predictor scores
  ~0%, before any real predictor's number is trusted.
- **Execution accuracy, not string match.** `eval/execution.py` compares
  result *sets* (order-insensitive unless the gold query has `ORDER BY`), so
  a query that's differently written but semantically equivalent still
  scores correct.
- **SQL execution is genuinely sandboxed**, not just prompted to behave: read
  queries are validated against a regex allowlist *and* executed against a
  true read-only SQLite connection (`mode=ro` at the OS level) — a bug in the
  regex guard still can't result in a write. This matters beyond the eval
  harness, because the same guard runs in the MCP server, where the SQL
  comes from a model responding to an agent's question, not a curated eval
  set.
- **The MCP server is a real, callable tool**, not a mock — verified
  end-to-end with a stub predictor: tool call → SQL generation → safe
  execution → real rows → logged to observability (see test output in the
  commit history / dev notes).
- **Observability is self-built and dependency-free** — `observability/logger.py`
  logs every call (latency, tokens, estimated cost, success/failure) to a
  local SQLite file, no external account needed, same pattern as
  `pr-review-agent`.

**Documented stand-in — real code, not yet run to completion in this repo:**
- **The fine-tune hasn't been executed here.** `training/finetune.py` is a
  complete, working LoRA fine-tuning pipeline (PEFT + transformers, prompt/label
  construction verified independent of any specific chat template), but
  actually running it needs a GPU with reasonable throughput — CPU/MPS work
  but are slow enough to make a 100+ example run impractical for a quick
  check. See **Fine-tuning** below for the recommended cloud path.
- **The frontier baseline numbers aren't populated.** `eval/baseline_frontier.py`
  is a working predictor against the Claude API (verified to import and parse
  correctly) — it just needs `ANTHROPIC_API_KEY` set to actually run and
  produce numbers. I'm not shipping API credentials in this repo.
- **The comparison report generator is proven, not the comparison itself.**
  `eval/report.py` was smoke-tested end-to-end against oracle/always-wrong
  predictors to confirm the report format and math are correct; the actual
  specialist-vs-frontier numbers require running both predictors for real
  (see **Running the full pipeline** below).

## Architecture

```
data/build_dataset.py ──▶ data/{train,eval}.jsonl   (139 examples, template-generated,
                                                       every gold SQL executed at build time)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                      ▼
training/finetune.py   eval/baseline_frontier.py   tests/test_harness_oracle.py
  (LoRA on a small        (prompt Claude Haiku/       (sanity-checks the harness
   open model)             Sonnet as the baseline)     itself before trusting scores)
        │                     │
        ▼                     │
training/merge_and_quantize.py
        │                     │
        ▼                     ▼
serving/ollama_predictor.py ──┴──▶ eval/harness.py ──▶ eval/report.py ──▶ COMPARISON.md
        │                          (execution-accuracy scoring,
        │                           same logic for every predictor)
        ▼
mcp_server/server.py  (nl_to_sql tool -- installable in Claude Desktop/Code)
        │
        ▼
observability/logger.py  (latency, tokens, cost -- local SQLite, no external account)
```

## Project structure

```
schema/           synthetic "ShopSphere" e-commerce DB (7 tables) + seeded generator
data/              templated gold (question, SQL) dataset -- every query build-time validated
eval/              execution-accuracy harness, frontier baseline, comparison report
training/          LoRA fine-tuning pipeline + LoRA-merge/quantize script
serving/           Ollama-backed and in-process HF predictors, Ollama Modelfile template
mcp_server/        the installable MCP tool (nl_to_sql)
observability/     self-built call logging (latency/tokens/cost), no external account
tests/             harness sanity checks (oracle predictor must score 100%)
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # base: anthropic, mcp, requests
python schema/generate_data.py           # build the seeded database
python data/build_dataset.py             # build + validate the gold dataset
python tests/test_harness_oracle.py      # confirm the eval harness itself is sound
```

`requirements-train.txt` adds torch/transformers/peft/trl for the fine-tuning
path — heavier, kept separate so the eval/serving/MCP path installs fast.

## Running the full pipeline

**1. Frontier baseline** (needs `ANTHROPIC_API_KEY`):
```bash
export ANTHROPIC_API_KEY="..."
python -m eval.baseline_frontier --model claude-haiku-4-5
# writes eval/results_claude-haiku-4-5.json
```

**2. Fine-tune the specialist** (needs a GPU for a real run — see below):
```bash
pip install -r requirements-train.txt
python -m training.finetune --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
python -m training.merge_and_quantize
# then follow the printed llama.cpp + ollama create instructions
```

**3. Score the specialist** the same way as the baseline:
```bash
python -c "
from eval.harness import run_eval, report_to_dict
from serving.ollama_predictor import OllamaPredictor
import json
report = run_eval(OllamaPredictor())
json.dump(report_to_dict(report), open('eval/results_specialist.json', 'w'), indent=2)
"
```

**4. Generate the comparison report:**
```bash
python -m eval.report eval/results_claude-haiku-4-5.json eval/results_specialist.json
```

### Fine-tuning: recommended path

CPU/MPS training works (`training/finetune.py` auto-detects the device) but
is slow enough that a full run isn't practical as a quick check. A single
cloud T4 (e.g. a Colab notebook, or any GPU rental) trains this dataset size
in a few minutes:

```bash
pip install -r requirements-train.txt
python -m training.finetune \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --epochs 3
```

## MCP server

```bash
# Backend defaults to a locally-served model via Ollama:
python -m mcp_server.server

# Or run against a prompted frontier model instead (no fine-tune needed --
# useful for trying the tool before training anything):
SQL_SPECIALIST_BACKEND=frontier SQL_SPECIALIST_MODEL=claude-haiku-4-5 \
  ANTHROPIC_API_KEY=... python -m mcp_server.server
```

Add to Claude Desktop's MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sql-specialist": {
      "command": "/absolute/path/to/sql-specialist-mcp/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/sql-specialist-mcp"
    }
  }
}
```

Then ask Claude something like *"Using the sql-specialist tool, which
customers have never placed an order?"* — it calls `nl_to_sql`, gets back
real rows from the database, and answers grounded in the actual data.

## Security notes

- SQL execution is read-only at two independent layers: a regex guard
  rejecting anything but `SELECT`/`WITH`, and a true OS-level read-only
  SQLite connection (`file:...?mode=ro`) as the backstop.
- The MCP server never executes anything the guard rejects, regardless of
  what the model or the calling agent asked for.
- No secrets are stored in this repo. `ANTHROPIC_API_KEY` is read from the
  environment only.

## What I'd build next

- Populate `COMPARISON.md` with real numbers once a GPU fine-tuning run
  completes (accuracy/latency/cost, specialist vs. Claude Haiku vs. Claude
  Sonnet).
- DPO on the categories where the SFT model's mistakes cluster, once real
  failure data exists.
- vLLM serving path for throughput comparison against the Ollama/GGUF path.
