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

**[Try the interactive demo](https://claude.ai/code/artifact/1509953f-d6ee-4a53-972b-60a55cf41731)**
— click through all 28 real eval questions and see the specialist's actual
generated SQL, latency, and result rows next to the frontier baseline. No
install required.

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

## Results

The full pipeline has been run end-to-end on real hardware, both sides: real
LoRA fine-tune, real merge, real GGUF quantization, real Ollama serving,
real eval — and a real frontier baseline against the live Claude API.
Base model: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (chosen for a fast iteration
loop on a laptop; see **Fine-tuning** below for the 1.5B path).

| Predictor | Accuracy | n | p50 latency | p95 latency | Cost / 1k calls |
|---|---|---|---|---|---|
| frontier: Claude Haiku 4.5 (prompted) | 53.6% | 28 | 1055ms | 1884ms | $1.06 |
| **sql-specialist (fine-tuned, quantized, local)** | **92.9%** | 28 | **207ms** | **371ms** | **$0.00** |

**Read this with the caveat, not just the headline.** I manually audited
every one of Claude Haiku's 13 measured "failures" against this eval set:
**zero were SQL logic errors.** All 13 were column-selection or row-order
convention mismatches — e.g. returning `(name, email)` when the gold answer
was just `(name)`, or correct rows in a different order than an `ORDER BY`
the original question never actually specified. The strict execution-accuracy
metric (`eval/execution.py` compares result rows column-for-column) scores
those identically to a genuinely wrong query, which the fine-tuned specialist
never produces because it memorized this dataset's exact conventions from
111 training examples — something a frontier model prompted zero-shot has no
way to know. Full failure-by-failure taxonomy in `COMPARISON.md`.

So: **the accuracy gap is real but partly an artifact of what the eval
rewards**, not purely a reasoning gap. **The latency and cost gap is not an
artifact** — 207ms/local/free vs. 1055ms/$1.06-per-1k-calls is the actual,
unhedged result of running a quantized 0.5B model locally instead of calling
an API, and it's the comparison this project's premise actually rests on.

The specialist's own 2 failures (out of 28) were genuine logic errors, not
formatting mismatches — hallucinating a plausible `orders.total` column that
doesn't exist in this schema, and dropping a table qualifier in one
multi-table `SELECT`. Training converged cleanly over 3 epochs (eval loss
0.060 → 0.048 → **0.008**), and the quantized model (988MB f16 →
**373MB q4_k_m**) serves through Ollama in ~200ms.

## What's real here

Being upfront about this matters more than it looks — it's the difference
between a project a recruiter can trust and one that reads like marketing.

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
- **The fine-tune is real, on this machine, verified converging.** LoRA
  (8.8M trainable params, 1.75% of the model) over 3 epochs, eval loss
  dropping monotonically each epoch. See **Engineering notes** below for two
  real bugs hit and fixed along the way.
- **SQL execution is genuinely sandboxed**, not just prompted to behave: read
  queries are validated against a regex allowlist *and* executed against a
  true read-only SQLite connection (`mode=ro` at the OS level) — a bug in the
  regex guard still can't result in a write. This matters beyond the eval
  harness, because the same guard runs in the MCP server, where the SQL
  comes from a model responding to an agent's question, not a curated eval
  set.
- **The MCP server is a real, callable tool serving the real fine-tuned
  model**, verified end-to-end: `nl_to_sql("Which employees have no manager
  assigned?")` → generates SQL via the quantized model over Ollama → executes
  it read-only → returns real rows → logs latency/cost to observability.
- **Observability is self-built and dependency-free** — `observability/logger.py`
  logs every call (latency, tokens, estimated cost, success/failure) to a
  local SQLite file, no external account needed, same pattern as
  `pr-review-agent`.

- **The frontier baseline is real too** — `eval/baseline_frontier.py` ran
  against the live Claude API (Claude Haiku 4.5), not just imported cleanly.
  Its "failures" turned out to reveal a real eval-methodology finding — see
  **Results** above and `COMPARISON.md` for the full manual failure audit.

## Engineering notes: two real bugs found running this for real

Actually executing the fine-tune (rather than leaving it as "should work in
theory") surfaced two genuine PyTorch memory bugs, both fixed in the current
code:

1. **MPS caching-allocator runaway.** Training on Apple Silicon's MPS backend
   via `transformers.Trainer` caused the process to balloon to 23GB RSS and
   hang, on dynamic per-batch padding — each distinct (batch, seq_len) shape
   gets its own memory pool in PyTorch's MPS allocator, which doesn't return
   freed memory to the OS. Fix: `--device cpu` override in `finetune.py`, and
   more fundamentally, fixed-length padding (below) so this class of bug
   can't recur on any backend.
2. **`Trainer`/`DataLoader` overhead, not the model.** A direct forward+backward
   pass timed at 1.6s/example; the same computation through `transformers.Trainer`
   left the process idle for minutes between logged steps with no
   corresponding compute. Root-caused by isolating the actual model+LoRA
   forward/backward with manual timing before assuming the bug was in model
   code. Fix: replaced `Trainer` with a ~40-line manual training loop
   (`training/finetune.py`) — same LoRA setup, direct control over the batch
   loop, no unexplained overhead. Also switched batch collation from
   dynamic-per-batch to fixed-length padding (every batch shaped identically),
   which independently fixed the allocator-fragmentation pattern from bug #1.

Neither fix is a workaround bolted on top — both are visible in
`training/finetune.py` as the only implementation, not an alternate path.

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

**2. Fine-tune the specialist** (this is what was actually run to produce the
results above — takes ~15 min of active compute on a laptop CPU, though wall
clock varies a lot with system load; a GPU is much faster, see below):
```bash
pip install -r requirements-train.txt
python -m training.finetune --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct --device cpu
python -m training.merge_and_quantize --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct
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

### Fine-tuning: scaling up

The results above use `Qwen2.5-Coder-0.5B-Instruct` on CPU, for a fast local
iteration loop. `training/finetune.py --base-model` accepts any HF causal-LM
repo (or a local directory) — `Qwen2.5-Coder-1.5B-Instruct` is a straightforward
swap for better quality, and a single cloud GPU (a T4 is enough for this
dataset size) trains either size in a couple of minutes instead of ~15:

```bash
pip install -r requirements-train.txt
python -m training.finetune \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --epochs 3
```

`--device {cuda,mps,cpu}` overrides auto-detection. MPS is auto-detected on
Apple Silicon but not recommended for this task yet — see **Engineering
notes** above.

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

- **Normalize the eval for column supersets** — score a prediction correct
  if the gold-requested columns' values are present, rather than requiring
  an exact column-for-column match. This is the fix implied by the failure
  taxonomy in `COMPARISON.md`; it would very likely close most of the
  measured 53.6%→92.9% gap and produce a comparison that isolates actual
  reasoning ability from convention-matching.
- Run `eval/baseline_frontier.py` against Claude Sonnet too, for a
  stronger-model comparison point (Haiku is the cheap/fast tier; Sonnet is
  the "how much does model strength alone close the gap" question).
- Fine-tune `Qwen2.5-Coder-1.5B-Instruct` on a GPU and compare accuracy against
  the 0.5B result (92.9%) to quantify the size/quality tradeoff directly.
- DPO targeting the specialist's two known failure modes (hallucinated
  columns, dropped table qualifiers in multi-joins) now that real failure
  data exists.
- vLLM serving path for throughput comparison against the Ollama/GGUF path.
