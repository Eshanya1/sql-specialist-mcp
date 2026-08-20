"""Baseline predictor: prompt a frontier Claude model with the schema and
question, ask for raw SQL back. This is the thing the fine-tuned specialist
has to beat on cost/latency at comparable accuracy -- see eval/report.py.

Run: python -m eval.baseline_frontier --model claude-haiku-4-5
"""
import argparse
import time

import anthropic

from eval.types import Prediction
from training.prompt_format import SYSTEM_PROMPT, extract_sql

# USD per 1M tokens (input, output). Standard list pricing -- verify current
# rates at https://www.anthropic.com/pricing before trusting cost figures;
# Claude Sonnet 5 in particular had introductory pricing ($2/$10) through
# 2026-08-31, not reflected here.
PRICING_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}

# Appended only to the frontier baseline's prompt, not the shared training
# prompt -- the specialist implicitly learned this column-selection
# convention from its training examples (every gold query selects exactly
# the columns the question asks for, never a defensive SELECT * or extra
# ID/email/timestamp columns "for context"). A frontier model prompted with
# the bare schema has no way to know that convention and, not unreasonably,
# tends to return more columns than asked for -- which execution-accuracy
# scoring (comparing result tuples position-for-position) then marks wrong
# even when the requested data is present and correct. Without this
# addendum, an early run scored the specialist at 92.9% vs frontier's 46.4%
# with a manual failure audit showing nearly every "failure" was an
# otherwise-correct query with extra columns, not a reasoning error --
# a misleading comparison this addendum exists to correct.
COLUMN_DISCIPLINE_ADDENDUM = (
    "\n\nSelect exactly the columns the question asks for, in the order asked, "
    "and nothing else -- no SELECT *, no extra ID/email/timestamp columns "
    "'for context'. If the question asks a yes/no or count question, return "
    "only that count/aggregate column."
)


class FrontierPredictor:
    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model
        self.name = f"frontier:{model}"
        self.client = anthropic.Anthropic()

    def predict(self, question: str) -> Prediction:
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT + COLUMN_DISCIPLINE_ADDENDUM,
                messages=[{"role": "user", "content": question}],
            )
        except Exception as e:
            return Prediction(sql="", latency_ms=(time.perf_counter() - t0) * 1000, error=str(e))

        latency_ms = (time.perf_counter() - t0) * 1000
        text = "".join(b.text for b in resp.content if b.type == "text")
        sql = extract_sql(text)

        cost = None
        if self.model in PRICING_PER_MTOK:
            in_price, out_price = PRICING_PER_MTOK[self.model]
            cost = (resp.usage.input_tokens / 1e6) * in_price + (resp.usage.output_tokens / 1e6) * out_price

        return Prediction(
            sql=sql, latency_ms=latency_ms,
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            cost_usd=cost,
        )


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).parent.parent))
    from eval.harness import run_eval, report_to_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5")
    args = parser.parse_args()

    report = run_eval(FrontierPredictor(args.model))
    out = _P(__file__).parent.parent / "eval" / f"results_{args.model}.json"
    out.write_text(json.dumps(report_to_dict(report), indent=2))
    print(f"\nWrote {out}")
