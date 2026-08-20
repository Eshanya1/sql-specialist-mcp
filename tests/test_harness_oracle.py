"""Sanity check: a predictor that returns the gold SQL verbatim must score
100% exec accuracy. If it doesn't, the comparison logic itself is broken --
this has to pass before any real predictor's score can be trusted.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.harness import load_eval_set, run_eval
from eval.types import Prediction


class OraclePredictor:
    name = "oracle (returns gold SQL)"

    def __init__(self, eval_set):
        self._gold = {ex["id"]: ex["sql"] for ex in eval_set}
        self._by_question = {ex["question"]: ex["sql"] for ex in eval_set}

    def predict(self, question: str) -> Prediction:
        t0 = time.perf_counter()
        sql = self._by_question[question]
        return Prediction(sql=sql, latency_ms=(time.perf_counter() - t0) * 1000)


class WrongPredictor:
    """Always returns a query that is valid SQL but wrong -- must score 0%."""
    name = "always-wrong"

    def predict(self, question: str) -> Prediction:
        return Prediction(sql="SELECT COUNT(*) FROM customers WHERE 1=0", latency_ms=0.1)


if __name__ == "__main__":
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval examples\n")

    report = run_eval(OraclePredictor(eval_set), eval_set, verbose=False)
    acc = report.accuracy()
    print(f"Oracle accuracy: {acc:.1%}")
    assert acc == 1.0, f"Oracle must score 100%, got {acc:.1%} -- harness comparison logic is broken"

    wrong_report = run_eval(WrongPredictor(), eval_set, verbose=False)
    wrong_acc = wrong_report.accuracy()
    print(f"Always-wrong accuracy: {wrong_acc:.1%}")
    assert wrong_acc < 0.05, f"Always-wrong should score ~0%, got {wrong_acc:.1%}"

    print("\nHarness sanity checks passed.")
