"""Run any predictor against the eval set and score it by execution accuracy.

A predictor is any object with `.predict(question: str) -> Prediction` and a
`.name` attribute. This lets the same harness score the frontier baseline
(eval/baseline_frontier.py) and the fine-tuned specialist (serving/) with
identical scoring logic, which is the whole point: the comparison report is
only meaningful if both sides were measured the same way.
"""
import json
from pathlib import Path

from eval.execution import UnsafeSQLError, execute_readonly, results_match
from eval.types import EvalReport, ExampleResult

HERE = Path(__file__).parent
DB_PATH = HERE.parent / "schema" / "shopsphere.db"


def load_eval_set(path: Path = None) -> list:
    path = path or (HERE.parent / "data" / "eval.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_eval(predictor, eval_set: list = None, db_path: Path = DB_PATH, verbose: bool = True) -> EvalReport:
    eval_set = eval_set if eval_set is not None else load_eval_set()
    report = EvalReport(predictor_name=predictor.name)

    for ex in eval_set:
        pred = predictor.predict(ex["question"])
        correct = False
        exec_error = None

        if pred.error:
            exec_error = f"predictor error: {pred.error}"
        else:
            try:
                gold_rows = execute_readonly(db_path, ex["sql"])
                pred_rows = execute_readonly(db_path, pred.sql)
                correct = results_match(ex["sql"], gold_rows, pred_rows)
            except UnsafeSQLError as e:
                exec_error = f"rejected unsafe SQL: {e}"
            except Exception as e:
                exec_error = f"execution error: {e}"

        result = ExampleResult(
            id=ex["id"], question=ex["question"], category=ex["category"],
            gold_sql=ex["sql"], pred=pred, correct=correct, exec_error=exec_error,
        )
        report.results.append(result)

        if verbose:
            mark = "OK" if correct else "X "
            note = f"  [{exec_error}]" if exec_error else ""
            print(f"[{mark}] {ex['category']:20s} {ex['question'][:60]:60s}{note}")

    if verbose:
        print(f"\n{predictor.name}: {report.accuracy():.1%} exec accuracy "
              f"({sum(r.correct for r in report.results)}/{len(report.results)})")
    return report


def report_to_dict(report: EvalReport) -> dict:
    return {
        "predictor": report.predictor_name,
        "accuracy": report.accuracy(),
        "n": len(report.results),
        "accuracy_by_category": report.accuracy_by_category(),
        "latency_ms": report.latency_stats_ms(),
        "total_cost_usd": report.total_cost_usd(),
        "cost_per_1k_usd": report.cost_per_1k_usd(),
        "failures": [
            {"id": r.id, "question": r.question, "category": r.category,
             "gold_sql": r.gold_sql, "pred_sql": r.pred.sql, "exec_error": r.exec_error}
            for r in report.results if not r.correct
        ],
    }
