"""Generate the data backing the shareable demo artifact: for every eval
example, capture the specialist's real generated SQL, real latency, and the
real result rows returned by executing it -- so the demo replays actual
recorded runs, not fabricated examples. Same "cassette" approach as
pr-review-agent's demo.

Run: python -m eval.generate_demo_data
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.execution import execute_readonly
from eval.harness import DB_PATH, load_eval_set
from serving.ollama_predictor import OllamaPredictor

MAX_ROWS = 6


def main():
    predictor = OllamaPredictor()
    eval_set = load_eval_set()
    records = []

    for ex in eval_set:
        pred = predictor.predict(ex["question"])
        record = {
            "id": ex["id"],
            "category": ex["category"],
            "question": ex["question"],
            "gold_sql": ex["sql"],
            "pred_sql": pred.sql,
            "latency_ms": round(pred.latency_ms, 1),
        }
        try:
            gold_rows = execute_readonly(DB_PATH, ex["sql"])
            pred_rows = execute_readonly(DB_PATH, pred.sql)
            from eval.execution import results_match
            record["correct"] = results_match(ex["sql"], gold_rows, pred_rows)
            record["result_rows"] = [list(r) for r in pred_rows[:MAX_ROWS]]
            record["result_row_count"] = len(pred_rows)
            record["result_truncated"] = len(pred_rows) > MAX_ROWS
        except Exception as e:
            record["correct"] = False
            record["result_rows"] = []
            record["result_row_count"] = 0
            record["result_truncated"] = False
            record["error"] = str(e)

        records.append(record)
        mark = "OK" if record["correct"] else "X "
        print(f"[{mark}] {ex['category']:20s} {ex['question'][:60]}")

    out = Path(__file__).parent / "demo_data.json"
    out.write_text(json.dumps(records, indent=2))
    n_correct = sum(r["correct"] for r in records)
    print(f"\n{n_correct}/{len(records)} correct. Wrote {out}")


if __name__ == "__main__":
    main()
