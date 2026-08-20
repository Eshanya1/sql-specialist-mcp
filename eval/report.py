"""Build the specialist-vs-frontier comparison report from one or more
eval/results_*.json files (written by eval/harness.py-based runners).

Run: python -m eval.report eval/results_claude-haiku-4-5.json eval/results_specialist.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load_reports(paths: list) -> list:
    return [json.loads(Path(p).read_text()) for p in paths]


def render_markdown(reports: list) -> str:
    lines = ["# NL-to-SQL specialist vs. frontier prompting — comparison report", ""]

    lines.append("| Predictor | Accuracy | n | p50 latency (ms) | p95 latency (ms) | Cost / 1k calls |")
    lines.append("|---|---|---|---|---|---|")
    for r in reports:
        lat = r["latency_ms"]
        cost = r.get("cost_per_1k_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "n/a (no cost data)"
        lines.append(
            f"| {r['predictor']} | {r['accuracy']:.1%} | {r['n']} "
            f"| {lat['p50']:.0f} | {lat['p95']:.0f} | {cost_str} |"
        )
    lines.append("")

    all_categories = sorted({c for r in reports for c in r["accuracy_by_category"]})
    if all_categories:
        lines.append("## Accuracy by query category")
        lines.append("")
        lines.append("| Category | " + " | ".join(r["predictor"] for r in reports) + " |")
        lines.append("|---|" + "---|" * len(reports))
        for cat in all_categories:
            row = [f"{r['accuracy_by_category'].get(cat, float('nan')):.0%}"
                   if cat in r["accuracy_by_category"] else "—" for r in reports]
            lines.append(f"| {cat} | " + " | ".join(row) + " |")
        lines.append("")

    for r in reports:
        if r["failures"]:
            lines.append(f"## Failures — {r['predictor']} ({len(r['failures'])})")
            lines.append("")
            for f in r["failures"][:10]:
                lines.append(f"- **{f['category']}** — _{f['question']}_")
                lines.append(f"  - gold: `{f['gold_sql']}`")
                lines.append(f"  - pred: `{f['pred_sql']}`")
                if f["exec_error"]:
                    lines.append(f"  - error: {f['exec_error']}")
            if len(r["failures"]) > 10:
                lines.append(f"  - ... and {len(r['failures']) - 10} more")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python -m eval.report <results1.json> [results2.json ...]")
        sys.exit(1)
    reports = load_reports(paths)
    report_md = render_markdown(reports)
    out = HERE.parent / "COMPARISON.md"
    out.write_text(report_md)
    print(report_md)
    print(f"\nWrote {out}")
