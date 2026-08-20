from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prediction:
    """What a predictor returns for one question."""
    sql: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None  # set if the predictor itself failed (e.g. API error)


@dataclass
class ExampleResult:
    id: str
    question: str
    category: str
    gold_sql: str
    pred: Prediction
    correct: bool
    exec_error: Optional[str] = None  # set if predicted SQL failed to execute / was rejected


@dataclass
class EvalReport:
    predictor_name: str
    results: list = field(default_factory=list)  # list[ExampleResult]

    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correct for r in self.results) / len(self.results)

    def accuracy_by_category(self) -> dict:
        cats = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.correct)
        return {c: sum(v) / len(v) for c, v in sorted(cats.items())}

    def latency_stats_ms(self) -> dict:
        vals = sorted(r.pred.latency_ms for r in self.results)
        if not vals:
            return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
        n = len(vals)
        return {
            "p50": vals[n // 2],
            "p95": vals[min(n - 1, int(n * 0.95))],
            "mean": sum(vals) / n,
        }

    def total_cost_usd(self) -> Optional[float]:
        costs = [r.pred.cost_usd for r in self.results if r.pred.cost_usd is not None]
        if not costs:
            return None
        return sum(costs)

    def cost_per_1k_usd(self) -> Optional[float]:
        total = self.total_cost_usd()
        if total is None or not self.results:
            return None
        return total / len(self.results) * 1000
