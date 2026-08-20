"""Predictor backed by a locally-served model via Ollama (GGUF, quantized).

This is the portable serving path: after training/merge_and_quantize.py and a
GGUF conversion (see its printed instructions), `ollama create sql-specialist
-f Modelfile` makes the fine-tuned model callable through Ollama's REST API
with no Python ML stack required at inference time -- anyone can run the demo
with just Ollama installed, no GPU, no torch/transformers.
"""
import time

import requests

from eval.types import Prediction
from training.prompt_format import build_chat_messages, extract_sql


class OllamaPredictor:
    def __init__(self, model: str = "sql-specialist", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.name = f"ollama:{model}"

    def predict(self, question: str) -> Prediction:
        messages = build_chat_messages(question)
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "options": {"temperature": 0}},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return Prediction(sql="", latency_ms=(time.perf_counter() - t0) * 1000, error=str(e))

        latency_ms = (time.perf_counter() - t0) * 1000
        sql = extract_sql(data["message"]["content"])
        return Prediction(
            sql=sql, latency_ms=latency_ms,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            cost_usd=0.0,  # self-hosted: no per-call API charge (compute/electricity not modeled)
        )
