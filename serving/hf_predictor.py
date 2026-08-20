"""In-process predictor over the merged HF model -- for sanity-testing the
fine-tune immediately after training.merge_and_quantize, before bothering
with GGUF conversion. Not the deployment path (that's OllamaPredictor); this
is the fast feedback loop while iterating on the fine-tune itself.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.types import Prediction
from training.prompt_format import build_chat_messages, extract_sql


class HFLocalPredictor:
    def __init__(self, model_dir: str):
        self.name = f"hf-local:{model_dir}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device)
        self.device = device

    def predict(self, question: str) -> Prediction:
        messages = build_chat_messages(question)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        sql = extract_sql(text)

        return Prediction(
            sql=sql, latency_ms=latency_ms,
            input_tokens=inputs["input_ids"].shape[1], output_tokens=len(new_tokens),
            cost_usd=0.0,
        )
