"""Merge a trained LoRA adapter into the base model's weights, producing a
standalone model directory ready for GGUF conversion / Ollama serving.

Quantization itself happens outside this script -- see README "Serving" for
the llama.cpp conversion step, which needs the llama.cpp repo cloned locally
(not pip-installable) and is documented rather than automated here.

Run: python -m training.merge_and_quantize --adapter-dir training/output/sql-specialist-lora
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-dir", default=str(HERE / "output" / "sql-specialist-lora"))
    parser.add_argument("--output-dir", default=str(HERE / "output" / "sql-specialist-merged"))
    args = parser.parse_args()

    print(f"Loading base model {args.base_model}...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading and merging LoRA adapter from {args.adapter_dir}...")
    merged = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = merged.merge_and_unload()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nMerged model saved to {args.output_dir}")
    print("\nNext step (GGUF conversion for Ollama/llama.cpp serving):")
    print("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-quantize")
    print("  pip install -r requirements/requirements-convert_hf_to_gguf.txt")
    print(f"  python convert_hf_to_gguf.py {args.output_dir} --outfile sql-specialist-f16.gguf --outtype f16")
    print("  # q4_k_m is a K-quant, not a direct convert_hf_to_gguf.py --outtype -- it needs the")
    print("  # compiled llama-quantize binary as a second step:")
    print("  ./build/bin/llama-quantize sql-specialist-f16.gguf sql-specialist.gguf q4_k_m")
    print("  cp serving/Modelfile.template Modelfile   # ChatML template matching Qwen2.5's chat format")
    print("  ollama create sql-specialist -f Modelfile")
    print("  # then: serving.OllamaPredictor() works against it directly")


if __name__ == "__main__":
    main()
