"""LoRA fine-tune a small open-weight model on the NL-to-SQL gold dataset.

Prompt/label construction is done manually (apply_chat_template twice -- once
for the full example, once for prompt-only) rather than via trl's
DataCollatorForCompletionOnlyLM, which matches on a hardcoded response-template
string that varies by tokenizer and is easy to get silently wrong. This
approach only assumes apply_chat_template exists and is deterministic, which
holds for any instruct-tuned HF model.

CPU/MPS work but are slow -- see README for the recommended cloud-GPU path.

Run: python -m training.finetune --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments,
)

from training.prompt_format import build_chat_messages

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def load_examples(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_dataset(examples: list, tokenizer, max_length: int) -> Dataset:
    """Tokenize each example once as a full chat turn and once as prompt-only
    (with the generation-prompt suffix), then mask the prompt-token labels to
    -100 so loss is computed only on the SQL completion."""
    input_ids_col, labels_col, attn_col = [], [], []
    for ex in examples:
        messages = build_chat_messages(ex["question"], ex["sql"])
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer(full_text, add_special_tokens=False,
                              truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                                truncation=True, max_length=max_length)["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        input_ids_col.append(full_ids)
        labels_col.append(labels)
        attn_col.append([1] * len(full_ids))

    return Dataset.from_dict({"input_ids": input_ids_col, "labels": labels_col, "attention_mask": attn_col})


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # bf16/fp16 training on MPS is still flaky as of PyTorch 2.x
    return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output-dir", default=str(HERE / "output" / "sql-specialist-lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()

    device, dtype = pick_device_and_dtype()
    if device == "cpu":
        print("WARNING: no CUDA or MPS device detected -- training on CPU will be "
              "very slow for anything beyond a smoke test. See README for the "
              "recommended cloud-GPU path (a single T4 is enough for this dataset size).")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_examples = load_examples(DATA_DIR / "train.jsonl")
    eval_examples = load_examples(DATA_DIR / "eval.jsonl")
    train_ds = build_dataset(train_examples, tokenizer, args.max_length)
    eval_ds = build_dataset(eval_examples, tokenizer, args.max_length)
    print(f"Train examples: {len(train_ds)}  Eval examples: {len(eval_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
