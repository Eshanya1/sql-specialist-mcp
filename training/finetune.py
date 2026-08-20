"""LoRA fine-tune a small open-weight model on the NL-to-SQL gold dataset.

Prompt/label construction is done manually (apply_chat_template twice -- once
for the full example, once for prompt-only) rather than via trl's
DataCollatorForCompletionOnlyLM, which matches on a hardcoded response-template
string that varies by tokenizer and is easy to get silently wrong. This
approach only assumes apply_chat_template exists and is deterministic, which
holds for any instruct-tuned HF model.

The training loop itself is hand-rolled rather than transformers.Trainer --
a direct forward/backward on this model measures ~1.6s/example even on CPU,
but Trainer's dataloader/scheduling layer was observed to leave the process
idle for minutes between steps with transformers 5.x on Apple Silicon. Fewer
moving parts, easier to reason about, and it's fast: with this loop, a 111-
example / 3-epoch run on a 0.5B model completes in a few minutes on CPU.

CPU/MPS work but are slow -- see README for the recommended cloud-GPU path.
MPS in particular has been observed to run away in memory on this task (a
PyTorch caching-allocator issue with dynamic-shape padding) -- pass
--device cpu if you hit that.

Run: python -m training.finetune --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.prompt_format import build_chat_messages

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def load_examples(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def tokenize_example(ex: dict, tokenizer, max_length: int) -> dict:
    """Tokenize once as a full chat turn and once as prompt-only (with the
    generation-prompt suffix), then mask the prompt-token labels to -100 so
    loss is computed only on the SQL completion."""
    messages = build_chat_messages(ex["question"], ex["sql"])
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "labels": labels}


def collate_batch(batch: list, pad_token_id: int, device: str, fixed_len: int) -> dict:
    """Pad every batch to the same fixed length (max_length), not to the
    longest sequence in the batch. Dynamic per-batch padding was observed to
    grow the CPU caching allocator's memory unboundedly over a run -- each
    distinct (batch_size, seq_len) shape gets its own memory pool that never
    gets released back to the OS, and with shuffled data every batch has a
    different longest sequence. Fixed-length padding means every batch has
    identical tensor shapes, so the allocator reuses one pool instead of
    fragmenting across hundreds of them. Same underlying failure mode as the
    MPS memory blowup (shape-driven allocator fragmentation), different backend."""
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        seq = ex["input_ids"][:fixed_len]
        lab = ex["labels"][:fixed_len]
        pad_len = fixed_len - len(seq)
        input_ids.append(seq + [pad_token_id] * pad_len)
        labels.append(lab + [-100] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(attention_mask, device=device),
    }


def pick_device_and_dtype(force_device: str = None):
    if force_device:
        dtype = torch.bfloat16 if force_device == "cuda" else torch.float32
        return force_device, dtype
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # bf16/fp16 training on MPS is still flaky as of PyTorch 2.x
    return "cpu", torch.float32


@torch.no_grad()
def evaluate(model, examples: list, pad_token_id: int, device: str, batch_size: int, fixed_len: int) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = collate_batch(examples[i:i + batch_size], pad_token_id, device, fixed_len)
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


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
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                         help="Override device auto-detection.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, dtype = pick_device_and_dtype(args.device)
    if device == "cpu":
        print("WARNING: no CUDA or MPS device detected -- training on CPU will be "
              "slow for larger models. See README for the recommended cloud-GPU path.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    train_examples = [tokenize_example(ex, tokenizer, args.max_length) for ex in load_examples(DATA_DIR / "train.jsonl")]
    eval_examples = [tokenize_example(ex, tokenizer, args.max_length) for ex in load_examples(DATA_DIR / "eval.jsonl")]
    print(f"Train examples: {len(train_examples)}  Eval examples: {len(eval_examples)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pad_id = tokenizer.pad_token_id

    micro_step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        random.shuffle(train_examples)
        running_loss, n_logged = 0.0, 0
        for i in range(0, len(train_examples), args.batch_size):
            batch = collate_batch(train_examples[i:i + args.batch_size], pad_id, device, args.max_length)
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            running_loss += out.loss.item()
            n_logged += 1
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if n_logged % 5 == 0 or i + args.batch_size >= len(train_examples):
                elapsed = time.time() - t_start
                print(f"epoch {epoch+1}/{args.epochs}  micro_step {micro_step}  "
                      f"loss {running_loss/n_logged:.4f}  elapsed {elapsed:.0f}s", flush=True)
                running_loss, n_logged = 0.0, 0

        # flush any partial accumulation at epoch end
        if micro_step % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        eval_loss = evaluate(model, eval_examples, pad_id, device, args.batch_size, args.max_length)
        print(f"=== epoch {epoch+1} done  eval_loss {eval_loss:.4f}  total_elapsed {time.time()-t_start:.0f}s ===", flush=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
