"""LoRA supervised fine-tuning CLI for the terminal-monitor Qwen2.5-0.5B model.

Trains a PEFT/LoRA adapter on **external JSONL chat data** and writes an adapter
directory that :class:`src.agent_llm.terminal_llm.TerminalLoRAInference` and
``src.validation.terminal_monitor_eval --adapter-dir`` load directly.

JSONL formats (one JSON object per line)::

    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "battery_low"}]}

    {"prompt": "<user text>", "response": "battery_low"}

    {"text": "<already formatted full text>"}     # no masking, advanced use

Correctness guarantees:

* The prompt is rendered with the tokenizer's chat template
  (``add_generation_prompt=True``); the completion is appended as the assistant
  turn plus the EOS token.  Prompt and completion are tokenized **separately**
  and concatenated, so the assistant-only label mask is exact and never depends
  on boundary re-tokenisation.
* Loss labels are ``-100`` on every prompt token and on padding, so the model is
  trained on the answer only.
* ``{"text": ...}`` examples are trained on all tokens and are reported as such
  in the run metadata.

Example::

    python -m src.agent_llm.train_lora \\
        --train-jsonl examples/llm/terminal_sft_example.jsonl \\
        --output-dir /tmp/qwen_terminal_lora \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --epochs 3 --batch-size 4 --lr 2e-4

Use ``--validate-only`` to check the JSONL without downloading a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.agent_llm import PLANNER_VERSION

SUPPORTED_ROLES = ("system", "user", "assistant")


class SFTDataError(ValueError):
    """Raised when the JSONL training data is malformed."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_sft_examples(path: Path) -> List[Dict[str, Any]]:
    """Parse and validate an SFT JSONL file into normalised examples."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training JSONL not found: {path}")
    examples: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTDataError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise SFTDataError(f"{path}:{lineno}: line must be a JSON object")
            examples.append(_normalise_example(obj, f"{path}:{lineno}"))
    if not examples:
        raise SFTDataError(f"{path}: no training examples found")
    return examples


def _normalise_example(obj: Dict[str, Any], where: str) -> Dict[str, Any]:
    if "messages" in obj:
        messages = obj["messages"]
        if not isinstance(messages, list) or not messages:
            raise SFTDataError(f"{where}: 'messages' must be a non-empty list")
        clean: List[Dict[str, str]] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise SFTDataError(f"{where}: messages[{i}] must be an object")
            role = msg.get("role")
            content = msg.get("content")
            if role not in SUPPORTED_ROLES:
                raise SFTDataError(
                    f"{where}: messages[{i}].role must be one of {list(SUPPORTED_ROLES)}, "
                    f"got {role!r}"
                )
            if not isinstance(content, str) or not content.strip():
                raise SFTDataError(f"{where}: messages[{i}].content must be a non-empty string")
            clean.append({"role": role, "content": content})
        if clean[-1]["role"] != "assistant":
            raise SFTDataError(f"{where}: the last message must have role 'assistant'")
        if len(clean) < 2:
            raise SFTDataError(f"{where}: need at least one prompt message plus the assistant answer")
        return {"kind": "chat", "messages": clean}

    if "prompt" in obj and "response" in obj:
        prompt, response = obj["prompt"], obj["response"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise SFTDataError(f"{where}: 'prompt' must be a non-empty string")
        if not isinstance(response, str) or not response.strip():
            raise SFTDataError(f"{where}: 'response' must be a non-empty string")
        return {
            "kind": "chat",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        }

    if "text" in obj:
        text = obj["text"]
        if not isinstance(text, str) or not text.strip():
            raise SFTDataError(f"{where}: 'text' must be a non-empty string")
        return {"kind": "text", "text": text}

    raise SFTDataError(
        f"{where}: expected 'messages', or 'prompt'+'response', or 'text'"
    )


def encode_example(
    tokenizer: Any,
    example: Dict[str, Any],
    max_seq_len: int,
    eos_token_id: int,
) -> Dict[str, List[int]]:
    """Tokenize one example with an exact assistant-only label mask."""
    if example["kind"] == "text":
        ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            raise SFTDataError("'text' example tokenized to zero tokens")
        if len(ids) > max_seq_len:
            raise SFTDataError(
                f"'text' example has {len(ids)} tokens > --max-seq-len {max_seq_len}"
            )
        return {"input_ids": ids, "labels": list(ids), "attention_mask": [1] * len(ids)}

    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    answer_text = messages[-1]["content"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise SFTDataError(f"assistant answer {answer_text!r} tokenized to zero tokens")
    if answer_ids[-1] != eos_token_id:
        answer_ids = answer_ids + [eos_token_id]

    ids = list(prompt_ids) + list(answer_ids)
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    if len(ids) > max_seq_len:
        raise SFTDataError(
            f"example has {len(ids)} tokens (prompt {len(prompt_ids)}, answer "
            f"{len(answer_ids)}) > --max-seq-len {max_seq_len}; raise --max-seq-len "
            f"instead of silently truncating the answer"
        )
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def describe_dataset(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    kinds: Dict[str, int] = {}
    n_assistant_msgs = 0
    for ex in examples:
        kinds[ex["kind"]] = kinds.get(ex["kind"], 0) + 1
        if ex["kind"] == "chat":
            n_assistant_msgs += 1
    return {
        "n_examples": len(examples),
        "kinds": kinds,
        "n_masked_examples": n_assistant_msgs,
        "n_unmasked_text_examples": kinds.get("text", 0),
    }


class _SFTDataset:
    """Minimal torch Dataset over pre-tokenized examples (no torch import needed here)."""

    def __init__(self, encoded: List[Dict[str, List[int]]]):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.encoded[idx]


def _make_collator(pad_token_id: int):
    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA SFT for the terminal-monitor Qwen2.5-0.5B model on external JSONL chat data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-jsonl", type=Path, required=True, help="External JSONL training data.")
    p.add_argument("--output-dir", type=Path, required=True, help="Adapter output directory.")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval-jsonl", type=Path, default=None, help="Optional held-out JSONL for a final eval loss.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size.")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        type=str,
        default="all-linear",
        help="Comma-separated module names, or 'all-linear' (PEFT default for Qwen2).",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA 4-bit loading (CUDA + bitsandbytes).")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--validate-only", action="store_true", help="Validate the JSONL and exit (no model download).")
    return p


def _resolve_dtype(torch: Any, dtype: str, use_cuda: bool) -> Any:
    if dtype == "auto":
        return torch.bfloat16 if use_cuda else torch.float32
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    resolved = mapping[dtype]
    if resolved is torch.float16 and not use_cuda:
        raise ValueError("float16 training requires CUDA; use --dtype float32 on CPU")
    return resolved


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _eval_loss(trainer_model: Any, dataloader: Any, device: str) -> Optional[float]:
    import torch

    trainer_model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = trainer_model(**batch).loss
            total += float(loss.item()) * batch["input_ids"].shape[0]
            count += batch["input_ids"].shape[0]
    trainer_model.train()
    return total / count if count else None


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        examples = load_sft_examples(args.train_jsonl)
        summary = describe_dataset(examples)
        eval_examples = load_sft_examples(args.eval_jsonl) if args.eval_jsonl else None
    except (SFTDataError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=== SFT data summary ===")
    print(json.dumps(summary, indent=2))
    if eval_examples is not None:
        print("eval:", json.dumps(describe_dataset(eval_examples)))

    if args.validate_only:
        print("--validate-only: data is well formed; no model was loaded.")
        return 0

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        print(
            f"ERROR: training requires transformers + peft + accelerate "
            f"(pip install -r requirements-llm.txt); missing: {exc.name or exc}",
            file=sys.stderr,
        )
        return 2

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but CUDA is not available", file=sys.stderr)
        return 2
    if args.load_in_4bit and not use_cuda:
        print("ERROR: --load-in-4bit requires CUDA", file=sys.stderr)
        return 2

    torch_dtype = _resolve_dtype(torch, args.dtype, use_cuda)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        print("ERROR: tokenizer has no eos token; cannot build the label mask", file=sys.stderr)
        return 2

    try:
        encoded = [encode_example(tokenizer, ex, args.max_seq_len, eos_token_id) for ex in examples]
        encoded_eval = (
            [encode_example(tokenizer, ex, args.max_seq_len, eos_token_id) for ex in eval_examples]
            if eval_examples
            else None
        )
    except SFTDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = {"": 0}
        except ImportError as exc:
            print(f"ERROR: --load-in-4bit requires bitsandbytes: {exc}", file=sys.stderr)
            return 2

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    target_modules: Any = args.target_modules
    if isinstance(target_modules, str) and target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / max(1, total):.3f}%)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs: Dict[str, Any] = dict(
        output_dir=str(args.output_dir / "trainer_state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        use_cpu=not use_cuda,
        fp16=use_cuda and torch_dtype is torch.float16,
        bf16=use_cuda and torch_dtype is torch.bfloat16,
        optim="adamw_torch",
    )
    try:
        training_args = TrainingArguments(**training_kwargs)
    except TypeError:  # older transformers renamed an argument
        training_kwargs.pop("use_cpu", None)
        training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_SFTDataset(encoded),
        data_collator=_make_collator(tokenizer.pad_token_id),
    )
    train_output = trainer.train()

    eval_loss = None
    if encoded_eval:
        import torch as _torch
        from torch.utils.data import DataLoader

        eval_loader = DataLoader(
            _SFTDataset(encoded_eval),
            batch_size=args.batch_size,
            collate_fn=_make_collator(tokenizer.pad_token_id),
        )
        eval_loss = _eval_loss(model, eval_loader, "cuda" if use_cuda else "cpu")

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "planner_version": PLANNER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "train_jsonl": str(args.train_jsonl),
        "train_jsonl_sha256": _file_sha256(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "dataset": summary,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_seq_len": args.max_seq_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "load_in_4bit": args.load_in_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dtype": str(torch_dtype).replace("torch.", ""),
            "device": "cuda" if use_cuda else "cpu",
        },
        "label_masking": (
            "assistant-only: prompt tokens and padding are -100; "
            "'text' examples are unmasked full-text LM"
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_loss": getattr(train_output, "training_loss", None),
        "eval_loss": eval_loss,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== LoRA training completed ===")
    print(f"adapter_dir: {args.output_dir}")
    print(f"train_loss: {metadata['train_loss']}")
    print(f"eval_loss: {eval_loss}")
    print(
        "Next: python3 -m src.validation.terminal_monitor_eval "
        f"--adapter-dir {args.output_dir} --data-csv <your_windows.csv>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
