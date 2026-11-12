"""Real local Qwen2.5-0.5B (+PEFT/LoRA) inference for the terminal monitor.

This module is the only place where the terminal-monitor LLM is executed.  It
loads a real base model with ``transformers`` and a real adapter with ``peft``,
runs greedy generation, and parses the emitted label.  There is no fallback
path: a missing adapter, a missing dependency, or an unparsable generation is
reported as such and never replaced by a guess.

Dependencies (see ``requirements-llm.txt``): ``transformers``, ``peft``,
``accelerate``, optional ``bitsandbytes`` for 4-bit loading.  ``torch`` is
already a core dependency.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.agent_llm.terminal_data import (
    EVENT_CLASSES,
    FIELDS,
    WindowDataset,
    label_index,
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """\
You are the terminal semantic monitor of an autonomous UAV fleet.

You receive one short telemetry window of a single UAV. Each step reports four
estimated quantities:
  accel_mps2     longitudinal acceleration estimate (m/s^2)
  speed_mps      ground speed estimate (m/s)
  local_density  local airspace occupancy / charging-congestion estimate in [0, 1]
  battery_pct    remaining battery percentage

Classify the window into exactly one event class from this closed list:
{classes}

Answer with the class name only, copied exactly from the list. Do not add any
explanation, punctuation, or extra words.
""".format(classes="\n".join(f"- {name}" for name in EVENT_CLASSES))

LABEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(name) for name in EVENT_CLASSES) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def render_window(window: np.ndarray) -> str:
    """Render an ``(L, 4)`` window as the exact text seen during training."""
    window = np.asarray(window, dtype=float)
    if window.ndim != 2 or window.shape[1] != len(FIELDS):
        raise ValueError(f"window must have shape (L, {len(FIELDS)}), got {window.shape}")
    lines = ["Telemetry window (oldest to newest):"]
    for step, row in enumerate(window):
        lines.append(
            f"step {step}: accel_mps2={row[0]:.3f} speed_mps={row[1]:.3f} "
            f"local_density={row[2]:.3f} battery_pct={row[3]:.2f}"
        )
    lines.append("Event class:")
    return "\n".join(lines)


def build_messages(window: np.ndarray) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_window(window)},
    ]


def parse_label(text: str) -> Optional[str]:
    """Parse a generated label; returns ``None`` when absent or ambiguous.

    Ambiguity (two different class names in one answer) is treated as a failure
    rather than resolved by ordering.
    """
    if not text:
        return None
    matches = {m.group(1).lower() for m in LABEL_PATTERN.finditer(text)}
    if len(matches) != 1:
        return None
    name = matches.pop()
    for canonical in EVENT_CLASSES:
        if canonical.lower() == name:
            return canonical
    return None


@dataclass
class InferenceResult:
    labels: List[Optional[str]]
    indices: List[int]  # -1 marks an unparsable generation
    raw_outputs: List[str]
    latencies_ms: List[float]
    batch_latencies_ms: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def n_parse_failures(self) -> int:
        return sum(1 for idx in self.indices if idx < 0)

    def as_metrics_block(self) -> Dict[str, Any]:
        arr = np.asarray(self.latencies_ms, dtype=float)
        return {
            "n_samples": self.n_samples,
            "parse_failures": self.n_parse_failures,
            "parse_failure_rate": round(self.n_parse_failures / max(1, self.n_samples), 4),
            "latency_ms_mean": round(float(arr.mean()), 4) if arr.size else None,
            "latency_ms_p95": round(float(np.percentile(arr, 95)), 4) if arr.size else None,
            "batch_size": self.batch_sizes[0] if self.batch_sizes else None,
            "n_batches": len(self.batch_latencies_ms),
        }


def _resolve_torch_dtype(torch: Any, dtype: str, device: str) -> Any:
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype {dtype!r}; expected auto|float32|float16|bfloat16")
    if dtype == "float16" and not device.startswith("cuda"):
        raise ValueError("float16 inference requires a CUDA device; use float32 on CPU")
    return mapping[dtype]


class TerminalLoRAInference:
    """Greedy label generation from a Qwen2.5 base model + LoRA adapter."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_dir: Optional[Path] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = 16,
        batch_size: int = 8,
        trust_remote_code: bool = False,
    ):
        """Load a real base model, and a real LoRA adapter when one is given.

        ``adapter_dir=None`` means base-model-only inference (the paper's
        "Base Tiny-LLM without LoRA" condition); it is never a silent fallback
        for a missing adapter, because a non-existent path raises.
        """
        adapter_path: Optional[Path] = None
        if adapter_dir is not None:
            adapter_path = Path(adapter_dir)
            if not adapter_path.exists():
                raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
            if not (adapter_path / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"{adapter_path} does not look like a PEFT adapter "
                    f"(missing adapter_config.json)"
                )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "local LLM inference requires transformers + peft "
                "(pip install -r requirements-llm.txt); missing: "
                f"{exc.name or exc}"
            ) from exc

        self.torch = torch
        self.base_model_id = base_model
        self.adapter_dir = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = _resolve_torch_dtype(torch, dtype, device)

        quantization_config = None
        if load_in_4bit:
            if not device.startswith("cuda"):
                raise ValueError("--load-in-4bit requires a CUDA device")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "4-bit loading requires transformers with bitsandbytes support"
                ) from exc
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "4-bit loading requires the 'bitsandbytes' package "
                    "(pip install -r requirements-llm.txt)"
                ) from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        tokenizer_source = (
            adapter_path
            if adapter_path is not None and (adapter_path / "tokenizer_config.json").exists()
            else base_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = {"": device}
        else:
            model_kwargs["torch_dtype"] = self.dtype
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if quantization_config is None:
            base.to(device)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            self.model = base
        self.model.eval()

    # -- public API ---------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model_id,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir is not None else None,
            "condition": "lora_adapter" if self.adapter_dir is not None else "base_model_only",
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "decoding": "greedy (do_sample=False, num_beams=1)",
        }

    def predict_windows(self, windows: np.ndarray) -> InferenceResult:
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 3 or windows.shape[2] != len(FIELDS):
            raise ValueError(
                f"windows must have shape (N, L, {len(FIELDS)}), got {windows.shape}"
            )
        labels: List[Optional[str]] = []
        indices: List[int] = []
        raws: List[str] = []
        latencies: List[float] = []
        batch_latencies: List[float] = []
        batch_sizes: List[int] = []

        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    build_messages(window), tokenize=False, add_generation_prompt=True
                )
                for window in batch
            ]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            batch_latencies.append(elapsed_ms)
            batch_sizes.append(len(batch))
            per_sample = elapsed_ms / len(batch)
            for text in texts:
                label = parse_label(text)
                raws.append(text)
                labels.append(label)
                indices.append(-1 if label is None else label_index(label))
                latencies.append(per_sample)

        return InferenceResult(
            labels=labels,
            indices=indices,
            raw_outputs=raws,
            latencies_ms=latencies,
            batch_latencies_ms=batch_latencies,
            batch_sizes=batch_sizes,
        )

    def predict_dataset(self, dataset: WindowDataset) -> InferenceResult:
        return self.predict_windows(dataset.windows)
