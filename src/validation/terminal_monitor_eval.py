"""Terminal semantic monitor & L0-L3 observation-degradation evaluation entry point.

Implements the four-level post-estimation observation perturbation model (Table 2),
the single-flight latest-snapshot monitor scheduler (size-1 overwrite slot),
explicitly named non-LLM baselines (rule threshold, compact MLP, GBDT), and the
real local Qwen2.5-0.5B (+PEFT/LoRA) inference path behind ``--adapter-dir``.

Truthfulness contract
---------------------
* ``--adapter-dir`` really loads ``transformers`` + ``peft`` and really scores
  the adapter's generations.  A missing/invalid adapter or an unparsable
  generation is reported (``parse_failure_rate``), never replaced by a guess.
* The built-in labelled windows are a **fixture** for smoke testing and are
  labelled ``fixture_synthetic`` in the report; they are not a paper
  reproduction.  Pass ``--data-csv`` to score real windows.
* The scheduler service time is taken from measured LLM inference latency, or
  replayed from a latency JSON file, or explicitly marked
  ``assumed_constant_not_measured``.  A replayed file is classified by its own
  provenance: ``replay_fixture_not_measured`` for example/fixture files and
  ``replay_user_supplied_measured`` for real external measurements (which this
  script does not verify).  The 2.8 s default is never reported as a measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agent_llm import PLANNER_VERSION
from src.agent_llm.terminal_data import (
    CRITICAL_EVENTS,
    EVENT_CLASSES,
    WindowDataset,
    apply_observation_degradation,
    load_window_csv,
    split_dataset,
    synthesize_fixture_windows,
)

# Baseline method identifiers.  These are classical/statistical baselines and
# are named as such so they can never be mistaken for the LLM monitor.
RULE_METHOD = "baseline_rule_threshold"
MLP_METHOD = "baseline_mlp"
GBDT_METHOD = "baseline_gbdt"
BASELINE_METHODS = (RULE_METHOD, MLP_METHOD, GBDT_METHOD)
DISPLAY_NAMES = {
    RULE_METHOD: "Rule-threshold baseline (non-LLM)",
    MLP_METHOD: "Compact MLP baseline (non-LLM)",
    GBDT_METHOD: "Gradient-boosted trees baseline (non-LLM)",
}

FIXTURE_NOTE = (
    "Synthetic prototype windows generated from hand-set class prototypes plus "
    "Gaussian jitter. FIXTURE ONLY: suitable for interface/CLI smoke tests, "
    "NOT a paper reproduction. Pass --data-csv with real windows for measurements."
)


@dataclass
class LatestSnapshotSlot:
    """Single-slot non-accumulating state window buffer (overwrite-while-busy)."""

    window: Optional[np.ndarray] = None
    timestamp_s: float = 0.0
    overwritten_count: int = 0

    def push(self, window: np.ndarray, timestamp_s: float) -> None:
        if self.window is not None:
            self.overwritten_count += 1
        self.window = window.copy()
        self.timestamp_s = timestamp_s

    def pop_latest(self) -> Tuple[Optional[np.ndarray], float]:
        w, ts = self.window, self.timestamp_s
        self.window = None
        return w, ts


def rule_threshold_predict(windows: np.ndarray) -> np.ndarray:
    """Deterministic threshold baseline over the window mean."""
    preds = []
    for w in windows:
        mean_a, mean_v, mean_rho, mean_bat = w.mean(axis=0)
        if mean_bat < 19.0:
            preds.append(EVENT_CLASSES.index("battery_insufficient"))
        elif mean_v > 16.2:
            preds.append(EVENT_CLASSES.index("overspeed"))
        elif mean_a > 7.6:
            preds.append(EVENT_CLASSES.index("strong_wind_high"))
        elif mean_rho > 0.80 and mean_bat < 38.0:
            preds.append(EVENT_CLASSES.index("charging_congested_high"))
        elif mean_v < 8.0 and mean_rho > 0.55:
            preds.append(EVENT_CLASSES.index("fog_low_visibility"))
        elif mean_rho > 0.66:
            preds.append(EVENT_CLASSES.index("charging_congested_low"))
        elif mean_bat < 32.0:
            preds.append(EVENT_CLASSES.index("battery_low"))
        elif mean_a > 6.2:
            preds.append(EVENT_CLASSES.index("strong_wind_low"))
        else:
            preds.append(EVENT_CLASSES.index("no_event"))
    return np.asarray(preds, dtype=int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Per-class macro F1 and critical-event recall.

    Predictions of ``-1`` (unparsable LLM generation) never match a class, so
    they are counted as errors instead of being dropped.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    f1s = []
    for c in range(len(EVENT_CLASSES)):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    crit_indices = {EVENT_CLASSES.index(name) for name in CRITICAL_EVENTS}
    crit_mask = np.isin(y_true, list(crit_indices))
    crit_recall = (
        float(np.mean(y_pred[crit_mask] == y_true[crit_mask])) if np.any(crit_mask) else 0.0
    )
    return {
        "event_macro_f1": round(float(np.mean(f1s)), 4),
        "critical_event_recall": round(crit_recall, 4),
    }


def simulate_single_flight_schedule(
    total_duration_s: float = 30.0,
    tick_hz: float = 1.0,
    service_time_s: float = 2.80,
    service_time_source: str = "assumed_constant_not_measured",
) -> Dict[str, Any]:
    """Single-flight monitor queue: a busy worker drops to the latest snapshot.

    ``service_time_s`` is a *parameter*; the caller is responsible for passing a
    measured value (see ``--latency-replay-json`` / measured adapter latency) or
    for labelling it as an assumption.
    """
    if service_time_s <= 0:
        raise ValueError(f"service_time_s must be positive, got {service_time_s}")
    slot = LatestSnapshotSlot()
    dt = 1.0 / tick_hz
    t = 0.0
    worker_free_at = 0.0
    completed = 0
    latencies: List[float] = []

    while t < total_duration_s - 1e-9:
        slot.push(np.zeros((4, 4)), timestamp_s=t)
        if t >= worker_free_at - 1e-9 and slot.window is not None:
            _, snap_ts = slot.pop_latest()
            worker_free_at = t + service_time_s
            if worker_free_at <= total_duration_s + service_time_s:
                completed += 1
                latencies.append((t - snap_ts) + service_time_s)
        t = round(t + dt, 6)

    return {
        "duration_s": total_duration_s,
        "tick_hz": tick_hz,
        "service_time_s": service_time_s,
        "service_time_source": service_time_source,
        "completed_inferences": completed,
        "overwritten_stale_windows": slot.overwritten_count,
        "mean_request_latency_s": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "max_queue_depth": 1,
    }


_FIXTURE_HINTS = (
    "example",
    "illustrative",
    "fixture",
    "synthetic",
    "placeholder",
    "dummy",
    "toy",
    "fake",
    "assumed",
)
_FIXTURE_PROVENANCE = {"fixture", "example", "synthetic", "not_measured", "assumed", "placeholder"}
_MEASURED_PROVENANCE = {"measured", "user_supplied_measured", "user-supplied", "external_measured"}


def _classify_latency_provenance(payload: Dict[str, Any], path: Path) -> Tuple[str, str, bool]:
    """Classify a replayed latency file as fixture vs user-supplied measurement.

    Returns ``(service_time_source, note, is_measurement)``.  We never claim to
    have measured values that arrived from a file, and we refuse to dress up an
    example/fixture file as a measurement.
    """
    declared = str(payload.get("source") or "").lower()
    explicit = str(payload.get("provenance") or "").strip().lower()
    is_fixture = (
        explicit in _FIXTURE_PROVENANCE
        or payload.get("fixture") is True
        or any(hint in declared for hint in _FIXTURE_HINTS)
    )
    if is_fixture:
        return (
            "replay_fixture_not_measured",
            (
                f"Latency values in {path} are declared/identified as FIXTURE or "
                f"example data, not measurements. The scheduler value is illustrative "
                f"only and must not be reported as a measured latency."
            ),
            False,
        )
    if explicit and explicit not in _MEASURED_PROVENANCE:
        raise ValueError(
            f"{path}: unrecognised 'provenance' {explicit!r}; use 'measured' for real "
            f"external measurements or 'fixture' for example data"
        )
    return (
        "replay_user_supplied_measured",
        (
            f"Latency values were supplied by the user in {path}; this script did not "
            f"measure them and does not verify their provenance."
        ),
        True,
    )


def _load_latency_replay(path: Path) -> Tuple[float, Dict[str, Any]]:
    """Read replayed per-sample latencies from JSON.

    Accepted shapes::

        {"latencies_ms": [...], "source": "...", "provenance": "measured"}
        {"service_time_ms": 123.4, "source": "...", "provenance": "fixture"}

    ``provenance`` is optional but recommended.  Without it, a ``fixture: true``
    flag or example/illustrative wording in ``source`` marks the file as a
    fixture; otherwise it is treated as user-supplied (unverified) measurement.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"latency replay file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: latency replay must be a JSON object")

    source, note, is_measurement = _classify_latency_provenance(payload, path)

    if "latencies_ms" in payload:
        values = np.asarray(payload["latencies_ms"], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{path}: 'latencies_ms' must be positive finite numbers")
        return float(values.mean()) / 1000.0, {
            "service_time_source": source,
            "is_measurement": is_measurement,
            "source_file": str(path),
            "n_measurements": int(values.size),
            "latency_ms_mean": round(float(values.mean()), 4),
            "latency_ms_p95": round(float(np.percentile(values, 95)), 4),
            "declared_source": payload.get("source"),
            "declared_provenance": payload.get("provenance"),
            "note": note,
        }
    if "service_time_ms" in payload:
        value = float(payload["service_time_ms"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: 'service_time_ms' must be positive")
        return value / 1000.0, {
            "service_time_source": source,
            "is_measurement": is_measurement,
            "source_file": str(path),
            "n_measurements": payload.get("n_measurements"),
            "latency_ms_mean": round(value, 4),
            "latency_ms_p95": payload.get("latency_ms_p95_ms"),
            "declared_source": payload.get("source"),
            "declared_provenance": payload.get("provenance"),
            "note": note,
        }
    raise ValueError(
        f"{path}: expected 'latencies_ms' or 'service_time_ms', found keys {sorted(payload)}"
    )


def _load_datasets(args: argparse.Namespace) -> Tuple[WindowDataset, WindowDataset, Dict[str, Any]]:
    if args.data_csv is not None:
        dataset = load_window_csv(args.data_csv)
        train, test, split_note = split_dataset(dataset, args.test_fraction, args.seed)
        data_source: Dict[str, Any] = {
            "kind": "user_csv",
            "fixture": False,
            "path": str(args.data_csv),
            "n_samples": len(dataset),
            "n_train": len(train),
            "n_test": len(test),
            "split": split_note,
            "class_counts": dataset.class_counts(),
        }
        return train, test, data_source

    train = synthesize_fixture_windows(args.samples_per_class, seed=args.seed)
    test = synthesize_fixture_windows(max(30, args.samples_per_class // 2), seed=args.seed + 101)
    data_source = {
        "kind": "fixture_synthetic",
        "fixture": True,
        "note": FIXTURE_NOTE,
        "n_train": len(train),
        "n_test": len(test),
        "seed": args.seed,
        "class_counts": train.class_counts(),
    }
    return train, test, data_source


def _method_slug(base_model: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in base_model).strip("-")
    return slug.lower() or "llm"


def _run_llm_condition(
    display_name: str,
    condition: str,
    base_model: str,
    adapter_dir: Optional[Path],
    args: argparse.Namespace,
    test: WindowDataset,
    levels: List[str],
    degradation_seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[float]]:
    """Load a real local model and score every degradation level."""
    from src.agent_llm.terminal_llm import TerminalLoRAInference

    monitor = TerminalLoRAInference(
        base_model=base_model,
        adapter_dir=adapter_dir,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    info = {"display_name": display_name, "method_class": condition, **monitor.describe()}
    per_level: Dict[str, Any] = {}
    all_latencies_ms: List[float] = []

    for level in levels:
        rng = np.random.default_rng(degradation_seed)
        windows = np.asarray(
            [apply_observation_degradation(w, level, rng) for w in test.windows], dtype=float
        )
        result = monitor.predict_windows(windows)
        pred = np.asarray(result.indices, dtype=int)
        metrics = compute_metrics(test.labels, pred)
        per_level[level] = {
            "display_name": display_name,
            "method_class": condition,
            **metrics,
            "latency_ms": round(float(np.mean(result.latencies_ms)), 4),
            "latency_ms_p95": round(float(np.percentile(result.latencies_ms, 95)), 4),
            "parse_failures": result.n_parse_failures,
            "parse_failure_rate": round(result.n_parse_failures / max(1, result.n_samples), 4),
            "n_samples": result.n_samples,
            "generation_examples": result.raw_outputs[:3],
        }
        all_latencies_ms.extend(result.latencies_ms)

    info["levels"] = list(levels)
    return per_level, info, all_latencies_ms


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate terminal semantic monitors across L0-L3 degradation. "
            "Baselines are non-LLM; the LLM monitor is real local Qwen2.5-0.5B "
            "(+PEFT/LoRA) inference when --adapter-dir is given."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--levels",
        nargs="+",
        default=["L0", "L1", "L2", "L3"],
        choices=["L0", "L1", "L2", "L3"],
    )
    p.add_argument(
        "--samples-per-class",
        type=int,
        default=80,
        help="Fixture samples per class (ignored with --data-csv).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="Real windows CSV (long or wide format); replaces the labelled fixture.",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Test fraction when the CSV has no 'split' column.",
    )
    # Local LLM condition
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="PEFT/LoRA adapter directory (really loaded and scored).",
    )
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--include-base-model",
        action="store_true",
        help="Also score the base model without the adapter.",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--load-in-4bit", action="store_true", help="4-bit inference (CUDA + bitsandbytes).")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--llm-method-name", type=str, default=None, help="Override the LLM method id in the report.")
    # Scheduler latency provenance
    p.add_argument(
        "--latency-replay-json",
        type=Path,
        default=None,
        help=(
            "Latency JSON ({'latencies_ms': [...]} or {'service_time_ms': ...}) with "
            "optional 'provenance': 'measured' (real external data, unverified) or "
            "'fixture' (example data). Example/fixture files are reported as "
            "replay_fixture_not_measured."
        ),
    )
    p.add_argument(
        "--assumed-service-time-s",
        type=float,
        default=2.80,
        help=(
            "Fallback constant used only when no measured latency is available; "
            "always labelled as an assumption in the report."
        ),
    )
    p.add_argument("--schedule-duration", type=float, default=30.0)
    p.add_argument("--tick-hz", type=float, default=1.0)
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        train, test, data_source = _load_datasets(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    x_train = train.windows.reshape(len(train), -1)
    y_train = train.labels

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=args.seed),
    )
    mlp.fit(x_train, y_train)
    gbdt = GradientBoostingClassifier(random_state=args.seed)
    gbdt.fit(x_train, y_train)

    report: Dict[str, Any] = {
        "meta": {
            "planner_version": PLANNER_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "report_schema": "terminal_monitor_eval/2",
            "data_source": data_source,
            "baseline_method_ids": list(BASELINE_METHODS),
            "baselines": {
                RULE_METHOD: DISPLAY_NAMES[RULE_METHOD],
                MLP_METHOD: DISPLAY_NAMES[MLP_METHOD],
                GBDT_METHOD: DISPLAY_NAMES[GBDT_METHOD],
            },
            "notes": [
                "Non-LLM methods are explicitly named baselines; they are not the LLM monitor.",
                "Unparsable LLM generations are scored as errors and reported via parse_failure_rate.",
            ],
        },
        "levels": {},
        "llm_conditions": {},
    }
    if data_source.get("fixture"):
        report["meta"]["notes"].append(FIXTURE_NOTE)

    degradation_seed = args.seed + 2026
    for level in args.levels:
        rng = np.random.default_rng(degradation_seed)
        x_deg = np.asarray(
            [apply_observation_degradation(w, level, rng) for w in test.windows], dtype=float
        )
        flat_deg = x_deg.reshape(len(x_deg), -1)

        t0 = time.perf_counter()
        pred_rule = rule_threshold_predict(x_deg)
        lat_rule = (time.perf_counter() - t0) * 1000.0 / len(x_deg)

        t0 = time.perf_counter()
        pred_mlp = mlp.predict(flat_deg)
        lat_mlp = (time.perf_counter() - t0) * 1000.0 / len(x_deg)

        t0 = time.perf_counter()
        pred_gbdt = gbdt.predict(flat_deg)
        lat_gbdt = (time.perf_counter() - t0) * 1000.0 / len(x_deg)

        report["levels"][level] = {
            RULE_METHOD: {
                "display_name": DISPLAY_NAMES[RULE_METHOD],
                "method_class": "non_llm_baseline",
                **compute_metrics(test.labels, pred_rule),
                "latency_ms": round(lat_rule, 4),
            },
            MLP_METHOD: {
                "display_name": DISPLAY_NAMES[MLP_METHOD],
                "method_class": "non_llm_baseline",
                **compute_metrics(test.labels, pred_mlp),
                "latency_ms": round(lat_mlp, 4),
            },
            GBDT_METHOD: {
                "display_name": DISPLAY_NAMES[GBDT_METHOD],
                "method_class": "non_llm_baseline",
                **compute_metrics(test.labels, pred_gbdt),
                "latency_ms": round(lat_gbdt, 4),
            },
        }

    measured_latencies_ms: List[float] = []
    try:
        if args.adapter_dir is not None:
            llm_method_id = args.llm_method_name or f"llm_lora_{_method_slug(args.base_model)}"
            per_level, info, latencies = _run_llm_condition(
                f"LoRA Qwen2.5-0.5B adapter ({args.adapter_dir})",
                "llm_local_adapter",
                args.base_model,
                args.adapter_dir,
                args,
                test,
                args.levels,
                degradation_seed,
            )
            for level, block in per_level.items():
                report["levels"][level][llm_method_id] = block
            report["llm_conditions"][llm_method_id] = info
            measured_latencies_ms.extend(latencies)

        if args.include_base_model:
            base_method_id = f"llm_base_{_method_slug(args.base_model)}"
            per_level, info, latencies = _run_llm_condition(
                "Base Qwen2.5-0.5B without LoRA",
                "llm_base_model_only",
                args.base_model,
                None,
                args,
                test,
                args.levels,
                degradation_seed,
            )
            for level, block in per_level.items():
                report["levels"][level][base_method_id] = block
            report["llm_conditions"][base_method_id] = info
            measured_latencies_ms.extend(latencies)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: LLM condition failed: {exc}", file=sys.stderr)
        print(
            "No report was written; refusing to report a partial/fabricated LLM result.",
            file=sys.stderr,
        )
        return 2

    # ---- scheduler service time provenance ---------------------------------
    if measured_latencies_ms:
        arr = np.asarray(measured_latencies_ms, dtype=float)
        service_time_s = float(arr.mean()) / 1000.0
        scheduler_meta: Dict[str, Any] = {
            "service_time_source": "measured_llm_inference",
            "n_measurements": int(arr.size),
            "latency_ms_mean": round(float(arr.mean()), 4),
            "latency_ms_p95": round(float(np.percentile(arr, 95)), 4),
            "note": (
                "Service time is the mean per-sample wall-clock latency of the "
                "measured LLM condition(s) above."
            ),
        }
    elif args.latency_replay_json is not None:
        try:
            service_time_s, replay_meta = _load_latency_replay(args.latency_replay_json)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        scheduler_meta = dict(replay_meta)
    else:
        service_time_s = float(args.assumed_service_time_s)
        scheduler_meta = {
            "service_time_source": "assumed_constant_not_measured",
            "n_measurements": 0,
            "note": (
                f"{service_time_s:.3f} s is an ASSUMED constant for the scheduling "
                f"simulation, NOT a measurement. Run with --adapter-dir (real "
                f"inference) or --latency-replay-json to use measured latency."
            ),
        }

    report["scheduler_profile"] = simulate_single_flight_schedule(
        total_duration_s=args.schedule_duration,
        tick_hz=args.tick_hz,
        service_time_s=service_time_s,
        service_time_source=scheduler_meta["service_time_source"],
    )
    report["scheduler_profile"]["provenance"] = scheduler_meta

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
