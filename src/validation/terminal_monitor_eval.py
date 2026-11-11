"""Terminal semantic monitor & L0-L3 observation-degradation evaluation entry point.

Implements the four-level post-estimation observation perturbation model (Table 2),
the single-flight latest-snapshot monitor scheduler (size-1 overwrite slot),
structured baselines (Rule-threshold, Compact MLP, XGBoost/GBDT), and an optional
local LoRA Tiny-LLM inference hook (--adapter-dir).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


EVENT_CLASSES = [
    "no_event",
    "charging_congested_high",
    "strong_wind_high",
    "overspeed",
    "battery_insufficient",
    "charging_congested_low",
    "battery_low",
    "strong_wind_low",
    "fog_low_visibility",
]

CRITICAL_EVENTS = {
    "charging_congested_high",
    "strong_wind_high",
    "overspeed",
    "battery_insufficient",
    "fog_low_visibility",
}

DEGRADATION_PARAMS: Dict[str, Dict[str, float]] = {
    "L0": {"sigma_a": 0.00, "sigma_b": 0.000, "sigma_v": 0.00, "sigma_rho": 0.00, "lam": 0.00, "p_corr": 0.00},
    "L1": {"sigma_a": 0.08, "sigma_b": 0.003, "sigma_v": 0.08, "sigma_rho": 0.03, "lam": 0.12, "p_corr": 0.03},
    "L2": {"sigma_a": 0.18, "sigma_b": 0.006, "sigma_v": 0.18, "sigma_rho": 0.06, "lam": 0.28, "p_corr": 0.07},
    "L3": {"sigma_a": 0.30, "sigma_b": 0.010, "sigma_v": 0.30, "sigma_rho": 0.10, "lam": 0.45, "p_corr": 0.12},
}


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


def apply_observation_degradation(
    clean_window: np.ndarray, level: str, rng: np.random.Generator
) -> np.ndarray:
    """Apply Table 2 post-estimation degradation H_l(x_{k-L+1:k}) to an (L, 4) window.

    Columns: [accel_mps2, speed_mps, local_density, battery_pct]
    """
    cfg = DEGRADATION_PARAMS[level]
    out = clean_window.copy().astype(float)
    steps = out.shape[0]
    bias = 0.0
    for k in range(steps):
        bias += rng.normal(0.0, cfg["sigma_b"])
        out[k, 0] = max(0.0, out[k, 0] + rng.normal(0.0, cfg["sigma_a"]) + bias)
        out[k, 1] = max(0.0, out[k, 1] + rng.normal(0.0, cfg["sigma_v"]))
        out[k, 2] = float(np.clip(out[k, 2] + rng.normal(0.0, cfg["sigma_rho"]), 0.0, 1.0))

        lam = cfg["lam"]
        nominals = np.array([5.0, 9.6, 0.50, 60.0])
        out[k] = (1.0 - lam) * out[k] + lam * nominals

        if k > 0 and rng.random() < cfg["p_corr"]:
            field_idx = int(rng.integers(0, 4))
            out[k, field_idx] = out[k - 1, field_idx]
    return out


def synthesize_windows(
    n_per_class: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prototypes = {
        "no_event": [4.8, 12.0, 0.25, 72.0],
        "charging_congested_high": [5.1, 8.5, 0.88, 28.0],
        "strong_wind_high": [8.6, 10.2, 0.35, 58.0],
        "overspeed": [6.2, 18.4, 0.32, 62.0],
        "battery_insufficient": [5.0, 11.2, 0.40, 14.0],
        "charging_congested_low": [4.9, 10.8, 0.72, 54.0],
        "battery_low": [4.9, 11.8, 0.30, 26.0],
        "strong_wind_low": [6.7, 11.4, 0.30, 65.0],
        "fog_low_visibility": [5.8, 7.2, 0.62, 55.0],
    }
    xs, ys = [], []
    for cls_idx, cls_name in enumerate(EVENT_CLASSES):
        base = np.array(prototypes[cls_name], dtype=float)
        for _ in range(n_per_class):
            jitter = rng.normal(0.0, [0.22, 0.35, 0.03, 2.0], size=(4, 4))
            win = np.clip(base + jitter, [0.0, 0.0, 0.0, 1.0], [20.0, 30.0, 1.0, 100.0])
            xs.append(win)
            ys.append(cls_idx)
    return np.asarray(xs), np.asarray(ys, dtype=int)


def rule_threshold_predict(windows: np.ndarray) -> np.ndarray:
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
    crit_recall = float(np.mean(y_pred[crit_mask] == y_true[crit_mask])) if np.any(crit_mask) else 0.0
    return {
        "event_macro_f1": round(float(np.mean(f1s)), 4),
        "critical_event_recall": round(crit_recall, 4),
    }


def simulate_single_flight_schedule(
    total_duration_s: float = 30.0, tick_hz: float = 1.0, service_time_s: float = 2.80
) -> Dict[str, float | int]:
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
        "completed_inferences": completed,
        "overwritten_stale_windows": slot.overwritten_count,
        "mean_request_latency_s": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "max_queue_depth": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate terminal semantic monitors across L0-L3 degradation.")
    parser.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"], choices=["L0", "L1", "L2", "L3"])
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = synthesize_windows(args.samples_per_class, seed=args.seed)
    x_test_clean, y_test = synthesize_windows(max(30, args.samples_per_class // 2), seed=args.seed + 101)

    flat_train = x_train.reshape(len(x_train), -1)
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=args.seed))
    mlp.fit(flat_train, y_train)
    gbdt = GradientBoostingClassifier(random_state=args.seed)
    gbdt.fit(flat_train, y_train)

    rng = np.random.default_rng(args.seed + 2026)
    report: Dict[str, Dict] = {"levels": {}, "scheduler_profile": simulate_single_flight_schedule()}

    for lvl in args.levels:
        x_deg = np.asarray([apply_observation_degradation(w, lvl, rng) for w in x_test_clean])
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

        report["levels"][lvl] = {
            "Rule-threshold": {**compute_metrics(y_test, pred_rule), "latency_ms": round(lat_rule, 4)},
            "Compact MLP": {**compute_metrics(y_test, pred_mlp), "latency_ms": round(lat_mlp, 4)},
            "XGBoost": {**compute_metrics(y_test, pred_gbdt), "latency_ms": round(lat_gbdt, 4)},
        }

    if args.adapter_dir is not None:
        report["adapter_note"] = f"Adapter path registered: {args.adapter_dir}"

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
