"""Window data model, CSV loading, and the synthetic *fixture* generator.

A terminal-monitor sample is an ``(L, 4)`` window over
``[accel_mps2, speed_mps, local_density, battery_pct]`` (Table 2 in the paper).

Real data is loaded from CSV with :func:`load_window_csv`; the prototype
generator :func:`synthesize_fixture_windows` exists only to smoke-test the
code path and is explicitly labelled as a fixture everywhere it surfaces.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FIELDS: Tuple[str, ...] = ("accel_mps2", "speed_mps", "local_density", "battery_pct")
DEFAULT_WINDOW_LEN = 4

EVENT_CLASSES: List[str] = [
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

# Table 2: observation-degradation levels L0-L3.
DEGRADATION_PARAMS: Dict[str, Dict[str, float]] = {
    "L0": {"sigma_a": 0.00, "sigma_b": 0.000, "sigma_v": 0.00, "sigma_rho": 0.00, "lam": 0.00, "p_corr": 0.00},
    "L1": {"sigma_a": 0.08, "sigma_b": 0.003, "sigma_v": 0.08, "sigma_rho": 0.03, "lam": 0.12, "p_corr": 0.03},
    "L2": {"sigma_a": 0.18, "sigma_b": 0.006, "sigma_v": 0.18, "sigma_rho": 0.06, "lam": 0.28, "p_corr": 0.07},
    "L3": {"sigma_a": 0.30, "sigma_b": 0.010, "sigma_v": 0.30, "sigma_rho": 0.10, "lam": 0.45, "p_corr": 0.12},
}

FIXTURE_PROTOTYPES: Dict[str, List[float]] = {
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


@dataclass
class WindowDataset:
    """A labelled window set with an optional explicit train/test split."""

    windows: np.ndarray  # (N, L, 4)
    labels: np.ndarray  # (N,) int indices into EVENT_CLASSES
    sample_ids: List[str]
    splits: Optional[List[str]] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.windows.ndim != 3 or self.windows.shape[2] != len(FIELDS):
            raise ValueError(
                f"windows must have shape (N, L, {len(FIELDS)}), got {self.windows.shape}"
            )
        if len(self.windows) != len(self.labels) or len(self.windows) != len(self.sample_ids):
            raise ValueError("windows, labels, and sample_ids must have the same length")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def subset(self, indices: Sequence[int]) -> "WindowDataset":
        idx = np.asarray(list(indices), dtype=int)
        return WindowDataset(
            windows=self.windows[idx],
            labels=self.labels[idx],
            sample_ids=[self.sample_ids[i] for i in idx],
            splits=[self.splits[i] for i in idx] if self.splits else None,
            source=self.source,
        )

    def split_by(self, split: str) -> "WindowDataset":
        if self.splits is None:
            raise ValueError(f"dataset {self.source!r} has no 'split' column")
        idx = [i for i, s in enumerate(self.splits) if s == split]
        if not idx:
            raise ValueError(f"dataset {self.source!r} has no samples with split={split!r}")
        return self.subset(idx)

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            name = EVENT_CLASSES[int(label)]
            counts[name] = counts.get(name, 0) + 1
        return counts


def label_index(name: str) -> int:
    try:
        return EVENT_CLASSES.index(name)
    except ValueError as exc:
        raise ValueError(
            f"unknown event label {name!r}; expected one of {EVENT_CLASSES}"
        ) from exc


def apply_observation_degradation(
    clean_window: np.ndarray, level: str, rng: np.random.Generator
) -> np.ndarray:
    """Apply the Table-2 post-estimation degradation H_l(x_{k-L+1:k}) to an (L, 4) window."""
    if level not in DEGRADATION_PARAMS:
        raise ValueError(f"unknown degradation level {level!r}; expected {list(DEGRADATION_PARAMS)}")
    cfg = DEGRADATION_PARAMS[level]
    out = clean_window.copy().astype(float)
    steps = out.shape[0]
    bias = 0.0
    nominals = np.array([5.0, 9.6, 0.50, 60.0])
    for k in range(steps):
        bias += rng.normal(0.0, cfg["sigma_b"])
        out[k, 0] = max(0.0, out[k, 0] + rng.normal(0.0, cfg["sigma_a"]) + bias)
        out[k, 1] = max(0.0, out[k, 1] + rng.normal(0.0, cfg["sigma_v"]))
        out[k, 2] = float(np.clip(out[k, 2] + rng.normal(0.0, cfg["sigma_rho"]), 0.0, 1.0))
        out[k] = (1.0 - cfg["lam"]) * out[k] + cfg["lam"] * nominals
        if k > 0 and rng.random() < cfg["p_corr"]:
            field_idx = int(rng.integers(0, 4))
            out[k, field_idx] = out[k - 1, field_idx]
    return out


def synthesize_fixture_windows(n_per_class: int, seed: int) -> WindowDataset:
    """Synthetic prototype windows around the Table-2 nominal classes.

    FIXTURE ONLY: these are hand-set prototypes plus Gaussian jitter.  They
    exercise the pipeline and must never be reported as a paper reproduction.
    """
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ids: List[str] = []
    for cls_idx, cls_name in enumerate(EVENT_CLASSES):
        base = np.array(FIXTURE_PROTOTYPES[cls_name], dtype=float)
        for j in range(n_per_class):
            jitter = rng.normal(0.0, [0.22, 0.35, 0.03, 2.0], size=(DEFAULT_WINDOW_LEN, len(FIELDS)))
            win = np.clip(
                base + jitter,
                [0.0, 0.0, 0.0, 1.0],
                [20.0, 30.0, 1.0, 100.0],
            )
            xs.append(win)
            ys.append(cls_idx)
            ids.append(f"fixture-{cls_name}-{j:03d}")
    return WindowDataset(
        windows=np.asarray(xs, dtype=float),
        labels=np.asarray(ys, dtype=int),
        sample_ids=ids,
        splits=None,
        source=f"fixture_synthetic(prototypes, n_per_class={n_per_class}, seed={seed})",
    )


# ---------------------------------------------------------------------------
# Real-data CSV loading
# ---------------------------------------------------------------------------
def _parse_float(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: expected a number, got {value!r}") from exc


def load_window_csv(path: Path) -> WindowDataset:
    """Load real windows from CSV.

    Long format (recommended)::

        sample_id,step,accel_mps2,speed_mps,local_density,battery_pct,label[,split]

    Wide format::

        sample_id,label[,split],f0,f1,...,f{4L-1}   # row-major step x field

    ``label`` must be one of :data:`EVENT_CLASSES`.  ``split`` is optional; when
    present it must contain ``train`` / ``test`` values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"window CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        records = list(reader)

    if not records:
        raise ValueError(f"window CSV is empty: {path}")

    if "step" in fieldnames:
        return _load_long(records, fieldnames, path)
    if "f0" in fieldnames:
        return _load_wide(records, fieldnames, path)
    raise ValueError(
        f"{path}: unrecognised schema. Long format needs columns "
        f"sample_id,step,{','.join(FIELDS)},label[,split]; wide format needs "
        f"sample_id,label[,split],f0..f(N-1). Found: {fieldnames}"
    )


def _load_long(
    records: List[Dict[str, Any]], fieldnames: List[str], path: Path
) -> WindowDataset:
    for col in ("sample_id", *FIELDS, "label"):
        if col not in fieldnames:
            raise ValueError(f"{path}: missing column {col!r} for long format")
    grouped: Dict[str, List[Tuple[int, List[float]]]] = {}
    labels: Dict[str, int] = {}
    splits: Dict[str, Optional[str]] = {}
    for rec in records:
        sid = str(rec["sample_id"]).strip()
        if not sid:
            raise ValueError(f"{path}: empty sample_id")
        step = int(float(rec["step"]))
        row = [_parse_float(rec[f], f"{path}:{sid}:{f}") for f in FIELDS]
        grouped.setdefault(sid, []).append((step, row))
        label_name = str(rec["label"]).strip()
        idx = label_index(label_name)
        if sid in labels and labels[sid] != idx:
            raise ValueError(f"{path}: sample {sid} has conflicting labels")
        labels[sid] = idx
        split = rec.get("split")
        splits[sid] = str(split).strip() if split not in (None, "") else None

    windows: List[np.ndarray] = []
    sample_ids: List[str] = []
    ys: List[int] = []
    split_list: List[str] = []
    for sid, steps in grouped.items():
        steps.sort(key=lambda item: item[0])
        expected = list(range(len(steps)))
        if [s for s, _ in steps] != expected:
            raise ValueError(
                f"{path}: sample {sid} must have steps 0..{len(steps) - 1} without gaps, "
                f"got {[s for s, _ in steps]}"
            )
        windows.append(np.asarray([row for _, row in steps], dtype=float))
        sample_ids.append(sid)
        ys.append(labels[sid])
        if splits.get(sid):
            split_list.append(splits[sid])  # type: ignore[arg-type]

    lens = {w.shape[0] for w in windows}
    if len(lens) != 1:
        raise ValueError(f"{path}: all samples must share one window length L, found {sorted(lens)}")
    return WindowDataset(
        windows=np.asarray(windows, dtype=float),
        labels=np.asarray(ys, dtype=int),
        sample_ids=sample_ids,
        splits=split_list if len(split_list) == len(sample_ids) else None,
        source=f"csv_long:{path}",
    )


def _load_wide(
    records: List[Dict[str, Any]], fieldnames: List[str], path: Path
) -> WindowDataset:
    f_cols = sorted(
        (name for name in fieldnames if name.startswith("f") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    if not f_cols:
        raise ValueError(f"{path}: wide format requires columns f0..f(N-1)")
    if len(f_cols) % len(FIELDS) != 0:
        raise ValueError(
            f"{path}: {len(f_cols)} feature columns is not a multiple of {len(FIELDS)}"
        )
    window_len = len(f_cols) // len(FIELDS)
    windows: List[np.ndarray] = []
    ys: List[int] = []
    ids: List[str] = []
    splits: List[str] = []
    for rec in records:
        sid = str(rec.get("sample_id", "")).strip()
        if not sid:
            raise ValueError(f"{path}: empty sample_id")
        values = [_parse_float(rec[name], f"{path}:{sid}:{name}") for name in f_cols]
        windows.append(np.asarray(values, dtype=float).reshape(window_len, len(FIELDS)))
        ys.append(label_index(str(rec["label"]).strip()))
        ids.append(sid)
        split = rec.get("split")
        if split not in (None, ""):
            splits.append(str(split).strip())
    return WindowDataset(
        windows=np.asarray(windows, dtype=float),
        labels=np.asarray(ys, dtype=int),
        sample_ids=ids,
        splits=splits if len(splits) == len(ids) else None,
        source=f"csv_wide:{path}",
    )


def split_dataset(
    dataset: WindowDataset, test_fraction: float, seed: int
) -> Tuple[WindowDataset, WindowDataset, str]:
    """Split into train/test, honouring an explicit ``split`` column when present."""
    if dataset.splits is not None:
        return (
            dataset.split_by("train"),
            dataset.split_by("test"),
            "explicit 'split' column (train/test)",
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    n_test = max(1, int(round(len(dataset) * test_fraction)))
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    return (
        dataset.subset(train_idx),
        dataset.subset(test_idx),
        f"random split with seed={seed}, test_fraction={test_fraction}",
    )
