from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_choice(rng, weights: dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(float(weight) for _, weight in items)
    threshold = rng.random() * total
    partial = 0.0
    for label, weight in items:
        partial += float(weight)
        if partial >= threshold:
            return label
    return items[-1][0]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * fraction
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[lower]
    weight = idx - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def point_in_box(point: tuple[float, float, float], region: dict) -> bool:
    x, y, z = point
    return (
        float(region["x_min"]) <= x <= float(region["x_max"])
        and float(region["y_min"]) <= y <= float(region["y_max"])
        and float(region["z_min"]) <= z <= float(region["z_max"])
    )


def region_label(point: tuple[float, float, float], regions: list[dict]) -> str:
    for region in regions:
        if point_in_box(point, region):
            return str(region["type"])
    return "normal_region"


def random_point_in_region(rng, region: dict) -> tuple[float, float, float]:
    return (
        rng.uniform(float(region["x_min"]), float(region["x_max"])),
        rng.uniform(float(region["y_min"]), float(region["y_max"])),
        rng.uniform(float(region["z_min"]), float(region["z_max"])),
    )


def bounded_random_point(rng, area: dict, altitude_range: list[float]) -> tuple[float, float, float]:
    return (
        rng.uniform(0.0, float(area["width"])),
        rng.uniform(0.0, float(area["height"])),
        rng.uniform(float(altitude_range[0]), float(altitude_range[1])),
    )
