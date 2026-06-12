from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from common import bounded_random_point, load_json, random_point_in_region, region_label, write_csv


TRACE_FIELDS = ["time", "uav_id", "x", "y", "z", "speed", "heading", "region_label"]


def _select_candidate(rng, scenario: dict, regions: list[dict], current: tuple[float, float, float]) -> tuple[float, float, float]:
    canyon_regions = [region for region in regions if region["type"] == "canyon_region"]
    hotspot_regions = [region for region in regions if region["type"] == "hotspot_region"]
    draw = rng.random()
    canyon_prob = float(scenario.get("canyon_entry_probability", 0.0))
    hotspot_prob = float(scenario.get("hotspot_crossing_probability", 0.0))
    if canyon_regions and draw < canyon_prob:
        return random_point_in_region(rng, rng.choice(canyon_regions))
    if hotspot_regions and draw < canyon_prob + hotspot_prob:
        return random_point_in_region(rng, rng.choice(hotspot_regions))
    return bounded_random_point(rng, scenario["area"], scenario["altitude_range"])


def _clamp_target(current: tuple[float, float, float], candidate: tuple[float, float, float], max_distance: float) -> tuple[float, float, float]:
    dx = candidate[0] - current[0]
    dy = candidate[1] - current[1]
    dz = candidate[2] - current[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance <= max_distance or distance == 0:
        return candidate
    scale = max_distance / distance
    return (
        current[0] + dx * scale,
        current[1] + dy * scale,
        current[2] + dz * scale,
    )


def build_trace_rows(scenario: dict, region_config: dict) -> list[dict]:
    rng = random.Random(int(scenario.get("seed", 0)))
    duration = float(scenario["duration_s"])
    step = float(scenario.get("trace_step_s", 1.0))
    max_waypoints = int(scenario.get("max_waypoints", 6))
    segment_duration = duration / max_waypoints
    min_speed = float(scenario["speed_mps"]["min"])
    max_speed = float(scenario["speed_mps"]["max"])
    regions = region_config.get("regions", [])
    rows: list[dict] = []

    for uav_id in range(int(scenario["uav_count"])):
        current = bounded_random_point(rng, scenario["area"], scenario["altitude_range"])
        waypoints: list[tuple[float, tuple[float, float, float], float]] = [(0.0, current, rng.uniform(min_speed, max_speed))]
        for segment_idx in range(max_waypoints):
            speed = rng.uniform(min_speed, max_speed)
            candidate = _select_candidate(rng, scenario, regions, current)
            target = _clamp_target(current, candidate, speed * segment_duration)
            current = target
            waypoints.append(((segment_idx + 1) * segment_duration, current, speed))

        waypoint_index = 0
        sample_count = int(round(duration / step))
        for sample_idx in range(sample_count + 1):
            sample_time = round(sample_idx * step, 6)
            while waypoint_index + 1 < len(waypoints) and sample_time > waypoints[waypoint_index + 1][0]:
                waypoint_index += 1
            start_time, start_point, speed = waypoints[waypoint_index]
            end_time, end_point, _ = waypoints[min(waypoint_index + 1, len(waypoints) - 1)]
            if end_time == start_time:
                ratio = 0.0
            else:
                ratio = min(max((sample_time - start_time) / (end_time - start_time), 0.0), 1.0)
            x = start_point[0] + (end_point[0] - start_point[0]) * ratio
            y = start_point[1] + (end_point[1] - start_point[1]) * ratio
            z = start_point[2] + (end_point[2] - start_point[2]) * ratio
            heading = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))
            rows.append(
                {
                    "time": f"{sample_time:.2f}",
                    "uav_id": uav_id,
                    "x": f"{x:.3f}",
                    "y": f"{y:.3f}",
                    "z": f"{z:.3f}",
                    "speed": f"{speed:.3f}",
                    "heading": f"{heading:.3f}",
                    "region_label": region_label((x, y, z), regions),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scenario = load_json(args.scenario_config)
    region_config = load_json(args.region_config)
    rows = build_trace_rows(scenario, region_config)
    write_csv(args.output, TRACE_FIELDS, rows)
    print(Path(args.output))


if __name__ == "__main__":
    main()
