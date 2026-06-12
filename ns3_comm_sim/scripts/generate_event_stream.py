from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import dump_json, load_json, weighted_choice, write_csv


EVENT_FIELDS = [
    "time",
    "uav_id",
    "event_type",
    "transaction_id",
    "priority",
    "deadline_ms",
    "max_retries",
    "request_packets",
    "response_packets",
]


def _priority_deadline(priority: int, base_deadline_ms: int) -> int:
    if priority >= 3:
        return max(50, int(base_deadline_ms * 0.8))
    if priority == 2:
        return base_deadline_ms
    return int(base_deadline_ms * 1.2)


def _task_profile(traffic: dict, task_name: str) -> dict:
    profiles = traffic.get("task_profiles", {})
    profile = profiles.get(task_name, {})
    return {
        "request_packets": int(profile.get("request_packets", 1)),
        "response_packets": int(profile.get("response_packets", 1)),
        "deadline_ms": int(profile.get("deadline_ms", 300)),
        "max_retries": int(profile.get("max_retries", 2)),
    }


def _append_event(
    rows: list[dict],
    *,
    time_s: float,
    uav_id: int,
    event_type: str,
    transaction_id: int,
    priority: int,
    profile: dict,
) -> None:
    rows.append(
        {
            "time": f"{time_s:.2f}",
            "uav_id": uav_id,
            "event_type": event_type,
            "transaction_id": transaction_id,
            "priority": priority,
            "deadline_ms": _priority_deadline(priority, int(profile["deadline_ms"])),
            "max_retries": int(profile["max_retries"]),
            "request_packets": int(profile["request_packets"]),
            "response_packets": int(profile["response_packets"]),
        }
    )


def build_event_rows(scenario: dict, traffic: dict) -> list[dict]:
    rng = random.Random(int(scenario.get("seed", 0)) + 17)
    duration = float(scenario["duration_s"])
    plan_low, plan_high = [float(value) for value in scenario.get("plan_start_window_s", [1.0, 4.0])]
    plan_interval_s = float(scenario.get("plan_interval_s", max(duration, 1.0)))
    plan_priority = int(scenario.get("plan_priority", 2))
    replan_rate_per_second = float(scenario.get("replan_rate_per_uav_per_min", 0.0)) / 60.0
    decouple_rate_per_second = float(scenario.get("decouple_rate_per_uav_per_min", 0.0)) / 60.0
    replan_priority_weights = {
        str(key): float(value)
        for key, value in scenario.get("replan_priority_weights", {"1": 1.0}).items()
    }
    decouple_priority_weights = {
        str(key): float(value)
        for key, value in scenario.get("decouple_priority_weights", {"3": 1.0}).items()
    }
    plan_profile = _task_profile(traffic, "plan")
    replan_profile = _task_profile(traffic, "replan")
    decouple_profile = _task_profile(traffic, "decouple")

    rows: list[dict] = []
    transaction_id = 1
    for uav_id in range(int(scenario["uav_count"])):
        current = rng.uniform(plan_low, plan_high)
        while current < duration - 1.0:
            _append_event(
                rows,
                time_s=current,
                uav_id=uav_id,
                event_type="plan",
                transaction_id=transaction_id,
                priority=plan_priority,
                profile=plan_profile,
            )
            transaction_id += 1
            current += plan_interval_s

    if replan_rate_per_second > 0.0:
        for uav_id in range(int(scenario["uav_count"])):
            current = max(plan_high + 1.0, 5.0)
            while current < duration - 1.0:
                current += rng.expovariate(replan_rate_per_second)
                if current >= duration - 1.0:
                    break
                priority = int(weighted_choice(rng, replan_priority_weights))
                _append_event(
                    rows,
                    time_s=current,
                    uav_id=uav_id,
                    event_type="replan",
                    transaction_id=transaction_id,
                    priority=priority,
                    profile=replan_profile,
                )
                transaction_id += 1

    if decouple_rate_per_second > 0.0:
        for uav_id in range(int(scenario["uav_count"])):
            current = max(plan_high + 2.0, 8.0)
            while current < duration - 1.0:
                current += rng.expovariate(decouple_rate_per_second)
                if current >= duration - 1.0:
                    break
                priority = int(weighted_choice(rng, decouple_priority_weights))
                _append_event(
                    rows,
                    time_s=current,
                    uav_id=uav_id,
                    event_type="decouple",
                    transaction_id=transaction_id,
                    priority=priority,
                    profile=decouple_profile,
                )
                transaction_id += 1

    rows.sort(key=lambda row: (float(row["time"]), int(row["uav_id"]), int(row["transaction_id"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--traffic-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--packet-profile-output", required=True)
    args = parser.parse_args()

    scenario = load_json(args.scenario_config)
    traffic = load_json(args.traffic_config)
    rows = build_event_rows(scenario, traffic)
    write_csv(args.output, EVENT_FIELDS, rows)
    dump_json(args.packet_profile_output, traffic)
    print(Path(args.output))


if __name__ == "__main__":
    main()
