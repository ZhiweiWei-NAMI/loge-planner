from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import dump_json, percentile, read_csv, write_csv


CONTROL_PORTS = {9000, 9100}
TASK_NAMES = ["plan", "replan", "decouple"]


SUMMARY_FIELDS = [
    "run_name",
    "scenario",
    "architecture",
    "total_transactions",
    "network_required_transactions",
    "successful_transactions",
    "plan_success_rate",
    "replan_success_rate",
    "decouple_success_rate",
    "mean_delay_ms",
    "p95_delay_ms",
    "p99_delay_ms",
    "penalty_mean_delay_ms",
    "penalty_p95_delay_ms",
    "penalty_p99_delay_ms",
    "plan_mean_delay_ms",
    "plan_p95_delay_ms",
    "plan_p99_delay_ms",
    "plan_penalty_mean_delay_ms",
    "plan_penalty_p95_delay_ms",
    "plan_penalty_p99_delay_ms",
    "replan_mean_delay_ms",
    "replan_p95_delay_ms",
    "replan_p99_delay_ms",
    "replan_penalty_mean_delay_ms",
    "replan_penalty_p95_delay_ms",
    "replan_penalty_p99_delay_ms",
    "decouple_mean_delay_ms",
    "decouple_p95_delay_ms",
    "decouple_p99_delay_ms",
    "decouple_penalty_mean_delay_ms",
    "decouple_penalty_p95_delay_ms",
    "decouple_penalty_p99_delay_ms",
    "retry_to_success",
    "plan_retry_to_success",
    "replan_retry_to_success",
    "decouple_retry_to_success",
    "stale_ratio",
    "plan_stale_ratio",
    "replan_stale_ratio",
    "decouple_stale_ratio",
    "timeout_ratio",
    "plan_timeout_ratio",
    "replan_timeout_ratio",
    "decouple_timeout_ratio",
    "local_execution_ratio",
    "replan_local_execution_ratio",
    "packet_delivery_ratio",
    "control_packet_delivery_ratio",
    "control_mean_flow_delay_ms",
    "control_mean_flow_jitter_ms",
    "mean_flow_delay_ms",
    "mean_flow_jitter_ms",
    "mean_snr_db",
]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_port(value: str) -> int:
    return _safe_int(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _filter_rows(rows: list[dict], event_type: str | None = None) -> list[dict]:
    if event_type is None:
        return list(rows)
    return [row for row in rows if row.get("event_type") == event_type]


def _successful_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if _safe_int(row.get("success", "0")) == 1]


def _successful_network_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if _safe_int(row.get("success", "0")) == 1 and _safe_int(row.get("network_required", "1")) == 1
    ]


def _delay_values(rows: list[dict]) -> list[float]:
    return [_safe_float(row.get("critical_path_delay_ms", "0")) for row in rows]


def _penalized_delay_value(row: dict) -> float:
    actual_delay = _safe_float(row.get("critical_path_delay_ms", "0"))
    deadline_ms = _safe_float(row.get("deadline_ms", "0"))
    success = _safe_int(row.get("success", "0")) == 1
    if success:
        return actual_delay
    return max(actual_delay, deadline_ms)


def _penalized_delay_values(rows: list[dict]) -> list[float]:
    return [_penalized_delay_value(row) for row in rows]


def _delay_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def _retry_to_success(rows: list[dict]) -> float:
    retry_values = [_safe_int(row.get("retry_count", "0")) for row in rows]
    return _mean([float(value) for value in retry_values])


def _is_control_flow(row: dict) -> bool:
    return _safe_port(row.get("source_port", "0")) in CONTROL_PORTS or _safe_port(row.get("destination_port", "0")) in CONTROL_PORTS


def summarize_run(raw_dir: str | Path) -> dict:
    raw_path = Path(raw_dir)
    txn_rows = read_csv(raw_path / "transactions.csv")
    flow_rows = read_csv(raw_path / "flow_summary.csv") if (raw_path / "flow_summary.csv").exists() else []
    snr_rows = read_csv(raw_path / "snr_samples.csv") if (raw_path / "snr_samples.csv").exists() else []

    task_rows = {task: _filter_rows(txn_rows, task) for task in TASK_NAMES}
    plan_rows = task_rows["plan"]
    replan_rows = task_rows["replan"]
    decouple_rows = task_rows["decouple"]

    successful_txns = _successful_rows(txn_rows)
    successful_network_txns = _successful_network_rows(txn_rows)
    successful_plan = _successful_rows(plan_rows)
    successful_replan = _successful_rows(replan_rows)
    successful_decouple = _successful_rows(decouple_rows)
    successful_plan_network = _successful_network_rows(plan_rows)
    successful_replan_network = _successful_network_rows(replan_rows)
    successful_decouple_network = _successful_network_rows(decouple_rows)

    delays = _delay_values(successful_txns)
    penalized_delays = _penalized_delay_values(txn_rows)
    plan_delays = _delay_values(successful_plan)
    plan_penalized_delays = _penalized_delay_values(plan_rows)
    replan_delays = _delay_values(successful_replan)
    replan_penalized_delays = _penalized_delay_values(replan_rows)
    decouple_delays = _delay_values(successful_decouple)
    decouple_penalized_delays = _penalized_delay_values(decouple_rows)

    overall_stats = _delay_stats(delays)
    overall_penalty_stats = _delay_stats(penalized_delays)
    plan_stats = _delay_stats(plan_delays)
    plan_penalty_stats = _delay_stats(plan_penalized_delays)
    replan_stats = _delay_stats(replan_delays)
    replan_penalty_stats = _delay_stats(replan_penalized_delays)
    decouple_stats = _delay_stats(decouple_delays)
    decouple_penalty_stats = _delay_stats(decouple_penalized_delays)

    stale_count = sum(_safe_int(row.get("stale", "0")) for row in txn_rows)
    timeout_count = sum(_safe_int(row.get("timeout", "0")) for row in txn_rows)
    local_execution_count = sum(_safe_int(row.get("local_execution", "0")) for row in txn_rows)
    network_required_count = sum(_safe_int(row.get("network_required", "1")) for row in txn_rows)
    plan_stale_count = sum(_safe_int(row.get("stale", "0")) for row in plan_rows)
    replan_stale_count = sum(_safe_int(row.get("stale", "0")) for row in replan_rows)
    decouple_stale_count = sum(_safe_int(row.get("stale", "0")) for row in decouple_rows)
    plan_timeout_count = sum(_safe_int(row.get("timeout", "0")) for row in plan_rows)
    replan_timeout_count = sum(_safe_int(row.get("timeout", "0")) for row in replan_rows)
    decouple_timeout_count = sum(_safe_int(row.get("timeout", "0")) for row in decouple_rows)
    replan_local_execution_count = sum(_safe_int(row.get("local_execution", "0")) for row in replan_rows)

    tx_packets = sum(_safe_int(row.get("tx_packets", "0")) for row in flow_rows)
    rx_packets = sum(_safe_int(row.get("rx_packets", "0")) for row in flow_rows)
    flow_delay_values = [_safe_float(row.get("mean_delay_ms", "0")) for row in flow_rows if _safe_int(row.get("rx_packets", "0")) > 0]
    flow_jitter_values = [_safe_float(row.get("mean_jitter_ms", "0")) for row in flow_rows if _safe_int(row.get("rx_packets", "0")) > 1]
    control_flow_rows = [row for row in flow_rows if _is_control_flow(row)]
    control_tx_packets = sum(_safe_int(row.get("tx_packets", "0")) for row in control_flow_rows)
    control_rx_packets = sum(_safe_int(row.get("rx_packets", "0")) for row in control_flow_rows)
    control_flow_delay_values = [_safe_float(row.get("mean_delay_ms", "0")) for row in control_flow_rows if _safe_int(row.get("rx_packets", "0")) > 0]
    control_flow_jitter_values = [_safe_float(row.get("mean_jitter_ms", "0")) for row in control_flow_rows if _safe_int(row.get("rx_packets", "0")) > 1]
    snr_values = [_safe_float(row.get("snr_db", "0")) for row in snr_rows]

    run_name = raw_path.name
    scenario = raw_path.parent.name if raw_path.parent != raw_path else run_name
    architecture = run_name.split("__")[-1] if "__" in run_name else run_name

    summary = {
        "run_name": run_name,
        "scenario": scenario,
        "architecture": architecture,
        "total_transactions": len(txn_rows),
        "network_required_transactions": network_required_count,
        "successful_transactions": sum(_safe_int(row.get("success", "0")) for row in txn_rows),
        "plan_success_rate": _ratio(len(successful_plan), len(plan_rows)),
        "replan_success_rate": _ratio(len(successful_replan), len(replan_rows)),
        "decouple_success_rate": _ratio(len(successful_decouple), len(decouple_rows)),
        "mean_delay_ms": overall_stats["mean"],
        "p95_delay_ms": overall_stats["p95"],
        "p99_delay_ms": overall_stats["p99"],
        "penalty_mean_delay_ms": overall_penalty_stats["mean"],
        "penalty_p95_delay_ms": overall_penalty_stats["p95"],
        "penalty_p99_delay_ms": overall_penalty_stats["p99"],
        "plan_mean_delay_ms": plan_stats["mean"],
        "plan_p95_delay_ms": plan_stats["p95"],
        "plan_p99_delay_ms": plan_stats["p99"],
        "plan_penalty_mean_delay_ms": plan_penalty_stats["mean"],
        "plan_penalty_p95_delay_ms": plan_penalty_stats["p95"],
        "plan_penalty_p99_delay_ms": plan_penalty_stats["p99"],
        "replan_mean_delay_ms": replan_stats["mean"],
        "replan_p95_delay_ms": replan_stats["p95"],
        "replan_p99_delay_ms": replan_stats["p99"],
        "replan_penalty_mean_delay_ms": replan_penalty_stats["mean"],
        "replan_penalty_p95_delay_ms": replan_penalty_stats["p95"],
        "replan_penalty_p99_delay_ms": replan_penalty_stats["p99"],
        "decouple_mean_delay_ms": decouple_stats["mean"],
        "decouple_p95_delay_ms": decouple_stats["p95"],
        "decouple_p99_delay_ms": decouple_stats["p99"],
        "decouple_penalty_mean_delay_ms": decouple_penalty_stats["mean"],
        "decouple_penalty_p95_delay_ms": decouple_penalty_stats["p95"],
        "decouple_penalty_p99_delay_ms": decouple_penalty_stats["p99"],
        "retry_to_success": _retry_to_success(successful_network_txns),
        "plan_retry_to_success": _retry_to_success(successful_plan_network),
        "replan_retry_to_success": _retry_to_success(successful_replan_network),
        "decouple_retry_to_success": _retry_to_success(successful_decouple_network),
        "stale_ratio": _ratio(stale_count, len(txn_rows)),
        "plan_stale_ratio": _ratio(plan_stale_count, len(plan_rows)),
        "replan_stale_ratio": _ratio(replan_stale_count, len(replan_rows)),
        "decouple_stale_ratio": _ratio(decouple_stale_count, len(decouple_rows)),
        "timeout_ratio": _ratio(timeout_count, len(txn_rows)),
        "plan_timeout_ratio": _ratio(plan_timeout_count, len(plan_rows)),
        "replan_timeout_ratio": _ratio(replan_timeout_count, len(replan_rows)),
        "decouple_timeout_ratio": _ratio(decouple_timeout_count, len(decouple_rows)),
        "local_execution_ratio": _ratio(local_execution_count, len(txn_rows)),
        "replan_local_execution_ratio": _ratio(replan_local_execution_count, len(replan_rows)),
        "packet_delivery_ratio": (rx_packets / tx_packets) if tx_packets else 0.0,
        "control_packet_delivery_ratio": (control_rx_packets / control_tx_packets) if control_tx_packets else 0.0,
        "control_mean_flow_delay_ms": _mean(control_flow_delay_values),
        "control_mean_flow_jitter_ms": _mean(control_flow_jitter_values),
        "mean_flow_delay_ms": _mean(flow_delay_values),
        "mean_flow_jitter_ms": _mean(flow_jitter_values),
        "mean_snr_db": _mean(snr_values),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    summary = summarize_run(args.raw_dir)
    if args.output_json:
        dump_json(args.output_json, summary)
    if args.output_csv:
        write_csv(args.output_csv, SUMMARY_FIELDS, [summary])
    else:
        print(summary)


if __name__ == "__main__":
    main()
