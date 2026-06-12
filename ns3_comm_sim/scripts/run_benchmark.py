from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from common import dump_json, ensure_dir, load_json, write_csv
from generate_event_stream import build_event_rows
from generate_uav_traces import TRACE_FIELDS, build_trace_rows
from parse_results import SUMMARY_FIELDS, summarize_run


REGION_FIELDS = [
    "region_id",
    "type",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "extra_loss_db",
    "extra_jitter_ms",
    "drop_probability",
    "building_height",
]

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


def write_region_csv(region_config: dict, output_path: Path) -> None:
    rows = []
    for region in region_config.get("regions", []):
        rows.append({field: region.get(field, "") for field in REGION_FIELDS})
    write_csv(output_path, REGION_FIELDS, rows)


def prepare_inputs(scenario_config: Path, region_config: Path, traffic_config: Path, output_root: Path) -> dict:
    scenario = load_json(scenario_config)
    regions = load_json(region_config)
    traffic = load_json(traffic_config)

    trace_rows = build_trace_rows(scenario, regions)
    event_rows = build_event_rows(scenario, traffic)

    trace_path = output_root / "inputs" / "traces" / f"{scenario['name']}.csv"
    event_path = output_root / "inputs" / "events" / f"{scenario['name']}.csv"
    packet_profile_path = output_root / "inputs" / "packet_profiles" / "control_messages.json"
    compiled_region_path = output_root / "inputs" / "regions" / f"{scenario['name']}.csv"

    write_csv(trace_path, TRACE_FIELDS, trace_rows)
    write_csv(event_path, EVENT_FIELDS, event_rows)
    dump_json(packet_profile_path, traffic)
    write_region_csv(regions, compiled_region_path)

    return {
        "scenario": scenario,
        "region": regions,
        "traffic": traffic,
        "trace_path": trace_path,
        "event_path": event_path,
        "packet_profile_path": packet_profile_path,
        "compiled_region_path": compiled_region_path,
    }


def run_subprocess(command: list[str], cwd: Path, log_file: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}")


def ensure_ns3_ready(ns3_root: Path, build_log: Path) -> None:
    run_subprocess(["./ns3", "configure", "--enable-examples", "--disable-tests"], ns3_root, build_log.parent / "configure.log")
    run_subprocess(["./ns3", "build", "uav_comm_benchmark"], ns3_root, build_log)


def locate_target(ns3_root: Path, target_name: str) -> str:
    result = subprocess.run(["./ns3", "show", "targets"], cwd=ns3_root, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    for line in result.stdout.splitlines():
        target_entry = line.strip().split()[0] if line.strip() else ""
        if target_entry.endswith(target_name) or target_entry == target_name or target_name in target_entry:
            return target_entry
    return target_name


def run_single_benchmark(
    scenario_config: Path,
    region_config: Path,
    traffic_config: Path,
    radio_config: Path,
    architecture_config: Path,
    ns3_root: Path,
) -> dict:
    ns3_root = ns3_root.expanduser().resolve()
    radio = load_json(radio_config)
    architecture = load_json(architecture_config)
    prep = prepare_inputs(scenario_config, region_config, traffic_config, ROOT)
    scenario = prep["scenario"]
    run_name = f"{scenario['name']}__{architecture['name']}"
    raw_dir = ensure_dir(ROOT / "results" / "raw" / scenario["name"] / run_name)
    parsed_dir = ensure_dir(ROOT / "results" / "parsed")
    ensure_dir(ROOT / "results" / "figures")
    ensure_dir(ROOT / "results" / "tables")

    local_source = ROOT / "ns3" / "uav_comm_benchmark.cc"
    scratch_source = ns3_root / "scratch" / "uav_comm_benchmark.cc"
    scratch_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_source, scratch_source)

    manifest = {
        "scenario_config": str(scenario_config),
        "region_config": str(region_config),
        "traffic_config": str(traffic_config),
        "radio_config": str(radio_config),
        "architecture_config": str(architecture_config),
        "trace_path": str(prep["trace_path"]),
        "event_path": str(prep["event_path"]),
        "region_csv": str(prep["compiled_region_path"]),
        "packet_profile_path": str(prep["packet_profile_path"]),
    }
    dump_json(raw_dir / "run_manifest.json", manifest)

    ensure_ns3_ready(ns3_root, raw_dir / "build.log")
    target = locate_target(ns3_root, "uav_comm_benchmark")

    command_string = (
        f"{target} "
        f"--traceFile={prep['trace_path']} "
        f"--eventFile={prep['event_path']} "
        f"--regionFile={prep['compiled_region_path']} "
        f"--outputDir={raw_dir} "
        f"--architecture={architecture['name']} "
        f"--simulationTime={scenario['duration_s']} "
        f"--packetSize={prep['traffic']['packet_size_bytes']} "
        f"--attemptTimeoutMs={prep['traffic']['attempt_timeout_ms']} "
        f"--localProcessingMs={prep['traffic']['processing_ms']['local']} "
        f"--edgeProcessingMs={prep['traffic']['processing_ms']['edge']} "
        f"--cloudProcessingMs={prep['traffic']['processing_ms']['cloud']} "
        f"--interRequestSpacingMs={prep['traffic']['inter_request_spacing_ms']} "
        f"--phyMode={radio['phy_mode']} "
        f"--controlMode={radio['control_mode']} "
        f"--txPowerDbm={radio['tx_power_dbm']} "
        f"--frequencyHz={radio['frequency_hz']} "
        f"--channelWidthMHz={radio['channel_width_mhz']} "
        f"--backhaulDelayMs={radio['cloud_backhaul_delay_ms']} "
        f"--backhaulRateMbps={radio['cloud_backhaul_rate_mbps']} "
        f"--backhaulPacketErrorRate={radio.get('cloud_backhaul_packet_error_rate', 0.0)} "
        f"--edgeApHeight={radio['edge_ap_height_m']} "
        f"--areaWidth={scenario['area']['width']} "
        f"--areaHeight={scenario['area']['height']} "
        f"--backgroundStaCount={scenario['background_sta_count']} "
        f"--backgroundContentionLoadMbps={prep['region'].get('background_contention_load_mbps', 0.0)} "
        f"--planTarget={architecture['plan_target']} "
        f"--replanTarget={architecture['replan_target']} "
        f"--decoupleTarget={architecture['decouple_target']} "
        f"--planResponseSource={architecture['plan_response_source']} "
        f"--replanResponseSource={architecture['replan_response_source']} "
        f"--decoupleResponseSource={architecture['decouple_response_source']} "
        f"--enablePcap={1 if radio.get('pcap', False) else 0} "
        f"--enableNetAnim={1 if radio.get('netanim', False) else 0} "
        f"--enableNetAnimPacketMetadata={1 if radio.get('netanim_packet_metadata', False) else 0} "
        f"--netAnimMobilityPollIntervalMs={radio.get('netanim_mobility_poll_interval_ms', 500.0)} "
        f"--seed={scenario['seed']}"
    )
    run_subprocess(["./ns3", "run", command_string], ns3_root, raw_dir / "ns3_run.log")

    summary = summarize_run(raw_dir)
    summary_csv = parsed_dir / f"{run_name}.csv"
    summary_json = parsed_dir / f"{run_name}.json"
    write_csv(summary_csv, SUMMARY_FIELDS, [summary])
    dump_json(summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--traffic-config", required=True)
    parser.add_argument("--radio-config", required=True)
    parser.add_argument("--architecture-config", required=True)
    parser.add_argument("--ns3-root", default=str(ROOT.parent / "ns3_install" / "ns-3.40"))
    args = parser.parse_args()

    summary = run_single_benchmark(
        Path(args.scenario_config),
        Path(args.region_config),
        Path(args.traffic_config),
        Path(args.radio_config),
        Path(args.architecture_config),
        Path(args.ns3_root),
    )
    print(summary)


if __name__ == "__main__":
    main()
