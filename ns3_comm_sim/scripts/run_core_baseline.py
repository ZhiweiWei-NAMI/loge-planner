from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from common import ensure_dir, write_csv
from parse_results import SUMMARY_FIELDS
from run_benchmark import ROOT, run_single_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns3-root", default=str(ROOT.parent / "ns3_install" / "ns-3.40"))
    args = parser.parse_args()

    scenario = ROOT / "configs" / "scenarios" / "core_baseline.json"
    region = ROOT / "configs" / "regions" / "city_canyon.json"
    traffic = ROOT / "configs" / "traffic" / "control_messages.json"
    radio = ROOT / "configs" / "radio" / "wifi_5ghz_fixed.json"
    arch_dir = ROOT / "configs" / "architectures"

    summaries = []
    for name in ["cloud.json", "cloud_edge.json", "cloud_edge_terminal.json"]:
        summaries.append(
            run_single_benchmark(
                scenario,
                region,
                traffic,
                radio,
                arch_dir / name,
                Path(args.ns3_root),
            )
        )

    parsed_dir = ensure_dir(ROOT / "results" / "parsed")
    tables_dir = ensure_dir(ROOT / "results" / "tables")
    summary_path = parsed_dir / "core_baseline_summary.csv"
    write_csv(summary_path, SUMMARY_FIELDS, summaries)
    write_csv(tables_dir / "core_baseline_summary.csv", SUMMARY_FIELDS, summaries)
    print(summary_path)


if __name__ == "__main__":
    main()
