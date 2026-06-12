from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from common import ensure_dir, write_csv
from run_benchmark import ROOT, run_single_benchmark
from parse_results import SUMMARY_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns3-root", default=str(ROOT.parent / "ns3_install" / "ns-3.40"))
    args = parser.parse_args()

    scenario = ROOT / "configs" / "scenarios" / "smoke_test.json"
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
    write_csv(parsed_dir / "smoke_test_summary.csv", SUMMARY_FIELDS, summaries)
    print(parsed_dir / "smoke_test_summary.csv")


if __name__ == "__main__":
    main()
