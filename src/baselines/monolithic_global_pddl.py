#!/usr/bin/env python3
"""Strong monolithic baseline: single pseudo-cluster + global PDDL solve."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.offline_pipeline.stage3_plan_monolithic_pddl import _extract_checkpoints_from_scenario
from src.offline_pipeline.stage3_plan_with_pddl import (
    _load_scenario,
    _load_workflow_init_info,
    run_stage3_planning,
)


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def _build_assignment_rows(
    scenario: Dict[str, Any],
    checkpoint_df: pd.DataFrame,
    workflow_start_times: Dict[str, float],
) -> List[Dict[str, Any]]:
    agent_ids = [str(agent.get("id")) for agent in scenario.get("agents", [])]
    center = {"x": 0.0, "y": 0.0, "z": 0.0}
    if not checkpoint_df.empty:
        xyz = checkpoint_df["location"].apply(lambda value: ast.literal_eval(str(value))).tolist()
        xs = [float(item[0]) for item in xyz]
        ys = [float(item[1]) for item in xyz]
        zs = [float(item[2]) for item in xyz]
        center = {
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "z": sum(zs) / len(zs),
        }
    earliest = min((workflow_start_times.get(str(wf_id), 0.0) for wf_id in checkpoint_df["workflow_id"].unique()), default=0.0)
    op_types = sorted(set(str(value) for value in checkpoint_df.get("operation_type", [])))
    return [{
        "cluster_group": 0,
        "center_x": center["x"],
        "center_y": center["y"],
        "center_z": center["z"],
        "earliest_start_time": earliest,
        "n_checkpoints": int(len(checkpoint_df)),
        "op_types": ";".join(op_types),
        "assigned_agents": ";".join(agent_ids),
        "assignment_reason": "all_agents_single_pseudo_cluster",
    }]


def prepare_monolithic_inputs(
    seed: int,
    scenario_name: str,
    input_root: Path,
    output_root: Path,
) -> Dict[str, Path]:
    run_dir_in = input_root / f"run_seed_{seed}"
    run_dir_out = output_root / f"run_seed_{seed}"
    run_dir_out.mkdir(parents=True, exist_ok=True)

    scenario_path = run_dir_in / "scenario.yaml"
    workflow_info_path = run_dir_in / "workflow_init_info.csv"
    scenario = _load_scenario(scenario_path)
    workflow_start_times = _load_workflow_init_info(workflow_info_path)

    checkpoint_df = _extract_checkpoints_from_scenario(scenario)
    checkpoint_df["cluster_group"] = 0
    cluster_path = run_dir_in / "checkpoint_clusters_monolithic_global_pddl.csv"
    checkpoint_df.to_csv(cluster_path, index=False)

    assignment_rows = _build_assignment_rows(scenario, checkpoint_df, workflow_start_times)
    assignment_path = run_dir_out / "cluster_assignment_monolithic_global_pddl.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(assignment_rows[0].keys()))
        writer.writeheader()
        for row in assignment_rows:
            writer.writerow(row)

    return {
        "scenario_path": scenario_path,
        "workflow_info_path": workflow_info_path,
        "cluster_path": cluster_path,
        "assignment_path": assignment_path,
        "run_dir_in": run_dir_in,
        "run_dir_out": run_dir_out,
    }


def _pddl_timeout_for_scenario(scenario_name: str) -> float:
    """Return PDDL solver timeout based on scenario size.
    
    Small scenarios can be solved monolithically with a longer timeout,
    while medium/large scenarios will timeout regardless (which IS the result).
    """
    if scenario_name.startswith("small"):
        return 600.0
    return 300.0


def run_monolithic_global_pddl(
    seed: int,
    scenario_name: str,
    input_root: Path,
    output_root: Path,
    topology_file: Path | None = None,
) -> Path:
    prepare_monolithic_inputs(seed, scenario_name, input_root, output_root)
    pddl_timeout = _pddl_timeout_for_scenario(scenario_name)
    return run_stage3_planning(
        seed=seed,
        cluster_method="monolithic_global_pddl",
        input_root=input_root,
        output_root=output_root,
        topology_file=topology_file,
        export_edge_routes=False,
        pddl_timeout=pddl_timeout,
    )


def run_baseline(seed: int, scenario: str, input_root: Path, output_root: Path, topology_file: Path | None = None) -> Path:
    return run_monolithic_global_pddl(
        seed=seed,
        scenario_name=scenario,
        input_root=input_root,
        output_root=output_root,
        topology_file=topology_file,
    )


def _suite_timeout_for_scenario(scenario_name: str) -> int:
    """Return outer suite stage3 timeout based on scenario size."""
    if scenario_name.startswith("small"):
        return 900
    return 600


def main() -> int:
    parser = argparse.ArgumentParser(description="Run monolithic global PDDL baseline")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--topology-file", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    input_root = _resolve_path(args.input_root or f"workflow_decouple/data/multi_seed_runs/{args.scenario}", project_root)
    output_root = _resolve_path(args.output_root or f"data/offline_plans/{args.scenario}", project_root)
    topology_file = _resolve_path(args.topology_file, project_root) if args.topology_file else None

    if args.prepare_only:
        outputs = prepare_monolithic_inputs(
            seed=args.seed,
            scenario_name=args.scenario,
            input_root=input_root,
            output_root=output_root,
        )
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0

    output_path = run_monolithic_global_pddl(
        seed=args.seed,
        scenario_name=args.scenario,
        input_root=input_root,
        output_root=output_root,
        topology_file=topology_file,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
