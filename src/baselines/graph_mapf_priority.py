#!/usr/bin/env python3
"""Conflict-aware graph baseline with prioritized routing and reservation metadata."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.offline_pipeline.stage3_plan_monolithic_pddl import _extract_checkpoints_from_scenario
from src.offline_pipeline.stage3_plan_with_pddl import (
    _load_scenario,
    _load_workflow_init_info,
    _make_topo_node_label,
    _snap_to_topo_node,
    build_agent_plans,
)
from src.simulation.environment.topology_manager import TopologyManager


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def _load_topology(run_dir_in: Path, scenario: Dict[str, Any]) -> TopologyManager:
    topo_path = run_dir_in / "graph_topology.json"
    if not topo_path.exists():
        env_cfg = scenario.get("environment", {}) or {}
        topo_ref = env_cfg.get("graph_topology_file")
        if topo_ref:
            topo_path = Path(topo_ref)
    bounds = {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "z_min": 0.0,
        "z_max": 1.0,
    }
    topo = TopologyManager(bounds=bounds)
    if topo_path.exists():
        topo.load_from_file(str(topo_path))
        return topo

    env_cfg = scenario.get("environment", {}) or {}
    sim_space = env_cfg.get("simulation_space", {}) or {}
    bounds = {
        "x_min": float(sim_space.get("x_min", 0.0)),
        "x_max": float(sim_space.get("x_max", 5000.0)),
        "y_min": float(sim_space.get("y_min", 0.0)),
        "y_max": float(sim_space.get("y_max", 5000.0)),
        "z_min": float(sim_space.get("z_min", 0.0)),
        "z_max": float(sim_space.get("z_max", 200.0)),
    }
    topo = TopologyManager(bounds=bounds)
    topo.generate_grid_topology(seed=42)
    topo.save_to_file(str(run_dir_in / "graph_topology.json"))
    return topo


def _edge_travel_time(topo: TopologyManager, u: str, v: str) -> float:
    edge = topo.edges.get((u, v))
    if not edge:
        return 1.0
    velocity = max(float(edge.base_velocity), 1e-3)
    return max(1.0, float(edge.length) / velocity)


def _shortest_path(topo: TopologyManager, start: str, goal: str) -> List[str]:
    if start == goal:
        return [start]
    try:
        return nx.shortest_path(
            topo.graph,
            source=start,
            target=goal,
            weight=lambda u, v, attrs: float(attrs.get("length", 1.0)) / max(float(attrs.get("base_velocity", 1.0)), 1e-3),
        )
    except nx.NetworkXNoPath:
        return [start, goal]


def _shortest_cost(topo: TopologyManager, start: str, goal: str) -> float:
    if start == goal:
        return 0.0
    try:
        return float(
            nx.shortest_path_length(
                topo.graph,
                source=start,
                target=goal,
                weight=lambda u, v, attrs: float(attrs.get("length", 1.0)) / max(float(attrs.get("base_velocity", 1.0)), 1e-3),
            )
        )
    except nx.NetworkXNoPath:
        if start in topo.nodes and goal in topo.nodes:
            a = topo.nodes[start]
            b = topo.nodes[goal]
            return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        return 1e6


def _get_operation_category(op_type: str) -> Optional[str]:
    op_type = op_type.lower()
    if op_type in ("compute", "move_to_compute"):
        return "compute"
    if op_type in ("communication", "communicate", "move_to_communication"):
        return "communicate"
    if op_type in ("inspection", "inspect", "move_to_inspection"):
        return "inspect"
    return None


@dataclass
class WorkflowRoute:
    workflow_id: str
    cluster_group: int
    earliest_start_time: float
    assigned_agent: str
    checkpoints: List[Dict[str, Any]]
    first_node: str
    last_node: str
    estimated_cost: float


def _build_workflow_routes(
    scenario: Dict[str, Any],
    checkpoint_df: pd.DataFrame,
    workflow_start_times: Dict[str, float],
    topo: TopologyManager,
) -> List[WorkflowRoute]:
    routes: List[WorkflowRoute] = []
    grouped = checkpoint_df.groupby("workflow_id")
    for cluster_group, (workflow_id, group) in enumerate(grouped):
        ordered = group.sort_values("checkpoint_idx").copy()
        checkpoints: List[Dict[str, Any]] = []
        last_node: Optional[str] = None
        total_cost = 0.0
        for _, row in ordered.iterrows():
            loc = ast.literal_eval(str(row["location"]))
            node_id = _snap_to_topo_node(float(loc[0]), float(loc[1]), float(loc[2]), topo)
            if node_id is None:
                continue
            if last_node is not None:
                total_cost += _shortest_cost(topo, last_node, node_id)
            checkpoints.append({
                "workflow_id": str(workflow_id),
                "operation_type": str(row["operation_type"]),
                "location": str(list(loc)),
                "checkpoint_idx": int(row.get("checkpoint_idx", len(checkpoints))),
                "cluster_group": cluster_group,
                "topo_node_id": node_id,
            })
            last_node = node_id
        if not checkpoints:
            continue
        routes.append(
            WorkflowRoute(
                workflow_id=str(workflow_id),
                cluster_group=cluster_group,
                earliest_start_time=float(workflow_start_times.get(str(workflow_id), 0.0)),
                assigned_agent="",
                checkpoints=checkpoints,
                first_node=checkpoints[0]["topo_node_id"],
                last_node=checkpoints[-1]["topo_node_id"],
                estimated_cost=total_cost,
            )
        )
    return routes


def _assign_workflows(
    scenario: Dict[str, Any],
    routes: List[WorkflowRoute],
    topo: TopologyManager,
) -> Tuple[List[WorkflowRoute], List[Dict[str, Any]]]:
    agent_positions: Dict[str, str] = {}
    agent_times: Dict[str, float] = {}
    for agent in scenario.get("agents", []):
        agent_id = str(agent.get("id"))
        pos = agent.get("initial_position", {}) or {}
        node_id = topo.get_nearest_node(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0)))
        if node_id is None:
            continue
        agent_positions[agent_id] = node_id
        agent_times[agent_id] = 0.0

    assignment_rows: List[Dict[str, Any]] = []
    for route in sorted(routes, key=lambda item: (item.earliest_start_time, item.workflow_id)):
        best_agent = None
        best_cost = float("inf")
        for agent_id, node_id in agent_positions.items():
            try:
                travel_cost = _shortest_cost(topo, node_id, route.first_node)
            except nx.NetworkXNoPath:
                continue
            composite = max(agent_times[agent_id], route.earliest_start_time) + travel_cost + route.estimated_cost
            if composite < best_cost:
                best_cost = composite
                best_agent = agent_id
        if best_agent is None:
            continue
        route.assigned_agent = best_agent
        agent_times[best_agent] = best_cost
        agent_positions[best_agent] = route.last_node

        xs, ys, zs = [], [], []
        for checkpoint in route.checkpoints:
            loc = ast.literal_eval(checkpoint["location"])
            xs.append(float(loc[0]))
            ys.append(float(loc[1]))
            zs.append(float(loc[2]))
        op_types = sorted({checkpoint["operation_type"] for checkpoint in route.checkpoints})
        assignment_rows.append(
            {
                "cluster_group": route.cluster_group,
                "center_x": sum(xs) / len(xs),
                "center_y": sum(ys) / len(ys),
                "center_z": sum(zs) / len(zs),
                "earliest_start_time": route.earliest_start_time,
                "n_checkpoints": len(route.checkpoints),
                "op_types": ";".join(op_types),
                "assigned_agents": best_agent,
                "assignment_reason": f"graph_shortest_path_cost={best_cost:.3f}",
            }
        )
    return routes, assignment_rows


def _reserve_path(
    topo: TopologyManager,
    path_nodes: List[str],
    start_time: float,
    edge_reservations: Dict[Tuple[str, str], List[Tuple[int, int, str]]],
    node_reservations: Dict[str, List[Tuple[int, int, str]]],
    agent_id: str,
) -> Tuple[float, List[str]]:
    if len(path_nodes) <= 1:
        return start_time, path_nodes
    current_start = start_time
    while True:
        probe = current_start
        occupancy: List[Tuple[Tuple[str, str], int, int]] = []
        conflict = False
        for u, v in zip(path_nodes, path_nodes[1:]):
            duration = _edge_travel_time(topo, u, v)
            slot_start = int(math.floor(probe))
            slot_end = int(math.ceil(probe + duration))
            if any(not (slot_end <= start or slot_start >= end) for start, end, _agent in edge_reservations[(u, v)]):
                conflict = True
                break
            if any(not (slot_end <= start or slot_start >= end) for start, end, _agent in node_reservations[v]):
                conflict = True
                break
            occupancy.append(((u, v), slot_start, slot_end))
            probe += duration
        if not conflict:
            probe = current_start
            for (u, v), slot_start, slot_end in occupancy:
                edge_reservations[(u, v)].append((slot_start, slot_end, agent_id))
                node_reservations[v].append((slot_start, slot_end, agent_id))
                probe += _edge_travel_time(topo, u, v)
            return probe, path_nodes
        current_start += 1.0


def _build_plan_data(
    scenario: Dict[str, Any],
    workflow_routes: List[WorkflowRoute],
    checkpoint_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    workflow_start_times: Dict[str, float],
    topo: TopologyManager,
    seed: int,
) -> Dict[str, Any]:
    agents_plans = build_agent_plans(
        scenario=scenario,
        checkpoint_df=checkpoint_df,
        assignment_df=assignment_df,
        workflow_start_times=workflow_start_times,
        topo=topo,
    )

    edge_reservations: Dict[Tuple[str, str], List[Tuple[int, int, str]]] = defaultdict(list)
    node_reservations: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    diagnostics: Dict[str, Any] = {"workflow_routes": {}, "conflict_aware": True}

    for agent in scenario.get("agents", []):
        agent_id = str(agent.get("id"))
        pos = agent.get("initial_position", {}) or {}
        current_node = topo.get_nearest_node(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0)))
        current_time = 0.0
        steps: List[Dict[str, Any]] = []
        assigned_routes = [route for route in workflow_routes if route.assigned_agent == agent_id]
        assigned_routes.sort(key=lambda item: (item.earliest_start_time, item.workflow_id))

        for route in assigned_routes:
            diagnostics["workflow_routes"][route.workflow_id] = {
                "agent_id": agent_id,
                "cluster_group": route.cluster_group,
                "earliest_start_time": route.earliest_start_time,
            }
            current_time = max(current_time, route.earliest_start_time)
            for checkpoint in route.checkpoints:
                node_id = checkpoint["topo_node_id"]
                if current_node is None:
                    current_node = node_id
                path_nodes = _shortest_path(topo, current_node, node_id)
                arrival_time, _ = _reserve_path(
                    topo,
                    path_nodes,
                    current_time,
                    edge_reservations,
                    node_reservations,
                    agent_id,
                )
                loc_label = _make_topo_node_label(node_id)
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "planned_time": round(arrival_time, 3),
                        "action_name": "movetoloc",
                        "parameters": [agent_id, _make_topo_node_label(current_node), loc_label],
                    }
                )
                action_category = _get_operation_category(checkpoint["operation_type"])
                if action_category == "compute":
                    action_name = "assistcompute"
                elif action_category == "communicate":
                    action_name = "assistcommunicate"
                elif action_category == "inspect":
                    action_name = "inspect"
                else:
                    action_name = ""
                if action_name:
                    action_time = arrival_time + 1.0
                    node_reservations[node_id].append((int(math.floor(action_time)), int(math.ceil(action_time)) + 1, agent_id))
                    steps.append(
                        {
                            "step": len(steps) + 1,
                            "planned_time": round(action_time, 3),
                            "action_name": action_name,
                            "parameters": [agent_id, loc_label],
                        }
                    )
                    current_time = action_time
                else:
                    current_time = arrival_time
                current_node = node_id
        agents_plans[agent_id]["result"]["solution"] = steps

    topology_data = {
        "node_coords": {node_id: [node.x, node.y, node.z] for node_id, node in topo.nodes.items()},
        "bounds": topo.bounds,
        "n_nodes": len(topo.nodes),
        "n_edges": len(topo.edges),
        "topo_aware": True,
        "edge_speed_limits": {
            f"{u},{v}": round(max(0.1, float(edge.base_velocity) * 0.9), 2)
            for (u, v), edge in topo.edges.items()
        },
    }

    return {
        "seed": seed,
        "method": "graph_mapf_priority",
        "agents": agents_plans,
        "topology": topology_data,
        "mapf_diagnostics": diagnostics,
    }


def run_graph_mapf_priority(
    seed: int,
    scenario_name: str,
    input_root: Path,
    output_root: Path,
) -> Path:
    run_dir_in = input_root / f"run_seed_{seed}"
    run_dir_out = output_root / f"run_seed_{seed}"
    run_dir_out.mkdir(parents=True, exist_ok=True)

    scenario = _load_scenario(run_dir_in / "scenario.yaml")
    workflow_start_times = _load_workflow_init_info(run_dir_in / "workflow_init_info.csv")
    topo = _load_topology(run_dir_in, scenario)

    checkpoint_df = _extract_checkpoints_from_scenario(scenario)
    routes = _build_workflow_routes(scenario, checkpoint_df, workflow_start_times, topo)
    routes, assignment_rows = _assign_workflows(scenario, routes, topo)

    cluster_rows: List[Dict[str, Any]] = []
    workflow_to_cluster = {route.workflow_id: route.cluster_group for route in routes}
    for _, row in checkpoint_df.iterrows():
        workflow_id = str(row["workflow_id"])
        cluster_rows.append({**row.to_dict(), "cluster_group": workflow_to_cluster.get(workflow_id, -1)})
    checkpoint_out = pd.DataFrame(cluster_rows)
    if checkpoint_out.empty or not assignment_rows:
        raise RuntimeError(f"graph_mapf_priority could not build any assigned route for {scenario_name}/seed_{seed}")

    cluster_path = run_dir_in / "checkpoint_clusters_graph_mapf_priority.csv"
    checkpoint_out.to_csv(cluster_path, index=False)

    assignment_path = run_dir_out / "cluster_assignment_graph_mapf_priority.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(assignment_rows[0].keys()))
        writer.writeheader()
        for row in assignment_rows:
            writer.writerow(row)

    assignment_df = pd.DataFrame(assignment_rows)
    plan_data = _build_plan_data(
        scenario=scenario,
        workflow_routes=routes,
        checkpoint_df=checkpoint_out,
        assignment_df=assignment_df,
        workflow_start_times=workflow_start_times,
        topo=topo,
        seed=seed,
    )

    output_path = run_dir_out / "pddl_plan_graph_mapf_priority.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(plan_data, fh, indent=2)

    diagnostics_path = run_dir_out / "graph_mapf_priority_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as fh:
        json.dump(plan_data["mapf_diagnostics"], fh, indent=2)
    return output_path


def run_baseline(seed: int, scenario: str, input_root: Path, output_root: Path) -> Path:
    return run_graph_mapf_priority(
        seed=seed,
        scenario_name=scenario,
        input_root=input_root,
        output_root=output_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run graph MAPF priority baseline")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    input_root = _resolve_path(args.input_root or f"workflow_decouple/data/multi_seed_runs/{args.scenario}", project_root)
    output_root = _resolve_path(args.output_root or f"data/offline_plans/{args.scenario}", project_root)

    if args.prepare_only:
        scenario = _load_scenario(input_root / f"run_seed_{args.seed}" / "scenario.yaml")
        workflow_start_times = _load_workflow_init_info(input_root / f"run_seed_{args.seed}" / "workflow_init_info.csv")
        topo = _load_topology(input_root / f"run_seed_{args.seed}", scenario)
        checkpoint_df = _extract_checkpoints_from_scenario(scenario)
        routes = _build_workflow_routes(scenario, checkpoint_df, workflow_start_times, topo)
        routes, assignment_rows = _assign_workflows(scenario, routes, topo)
        workflow_to_cluster = {route.workflow_id: route.cluster_group for route in routes}
        checkpoint_out = checkpoint_df.copy()
        checkpoint_out["cluster_group"] = checkpoint_out["workflow_id"].map(lambda wf_id: workflow_to_cluster.get(str(wf_id), -1))
        run_dir_in = input_root / f"run_seed_{args.seed}"
        run_dir_out = output_root / f"run_seed_{args.seed}"
        run_dir_out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir_in / "checkpoint_clusters_graph_mapf_priority.csv"
        checkpoint_out.to_csv(checkpoint_path, index=False)
        assignment_path = run_dir_out / "cluster_assignment_graph_mapf_priority.csv"
        pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)
        print(json.dumps({"checkpoint_csv": str(checkpoint_path), "assignment_csv": str(assignment_path)}, indent=2))
        return 0

    output_path = run_graph_mapf_priority(
        seed=args.seed,
        scenario_name=args.scenario,
        input_root=input_root,
        output_root=output_root,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
