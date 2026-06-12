"""Stage 3 Monolithic: Generate PDDL plans without clustering.

This script implements a "monolithic" baseline that assigns checkpoints directly
to agents without any clustering step. Each checkpoint is assigned to the
nearest available agent using greedy assignment.

Key Differences from Clustering-based Stage 3:
----------------------------------------------
- No cluster grouping: each checkpoint is treated independently
- Greedy nearest-agent assignment: each checkpoint goes to closest agent
- No cluster_meta aggregation: workflow_id mapping is per-checkpoint

Output Format:
--------------
Same as stage3_plan_with_pddl.py, but with different assignment logic.
"""

import argparse
import ast
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import fcntl
import time

logger = logging.getLogger(__name__)


def _resolve_under_project(path_str: str, project_root: Path) -> Path:
    """Resolve a path which can be absolute or project-root-relative."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / path_str


def _parse_location(loc_str: str) -> Tuple[float, float, float]:
    """Parse location field stored as string like "[x, y, z]"."""
    try:
        value = ast.literal_eval(loc_str)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return float(value[0]), float(value[1]), float(value[2])
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def _make_position_label(x: float, y: float, z: float) -> str:
    """Create a position label from coordinates (rounded to integers)."""
    return f"position_{int(round(x))}_{int(round(y))}_{int(round(z))}"


def _load_scenario(scenario_path: Path, retry_count: int = 3, retry_delay: float = 0.1) -> Dict[str, Any]:
    """Load scenario.yaml with file locking and retry to prevent concurrent access issues."""
    last_error = None
    for attempt in range(retry_count):
        try:
            with scenario_path.open("r", encoding="utf-8") as f:
                # Acquire shared lock
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read()
                    if '\x00' in content:
                        raise ValueError("File contains null bytes, may be corrupted")
                    data = yaml.safe_load(content) or {}
                    return data
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            last_error = e
            logger.warning("Failed to load scenario (attempt %d/%d): %s", attempt + 1, retry_count, e)
            if attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"Failed to load scenario after {retry_count} attempts: {last_error}")


def _load_workflow_init_info(workflow_init_path: Path) -> Dict[str, float]:
    """Load workflow_init_info.csv and return workflow_id -> start_time mapping."""
    df = pd.read_csv(workflow_init_path)
    return df.set_index("workflow_id")["start_time"].to_dict()


def _get_operation_category(op_type: str) -> Optional[str]:
    """Map operation_type to PDDL goal category."""
    op_type = op_type.lower()
    if op_type in ("compute", "move_to_compute"):
        return "compute"
    if op_type in ("communication", "communicate", "move_to_communication"):
        return "communicate"
    if op_type in ("inspection", "inspect", "move_to_inspection"):
        return "inspect"
    return None


def _extract_checkpoints_from_scenario(scenario: Dict[str, Any]) -> pd.DataFrame:
    """Extract all checkpoints from scenario.yaml workflows.
    
    This is used for monolithic planning where we don't have pre-clustered
    checkpoint files.
    """
    records = []
    
    for wf in scenario.get("workflows", []):
        wf_id = str(wf.get("id", ""))
        wf_type = str(wf.get("type", "")).lower()
        props = wf.get("properties", {}) or {}
        
        if wf_type == "inspection":
            # Inspection workflows have multiple inspection_points
            insp_points = props.get("inspection_points", [])
            for i, point in enumerate(insp_points):
                # Handle both dict {x,y,z} and list [x,y,z] formats
                if isinstance(point, dict):
                    loc = [point.get("x", 0), point.get("y", 0), point.get("z", 0)]
                elif isinstance(point, (list, tuple)) and len(point) >= 3:
                    loc = list(point)
                else:
                    continue
                records.append({
                    "workflow_id": wf_id,
                    "operation_type": "inspection",
                    "location": str(loc),
                    "checkpoint_idx": i,
                })
        
        elif wf_type == "communication":
            # Communication workflows have a single location
            loc = props.get("location") or props.get("target_location")
            if isinstance(loc, dict):
                loc = [loc.get("x", 0), loc.get("y", 0), loc.get("z", 0)]
            if loc:
                records.append({
                    "workflow_id": wf_id,
                    "operation_type": "communication",
                    "location": str(list(loc) if isinstance(loc, (list, tuple)) else loc),
                    "checkpoint_idx": 0,
                })
        
        elif wf_type == "computation":
            # Computation workflows have compute_location
            loc = props.get("compute_location") or props.get("location") or props.get("target_location")
            if isinstance(loc, dict):
                loc = [loc.get("x", 0), loc.get("y", 0), loc.get("z", 0)]
            if loc and isinstance(loc, (list, tuple)):
                records.append({
                    "workflow_id": wf_id,
                    "operation_type": "compute",
                    "location": str(list(loc)),
                    "checkpoint_idx": 0,
                })
        
        elif wf_type == "logistics":
            # Logistics workflows have pickup and delivery locations
            pickup = props.get("pickup_location")
            delivery = props.get("delivery_location")
            
            if pickup:
                if isinstance(pickup, dict):
                    pickup = [pickup.get("x", 0), pickup.get("y", 0), pickup.get("z", 0)]
                records.append({
                    "workflow_id": wf_id,
                    "operation_type": "pickup",
                    "location": str(list(pickup)),
                    "checkpoint_idx": 0,
                })
            
            if delivery:
                if isinstance(delivery, dict):
                    delivery = [delivery.get("x", 0), delivery.get("y", 0), delivery.get("z", 0)]
                records.append({
                    "workflow_id": wf_id,
                    "operation_type": "delivery",
                    "location": str(list(delivery)),
                    "checkpoint_idx": 1,
                })
    
    return pd.DataFrame.from_records(records)


def build_agent_plans_monolithic(
    scenario: Dict[str, Any],
    checkpoint_df: pd.DataFrame,
    workflow_start_times: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Build per-agent plan data structures using greedy nearest-agent assignment.
    
    Each checkpoint is assigned to the nearest agent.
    """
    # Build agent initial positions
    agent_positions: Dict[str, Tuple[float, float, float]] = {}
    for agent in scenario.get("agents", []):
        agent_id = str(agent.get("id"))
        pos = agent.get("initial_position", {}) or {}
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        z = float(pos.get("z", 0.0))
        agent_positions[agent_id] = (x, y, z)

    # Parse checkpoint locations
    checkpoint_df = checkpoint_df.copy()
    checkpoint_df["loc_x"] = 0.0
    checkpoint_df["loc_y"] = 0.0
    checkpoint_df["loc_z"] = 0.0
    
    for idx, row in checkpoint_df.iterrows():
        x, y, z = _parse_location(row["location"])
        checkpoint_df.at[idx, "loc_x"] = x
        checkpoint_df.at[idx, "loc_y"] = y
        checkpoint_df.at[idx, "loc_z"] = z

    # Greedy assignment: each checkpoint to nearest agent
    agent_ids = list(agent_positions.keys())
    agent_pos_array = np.array([agent_positions[a] for a in agent_ids])
    
    agent_checkpoints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for _, row in checkpoint_df.iterrows():
        ckp_pos = np.array([row["loc_x"], row["loc_y"], row["loc_z"]])
        
        # Find nearest agent
        distances = np.linalg.norm(agent_pos_array - ckp_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_agent = agent_ids[nearest_idx]
        
        ckp_info = {
            "workflow_id": str(row["workflow_id"]),
            "operation_type": str(row["operation_type"]),
            "x": row["loc_x"],
            "y": row["loc_y"],
            "z": row["loc_z"],
            "start_time": workflow_start_times.get(str(row["workflow_id"]), 0.0),
        }
        
        agent_checkpoints[nearest_agent].append(ckp_info)

    # Build plan structure for each agent
    agents_plans: Dict[str, Dict[str, Any]] = {}
    
    for agent_id, pos in agent_positions.items():
        checkpoints = agent_checkpoints.get(agent_id, [])
        
        # Collect all unique locations
        locations_set = set()
        agent_home_label = _make_position_label(pos[0], pos[1], pos[2])
        locations_set.add(agent_home_label)
        
        # Build cluster_meta and goal lists
        cluster_meta: Dict[str, Dict[str, Any]] = {}
        compute_locations: List[str] = []
        communicate_locations: List[str] = []
        inspect_locations: List[str] = []
        
        # Group checkpoints by location label
        loc_to_ckps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for ckp in checkpoints:
            loc_label = _make_position_label(ckp["x"], ckp["y"], ckp["z"])
            locations_set.add(loc_label)
            loc_to_ckps[loc_label].append(ckp)
        
        # Build cluster_meta and goal lists
        for loc_label, ckps in loc_to_ckps.items():
            workflow_ids = list(set(c["workflow_id"] for c in ckps))
            op_types = list(set(c["operation_type"] for c in ckps))
            earliest_start = min(c["start_time"] for c in ckps)
            
            cluster_meta[loc_label] = {
                "cluster_group": -1,  # No clustering
                "earliest_start_time": earliest_start,
                "n_checkpoints": len(ckps),
                "op_types": op_types,
                "workflow_ids": workflow_ids,
            }
            
            # Add to goal lists based on operation types
            for ckp in ckps:
                category = _get_operation_category(ckp["operation_type"])
                if category == "compute" and loc_label not in compute_locations:
                    compute_locations.append(loc_label)
                elif category == "communicate" and loc_label not in communicate_locations:
                    communicate_locations.append(loc_label)
                elif category == "inspect" and loc_label not in inspect_locations:
                    inspect_locations.append(loc_label)
        
        # Build distance matrix
        locations = list(locations_set)
        loc_coords = {agent_home_label: pos}
        for loc_label, ckps in loc_to_ckps.items():
            ckp = ckps[0]
            loc_coords[loc_label] = (ckp["x"], ckp["y"], ckp["z"])
        
        distance_matrix: Dict[str, float] = {}
        for loc1 in locations:
            for loc2 in locations:
                if loc1 != loc2:
                    p1 = loc_coords.get(loc1, (0, 0, 0))
                    p2 = loc_coords.get(loc2, (0, 0, 0))
                    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                    distance_matrix[f"{loc1},{loc2}"] = float(dist)
        
        agents_plans[agent_id] = {
            "init_params": {
                "locations": locations,
                "distance_matrix": distance_matrix,
                "drone_locations": {agent_id: agent_home_label},
                "package_locations": {},
            },
            "goal_params": {
                "compute_locations": compute_locations,
                "communicate_locations": communicate_locations,
                "inspect_locations": inspect_locations,
            },
            "result": {
                "solution": []
            },
            "cluster_meta": cluster_meta,
        }
    
    return agents_plans


def solve_pddl_for_agent(
    agent_id: str,
    agent_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Solve PDDL problem using greedy nearest-neighbor."""
    init_params = agent_plan["init_params"]
    goal_params = agent_plan["goal_params"]
    
    all_goals = []
    for loc in goal_params.get("compute_locations", []):
        all_goals.append(("assistcompute", loc))
    for loc in goal_params.get("communicate_locations", []):
        all_goals.append(("assistcommunicate", loc))
    for loc in goal_params.get("inspect_locations", []):
        all_goals.append(("inspect", loc))
    
    if not all_goals:
        return []
    
    drone_locs = init_params.get("drone_locations", {})
    current_loc = drone_locs.get(agent_id, list(drone_locs.values())[0] if drone_locs else "base")
    
    solution = []
    remaining = list(all_goals)
    step = 1
    
    distance_matrix = init_params.get("distance_matrix", {})
    
    while remaining:
        best_idx = 0
        best_dist = float("inf")
        
        for i, (action, loc) in enumerate(remaining):
            key = f"{current_loc},{loc}"
            dist = distance_matrix.get(key, float("inf"))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        action, next_loc = remaining.pop(best_idx)
        
        if current_loc != next_loc:
            solution.append({
                "step": step,
                "action_name": "movetoloc",
                "parameters": [agent_id, current_loc, next_loc],
            })
            step += 1
            current_loc = next_loc
        
        solution.append({
            "step": step,
            "action_name": action,
            "parameters": [agent_id, next_loc],
        })
        step += 1
    
    return solution


def run_stage3_monolithic(
    seed: int,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Run Stage 3 monolithic planning.
    
    Returns path to pddl_plan_monolithic.json.
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    
    run_dir_in = input_root / f"run_seed_{seed}"
    run_dir_out = output_root / f"run_seed_{seed}"
    
    scenario_path = run_dir_in / "scenario.yaml"
    workflow_init_path = run_dir_in / "workflow_init_info.csv"
    
    for p in [scenario_path, workflow_init_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    
    logger.info("=" * 70)
    logger.info("Stage 3 Monolithic: Generate PDDL plans (seed=%s)", seed)
    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info("=" * 70)
    
    # Load data
    scenario = _load_scenario(scenario_path)
    workflow_start_times = _load_workflow_init_info(workflow_init_path)
    
    # Extract checkpoints from scenario (no clustering file needed)
    checkpoint_df = _extract_checkpoints_from_scenario(scenario)
    logger.info("Extracted %d checkpoints from scenario", len(checkpoint_df))
    
    # Build agent plans
    agents_plans = build_agent_plans_monolithic(
        scenario=scenario,
        checkpoint_df=checkpoint_df,
        workflow_start_times=workflow_start_times,
    )
    
    logger.info("Built plans for %d agents", len(agents_plans))
    
    # Solve PDDL for each agent
    total_goals = 0
    for agent_id, agent_plan in agents_plans.items():
        solution = solve_pddl_for_agent(agent_id, agent_plan)
        agent_plan["result"]["solution"] = solution
        
        n_compute = len(agent_plan["goal_params"].get("compute_locations", []))
        n_comm = len(agent_plan["goal_params"].get("communicate_locations", []))
        n_insp = len(agent_plan["goal_params"].get("inspect_locations", []))
        total_goals += n_compute + n_comm + n_insp
    
    logger.info("Total goals across all agents: %d", total_goals)
    
    # Write output
    run_dir_out.mkdir(parents=True, exist_ok=True)
    output_path = run_dir_out / "pddl_plan_monolithic.json"
    
    output_data = {
        "seed": seed,
        "method": "monolithic",
        "agents": agents_plans,
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("Saved PDDL plan to: %s", output_path)
    
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    parser = argparse.ArgumentParser(
        description="Stage 3 Monolithic: Generate PDDL plans without clustering"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used in Stage 1",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/offline_init",
        help="Root directory containing Stage 1 outputs",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/offline_plans",
        help="Root directory for outputs",
    )
    
    args = parser.parse_args()
    
    input_root = _resolve_under_project(args.input_root, project_root)
    output_root = _resolve_under_project(args.output_root, project_root)
    
    output_path = run_stage3_monolithic(
        seed=args.seed,
        input_root=input_root,
        output_root=output_root,
    )
    
    print("\n=== Stage 3 Monolithic Completed ===")
    print(f"PDDL plan file: {output_path}")


if __name__ == "__main__":
    main()
