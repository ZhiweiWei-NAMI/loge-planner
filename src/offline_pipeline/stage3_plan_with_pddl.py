"""Stage 3: Generate PDDL plans from cluster assignments.

This script takes the outputs from Stage 1 (checkpoint clustering) and Stage 2
(cluster-to-agent assignments) to produce a complete PDDL plan JSON file.

Key Design Choice:
------------------
Each checkpoint is treated as a separate PDDL location and goal, NOT aggregated
into cluster centers. This ensures that:
- Inspection workflows with N checkpoints get N separate (moveto + inspect) actions
- Communication workflows get their own communicate actions at their checkpoint locations
- Computation workflows get their own compute actions at their checkpoint locations

The cluster_meta field still records which workflows are associated with each location,
enabling OfflineTaskLoader to correctly assign tasks to workflows.

Output Format:
--------------
{
  "seed": <int>,
  "method": <str>,
  "agents": {
    "<agent_id>": {
      "init_params": {
        "locations": [...],
        "distance_matrix": {...},
        "drone_locations": {...},
        "package_locations": {}
      },
      "goal_params": {
        "compute_locations": [...],
        "communicate_locations": [...],
        "inspect_locations": [...]
      },
      "result": {
        "solution": [...]
      },
      "cluster_meta": {
        "<loc_label>": {
          "cluster_group": <int>,
          "earliest_start_time": <float>,
          "n_checkpoints": 1,
          "op_types": [...],
          "workflow_ids": [...]
        }
      }
    }
  }
}
"""

import argparse
import ast
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import fcntl
import time

from src.pddl_solver import PDDLSolver
from src.simulation.environment.topology_manager import TopologyManager
from src.offline_pipeline.edge_planner import EdgePlanner

logger = logging.getLogger(__name__)


def _generate_traffic_profiles(topo: TopologyManager, output_path: Path, seed: int = 42) -> None:
    """生成时变交通配置文件，模拟高峰时段限速
    
    Args:
        topo: TopologyManager实例
        output_path: 输出CSV路径
        seed: 随机种子
    
    高峰时段模拟:
    - 早高峰 (0-600s): 30%走廊限速至 0.33 (约5 m/s)
    - 正常时段 (600-1200s): 正常速度
    - 晚高峰 (1200-1800s): 30%走廊限速至 0.5 (约7.5 m/s)
    """
    random.seed(seed)
    
    edges = list(topo.edges.keys())
    if not edges:
        logger.warning("No edges in topology, cannot generate traffic profiles")
        return
    
    # 选择20%的边作为拥堵走廊
    n_congested = max(1, int(len(edges) * 0.2))
    congested_edges = random.sample(edges, min(n_congested, len(edges)))
    
    profiles = []
    for (u, v) in congested_edges:
        # 早高峰: 0-300s, 严重限速 0.1 (约1.5 m/s)
        profiles.append((u, v, 0.0, 300.0, 0.1, 0))
        # 正常时段: 300-1800s, 速度恢复
        profiles.append((u, v, 300.0, 1800.0, 1.0, 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['u', 'v', 't_start', 't_end', 'speed_factor', 'blocked'])
        for row in profiles:
            writer.writerow(row)
    
    logger.info("Generated traffic profiles for %d congested edges (30%% of %d total)", 
               len(congested_edges), len(edges))


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


def _make_topo_node_label(node_id: str) -> str:
    """Create a location label from topology node ID."""
    return f"topo_{node_id}"


def _snap_to_topo_node(
    x: float, y: float, z: float,
    topo: Optional[TopologyManager],
    type_filter: Optional[List[str]] = None
) -> Optional[str]:
    """Snap coordinates to nearest topology node.
    
    Args:
        x, y, z: Coordinates to snap
        topo: TopologyManager instance (if None, returns None)
        type_filter: Optional list of node types to filter
    
    Returns:
        Node ID if found, None otherwise
    """
    if topo is None:
        return None
    return topo.get_nearest_node(x, y, z, type_filter=type_filter)


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


def _load_checkpoint_clusters(checkpoint_path: Path) -> pd.DataFrame:
    """Load checkpoint_clusters_<method>.csv."""
    return pd.read_csv(checkpoint_path)


def _load_cluster_assignments(assignment_path: Path) -> pd.DataFrame:
    """Load cluster_assignment_<method>.csv."""
    return pd.read_csv(assignment_path)


#: operation category -> PDDL action name (see config/pddl/tccn_domain.pddl)
OP_CATEGORY_TO_ACTION = {
    "compute": "assistcompute",
    "communicate": "assistcommunicate",
    "inspect": "inspect",
}


def _get_operation_category(op_type: str) -> Optional[str]:
    """Map operation_type to PDDL goal category.
    
    Returns: 'compute', 'communicate', 'inspect', or None (for other types)
    """
    op_type = op_type.lower()
    if op_type in ("compute", "move_to_compute"):
        return "compute"
    if op_type in ("communication", "communicate", "move_to_communication"):
        return "communicate"
    if op_type in ("inspection", "inspect", "move_to_inspection"):
        return "inspect"
    return None


def build_agent_plans(
    scenario: Dict[str, Any],
    checkpoint_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    workflow_start_times: Dict[str, float],
    topo: Optional[TopologyManager] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build per-agent plan data structures.
    
    Each checkpoint becomes a separate location in the PDDL problem.
    If topo is provided, checkpoint coordinates are snapped to nearest topo nodes.
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

    # Build cluster_group -> assigned_agents mapping
    cluster_to_agents: Dict[int, List[str]] = {}
    for _, row in assignment_df.iterrows():
        cg = int(row["cluster_group"])
        agents_str = row.get("assigned_agents", "")
        agents = [a.strip() for a in agents_str.split(";") if a.strip()]
        cluster_to_agents[cg] = agents

    # Parse checkpoint locations and group by cluster
    checkpoint_df = checkpoint_df.copy()
    checkpoint_df["loc_x"] = 0.0
    checkpoint_df["loc_y"] = 0.0
    checkpoint_df["loc_z"] = 0.0
    
    for idx, row in checkpoint_df.iterrows():
        x, y, z = _parse_location(row["location"])
        checkpoint_df.at[idx, "loc_x"] = x
        checkpoint_df.at[idx, "loc_y"] = y
        checkpoint_df.at[idx, "loc_z"] = z

    # For each agent, collect checkpoints from assigned clusters
    agent_checkpoints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for _, row in checkpoint_df.iterrows():
        cg = int(row["cluster_group"])
        assigned_agents = cluster_to_agents.get(cg, [])
        if not assigned_agents:
            continue
        
        ckp_info = {
            "workflow_id": str(row["workflow_id"]),
            "operation_type": str(row["operation_type"]),
            "x": row["loc_x"],
            "y": row["loc_y"],
            "z": row["loc_z"],
            "cluster_group": cg,
            "start_time": workflow_start_times.get(str(row["workflow_id"]), 0.0),
            "checkpoint_idx": int(row.get("checkpoint_idx", 0)),  # Track specific checkpoint index
        }
        
        # Assign to all candidate agents - PDDL solver will handle multi-agent planning
        for agent_id in assigned_agents:
            agent_checkpoints[agent_id].append(ckp_info)

    # Build plan structure for each agent
    agents_plans: Dict[str, Dict[str, Any]] = {}
    
    for agent_id, pos in agent_positions.items():
        checkpoints = agent_checkpoints.get(agent_id, [])
        
        # Collect all unique locations
        locations_set = set()
        # Snap agent home position to topo node
        agent_home_node_id = _snap_to_topo_node(pos[0], pos[1], pos[2], topo)
        if not agent_home_node_id:
            raise RuntimeError(f"Cannot snap agent {agent_id} home position ({pos[0]}, {pos[1]}, {pos[2]}) to topo node.")
        agent_home_label = _make_topo_node_label(agent_home_node_id)
        locations_set.add(agent_home_label)
        
        # Build cluster_meta and goal lists
        cluster_meta: Dict[str, Dict[str, Any]] = {}
        compute_locations: List[str] = []
        communicate_locations: List[str] = []
        inspect_locations: List[str] = []
        
        # Group checkpoints by location label to handle duplicates
        loc_to_ckps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for ckp in checkpoints:
            # Snap to topo node (topology is required)
            topo_node_id = _snap_to_topo_node(ckp["x"], ckp["y"], ckp["z"], topo)
            if not topo_node_id:
                raise RuntimeError(f"Cannot snap checkpoint ({ckp['x']}, {ckp['y']}, {ckp['z']}) to topo node. Topology is required.")
            loc_label = _make_topo_node_label(topo_node_id)
            ckp["topo_node_id"] = topo_node_id
            locations_set.add(loc_label)
            loc_to_ckps[loc_label].append(ckp)
        
        # Build cluster_meta and goal lists
        for loc_label, ckps in loc_to_ckps.items():
            workflow_ids = list(set(c["workflow_id"] for c in ckps))
            op_types = list(set(c["operation_type"] for c in ckps))
            cluster_groups = list(set(c["cluster_group"] for c in ckps))
            earliest_start = min(c["start_time"] for c in ckps)
            
            # Extract topo_node_id if present (all ckps at same loc should have same node)
            topo_node_ids = [c.get("topo_node_id") for c in ckps if c.get("topo_node_id")]
            topo_node_id = topo_node_ids[0] if topo_node_ids else None
            
            # Store original coordinates for runtime resolution
            first_ckp = ckps[0]
            original_coords = [first_ckp["x"], first_ckp["y"], first_ckp["z"]]
            
            # Build checkpoint_indices: workflow_id -> list of actual point indices at this location
            # This ensures runtime_loader only generates tasks for specific checkpoints, not all
            # NOTE: checkpoint_idx in CSV includes both move_to_* and actual operation checkpoints
            # For inspection/compute/communicate, actual_point_index = checkpoint_idx // 2
            checkpoint_indices: Dict[str, List[int]] = defaultdict(list)
            for c in ckps:
                wf_id = c["workflow_id"]
                ckp_idx = c.get("checkpoint_idx", 0)
                op_type = c.get("operation_type", "").lower()
                
                # Only include actual operation checkpoints (not move_to_* variants)
                # and calculate the actual point index
                if op_type in ("inspection", "inspect"):
                    actual_point_idx = ckp_idx // 2  # move_to + inspection pairs
                    if actual_point_idx not in checkpoint_indices[wf_id]:
                        checkpoint_indices[wf_id].append(actual_point_idx)
                elif op_type in ("compute",):
                    actual_point_idx = ckp_idx // 2  # move_to + compute pairs
                    if actual_point_idx not in checkpoint_indices[wf_id]:
                        checkpoint_indices[wf_id].append(actual_point_idx)
                elif op_type in ("communication", "communicate"):
                    actual_point_idx = ckp_idx // 2  # move_to + communicate pairs
                    if actual_point_idx not in checkpoint_indices[wf_id]:
                        checkpoint_indices[wf_id].append(actual_point_idx)
            
            cluster_meta[loc_label] = {
                "cluster_group": cluster_groups[0] if len(cluster_groups) == 1 else cluster_groups,
                "earliest_start_time": earliest_start,
                "n_checkpoints": len(ckps),
                "op_types": op_types,
                "workflow_ids": workflow_ids,
                "checkpoint_indices": dict(checkpoint_indices),  # workflow_id -> [checkpoint_idx, ...]
                "topo_node_id": topo_node_id,
                "original_coords": original_coords,
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
                "solution": []  # Will be filled by PDDL solver
            },
            "cluster_meta": cluster_meta,
        }
    
    return agents_plans


def solve_pddl_for_cluster(
    cluster_id: int,
    candidate_drones: List[str],
    drone_positions: Dict[str, Tuple[float, float, float]],
    checkpoints: List[Dict[str, Any]],
    domain_file: Path,
    topo: Optional['TopologyManager'] = None,
    pddl_timeout: float = 60.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Solve PDDL for a single cluster with multiple candidate drones.
    
    This enables true multi-agent planning where the PDDL solver decides
    which drone handles which checkpoint.
    
    Args:
        cluster_id: The cluster group ID
        candidate_drones: List of candidate drone/agent IDs for this cluster
        drone_positions: Initial positions of all candidate drones
        checkpoints: List of checkpoint dicts with x, y, z, operation_type, workflow_id
        domain_file: Path to PDDL domain file
        topo: TopologyManager instance for snapping positions to nodes
    
    Returns:
        Dict mapping drone_id -> list of solution steps for that drone
    """
    if not candidate_drones or not checkpoints:
        return {d: [] for d in candidate_drones}
    
    # Add drone home positions
    drone_locations: Dict[str, str] = {}
    for drone_id in candidate_drones:
        pos = drone_positions.get(drone_id)
        if not pos:
            continue
        # Snap to topo node
        node_id = _snap_to_topo_node(pos[0], pos[1], pos[2], topo)
        if not node_id:
            continue
        drone_locations[drone_id] = _make_topo_node_label(node_id)
    
    if not drone_locations:
        raise RuntimeError(
            f"cluster {cluster_id}: no candidate drone could be snapped to a topology node "
            f"(drones={candidate_drones})"
        )
    
    # Add checkpoint locations and categorize goals
    compute_locations: List[str] = []
    communicate_locations: List[str] = []
    inspect_locations: List[str] = []
    
    for ckp in checkpoints:
        # Snap to topo node
        node_id = _snap_to_topo_node(ckp["x"], ckp["y"], ckp["z"], topo)
        if not node_id:
            continue
        loc_label = _make_topo_node_label(node_id)
        
        category = _get_operation_category(ckp["operation_type"])
        if category == "compute" and loc_label not in compute_locations:
            compute_locations.append(loc_label)
        elif category == "communicate" and loc_label not in communicate_locations:
            communicate_locations.append(loc_label)
        elif category == "inspect" and loc_label not in inspect_locations:
            inspect_locations.append(loc_label)
    
    # Build the solver's connectivity from the REAL airspace graph: the location
    # universe is the whole topology and the distance matrix contains only
    # existing corridors.  PDDLSolver derives `connected` from this matrix (plus
    # an MST over it), so no corridor that does not exist can be flown.  This is
    # the same graph the A* router in EdgePlanner uses.
    if topo is None:
        raise RuntimeError(
            f"cluster {cluster_id}: topology is required to build the PDDL connectivity graph"
        )
    locations = [_make_topo_node_label(node_id) for node_id in topo.nodes]
    distance_matrix: Dict[Tuple[str, str], float] = {
        (_make_topo_node_label(u), _make_topo_node_label(v)): float(edge.length)
        for (u, v), edge in topo.edges.items()
        if u in topo.nodes and v in topo.nodes
    }
    
    # Build solver params
    solver_init_params = {
        "locations": locations,
        "distance_matrix": distance_matrix,
        "drone_locations": drone_locations,
        "package_locations": {},
        "packages_on_drone": {},
    }
    
    solver_goal_params = {
        "package_destinations": {},
        "compute_locations": compute_locations,
        "communicate_locations": communicate_locations,
        "inspect_locations": inspect_locations,
    }
    
    total_goals = len(compute_locations) + len(communicate_locations) + len(inspect_locations)
    
    if total_goals == 0:
        return {d: [] for d in candidate_drones}
    
    # Solve with PDDL (带超时)
    solver = PDDLSolver(str(domain_file))
    # only real corridors may be `connected`: no pair outside the matrix is added
    connectivity_threshold = max(distance_matrix.values(), default=1.0) + 1.0
    result = solver.solve(
        solver_init_params,
        solver_goal_params,
        distance_threshold=connectivity_threshold,
        timeout=pddl_timeout,
    )
    
    if not result.get("success"):
        error_msg = result.get('error', 'Unknown error')
        raise RuntimeError(
            f"cluster {cluster_id}: PDDL planning failed ({error_msg}); "
            f"drones={candidate_drones}, goals={total_goals}"
        )
    
    # Parse solution and split by drone
    raw_solution = result.get("solution") or []
    drone_solutions: Dict[str, List[Dict[str, Any]]] = {d: [] for d in candidate_drones}
    drone_steps: Dict[str, int] = {d: 1 for d in candidate_drones}
    
    for item in raw_solution:
        action_name_raw = (item.get("action_name") or item.get("action") or "").lower()
        params = [str(p) for p in (item.get("parameters") or [])]
        
        # First param is usually the drone
        if not params:
            continue
        drone_id = params[0]
        
        if drone_id not in drone_solutions:
            # Map back to original agent ID if needed
            for d in candidate_drones:
                if d.lower() == drone_id.lower() or drone_id in d or d in drone_id:
                    drone_id = d
                    break
        
        if drone_id not in drone_solutions:
            continue
        
        step = drone_steps[drone_id]
        
        if action_name_raw == "movetoloc" and len(params) >= 3:
            drone_solutions[drone_id].append({
                "step": step,
                "action_name": "movetoloc",
                "parameters": [params[0], params[1], params[2]],
            })
            drone_steps[drone_id] += 1
        elif action_name_raw == "assistcompute" and len(params) >= 2:
            drone_solutions[drone_id].append({
                "step": step,
                "action_name": "assistcompute",
                "parameters": [params[0], params[1]],
            })
            drone_steps[drone_id] += 1
        elif action_name_raw == "assistcommunicate" and len(params) >= 2:
            drone_solutions[drone_id].append({
                "step": step,
                "action_name": "assistcommunicate",
                "parameters": [params[0], params[1]],
            })
            drone_steps[drone_id] += 1
        elif action_name_raw == "inspect" and len(params) >= 2:
            drone_solutions[drone_id].append({
                "step": step,
                "action_name": "inspect",
                "parameters": [params[0], params[1]],
            })
            drone_steps[drone_id] += 1
    
    return drone_solutions


def run_stage3_planning(
    seed: int,
    cluster_method: str,
    input_root: Path,
    output_root: Path,
    topology_file: Optional[Path] = None,
    export_edge_routes: bool = True,
    topo_aware: bool = False,
    pddl_timeout: float = 60.0,
) -> Path:
    """Run Stage 3 planning: generate PDDL plans for all agents.
    
    Args:
        topo_aware: If True, estimate speed limits from topology edges and include
                    them in the plan JSON for runtime MoveToTask speed configuration.
    
    Returns path to pddl_plan_<method>.json.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    
    run_dir_in = input_root / f"run_seed_{seed}"
    run_dir_out = output_root / f"run_seed_{seed}"
    
    # Input files
    scenario_path = run_dir_in / "scenario.yaml"
    workflow_init_path = run_dir_in / "workflow_init_info.csv"
    checkpoint_path = run_dir_in / f"checkpoint_clusters_{cluster_method}.csv"
    assignment_path = run_dir_out / f"cluster_assignment_{cluster_method}.csv"
    
    # Validate inputs
    for p in [scenario_path, workflow_init_path, checkpoint_path, assignment_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    
    logger.info("=" * 70)
    logger.info("Stage 3: Generate PDDL plans (seed=%s, method=%s)", seed, cluster_method)
    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info("=" * 70)
    
    # Load data
    scenario = _load_scenario(scenario_path)
    workflow_start_times = _load_workflow_init_info(workflow_init_path)
    checkpoint_df = _load_checkpoint_clusters(checkpoint_path)
    assignment_df = _load_cluster_assignments(assignment_path)
    
    logger.info("Loaded %d checkpoints, %d cluster assignments", 
                len(checkpoint_df), len(assignment_df))
    
    # Load topology for snap-to-node and edge route planning
    topo: Optional[TopologyManager] = None
    planner: Optional[EdgePlanner] = None
    topology_path: Optional[Path] = None

    if topology_file is not None:
        topology_path = topology_file
    else:
        env_cfg = scenario.get("environment", {}) or {}
        topo_path_str = env_cfg.get("graph_topology_file")
        if topo_path_str:
            topology_path = _resolve_under_project(str(topo_path_str), project_root)
        else:
            # Auto-detect topology file in run_dir
            default_topo = run_dir_in / "graph_topology.json"
            if default_topo.exists():
                topology_path = default_topo
                logger.info("Auto-detected topology file: %s", topology_path)

    if topology_path is not None and topology_path.exists():
        topo = TopologyManager(bounds={"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0})
        topo.load_from_file(str(topology_path))
        logger.info("Loaded topology with %d nodes, %d edges", len(topo.nodes), len(topo.edges))

        # Always generate/load traffic profiles for runtime simulation
        # Both topo-aware and non-topo-aware modes face the same traffic conditions
        # The difference is: topo-aware PLANS considering traffic, non-topo-aware doesn't
        traffic_csv = topology_path.with_name(f"{topology_path.stem}_traffic.csv")
        if not traffic_csv.exists():
            # Generate traffic profiles for runtime (same for both modes)
            _generate_traffic_profiles(topo, traffic_csv, seed)
            logger.info("Generated traffic profiles: %s", traffic_csv)
        
        # Load traffic profiles into topology
        topo.load_traffic_from_csv(str(traffic_csv))
        logger.info("Loaded traffic profiles from %s (%d edges with time-varying speed)", 
                   traffic_csv, len(topo.traffic_profiles))
        
        # IMPORTANT: Only topo-aware mode uses traffic profiles during path PLANNING
        # Non-topo-aware plans static shortest paths, then faces traffic at runtime
        if not topo_aware:
            # Clear traffic profiles from topology for planning (but keep file for runtime)
            topo.traffic_profiles = {}
            logger.info("Non-topo-aware mode: planning with static topology (runtime will use traffic)")
    else:
        # Never invent an airspace: topology generation is Stage 1's explicit job.
        raise FileNotFoundError(
            "Stage 3 requires a topology graph, but none was found "
            f"(looked for scenario.environment.graph_topology_file and {run_dir_in / 'graph_topology.json'}). "
            "Generate it explicitly in Stage 1 (src.offline_pipeline.stage1_precluster_from_config) "
            "or pass --topology-file."
        )

    # Build agent plans with topo snap-to-node
    agents_plans = build_agent_plans(
        scenario=scenario,
        checkpoint_df=checkpoint_df,
        assignment_df=assignment_df,
        workflow_start_times=workflow_start_times,
        topo=topo,
    )
    
    logger.info("Built plans for %d agents", len(agents_plans))

    # Export edge routes if requested
    if export_edge_routes and topo is not None:
        planner = EdgePlanner(topo)
        for agent_id, agent_plan in list(agents_plans.items()):
            agents_plans[agent_id] = planner.export_routes_for_agent_plan(
                agent_plan=agent_plan,
                scenario=scenario,
                workflow_start_times=workflow_start_times,
            )
    
    # Hungarian: 跳过PDDL，直接按workflow生成顺序计划
    if cluster_method == "hungarian":
        logger.info(f"Method {cluster_method}: 跳过PDDL，直接生成workflow顺序计划")
        # 按workflow分组checkpoints，每个workflow的ckps按checkpoint_idx排序后顺序执行
        workflow_agent_map = {}  # workflow_id -> agent_id
        for _, row in assignment_df.iterrows():
            agents_str = row.get("assigned_agents", "")
            agents = [a.strip() for a in agents_str.split(";") if a.strip()]
            if agents:
                # 从checkpoint_df找这个cluster的workflows
                cg = int(row["cluster_group"])
                for _, ckp_row in checkpoint_df[checkpoint_df["cluster_group"] == cg].iterrows():
                    wf_id = str(ckp_row["workflow_id"])
                    if wf_id not in workflow_agent_map:
                        workflow_agent_map[wf_id] = agents[0]
        
        # 为每个agent生成顺序计划
        agent_solutions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for wf_id, agent_id in workflow_agent_map.items():
            wf_ckps = checkpoint_df[checkpoint_df["workflow_id"] == wf_id].sort_values("checkpoint_idx")
            for _, row in wf_ckps.iterrows():
                x, y, z = _parse_location(row["location"])
                node_id = _snap_to_topo_node(x, y, z, topo)
                if not node_id:
                    continue
                loc_label = _make_topo_node_label(node_id)
                op_type = str(row["operation_type"])
                category = _get_operation_category(op_type)
                
                # Move to location
                step = len(agent_solutions[agent_id]) + 1
                agent_solutions[agent_id].append({
                    "step": step,
                    "action_name": "movetoloc", 
                    "parameters": [agent_id, "current", loc_label]
                })
                step += 1
                
                # Perform action
                if category == "compute":
                    agent_solutions[agent_id].append({
                        "step": step, "action_name": "assistcompute",
                        "parameters": [agent_id, loc_label]
                    })
                elif category == "communicate":
                    agent_solutions[agent_id].append({
                        "step": step, "action_name": "assistcommunicate", 
                        "parameters": [agent_id, loc_label]
                    })
                elif category == "inspect":
                    agent_solutions[agent_id].append({
                        "step": step, "action_name": "inspect",
                        "parameters": [agent_id, loc_label]
                    })
        
        # 更新agents_plans
        for agent_id, agent_plan in agents_plans.items():
            agent_plan["result"]["solution"] = agent_solutions.get(agent_id, [])
        
        logger.info(f"Generated sequential plans for {len(workflow_agent_map)} workflows -> {len(set(workflow_agent_map.values()))} agents")
    else:
        # Multi-agent PDDL planning for other methods
        domain_file = project_root / "config" / "pddl" / "tccn_domain.pddl"
        if not domain_file.exists():
            raise FileNotFoundError(f"PDDL domain file not found: {domain_file}")
        
        # Build agent positions
        agent_positions: Dict[str, Tuple[float, float, float]] = {}
        for agent in scenario.get("agents", []):
            agent_id = str(agent.get("id"))
            pos = agent.get("initial_position", {}) or {}
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
            agent_positions[agent_id] = (x, y, z)
        
        # Build cluster -> agents and cluster -> checkpoints mappings
        cluster_to_agents: Dict[int, List[str]] = {}
        for _, row in assignment_df.iterrows():
            cg = int(row["cluster_group"])
            agents_str = row.get("assigned_agents", "")
            agents = [a.strip() for a in agents_str.split(";") if a.strip()]
            cluster_to_agents[cg] = agents
        
        # Parse checkpoint locations
        checkpoint_df_copy = checkpoint_df.copy()
        checkpoint_df_copy["loc_x"] = 0.0
        checkpoint_df_copy["loc_y"] = 0.0
        checkpoint_df_copy["loc_z"] = 0.0
        for idx, row in checkpoint_df_copy.iterrows():
            x, y, z = _parse_location(row["location"])
            checkpoint_df_copy.at[idx, "loc_x"] = x
            checkpoint_df_copy.at[idx, "loc_y"] = y
            checkpoint_df_copy.at[idx, "loc_z"] = z
        
        # Group checkpoints by cluster
        cluster_checkpoints: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in checkpoint_df_copy.iterrows():
            cg = int(row["cluster_group"])
            ckp_info = {
                "workflow_id": str(row["workflow_id"]),
                "operation_type": str(row["operation_type"]),
                "x": row["loc_x"],
                "y": row["loc_y"],
                "z": row["loc_z"],
                "cluster_group": cg,
                "start_time": workflow_start_times.get(str(row["workflow_id"]), 0.0),
            }
            cluster_checkpoints[cg].append(ckp_info)
        
        # Solve PDDL per cluster (multi-agent planning)
        agent_solutions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_goals = 0
        cluster_failures: List[str] = []
        
        for cg, ckps in cluster_checkpoints.items():
            candidate_drones = cluster_to_agents.get(cg, [])
            if not candidate_drones:
                logger.warning(f"Cluster {cg} has no assigned agents, skipping")
                continue
            
            # Count goals for this cluster (using topo format)
            goal_labels = set()
            for c in ckps:
                node_id = _snap_to_topo_node(c["x"], c["y"], c["z"], topo)
                if node_id:
                    goal_labels.add(_make_topo_node_label(node_id))
            n_goals = len(goal_labels)
            total_goals += n_goals
            expected_goals = set()
            for c in ckps:
                node_id = _snap_to_topo_node(c["x"], c["y"], c["z"], topo)
                category = _get_operation_category(str(c["operation_type"]))
                if node_id and category in OP_CATEGORY_TO_ACTION:
                    expected_goals.add(
                        (OP_CATEGORY_TO_ACTION[category], _make_topo_node_label(node_id))
                    )
            
            # Solve multi-agent PDDL for this cluster (failure propagates, never masked)
            try:
                drone_solutions = solve_pddl_for_cluster(
                    cluster_id=cg,
                    candidate_drones=candidate_drones,
                    drone_positions=agent_positions,
                    checkpoints=ckps,
                    domain_file=domain_file,
                    topo=topo,
                    pddl_timeout=pddl_timeout,
                )
            except RuntimeError as exc:
                logger.error("Stage 3 PDDL failure: %s", exc)
                cluster_failures.append(str(exc))
                continue
            
            # An idle candidate drone is legitimate (the solver may hand the whole
            # cluster to one drone), so check the returned plan actually satisfies
            # every goal instead of demanding steps from every drone.
            achieved = set()
            for steps in drone_solutions.values():
                for step in steps:
                    params = [str(p) for p in (step.get("parameters") or [])]
                    name = str(step.get("action_name") or "").lower()
                    if name in OP_CATEGORY_TO_ACTION.values() and len(params) >= 2:
                        achieved.add((name, params[1]))
            missing_goals = sorted(
                goal for goal in expected_goals if goal not in achieved
            )
            if missing_goals:
                cluster_failures.append(
                    f"cluster {cg}: PDDL plan does not satisfy goals {missing_goals}"
                )
                continue

            # Aggregate solutions by agent
            for drone_id, solution_steps in drone_solutions.items():
                agent_solutions[drone_id].extend(solution_steps)
            
            logger.debug(f"Cluster {cg}: {len(candidate_drones)} drones, {n_goals} goals")
        
        if cluster_failures:
            raise RuntimeError(
                f"Stage 3 PDDL planning failed for {len(cluster_failures)} cluster(s); "
                f"refusing to emit a plan file. First failure: {cluster_failures[0]}"
            )

        # Update agents_plans with aggregated solutions
        for agent_id, agent_plan in agents_plans.items():
            solution = agent_solutions.get(agent_id, [])
            # Re-number steps sequentially for each agent
            for i, step in enumerate(solution, 1):
                step["step"] = i
            agent_plan["result"]["solution"] = solution
            
            n_compute = len(agent_plan["goal_params"].get("compute_locations", []))
            n_comm = len(agent_plan["goal_params"].get("communicate_locations", []))
            n_insp = len(agent_plan["goal_params"].get("inspect_locations", []))
            
            if solution:
                logger.debug("Agent %s: %d original goals, %d solution steps from multi-agent planning",
                            agent_id, n_compute + n_comm + n_insp, len(solution))
        
        logger.info("Total unique goals across all clusters: %d", total_goals)
    
    # Write output
    run_dir_out.mkdir(parents=True, exist_ok=True)
    output_path = run_dir_out / f"pddl_plan_{cluster_method}.json"
    
    # Build topology node_coords mapping for runtime resolution
    topology_data: Optional[Dict[str, Any]] = None
    if topo is not None:
        node_coords = {}
        for node_id, node in topo.nodes.items():
            node_coords[node_id] = [node.x, node.y, node.z]
        
        # Build edge speed limits for topo-aware mode
        edge_speed_limits: Dict[str, float] = {}
        if topo_aware:
            logger.info("Topo-aware mode: estimating speed limits from topology edges")
            for (u, v), edge in topo.edges.items():
                # Use base_velocity as the speed limit (considering potential congestion)
                # Apply a safety margin of 90% to avoid speed violations
                speed_limit = edge.base_velocity * 0.9
                edge_speed_limits[f"{u},{v}"] = round(speed_limit, 2)
            logger.info("Computed speed limits for %d edges (avg: %.2f m/s)", 
                       len(edge_speed_limits),
                       sum(edge_speed_limits.values()) / max(len(edge_speed_limits), 1))
        
        topology_data = {
            "node_coords": node_coords,
            "bounds": topo.bounds,
            "n_nodes": len(topo.nodes),
            "n_edges": len(topo.edges),
            "topo_aware": topo_aware,
            "edge_speed_limits": edge_speed_limits if topo_aware else {},
        }
    
    output_data = {
        "seed": seed,
        "method": cluster_method,
        "agents": agents_plans,
        "topology": topology_data,
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
        description="Stage 3: Generate PDDL plans from cluster assignments"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used in Stage 1 and 2",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="rule_kmeans",
        help="Clustering method name (e.g., rule_kmeans, kmeans)",
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
        help="Root directory for Stage 2 and 3 outputs",
    )
    parser.add_argument(
        "--topology-file",
        type=str,
        default=None,
        help="Optional path to topology JSON; overrides scenario.environment.graph_topology_file if provided",
    )
    parser.add_argument(
        "--export-edge-routes",
        action="store_true",
        help="Enable EdgePlanner to export edge_routes into the agent plans",
    )
    parser.add_argument(
        "--topo-aware",
        action="store_true",
        help="Enable topology-aware speed estimation for MoveToTask (estimates speed limits from corridor constraints)",
    )
    
    args = parser.parse_args()
    
    input_root = _resolve_under_project(args.input_root, project_root)
    output_root = _resolve_under_project(args.output_root, project_root)

    topo_path: Optional[Path] = None
    if args.topology_file is not None:
        topo_path = _resolve_under_project(args.topology_file, project_root)
    
    output_path = run_stage3_planning(
        seed=args.seed,
        cluster_method=args.cluster_method,
        input_root=input_root,
        output_root=output_root,
        topology_file=topo_path,
        export_edge_routes=args.export_edge_routes,
        topo_aware=args.topo_aware,
    )
    
    print("\n=== Stage 3 Completed ===")
    print(f"PDDL plan file: {output_path}")


if __name__ == "__main__":
    main()
