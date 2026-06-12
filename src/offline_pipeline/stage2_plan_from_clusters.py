"""Stage 2: Plan from clustered checkpoints.

This script performs a very simple offline planning step based purely on
cluster locations and workflow start times:

1) Load scenario.yaml (agent initial positions).
2) Load workflow_init_info.csv (workflow start_time).
3) Load checkpoint_clusters_<method>.csv (cluster_group per checkpoint).
4) For each cluster_group, compute:
   - Spatial center (mean of checkpoint locations).
   - Earliest start time among its workflows.
5) Sort clusters by earliest start time.
6) For each cluster in order, assign the K nearest agents to this cluster,
   but enforce a cooldown period per agent:

   - An agent that has just been assigned to a cluster at time T will not be
     considered for other clusters whose earliest_start_time < T + cooldown.

The output is a simple cluster-level assignment file that can be used by
Stage 3 execution logic to decide which agents are primarily responsible
for tasks belonging to each cluster.
"""

import argparse
import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


@dataclass
class ClusterInfo:
    cluster_group: int
    center_x: float
    center_y: float
    center_z: float
    earliest_start_time: float
    n_checkpoints: int
    op_types: List[str] = None  # Operation types in this cluster


@dataclass
class AgentInfo:
    agent_id: str
    x: float
    y: float
    z: float


def _parse_location(loc_str: str) -> Tuple[float, float, float]:
    """Parse location field stored as string like "[x, y, z]".

    Returns (x, y, z). If parsing fails, returns (0.0, 0.0, 0.0).
    """
    try:
        value = ast.literal_eval(loc_str)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            x, y, z = value[0], value[1], value[2]
            return float(x), float(y), float(z)
    except Exception:
        logger.warning("Failed to parse location string: %s", loc_str)
    return 0.0, 0.0, 0.0


def _load_agents_from_scenario(scenario_path: Path, retry_count: int = 3, retry_delay: float = 0.1) -> List[AgentInfo]:
    """Load agent initial positions from scenario.yaml with file locking and retry."""
    logger.info("Loading scenario from: %s", scenario_path)
    last_error = None
    data = None
    for attempt in range(retry_count):
        try:
            with scenario_path.open("r", encoding="utf-8") as f:
                # Acquire shared lock
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read()
                    if '\x00' in content:
                        raise ValueError("File contains null bytes, may be corrupted")
                    data = yaml.safe_load(content)
                    break
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            last_error = e
            logger.warning("Failed to load scenario (attempt %d/%d): %s", attempt + 1, retry_count, e)
            if attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))
    
    if data is None:
        raise RuntimeError(f"Failed to load scenario after {retry_count} attempts: {last_error}")

    agents_raw = data.get("agents", [])
    agents: List[AgentInfo] = []
    for a in agents_raw:
        agent_id = str(a.get("id"))
        pos = a.get("initial_position", {}) or {}
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        z = float(pos.get("z", 0.0))
        agents.append(AgentInfo(agent_id=agent_id, x=x, y=y, z=z))

    logger.info("Loaded %d agents from scenario", len(agents))
    return agents


def _compute_cluster_infos(
    checkpoint_clusters_path: Path,
    workflow_init_info_path: Path,
) -> List[ClusterInfo]:
    """Compute per-cluster spatial center and earliest start time.

    - Parse checkpoint locations from checkpoint_clusters_<method>.csv.
    - Join workflow start_time from workflow_init_info.csv via workflow_id.
    - Group by cluster_group.
    """
    logger.info("Loading clustered checkpoints from: %s", checkpoint_clusters_path)
    df_ckp = pd.read_csv(checkpoint_clusters_path)

    if "cluster_group" not in df_ckp.columns:
        raise ValueError(
            f"Expected column 'cluster_group' in {checkpoint_clusters_path}, "
            f"got columns: {list(df_ckp.columns)}"
        )

    logger.info("Loading workflow init info from: %s", workflow_init_info_path)
    df_wf = pd.read_csv(workflow_init_info_path)
    if "workflow_id" not in df_wf.columns or "start_time" not in df_wf.columns:
        raise ValueError(
            f"workflow_init_info.csv must contain 'workflow_id' and 'start_time' columns, "
            f"got: {list(df_wf.columns)}"
        )

    start_time_map: Dict[str, float] = (
        df_wf.set_index("workflow_id")["start_time"].to_dict()
    )

    # Parse locations and attach start_time
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    start_times: List[float] = []

    for _, row in df_ckp.iterrows():
        x, y, z = _parse_location(row["location"])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        wf_id = str(row["workflow_id"])
        st = start_time_map.get(wf_id, 0.0)
        start_times.append(float(st))

    df_ckp = df_ckp.copy()
    df_ckp["loc_x"] = xs
    df_ckp["loc_y"] = ys
    df_ckp["loc_z"] = zs
    df_ckp["start_time"] = start_times

    cluster_infos: List[ClusterInfo] = []

    for cluster_group, g in df_ckp.groupby("cluster_group"):
        center_x = float(g["loc_x"].mean())
        center_y = float(g["loc_y"].mean())
        center_z = float(g["loc_z"].mean())
        earliest_start = float(g["start_time"].min())
        n_ckp = int(len(g))

        # Collect operation types for this cluster
        op_types = list(g["operation_type"].unique()) if "operation_type" in g.columns else []
        
        cluster_infos.append(
            ClusterInfo(
                cluster_group=int(cluster_group),
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                earliest_start_time=earliest_start,
                n_checkpoints=n_ckp,
                op_types=op_types,
            )
        )

    # Sort by earliest start time (ascending)
    cluster_infos.sort(key=lambda c: (c.earliest_start_time, c.cluster_group))
    logger.info("Prepared %d clusters (sorted by earliest start time)", len(cluster_infos))
    return cluster_infos


def _assign_clusters_to_agents(
    clusters: List[ClusterInfo],
    agents: List[AgentInfo],
    k: int,
    cooldown: float,
    default_params: dict = None,
    matching_rules: List[dict] = None,
) -> pd.DataFrame:
    """Assign each cluster to up to K nearest agents with cooldown.

    Now supports fine-grained rules that can override k and cooldown per cluster
    based on cluster properties (n_checkpoints, op_types).
    """
    if not agents:
        raise ValueError("No agents available for assignment.")
    
    default_params = default_params or {}
    matching_rules = matching_rules or []
    
    # Load limits from default params
    max_clusters_per_agent = default_params.get("max_clusters_per_agent", 999)
    max_checkpoints_per_agent = default_params.get("max_checkpoints_per_agent", 999)

    agent_ids = [a.agent_id for a in agents]
    agent_pos_init = np.array([[a.x, a.y, a.z] for a in agents], dtype=float)
    agent_current_pos = agent_pos_init.copy()
    next_available: Dict[str, float] = {aid: 0.0 for aid in agent_ids}
    
    # Track agent load to prevent overload
    agent_cluster_count: Dict[str, int] = {aid: 0 for aid in agent_ids}
    agent_checkpoint_count: Dict[str, int] = {aid: 0 for aid in agent_ids}

    records: List[Dict] = []

    for c in clusters:
        center = np.array([c.center_x, c.center_y, c.center_z], dtype=float)
        t0 = c.earliest_start_time
        
        # Apply fine-grained rules to get per-cluster matching params
        cluster_params = _get_matching_params_for_cluster(c, default_params, matching_rules)
        cluster_k = cluster_params.get("k_neighbors", k)
        cluster_cooldown = cluster_params.get("cooldown", cooldown)

        # 1) candidates respecting cooldown AND load limits
        candidate_indices: List[int] = [
            i for i, aid in enumerate(agent_ids)
            if next_available.get(aid, 0.0) <= t0
            and agent_cluster_count.get(aid, 0) < max_clusters_per_agent
            and agent_checkpoint_count.get(aid, 0) < max_checkpoints_per_agent
        ]

        use_recovery_assignment = False
        if not candidate_indices:
            # Recovery assignment: use all agents when cooldown constraints leave no candidate.
            use_recovery_assignment = True
            candidate_indices = list(range(len(agent_ids)))
            logger.warning(
                "No agents available under cooldown for cluster %s at t=%.1f; "
                "using all agents for recovery assignment.",
                c.cluster_group,
                t0,
            )

        # 使用当前位置（而非初始位置）计算距离
        cand_pos = agent_current_pos[candidate_indices]
        dists = np.linalg.norm(cand_pos - center, axis=1)
        order = np.argsort(dists)
    
        chosen_indices: List[int] = []
        for idx in order:
            global_idx = candidate_indices[idx]
            chosen_indices.append(global_idx)
            if len(chosen_indices) >= cluster_k:
                break

        chosen_agents = [agent_ids[i] for i in chosen_indices]

        # Update cooldown and load tracking
        if not use_recovery_assignment:
            for aid in chosen_agents:
                next_available[aid] = t0 + cluster_cooldown
        
        # Update agent load counters
        for aid in chosen_agents:
            agent_cluster_count[aid] = agent_cluster_count.get(aid, 0) + 1
            agent_checkpoint_count[aid] = agent_checkpoint_count.get(aid, 0) + c.n_checkpoints
        
        # Update agent positions to cluster center
        for i in chosen_indices:
            agent_current_pos[i] = center

        records.append(
            {
                "cluster_group": c.cluster_group,
                "earliest_start_time": t0,
                "center_x": c.center_x,
                "center_y": c.center_y,
                "center_z": c.center_z,
                "n_checkpoints": c.n_checkpoints,
                "assigned_agents": ";".join(chosen_agents),
            }
        )

    df_assign = pd.DataFrame.from_records(records)
    return df_assign


def _load_agent_matching_params(
    scenario_name: str,
    project_root: Path,
    rules_config: Path | None = None,
) -> Tuple[dict, List[dict]]:
    """Load agent_matching parameters and fine-grained rules from the config file.
    
    Returns:
        Tuple of (default_params, matching_rules)
    """
    rules_config_path = rules_config
    if rules_config_path is None:
        rules_config_path = project_root / "analysis" / "rules" / f"rule_kmeans_config_{scenario_name}.yaml"
    elif not rules_config_path.is_absolute():
        rules_config_path = project_root / rules_config_path
    if not rules_config_path.exists():
        logger.info("No rules config found at %s, using defaults", rules_config_path)
        return {}, []
    
    try:
        with rules_config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        recursive_params = config.get("recursive_params", {})
        agent_matching = recursive_params.get("agent_matching", {})
        agent_matching_rules = recursive_params.get("agent_matching_rules", [])
        if agent_matching:
            logger.info("Loaded agent_matching params: %s", agent_matching)
        if agent_matching_rules:
            logger.info("Loaded %d fine-grained agent matching rules", len(agent_matching_rules))
        return agent_matching, agent_matching_rules
    except Exception as e:
        logger.warning("Failed to load rules config: %s", e)
        return {}, []


def _get_matching_params_for_cluster(cluster: ClusterInfo, default_params: dict, rules: List[dict]) -> dict:
    """Apply fine-grained rules to get agent matching params for a specific cluster."""
    params = default_params.copy()
    
    for rule in rules:
        conditions = rule.get("if", {})
        match = True
        
        # Check n_checkpoints condition
        if "n_checkpoints" in conditions:
            cond = conditions["n_checkpoints"]
            if cond.startswith("<="):
                if not (cluster.n_checkpoints <= int(cond[2:].strip())):
                    match = False
            elif cond.startswith(">="):
                if not (cluster.n_checkpoints >= int(cond[2:].strip())):
                    match = False
        
        # Check op_types_contains condition
        if "op_types_contains" in conditions and match:
            required_op = conditions["op_types_contains"].lower()
            if cluster.op_types:
                op_types_lower = [op.lower() for op in cluster.op_types]
                if not any(required_op in op for op in op_types_lower):
                    match = False
            else:
                match = False
        
        # Apply rule if matched
        if match:
            then_params = rule.get("then", {})
            params.update(then_params)
            break  # First matching rule wins
    
    return params


def run_stage2_planning(
    seed: int,
    cluster_method: str,
    input_root: Path,
    output_root: Path,
    k_neighbors: int,
    cooldown: float,
    rules_config: Path | None = None,
) -> Path:
    """Run Stage 2 planning for a given seed and clustering method.

    Returns path to cluster_assignment_<method>.csv under run_seed_<seed>.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    input_root = input_root.resolve()
    output_root = output_root.resolve()

    run_dir_in = input_root / f"run_seed_{seed}"
    if not run_dir_in.exists():
        raise FileNotFoundError(f"Input run directory not found: {run_dir_in}")

    scenario_path = run_dir_in / "scenario.yaml"
    workflow_init_info_path = run_dir_in / "workflow_init_info.csv"
    checkpoint_clusters_path = run_dir_in / f"checkpoint_clusters_{cluster_method}.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(f"scenario.yaml not found under {run_dir_in}")
    if not workflow_init_info_path.exists():
        raise FileNotFoundError(f"workflow_init_info.csv not found under {run_dir_in}")
    if not checkpoint_clusters_path.exists():
        raise FileNotFoundError(
            f"checkpoint_clusters_{cluster_method}.csv not found under {run_dir_in}"
        )

    # Load agent_matching parameters and fine-grained rules from HtT config
    scenario_name = input_root.name
    agent_matching_params, agent_matching_rules = _load_agent_matching_params(
        scenario_name,
        project_root,
        rules_config=rules_config,
    )
    
    # Override defaults with rules config values if present
    k_neighbors = agent_matching_params.get("k_neighbors", k_neighbors)
    cooldown = agent_matching_params.get("cooldown", cooldown)

    logger.info("=" * 70)
    logger.info("Stage 2: Plan from clustered checkpoints (seed=%s, method=%s)", seed, cluster_method)
    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info("K nearest agents per cluster: %d", k_neighbors)
    logger.info("Agent cooldown (s): %.1f", cooldown)
    logger.info("=" * 70)

    agents = _load_agents_from_scenario(scenario_path)
    clusters = _compute_cluster_infos(
        checkpoint_clusters_path=checkpoint_clusters_path,
        workflow_init_info_path=workflow_init_info_path,
    )

    df_assign = _assign_clusters_to_agents(
        clusters=clusters,
        agents=agents,
        k=k_neighbors,
        cooldown=cooldown,
        default_params=agent_matching_params,
        matching_rules=agent_matching_rules,
    )

    run_dir_out = output_root / f"run_seed_{seed}"
    run_dir_out.mkdir(parents=True, exist_ok=True)

    output_path = run_dir_out / f"cluster_assignment_{cluster_method}.csv"
    df_assign.to_csv(output_path, index=False)
    logger.info("Saved cluster assignment to: %s", output_path)

    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: Load scenario.yaml + checkpoint_clusters_<method>.csv "
            "and assign each cluster to K nearest agents with a cooldown."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used in Stage 1 (selects run_seed_<seed> directory under input-root).",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="kmeans",
        help=(
            "Name of clustering method used in Stage 1, "
            "e.g., kmeans, triplet, rl, rule_kmeans. "
            "Determines which checkpoint_clusters_<method>.csv to load."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/offline_init",
        help="Root directory containing run_seed_<seed> from Stage 1 (relative to project root by default).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/offline_plans",
        help="Root directory to store Stage 2 outputs (relative to project root by default).",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=3,
        help="Number of nearest agents to assign per cluster.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=10.0,
        help="Cooldown in seconds before an agent can be assigned to a new cluster (based on earliest start time).",
    )
    parser.add_argument(
        "--rules-config",
        type=str,
        default=None,
        help="Optional path to a rule_kmeans_config YAML used to override scenario-based agent matching lookup.",
    )

    args = parser.parse_args()

    seed = args.seed
    cluster_method = args.cluster_method

    input_root = _resolve_under_project(args.input_root, project_root)
    output_root = _resolve_under_project(args.output_root, project_root)
    rules_config_path = _resolve_under_project(args.rules_config, project_root) if args.rules_config else None

    output_path = run_stage2_planning(
        seed=seed,
        cluster_method=cluster_method,
        input_root=input_root,
        output_root=output_root,
        k_neighbors=args.k_neighbors,
        cooldown=args.cooldown,
        rules_config=rules_config_path,
    )

    print("\n=== Stage 2 Completed ===")
    print(f"Cluster assignment file: {output_path}")


if __name__ == "__main__":
    main()
