"""Runtime loader for offline Stage 3 PDDL plans.

This module converts per-agent PDDL solutions produced by
`stage3_plan_with_pddl.py` into concrete task dictionaries compatible
with AirFogSim's `Agent.add_task_to_queue` API.

Design goals
------------
- No online model involvement at runtime.
- One PDDL location can correspond to multiple workflows, but each
  concrete task carries exactly one `workflow_id`.
- Generated task dicts follow the same schema as structured planners:

    {
        "component": "ComponentName",   # e.g. "Mobility", "Compute"
        "task_class": "TaskClassName", # e.g. "MoveToTask"
        "task_name": "DescriptiveName",
        "target_state": {"position": [x, y, z]},
        "properties": {...},
        "workflow_id": "workflow_xxx",
        "priority": "high" | "normal" | "low",
        "preemptive": bool
    }

Usage
-----
Example of loading tasks and pushing them into an agent's queue::

    from pathlib import Path
    from src.offline_pipeline.runtime_loader import OfflineTaskLoader

    loader = OfflineTaskLoader(
        plan_path=Path("data/offline_plans/run_seed_42/pddl_plan_htt_recursive.json")
    )

    for agent_id, agent in all_agents.items():
        offline_tasks = loader.get_tasks_for_agent(agent_id)
        for t in offline_tasks:
            agent.add_task_to_queue(
                component_name=t["component"],
                task_name=t["task_name"],
                task_class=t["task_class"],
                priority=t.get("priority"),
                preemptive=t.get("preemptive", False),
                target_state=t.get("target_state"),
                properties=t.get("properties"),
                workflow_id=t.get("workflow_id"),
            )

This keeps integration simple and avoids modifying the core Agent
scheduling logic.
"""

from __future__ import annotations

import json
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _ClusterMeta:
    """Lightweight view of per-location cluster metadata.

    This mirrors the structure written by ``stage3_plan_with_pddl.py``::

        cluster_meta[loc_label] = {
            "cluster_group": int,
            "earliest_start_time": float,
            "n_checkpoints": int,
            "op_types": List[str],
            "workflow_ids": List[str],
            "topo_node_id": Optional[str],
            "original_coords": Optional[List[float]],
        }
    """

    cluster_group: int
    earliest_start_time: float
    n_checkpoints: int
    op_types: List[str]
    workflow_ids: List[str]
    topo_node_id: Optional[str] = None
    original_coords: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_ClusterMeta":
        # Handle cluster_group which may be int or list (when multiple clusters share location)
        cg = d.get("cluster_group", -1)
        if isinstance(cg, (list, tuple)):
            cg = cg[0] if cg else -1
        return cls(
            cluster_group=int(cg),
            earliest_start_time=float(d.get("earliest_start_time", 0.0)),
            n_checkpoints=int(d.get("n_checkpoints", 0)),
            op_types=list(d.get("op_types", [])),
            workflow_ids=list(d.get("workflow_ids", [])),
            topo_node_id=d.get("topo_node_id"),
            original_coords=d.get("original_coords"),
        )


class OfflineTaskLoader:
    """Load Stage 3 PDDL planning results and build per-agent task lists.

    Parameters
    ----------
    plan_path:
        Path to the JSON output produced by ``stage3_plan_with_pddl.py``,
        e.g. ``data/offline_plans/run_seed_42/pddl_plan_htt_recursive.json``.
    """

    def __init__(self, plan_path: Path | str) -> None:
        self.plan_path = Path(plan_path)
        if not self.plan_path.exists():
            raise FileNotFoundError(f"PDDL plan JSON not found: {self.plan_path}")

        with self.plan_path.open("r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = json.load(f)

        # Load scenario workflow properties to adapt task generation
        self._scenario_wf: Dict[str, Dict[str, Any]] = self._load_scenario_workflows()

        self.agent_tasks: Dict[str, List[Dict[str, Any]]] = {}
        self._build_all_agent_tasks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tasks_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return a *copy* of all offline tasks for a given agent.

        The returned list is ordered following the PDDL solution steps but
        does not enforce earliest_start_time. Callers can implement their
        own scheduling policy if needed.
        """

        tasks = self.agent_tasks.get(agent_id, [])
        return [dict(t) for t in tasks]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_all_agent_tasks(self) -> None:
        agents_section = self._data.get("agents", {}) or {}
        if not agents_section:
            logger.warning(
                "OfflineTaskLoader: no 'agents' section found in %s", self.plan_path
            )

        for agent_id, agent_plan in agents_section.items():
            try:
                self.agent_tasks[agent_id] = self._build_tasks_for_agent(agent_id, agent_plan)
                logger.info(
                    "OfflineTaskLoader: built %d tasks for agent %s",
                    len(self.agent_tasks[agent_id]),
                    agent_id,
                )
            except Exception as e:  # pragma: no cover - defensive logging
                logger.error(
                    "OfflineTaskLoader: failed to build tasks for agent %s: %s",
                    agent_id,
                    e,
                )
                self.agent_tasks[agent_id] = []

        # After per-agent build, append logistics task chains for workflow owners based on scenario.yaml
        try:
            self._append_logistics_tasks_for_owners()
        except Exception as e:
            logger.error("OfflineTaskLoader: failed to append logistics tasks: %s", e)

    def _get_workflow_type(self, wf_id: str) -> Optional[str]:
        """Get the workflow type from scenario metadata.

        Returns one of: 'logistics', 'communication', 'computation', 'inspection', or None.
        """
        wf_meta = self._scenario_wf.get(wf_id, {})
        wf_type = wf_meta.get('type', '').lower() if wf_meta else ''
        # Normalize common variations
        if wf_type in ('compute', 'computation'):
            return 'computation'
        if wf_type in ('comm', 'communication'):
            return 'communication'
        if wf_type in ('inspect', 'inspection'):
            return 'inspection'
        if wf_type in ('logistics', 'delivery'):
            return 'logistics'
        return wf_type or None

    def _build_tasks_for_agent(self, agent_id: str, agent_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert one agent's PDDL solution into a list of task dicts.

        Each PDDL action (assistcompute/assistcommunicate/inspect) is mapped
        to tasks only for workflows whose type matches the action semantics:
        - assistcompute   -> only 'computation' workflows
        - assistcommunicate -> only 'communication' workflows
        - inspect         -> only 'inspection' workflows

        This ensures that e.g. an InspectionWorkflow does not receive
        CommunicationTask just because they share a cluster location.
        """

        init_params = agent_plan.get("init_params", {}) or {}
        locations = init_params.get("locations", []) or []

        # Per-location cluster metadata from Stage 3 (needed for topo node resolution)
        raw_cluster_meta = agent_plan.get("cluster_meta", {}) or {}
        cluster_meta: Dict[str, _ClusterMeta] = {
            loc: _ClusterMeta.from_dict(meta)
            for loc, meta in raw_cluster_meta.items()
        }

        # Build location label -> coordinate mapping (topo_<node_id> format only)
        loc_coords: Dict[str, List[float]] = {}
        for loc in locations:
            meta = cluster_meta.get(loc)
            loc_coords[loc] = self._parse_topo_label(loc, meta)

        # Optional精细路径：来自 Stage3 EdgePlanner 的 edge_routes
        edge_routes = agent_plan.get("edge_routes", {}) or {}

        solution = agent_plan.get("result", {}).get("solution", []) or []

        tasks: List[Dict[str, Any]] = []
        movement_emitted: set[tuple[str, str]] = set()  # (workflow_id, loc_label)

        for step in solution:
            action_name = step.get("action_name") or step.get("action") or ""
            action_name = action_name.lower()
            params = step.get("parameters", []) or []

            if not action_name:
                continue

            if action_name == "assistcompute":
                # Expected parameters: [agent_id, location_label]
                if len(params) >= 2:
                    loc_label = params[1]
                    meta = cluster_meta.get(loc_label)
                    pos = loc_coords.get(loc_label) or self._parse_topo_label(loc_label, meta)
                    workflow_ids = meta.workflow_ids if meta else []

                    # Only generate ComputeTask for 'computation' type workflows
                    for wf_id in workflow_ids:
                        if self._get_workflow_type(wf_id) != 'computation':
                            continue
                        key = (wf_id, loc_label)
                        if key not in movement_emitted:
                            tasks.append(self._make_move_task(agent_id, loc_label, pos, workflow_id=wf_id, edge_routes=edge_routes))
                            movement_emitted.add(key)
                        tasks.append(self._make_compute_task(agent_id, wf_id, loc_label, pos, meta))

            elif action_name == "assistcommunicate":
                # Expected parameters: [agent_id, location_label]
                if len(params) >= 2:
                    loc_label = params[1]
                    meta = cluster_meta.get(loc_label)
                    pos = loc_coords.get(loc_label) or self._parse_topo_label(loc_label, meta)
                    workflow_ids = meta.workflow_ids if meta else []

                    # Only generate CommunicationTask for 'communication' type workflows
                    for wf_id in workflow_ids:
                        if self._get_workflow_type(wf_id) != 'communication':
                            continue
                        key = (wf_id, loc_label)
                        if key not in movement_emitted:
                            tasks.append(self._make_move_task(agent_id, loc_label, pos, workflow_id=wf_id, edge_routes=edge_routes))
                            movement_emitted.add(key)
                        tasks.append(self._make_communication_task(agent_id, wf_id, loc_label, pos, meta))

            elif action_name == "inspect":
                # Expected parameters: [agent_id, location_label]
                if len(params) >= 2:
                    loc_label = params[1]
                    meta = cluster_meta.get(loc_label)
                    pos = loc_coords.get(loc_label) or self._parse_topo_label(loc_label, meta)
                    workflow_ids = meta.workflow_ids if meta else []
                    
                    # Get checkpoint_indices from cluster_meta (workflow_id -> [checkpoint_idx, ...])
                    # This ensures we only generate tasks for checkpoints at THIS location,
                    # not all inspection points from the workflow
                    checkpoint_indices_map = {}
                    if meta is not None:
                        raw_meta = raw_cluster_meta.get(loc_label, {})
                        checkpoint_indices_map = raw_meta.get("checkpoint_indices", {}) or {}

                    # Only generate InspectionTask for 'inspection' type workflows
                    for wf_id in workflow_ids:
                        if self._get_workflow_type(wf_id) != 'inspection':
                            continue
                        
                        # Get checkpoint indices for this workflow at this location
                        ckp_indices = checkpoint_indices_map.get(wf_id, [])
                        
                        if ckp_indices:
                            # Generate task only for checkpoints assigned to this location
                            wf_meta = self._scenario_wf.get(wf_id, {})
                            wf_props = wf_meta.get('properties', {}) or {}
                            inspection_points = wf_props.get('inspection_points', [])
                            
                            for point_idx in ckp_indices:
                                # Get position from inspection_points if available
                                if inspection_points and point_idx < len(inspection_points):
                                    point = inspection_points[point_idx]
                                    if isinstance(point, dict):
                                        point_pos = [float(point.get('x', 0)), float(point.get('y', 0)), float(point.get('z', 0))]
                                    else:
                                        point_pos = list(point) if isinstance(point, (list, tuple)) else pos
                                else:
                                    point_pos = pos
                                
                                # Use workflow_id + point_index as unique key to avoid duplicates
                                key = (wf_id, f"point_{point_idx}")
                                if key not in movement_emitted:
                                    tasks.append(self._make_move_task(agent_id, f"{loc_label}_pt{point_idx}", point_pos, workflow_id=wf_id, edge_routes=edge_routes))
                                    movement_emitted.add(key)
                                tasks.append(self._make_inspection_task(agent_id, wf_id, loc_label, point_pos, meta, point_index=point_idx))
                        else:
                            # Fallback: no checkpoint_indices in cluster_meta, use location position
                            key = (wf_id, loc_label)
                            if key not in movement_emitted:
                                tasks.append(self._make_move_task(agent_id, loc_label, pos, workflow_id=wf_id, edge_routes=edge_routes))
                                movement_emitted.add(key)
                            tasks.append(self._make_inspection_task(agent_id, wf_id, loc_label, pos, meta, point_index=0))

            else:
                # Unknown action types are ignored for now.
                logger.debug(
                    "OfflineTaskLoader: ignoring unsupported action '%s' for agent %s",
                    action_name,
                    agent_id,
                )

        return tasks

    # ------------------------------------------------------------------
    # Task factory helpers
    # ------------------------------------------------------------------

    def _parse_topo_label(self, label: str, meta: Optional[_ClusterMeta] = None) -> List[float]:
        """Parse topo node label into [x, y, z].

        Only supports ``topo_<node_id>`` format. Resolves coordinates via topology.node_coords
        or meta.original_coords.

        Raises RuntimeError if label format is invalid or node cannot be resolved.
        """
        if not label.startswith("topo_"):
            raise RuntimeError(f"Invalid topo label format: {label}. Expected 'topo_<node_id>'.")
        
        node_id = label[5:]  # Strip "topo_" prefix
        
        # Resolve from topology.node_coords
        topo_data = self._data.get("topology", {}) or {}
        node_coords = topo_data.get("node_coords", {}) or {}
        if node_id in node_coords:
            coord = node_coords[node_id]
            if isinstance(coord, (list, tuple)) and len(coord) == 3:
                return [float(coord[0]), float(coord[1]), float(coord[2])]
        
        # Fallback to meta.original_coords if available
        if meta is not None and meta.original_coords:
            return list(meta.original_coords)
        
        raise RuntimeError(f"Cannot resolve topo node '{node_id}'. Check topology data.")

    def _make_move_task(self, agent_id: str, loc_label: str, pos: List[float], workflow_id: Optional[str] = None, edge_routes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a MoveToTask entry.

        Movement tasks are considered navigation primitives and **do not
        carry a specific workflow_id**. Workflows observe the agent's
        position directly via their state triggers.
        
        If topo_aware mode is enabled in the plan, this method also sets
        the recommended speed based on topology edge speed limits.
        """

        properties: Dict[str, Any] = {
            "movement_type": "path_following",
            "target_position": pos,
        }
        
        # Check if topo_aware mode is enabled
        topo_meta = self._data.get("topology", {}) or {}
        topo_aware = topo_meta.get("topo_aware", False)
        edge_speed_limits = topo_meta.get("edge_speed_limits", {}) or {}
        node_coords = topo_meta.get("node_coords", {}) or {}
        
        path_nodes = []  # Track nodes for speed limit estimation
        segment_speeds = []  # Track time-varying speeds from path planning

        # 如果存在 EdgePlanner 生成的路径，则尽可能提取节点坐标作为waypoints
        if edge_routes:
            try:
                # 策略1：直接匹配 loc_label (如 topo_waypoint_30)
                candidate_keys = [k for k in edge_routes.keys() if k.strip().endswith(f"-> {loc_label}")]
                
                # 策略2：去掉 _ptX 后缀再匹配 (如 topo_waypoint_44_pt3 -> topo_waypoint_44)
                if not candidate_keys and loc_label.startswith("topo_"):
                    import re
                    base_label = re.sub(r'_pt\d+$', '', loc_label)
                    if base_label != loc_label:
                        candidate_keys = [k for k in edge_routes.keys() if k.strip().endswith(f"-> {base_label}")]
                
                # 策略3：如果 loc_label 不匹配，尝试通过目标位置找到最近的 topo 节点
                if not candidate_keys and node_coords:
                    min_dist = float('inf')
                    nearest_node = None
                    for node_id, coord in node_coords.items():
                        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            dist = ((coord[0] - pos[0])**2 + (coord[1] - pos[1])**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                nearest_node = node_id
                    if nearest_node and min_dist < 100:
                        topo_label = f"topo_{nearest_node}"
                        candidate_keys = [k for k in edge_routes.keys() if k.strip().endswith(f"-> {topo_label}")]
                
                if candidate_keys:
                    route_key = sorted(candidate_keys)[0]
                    route = edge_routes.get(route_key, {}) or {}
                    nodes = route.get("nodes", []) or []
                    segments = route.get("segments", []) or []
                    path_nodes = nodes  # Save for speed limit estimation
                    
                    # Extract time-varying speeds from segments (v_ref considers time-varying conditions)
                    for seg in segments:
                        v_ref = seg.get("v_ref", 0)
                        if v_ref > 0:
                            segment_speeds.append(v_ref)
                    
                    waypoints = []
                    waypoint_nodes = []  # 存储节点ID用于运行时速度查询
                    for nid in nodes:
                        coord = node_coords.get(nid)
                        if isinstance(coord, (list, tuple)) and len(coord) == 3:
                            waypoints.append([float(coord[0]), float(coord[1]), float(coord[2])])
                            waypoint_nodes.append(nid)
                    if waypoints:
                        properties["movement_type"] = "waypoint"
                        properties["waypoints"] = waypoints
                        properties["waypoint_nodes"] = waypoint_nodes  # 节点ID列表
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Failed to decode edge_routes for agent %s at %s: %s", agent_id, loc_label, e)
        
        # Estimate speed limit for topo_aware mode
        if topo_aware:
            min_speed_limit = float('inf')
            
            # Priority 1: Use time-varying v_ref from path planning segments
            # This considers traffic profiles, weather, congestion at planning time
            if segment_speeds:
                min_speed_limit = min(segment_speeds)
                logger.debug("MoveToTask %s: using segment v_ref min=%.2f m/s (time-varying)", 
                           loc_label, min_speed_limit)
            
            # Priority 2: Otherwise use static edge_speed_limits.
            elif edge_speed_limits and path_nodes and len(path_nodes) >= 2:
                for i in range(len(path_nodes) - 1):
                    edge_key = f"{path_nodes[i]},{path_nodes[i+1]}"
                    if edge_key in edge_speed_limits:
                        min_speed_limit = min(min_speed_limit, edge_speed_limits[edge_key])
            
            # Priority 3: Use average speed limit as a final estimate.
            if min_speed_limit == float('inf') and edge_speed_limits:
                min_speed_limit = sum(edge_speed_limits.values()) / len(edge_speed_limits)
            
            if min_speed_limit != float('inf'):
                properties["recommended_speed"] = round(min_speed_limit, 2)
                properties["topo_aware"] = True

        return {
            "component": "Mobility",  # matches MobilityComponent name
            "task_class": "MoveToTask",
            "task_name": f"MoveTo_{loc_label}",
            "target_state": {"position": pos},
            "properties": properties,
            "workflow_id": workflow_id,
            "priority": "normal",
            "preemptive": False,
        }

    def _make_compute_task(
        self,
        agent_id: str,
        workflow_id: str,
        loc_label: str,
        pos: List[float],
        meta: Optional[_ClusterMeta],
    ) -> Dict[str, Any]:
        """Create a ComputeTask entry for one workflow at a location."""

        n_ckp = meta.n_checkpoints if meta is not None else 1
        wf_props = self._scenario_wf.get(workflow_id, {}).get("properties", {})
        required_gflops = float(wf_props.get("required_gflops", float(n_ckp) * 100.0))
        workload_profile = wf_props.get("workload_profile", {})
        job_id = wf_props.get("job_id", f"{workflow_id}_cluster_{meta.cluster_group if meta else 'na'}")

        return {
            "component": "Compute",  # matches ComputeComponent name
            "task_class": "ComputeTask",
            "task_name": f"Compute_{workflow_id}_at_{loc_label}",
            "target_state": {"position": pos},
            "properties": {
                "job_id": job_id,
                "required_gflops": required_gflops,
                "workload_profile": workload_profile,
            },
            "workflow_id": workflow_id,
            "priority": "high",
            "preemptive": False,
        }

    def _make_communication_task(
        self,
        agent_id: str,
        workflow_id: str,
        loc_label: str,
        pos: List[float],
        meta: Optional[_ClusterMeta],
    ) -> Dict[str, Any]:
        """Create a CommunicationTask entry for one workflow at a location."""

        n_ckp = meta.n_checkpoints if meta is not None else 1
        wf_props = self._scenario_wf.get(workflow_id, {}).get("properties", {})
        data_size_mb = float(wf_props.get("data_size_mb", float(n_ckp) * 10.0))

        return {
            "component": "Communication",  # matches CommunicationComponent name
            "task_class": "CommunicationTask",
            "task_name": f"Comm_{workflow_id}_at_{loc_label}",
            "target_state": {"position": pos},
            "properties": {
                "data_id": f"{workflow_id}_cluster_{meta.cluster_group if meta else 'na'}",
                "data_size_mb": data_size_mb,
                "target_location": pos,
            },
            "workflow_id": workflow_id,
            "priority": "high",
            "preemptive": False,
        }

    def _make_inspection_task(
        self,
        agent_id: str,
        workflow_id: str,
        loc_label: str,
        pos: List[float],
        meta: Optional[_ClusterMeta],
        point_index: int = 0,
    ) -> Dict[str, Any]:
        """Create an InspectionTask entry for one workflow at a location."""

        return {
            "component": "Inspection",  # matches InspectionComponent name
            "task_class": "InspectionTask",
            "task_name": f"Inspect_{workflow_id}_at_{loc_label}",
            "target_state": {"position": pos},
            "properties": {
                "point_index": point_index,
                "waypoint": pos,
            },
            "workflow_id": workflow_id,
            "priority": "high",
            "preemptive": False,
        }

    # ------------------------------------------------------------------
    # Scenario loader helpers and logistics task synthesis
    # ------------------------------------------------------------------
    def _load_scenario_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Load scenario.yaml (if available) and return a mapping:
        { workflow_id: { 'type': str, 'properties': dict, 'agent_id': str } }
        """
        try:
            # Prefer seed from plan content; otherwise infer it from directory name.
            seed = self._data.get("seed", None)
            if seed is not None:
                run_dir_name = f"run_seed_{seed}"
            else:
                # plan_path = .../offline_plans/scenario_name/run_seed_XX/pddl_plan_*.json
                run_dir_name = self.plan_path.parent.name

            # Plan path structure: data/offline_plans/{scenario}_{method}/run_seed_{seed}/pddl_plan_{method}.json
            # plan_path.parents: [0]=run_seed_X, [1]=scenario_method, [2]=offline_plans, [3]=data
            scenario_dir_name = self.plan_path.parents[1].name
            
            # Strip method suffix from scenario name (e.g., small_logistics_inspection_kmeans -> small_logistics_inspection)
            # Common suffixes to strip
            for suffix in ['_kmeans', '_htt', '_rule_kmeans', '_triplet', '_rl']:
                if scenario_dir_name.endswith(suffix):
                    scenario_name = scenario_dir_name[:-len(suffix)]
                    break
            else:
                scenario_name = scenario_dir_name
            
            # Search multiple possible locations for scenario.yaml
            data_root = self.plan_path.parents[3]
            project_root = data_root.parent  # Go up from 'data' to project root
            
            candidate_dirs = [
                data_root / 'offline_init' / scenario_name / run_dir_name,
                project_root / 'workflow_decouple' / 'data' / 'multi_seed_runs' / scenario_name / run_dir_name,
                data_root / 'multi_seed_runs' / scenario_name / run_dir_name,
                # Also try with the full directory name (scenario_method)
                project_root / 'workflow_decouple' / 'data' / 'multi_seed_runs' / scenario_dir_name / run_dir_name,
            ]
            
            scenario = None
            for run_dir in candidate_dirs:
                scenario_path = run_dir / 'scenario.yaml'
                if scenario_path.exists():
                    with scenario_path.open('r', encoding='utf-8') as f:
                        scenario = yaml.safe_load(f) or {}
                    logger.info("OfflineTaskLoader: loaded scenario from %s", scenario_path)
                    break
            
            if scenario is None:
                logger.warning("OfflineTaskLoader: scenario.yaml not found in any candidate location")
                return {}

            wf_map: Dict[str, Dict[str, Any]] = {}
            for wf in scenario.get('workflows', []) or []:
                wid = wf.get('id')
                if not wid:
                    continue
                wf_map[wid] = {
                    'type': wf.get('type'),
                    'properties': wf.get('properties', {}) or {},
                    'agent_id': wf.get('agent_id')
                }
            return wf_map
        except Exception as e:
            logger.error("OfflineTaskLoader: failed to load scenario workflows: %s", e)
            return {}

    def _append_logistics_tasks_for_owners(self) -> None:
        """Synthesize minimal pickup/transport/delivery chains for logistics workflows.

        This uses scenario.yaml so that tasks align with each workflow's true
        owner agent and property requirements.
        """
        if not self._scenario_wf:
            return

        for wf_id, meta in self._scenario_wf.items():
            if (meta or {}).get('type') != 'logistics':
                continue

            owner_id = meta.get('agent_id')
            props = (meta or {}).get('properties') or {}
            if not owner_id or not props:
                continue

            pickup = props.get('pickup_location')
            delivery = props.get('delivery_location')
            payload_id = props.get('payload_id')
            payload_weight = props.get('payload_weight', 1.0)

            def _to_pos(v):
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    return [float(v[0]), float(v[1]), float(v[2])]
                if isinstance(v, dict) and all(k in v for k in ('x','y','z')):
                    return [float(v['x']), float(v['y']), float(v['z'])]
                return None

            pickup_pos = _to_pos(pickup)
            delivery_pos = _to_pos(delivery)
            if pickup_pos is None or delivery_pos is None or not payload_id:
                continue

            # Ensure task list exists for owner
            if owner_id not in self.agent_tasks:
                self.agent_tasks[owner_id] = []

            # Avoid duplicating if already present
            existing = self.agent_tasks[owner_id]
            if any(t.get('workflow_id') == wf_id and t.get('task_class') in ('PickupTask','DeliveryTask') for t in existing):
                continue

            # Build minimal chain: move->pickup->move->delivery
            self.agent_tasks[owner_id].extend([
                self._make_move_task(owner_id, f"pickup_{wf_id}", pickup_pos, workflow_id=wf_id),
                self._make_pickup_task(owner_id, wf_id, pickup_pos, payload_id, float(payload_weight)),
                self._make_move_task(owner_id, f"delivery_{wf_id}", delivery_pos, workflow_id=wf_id),
                self._make_delivery_task(owner_id, wf_id, delivery_pos, payload_id, float(payload_weight)),
            ])

    def _make_pickup_task(
        self,
        agent_id: str,
        workflow_id: str,
        pos: List[float],
        payload_id: str,
        payload_weight: float,
    ) -> Dict[str, Any]:
        return {
            "component": "Logistics",
            "task_class": "PickupTask",
            "task_name": f"Pickup_{workflow_id}",
            "target_state": {"position": pos},
            "properties": {
                "payload_id": payload_id,
                "payload_weight": payload_weight,
            },
            "workflow_id": workflow_id,
            "priority": "high",
            "preemptive": False,
        }

    def _make_delivery_task(
        self,
        agent_id: str,
        workflow_id: str,
        pos: List[float],
        payload_id: str,
        payload_weight: float,
    ) -> Dict[str, Any]:
        return {
            "component": "Logistics",
            "task_class": "DeliveryTask",
            "task_name": f"Delivery_{workflow_id}",
            "target_state": {"position": pos},
            "properties": {
                "payload_id": payload_id,
                "payload_weight": payload_weight,
            },
            "workflow_id": workflow_id,
            "priority": "high",
            "preemptive": False,
        }
