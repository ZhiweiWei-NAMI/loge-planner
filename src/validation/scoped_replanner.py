"""Validator-gated scoped replanning with real PDDL solving and real A* routing.

This module is the executable form of the scoped-repair procedure used by
``scoped_replan_benchmark.py``.  Nothing here is simulated:

* task ordering / assignment comes from a **real PDDL solve**
  (:class:`src.pddl_solver.pddl_solver.PDDLSolver`, unified_planning +
  fast-downward) over the airspace graph;
* every ``movetoloc`` leg is routed by the **time-varying A*** of
  :class:`src.offline_pipeline.edge_planner.EdgePlanner` on the real graph,
  honouring blocked corridors;
* the splice validator (``c_state``/``c_time``/``c_dep``/``c_res``/``c_rsv``)
  re-simulates the merged plan and rejects it when any condition fails;
* rejected repairs expand the scope (bounded by ``k_max``, ``rho_max``,
  ``b_rep_s``, ``c_max``) and, if the bound is hit, a **real global PDDL solve**
  is attempted.  When that also fails the case is reported as ``infeasible`` --
  never as a success.

The only input is an explicit snapshot: an airspace graph, the committed task
schedule, and a semantic alert.  See ``examples/replanning/`` for the bundled
deterministic demonstration fixtures for smoke testing.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.offline_pipeline.edge_planner import EdgePlanner
from src.pddl_solver.pddl_solver import PDDLSolver
from src.simulation.environment.topology_manager import TopologyManager

#: operation type -> PDDL action name (see config/pddl/tccn_domain.pddl)
OP_TO_ACTION = {
    "compute": "assistcompute",
    "communicate": "assistcommunicate",
    "inspect": "inspect",
}
#: PDDL action name -> operation type
ACTION_TO_OP = {action: op for op, action in OP_TO_ACTION.items()}
#: operation type -> PDDLSolver goal parameter key
OP_TO_GOAL_KEY = {
    "compute": "compute_locations",
    "communicate": "communicate_locations",
    "inspect": "inspect_locations",
}

#: unreachable/invalid sentinel distance; pairs not in the graph are simply absent
INF_DISTANCE = 1.0e9


@dataclass(frozen=True)
class Task:
    """A committed workflow task (one checkpoint) in the snapshot."""

    task_id: str
    workflow_id: str
    agent_id: str
    node: str
    op_type: str
    release: float = 0.0
    deadline: float = float("inf")
    service_s: float = 60.0


@dataclass
class PlanStep:
    """One executed step of a grounded plan.

    ``segments`` holds the real A* output for ``movetoloc`` steps:
    ``(u, v, t_start, t_end)`` per traversed edge.  For service actions the
    tuple list is empty and ``nodes`` is a single node.
    """

    agent_id: str
    action: str
    nodes: Tuple[str, ...]
    t_start: float
    t_end: float
    task_id: Optional[str] = None
    segments: Tuple[Tuple[str, str, float, float], ...] = ()
    #: arrival time; for service steps the wait arrival -> t_start occupies the node
    t_arrive: Optional[float] = None

    @property
    def target(self) -> str:
        return self.nodes[-1]

    def node_occupancies(self) -> List[Tuple[str, float, float]]:
        """Intervals during which this step occupies graph nodes.

        Service steps occupy their node from *arrival* (not from the service
        start), so a drone waiting for a release time or a dependency still
        holds the node in the capacity / reservation check.
        """
        if not self.segments:
            start = self.t_arrive if self.t_arrive is not None else self.t_start
            return [(self.nodes[-1], start, self.t_end)]
        occ: List[Tuple[str, float, float]] = []
        for u, _v, t0, t1 in self.segments:
            occ.append((u, t0, t1))
        last_u, _last_v, _t0, t1 = self.segments[-1]
        occ.append((self.nodes[-1], t1, self.t_end))
        return occ


@dataclass
class ValidationReport:
    """Splice validation outcome (V_splice in the paper)."""

    valid: bool
    c_state: bool = True
    c_time: bool = True
    c_dep: bool = True
    c_res: bool = True
    c_rsv: bool = True
    conflict_tasks: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "c_state": self.c_state,
            "c_time": self.c_time,
            "c_dep": self.c_dep,
            "c_res": self.c_res,
            "c_rsv": self.c_rsv,
            "conflict_tasks": list(self.conflict_tasks),
            "reasons": list(self.reasons),
        }


@dataclass
class SolveRecord:
    """One real PDDL solve attempt, kept for traceability."""

    stage: str
    scope_tasks: List[str]
    scope_agents: List[str]
    backend: str
    success: bool
    error: Optional[str]
    n_actions: int
    wall_ms: float
    problem_file: Optional[str] = None
    solver_log: str = ""


@dataclass
class ReplanResult:
    case_id: str
    status: str
    backend_chain: List[str]
    scope_initial: List[str]
    scope_final: List[str]
    expansion_rounds: int
    global_recovery: bool
    validation: Dict[str, Any]
    baseline_plan: List[Dict[str, Any]]
    repaired_plan: List[Dict[str, Any]]
    changed_routes: List[Dict[str, Any]]
    blocked_edges: List[List[str]]
    solves: List[Dict[str, Any]]
    timings_ms: Dict[str, float]
    notes: List[str] = field(default_factory=list)


class ScopedReplanner:
    """Algorithm-2 style scoped repair over a real PDDL + A* stack."""

    def __init__(
        self,
        graph: Dict[str, Any],
        snapshot: Dict[str, Any],
        alert: Dict[str, Any],
        domain_file: Optional[Path] = None,
        k_max: int = 3,
        b_rep_s: float = 30.0,
        rho_max: float = 0.6,
        c_max: int = 8,
        pddl_timeout: float = 30.0,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        self.graph = graph
        self.snapshot = snapshot
        self.alert = alert
        self.domain_file = Path(domain_file) if domain_file else None
        self.k_max = k_max
        self.b_rep_s = b_rep_s
        self.rho_max = rho_max
        self.c_max = c_max
        self.pddl_timeout = pddl_timeout
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

        self.topo = self._build_topology(graph)
        self.planner = EdgePlanner(self.topo, time_step=1.0)

        self.tasks: Dict[str, Task] = {
            str(t["task_id"]): Task(
                task_id=str(t["task_id"]),
                workflow_id=str(t["workflow_id"]),
                agent_id=str(t["agent_id"]),
                node=str(t["node"]),
                op_type=str(t["op_type"]),
                release=float(t.get("release", 0.0)),
                deadline=float(t.get("deadline", float("inf"))),
                service_s=float(t.get("service_s", 60.0)),
            )
            for t in snapshot.get("tasks", [])
        }
        self.deps: List[Tuple[str, str]] = [
            (str(u), str(v)) for u, v in snapshot.get("dependencies", [])
        ]
        self.deps_in: Dict[str, List[str]] = {}
        for u, v in self.deps:
            self.deps_in.setdefault(v, []).append(u)
        self.agent_home: Dict[str, str] = {
            str(a): str(info["node"]) for a, info in snapshot.get("agents", {}).items()
        }
        self.t0 = float(snapshot.get("t0", 0.0))

        self._validate_identifiers()

        self._astar_calls = 0
        self._astar_ms = 0.0
        self._pddl_ms = 0.0
        self._solves: List[SolveRecord] = []
        self._blocked_edges: List[List[str]] = []

    # ------------------------------------------------------------------ setup
    def _validate_identifiers(self) -> None:
        """PDDLSolver emits raw identifiers, so names must already be valid PDDL."""
        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        names = set(self.topo.nodes) | set(self.agent_home)
        for task in self.tasks.values():
            names.add(task.node)
            names.add(task.agent_id)
        bad = sorted(n for n in names if not pattern.match(n))
        if bad:
            raise ValueError(
                f"identifiers must match {pattern.pattern} (lowercase PDDL names): {bad}"
            )

    def _build_topology(self, graph: Dict[str, Any]) -> TopologyManager:
        bounds = graph.get("bounds") or {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
        }
        topo = TopologyManager(bounds=bounds)
        for node in graph.get("nodes", []):
            topo.add_node(
                str(node["id"]),
                float(node["x"]),
                float(node["y"]),
                float(node.get("z", 0.0)),
                str(node.get("type", "waypoint")),
                capacity=int(node.get("capacity", 1)),
            )
        for edge in graph.get("edges", []):
            topo.add_edge(
                str(edge["u"]),
                str(edge["v"]),
                base_velocity=float(edge.get("base_velocity", 15.0)),
                rho_jam=float(edge.get("rho_jam", 0.5)),
                capacity=int(edge.get("capacity", 1)),
            )
        return topo

    def apply_alert(self) -> None:
        """Make the alert's blocked corridors real for A*, PDDL and the validator.

        A closed corridor is closed in **both** directions by default: closing
        only one direction would let another drone fly the same physical
        corridor backwards and the splice validator would still report the plan
        as valid. Set ``"directional": true`` in the alert to close exactly the
        listed directions instead.
        """
        directional = bool(self.alert.get("directional", False))
        blocked = set()
        for pair in self.alert.get("blocked_edges", []):
            u, v = str(pair[0]), str(pair[1])
            blocked.add((u, v))
            if not directional:
                blocked.add((v, u))
        self.topo.weather_blocked_edges = blocked
        self._blocked_edges = [list(pair) for pair in sorted(blocked)]

    def _allowed_pairs(self) -> List[Tuple[str, str, float]]:
        """Graph edges that are still traversable (blocked corridors removed)."""
        blocked = self.topo.weather_blocked_edges
        pairs = []
        for (u, v), edge in self.topo.edges.items():
            if (u, v) in blocked:
                continue
            pairs.append((u, v, float(edge.length)))
        return pairs

    # ------------------------------------------------------------------- A*
    def route(self, start: str, goal: str, t_start: float):
        """Real time-varying A* route; ``None`` when the graph has no path."""
        begin = time.perf_counter()
        result = self.planner.plan_route(start, goal, t_start, density_fn=None)
        self._astar_ms += (time.perf_counter() - begin) * 1000.0
        self._astar_calls += 1
        if not result.nodes:
            return None
        return result

    def _agent_state(self, agent: str, frozen: Sequence[PlanStep]) -> Tuple[str, float]:
        """Position/time of ``agent`` after its last frozen step."""
        last: Optional[PlanStep] = None
        for step in frozen:
            if step.agent_id == agent and (last is None or step.t_end > last.t_end):
                last = step
        if last is None:
            return self.agent_home[agent], self.t0
        return last.target, last.t_end

    # ------------------------------------------------------------------ PDDL
    def _pddl_problem(
        self, agent: str, task_ids: Sequence[str], frozen: Sequence[PlanStep]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """Build PDDLSolver init/goal params for one agent.

        The whole airspace graph is the location universe and only traversable
        corridors enter the distance matrix, so the PDDL ``connected`` relation
        is exactly the A* graph minus the alert's blocked corridors.  One solver
        call per agent keeps the committed task-to-drone assignment fixed while
        the planner still chooses the visit order.
        """
        locations = sorted(self.topo.nodes)
        allowed = self._allowed_pairs()
        distance_matrix = {(u, v): length for u, v, length in allowed}
        max_len = max((length for _u, _v, length in allowed), default=1.0)
        threshold = max_len + 1.0

        node, _t = self._agent_state(agent, frozen)
        if node not in self.topo.nodes:
            raise RuntimeError(f"agent {agent} starts at unknown node {node}")

        goal_params: Dict[str, List[str]] = {key: [] for key in OP_TO_GOAL_KEY.values()}
        for tid in task_ids:
            task = self.tasks[tid]
            key = OP_TO_GOAL_KEY[task.op_type]
            if task.node not in goal_params[key]:
                goal_params[key].append(task.node)

        init_params = {
            "locations": locations,
            "distance_matrix": distance_matrix,
            "drone_locations": {agent: node},
            "package_locations": {},
            "packages_on_drone": {},
        }
        return init_params, goal_params, threshold

    def _order_from_solution(
        self, solution: Sequence[Dict[str, Any]], task_ids: Sequence[str]
    ) -> Optional[List[str]]:
        """Map the PDDL action sequence back onto this agent's scope tasks."""
        pending: Dict[Tuple[str, str], List[str]] = {}
        for tid in task_ids:
            task = self.tasks[tid]
            pending.setdefault((task.op_type, task.node), []).append(tid)

        order: List[str] = []
        for item in solution:
            name = str(item.get("action_name") or "").lower()
            params = [str(p) for p in (item.get("parameters") or [])]
            if name == "movetoloc" or len(params) < 2:
                continue
            op_type = ACTION_TO_OP.get(name)
            if op_type is None:
                continue
            bucket = pending.get((op_type, params[1]))
            if not bucket:
                continue
            # one visit may satisfy several identical checkpoints at that node
            order.extend(bucket)
            bucket.clear()

        if any(pending.values()):
            return None  # a scope goal was never achieved: mapping failure
        return order

    def _solve_one(
        self, stage: str, agent: str, task_ids: Sequence[str], frozen: Sequence[PlanStep]
    ) -> Tuple[Optional[List[str]], SolveRecord]:
        init_params, goal_params, threshold = self._pddl_problem(agent, task_ids, frozen)
        solver = PDDLSolver(str(self.domain_file) if self.domain_file else None)
        log_buffer = io.StringIO()
        begin = time.perf_counter()
        with contextlib.redirect_stdout(log_buffer):
            result = solver.solve(
                init_params,
                goal_params,
                distance_threshold=threshold,
                timeout=self.pddl_timeout,
            )
        wall_ms = (time.perf_counter() - begin) * 1000.0
        self._pddl_ms += wall_ms

        solver_log = log_buffer.getvalue()
        problem_file: Optional[str] = None
        if self.artifacts_dir is not None:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{self.alert.get('case_id', 'case')}_{stage}_{agent}"
            if solver.last_problem_content:
                problem_file = str(self.artifacts_dir / f"{slug}.problem.pddl")
                Path(problem_file).write_text(solver.last_problem_content, encoding="utf-8")
            (self.artifacts_dir / f"{slug}.solver.log").write_text(solver_log, encoding="utf-8")

        record = SolveRecord(
            stage=f"{stage}[{agent}]",
            scope_tasks=sorted(task_ids),
            scope_agents=[agent],
            backend=str(result.get("solver_used") or "unavailable"),
            success=bool(result.get("success")),
            error=None if result.get("success") else str(result.get("error")),
            n_actions=int(result.get("total_steps") or 0),
            wall_ms=round(wall_ms, 3),
            problem_file=problem_file,
            solver_log=solver_log,
        )
        self._solves.append(record)
        if not record.success:
            return None, record

        order = self._order_from_solution(list(result.get("solution") or []), task_ids)
        if order is None:
            record.success = False
            record.error = "PDDL plan does not achieve every scope goal (mapping failure)"
            return None, record
        return order, record

    def _solve_agents(
        self,
        stage: str,
        task_ids: Sequence[str],
        agents: Sequence[str],
        frozen: Sequence[PlanStep],
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Real PDDL solve per agent; returns agent -> ordered task ids."""
        orders: Dict[str, List[str]] = {}
        errors: List[str] = []
        for agent in agents:
            agent_tasks = [tid for tid in task_ids if self.tasks[tid].agent_id == agent]
            if not agent_tasks:
                continue
            order, record = self._solve_one(stage, agent, agent_tasks, frozen)
            if order is None:
                errors.append(f"{agent}: {record.error}")
            else:
                orders[agent] = order
        return orders, errors

    def _backend_chain(self) -> str:
        return ",".join(sorted({rec.backend for rec in self._solves}))

    def _topo_reorder(self, task_ids: Sequence[str]) -> List[str]:
        """Stable dependency-preserving reorder inside one agent's own queue."""
        remaining = list(task_ids)
        ordered: List[str] = []
        placed: set[str] = set()
        while remaining:
            progressed = False
            for tid in list(remaining):
                preds = [p for p in self.deps_in.get(tid, []) if p in task_ids]
                if all(p in placed for p in preds):
                    ordered.append(tid)
                    placed.add(tid)
                    remaining.remove(tid)
                    progressed = True
            if not progressed:  # dependency cycle inside the queue: keep PDDL order
                ordered.extend(remaining)
                break
        return ordered

    # -------------------------------------------------------------- schedule
    def _schedule_scope(
        self,
        agent_tasks: Dict[str, List[str]],
        frozen: Sequence[PlanStep],
        task_end: Dict[str, float],
    ) -> Optional[List[PlanStep]]:
        """Ground PDDL ordering into a timed plan using real A* routes."""
        steps: List[PlanStep] = []
        for agent in sorted(agent_tasks):
            order = self._topo_reorder(agent_tasks[agent])
            node, t_ready = self._agent_state(agent, frozen)
            for tid in order:
                task = self.tasks[tid]
                route = self.route(node, task.node, t_ready)
                if route is None:
                    return None
                dep_end = max([task_end.get(p, 0.0) for p in self.deps_in.get(tid, [])] or [0.0])
                t_start = max(route.total_time + t_ready, task.release, dep_end)
                t_arrive = t_ready + route.total_time
                if route.nodes and len(route.nodes) > 1:
                    steps.append(
                        PlanStep(
                            agent_id=agent,
                            action="movetoloc",
                            nodes=tuple(route.nodes),
                            t_start=t_ready,
                            t_end=t_arrive,
                            task_id=tid,
                            segments=tuple(
                                (seg.edge[0], seg.edge[1], seg.t_start, seg.t_end)
                                for seg in route.segments
                            ),
                            t_arrive=t_arrive,
                        )
                    )
                steps.append(
                    PlanStep(
                        agent_id=agent,
                        action=OP_TO_ACTION[task.op_type],
                        nodes=(task.node,),
                        t_start=t_start,
                        t_end=t_start + task.service_s,
                        task_id=tid,
                        t_arrive=t_arrive,
                    )
                )
                task_end[tid] = t_start + task.service_s
                node = task.node
                t_ready = t_start + task.service_s
        return steps

    def _validate(
        self, steps: Sequence[PlanStep], task_end: Dict[str, float]
    ) -> ValidationReport:
        """Re-simulate the merged plan and check every splice condition."""
        report = ValidationReport(valid=True)
        by_agent: Dict[str, List[PlanStep]] = {}
        for step in steps:
            by_agent.setdefault(step.agent_id, []).append(step)
        for agent, agent_steps in by_agent.items():
            agent_steps.sort(key=lambda s: (s.t_start, s.t_end))
            current = self.agent_home.get(agent)
            for step in agent_steps:
                if step.action == "movetoloc":
                    if not step.segments or step.nodes[0] != current:
                        report.c_state = False
                        if step.task_id:
                            report.conflict_tasks.append(step.task_id)
                        report.reasons.append(
                            f"{agent}: movetoloc starts at {step.nodes[0]} but drone is at {current}"
                        )
                    else:
                        for u, v, t0, _t1 in step.segments:
                            if (u, v) not in self.topo.edges:
                                report.c_state = False
                                if step.task_id:
                                    report.conflict_tasks.append(step.task_id)
                                report.reasons.append(f"{agent}: edge {u}->{v} does not exist")
                            elif not self.topo.is_edge_available(u, v, t0):
                                report.c_state = False
                                if step.task_id:
                                    report.conflict_tasks.append(step.task_id)
                                report.reasons.append(
                                    f"{agent}: edge {u}->{v} blocked at t={t0:.1f} (task {step.task_id})"
                                )
                    current = step.nodes[-1]
                else:
                    if current != step.target:
                        report.c_state = False
                        if step.task_id:
                            report.conflict_tasks.append(step.task_id)
                        report.reasons.append(
                            f"{agent}: {step.action} at {step.target} while drone is at {current}"
                        )

        for tid, step_end in task_end.items():
            task = self.tasks[tid]
            if step_end > task.deadline + 1e-6:
                report.c_time = False
                report.conflict_tasks.append(tid)
                report.reasons.append(
                    f"{tid}: ends at {step_end:.1f}s > deadline {task.deadline:.1f}s"
                )

        for u, v in self.deps:
            if u in task_end and v in task_end and task_end[u] > task_end[v] + 1e-6:
                report.c_dep = False
                for tid in (u, v):
                    if tid not in report.conflict_tasks:
                        report.conflict_tasks.append(tid)
                report.reasons.append(
                    f"dependency {u}->{v} violated ({task_end[u]:.1f}s > {task_end[v]:.1f}s)"
                )

        report.c_res, res_conflicts = self._capacity_conflicts(
            steps, kind="node", default_capacity=1
        )
        report.c_rsv, rsv_conflicts = self._capacity_conflicts(
            steps, kind="edge", default_capacity=1
        )
        for tid in res_conflicts + rsv_conflicts:
            if tid and tid not in report.conflict_tasks:
                report.conflict_tasks.append(tid)
        if not report.c_res:
            report.reasons.append("node capacity exceeded by concurrent occupancy")
        if not report.c_rsv:
            report.reasons.append("corridor reservation overlap (edge capacity exceeded)")

        report.valid = all(
            [report.c_state, report.c_time, report.c_dep, report.c_res, report.c_rsv]
        )
        report.conflict_tasks = sorted(set(report.conflict_tasks))
        return report

    def _capacity_conflicts(
        self, steps: Sequence[PlanStep], kind: str, default_capacity: int
    ) -> Tuple[bool, List[str]]:
        """Sweep-line capacity check over node occupancy or directed corridors."""
        intervals: Dict[Tuple[str, ...], List[Tuple[float, float, Optional[str]]]] = {}
        for step in steps:
            if kind == "node":
                for node, t0, t1 in step.node_occupancies():
                    cap = int(self.topo.nodes[node].capacity) if node in self.topo.nodes else 1
                    intervals.setdefault((node, str(cap)), []).append((t0, t1, step.task_id))
            else:
                for u, v, t0, t1 in step.segments:
                    edge = self.topo.edges.get((u, v))
                    cap = int(edge.capacity) if edge is not None else default_capacity
                    intervals.setdefault((f"{u}->{v}", str(cap)), []).append((t0, t1, step.task_id))

        ok = True
        conflicts: List[str] = []
        for (_key, cap_str), items in intervals.items():
            capacity = int(cap_str)
            events: List[Tuple[float, int]] = []
            for t0, t1, _tid in items:
                events.append((t0, 1))
                events.append((t1, -1))
            events.sort()
            concurrent = 0
            for _t, delta in events:
                concurrent += delta
                if concurrent > capacity:
                    ok = False
                    for t0, t1, tid in items:
                        if tid and t0 <= _t < t1:
                            conflicts.append(tid)
                    break
        return ok, sorted(set(conflicts))

    # ------------------------------------------------------------ algorithm
    def _seed_scope(self, remaining: Sequence[str]) -> List[str]:
        """A0 ∪ descendants ∪ same-agent tasks, restricted to remaining tasks."""
        wf = self.alert.get("workflow_id")
        remaining_set = set(remaining)
        anchors = [
            tid
            for tid in remaining
            if (wf is None or self.tasks[tid].workflow_id == str(wf))
            and (self.alert.get("node") is None or self.tasks[tid].node == self.alert.get("node"))
        ]
        if not anchors:  # fall back to the workflow's remaining tasks
            anchors = [tid for tid in remaining if self.tasks[tid].workflow_id == str(wf)]

        scope = set(anchors)
        adj: Dict[str, List[str]] = {}
        for u, v in self.deps:
            adj.setdefault(u, []).append(v)
        stack = list(anchors)
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, []):
                if nxt not in scope:
                    scope.add(nxt)
                    stack.append(nxt)

        agents = {self.tasks[tid].agent_id for tid in scope}
        for tid in remaining_set:
            if self.tasks[tid].agent_id in agents:
                scope.add(tid)
        return sorted(scope & remaining_set)

    def _expand_scope(self, scope: Sequence[str], conflicts: Sequence[str]) -> List[str]:
        """Add conflicting tasks and their whole agent queues to the scope."""
        remaining = {tid for tid in self.tasks if tid not in scope}
        grown = set(scope)
        for tid in conflicts:
            if tid in self.tasks and tid in remaining:
                grown.add(tid)
                grown |= {o for o in remaining if self.tasks[o].agent_id == self.tasks[tid].agent_id}
        return sorted(grown)

    def baseline(self) -> Tuple[List[PlanStep], ValidationReport, str]:
        """Committed plan: real per-agent PDDL solve on the un-alerted graph."""
        task_ids = sorted(self.tasks)
        agents = sorted(self.agent_home)
        orders, errors = self._solve_agents("baseline", task_ids, agents, [])
        backend = self._backend_chain()
        if errors:
            return (
                [],
                ValidationReport(valid=False, reasons=[f"baseline PDDL failed: {errors[0]}"]),
                backend,
            )
        task_end: Dict[str, float] = {}
        steps = self._schedule_scope(orders, [], task_end)
        if steps is None:
            return (
                [],
                ValidationReport(valid=False, reasons=["baseline A* found no route"]),
                backend,
            )
        return steps, self._validate(steps, task_end), backend

    def replan(
        self, baseline_steps: Sequence[PlanStep], baseline_end: Dict[str, float]
    ) -> ReplanResult:
        """Run bounded scope expansion, then real global recovery."""
        total_start = time.perf_counter()
        window = self.alert.get("window") or [self.t0, float("inf")]
        alert_start = float(window[0])
        remaining = sorted(
            tid for tid in self.tasks if baseline_end.get(tid, float("inf")) > alert_start
        )
        scope = self._seed_scope(remaining)
        scope_initial = list(scope)
        notes: List[str] = []
        validation = ValidationReport(valid=False, reasons=["not attempted"])
        repaired: List[PlanStep] = []
        rounds = 0
        status = "infeasible"
        global_recovery = False

        for k in range(self.k_max + 1):
            rounds = k
            scope_agents = sorted({self.tasks[tid].agent_id for tid in scope})
            frozen = [s for s in baseline_steps if s.task_id not in set(scope)]
            orders, errors = self._solve_agents(f"scope_r{k}", scope, scope_agents, frozen)
            if errors:
                # a PDDL failure is not a scope problem: go straight to global recovery
                notes.append(f"round {k}: PDDL solve failed ({errors[0]})")
                break

            task_end = {tid: t for tid, t in baseline_end.items() if tid not in set(scope)}
            scheduled = self._schedule_scope(orders, frozen, task_end)
            if scheduled is None:
                notes.append(f"round {k}: A* found no route for the repaired scope")
                validation = ValidationReport(
                    valid=False,
                    c_state=False,
                    reasons=["A* found no route for at least one repaired task"],
                )
                if k >= self.k_max:
                    break
                continue

            merged = list(frozen) + scheduled
            validation = self._validate(merged, task_end)
            if validation.valid:
                repaired = merged
                status = "splice_accepted"
                notes.append(f"round {k}: splice accepted (scope {len(scope)} tasks)")
                break

            notes.append(
                f"round {k}: splice rejected -> {', '.join(validation.reasons[:3])}"
            )
            ratio = len(scope) / max(1, len(self.tasks))
            elapsed = time.perf_counter() - total_start
            if (
                ratio > self.rho_max
                or elapsed > self.b_rep_s
                or len(validation.conflict_tasks) > self.c_max
                or k >= self.k_max
            ):
                break
            grown = self._expand_scope(scope, validation.conflict_tasks)
            if grown == list(scope):
                notes.append("round %d: no scope growth possible" % k)
                break
            scope = grown

        if status != "splice_accepted":
            # real global recovery attempt over the full remaining problem
            all_tasks = sorted(self.tasks)
            all_agents = sorted(self.agent_home)
            orders, errors = self._solve_agents("global", all_tasks, all_agents, [])
            if errors:
                status = "infeasible"
                notes.append(f"global PDDL solve failed: {errors[0]}")
                repaired = list(baseline_steps)
                validation = ValidationReport(
                    valid=False,
                    reasons=[f"global PDDL solve failed: {errors[0]}"],
                )
            else:
                task_end: Dict[str, float] = {}
                scheduled = self._schedule_scope(orders, [], task_end)
                if scheduled is None:
                    status = "infeasible"
                    notes.append("global plan found by PDDL but A* found no route")
                    repaired = list(baseline_steps)
                    validation = ValidationReport(
                        valid=False, c_state=False, reasons=["global A* route infeasible"]
                    )
                else:
                    validation = self._validate(scheduled, task_end)
                    repaired = scheduled
                    if validation.valid:
                        status = "global_recovered"
                        global_recovery = True
                        notes.append("global replan solved and validated")
                    else:
                        status = "infeasible"
                        notes.append(
                            "global plan rejected by validator: "
                            + "; ".join(validation.reasons[:3])
                        )

        changed = self._changed_routes(baseline_steps, repaired)
        elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        return ReplanResult(
            case_id=str(self.alert.get("case_id", "case")),
            status=status,
            backend_chain=sorted({rec.backend for rec in self._solves}),
            scope_initial=scope_initial,
            scope_final=sorted(scope),
            expansion_rounds=rounds,
            global_recovery=global_recovery,
            validation=validation.to_dict(),
            baseline_plan=[self._step_dict(s) for s in baseline_steps],
            repaired_plan=[self._step_dict(s) for s in repaired],
            changed_routes=changed,
            blocked_edges=self._blocked_edges,
            solves=[self._solve_dict(r) for r in self._solves],
            timings_ms={
                "total": round(elapsed_ms, 3),
                "pddl": round(self._pddl_ms, 3),
                "astar": round(self._astar_ms, 3),
                "astar_calls": float(self._astar_calls),
            },
            notes=notes,
        )

    @staticmethod
    def _step_dict(step: PlanStep) -> Dict[str, Any]:
        return {
            "agent": step.agent_id,
            "action": step.action,
            "task": step.task_id,
            "nodes": list(step.nodes),
            "t_start": round(step.t_start, 3),
            "t_end": round(step.t_end, 3),
        }

    @staticmethod
    def _solve_dict(record: SolveRecord) -> Dict[str, Any]:
        return {
            "stage": record.stage,
            "backend": record.backend,
            "success": record.success,
            "error": record.error,
            "n_actions": record.n_actions,
            "scope_tasks": record.scope_tasks,
            "scope_agents": record.scope_agents,
            "wall_ms": record.wall_ms,
            "problem_file": record.problem_file,
        }

    def _changed_routes(
        self, baseline: Sequence[PlanStep], repaired: Sequence[PlanStep]
    ) -> List[Dict[str, Any]]:
        """Tasks whose real A* route or timing changed after the repair."""
        old = {s.task_id: s for s in baseline if s.task_id and s.action == "movetoloc"}
        new = {s.task_id: s for s in repaired if s.task_id and s.action == "movetoloc"}
        changed = []
        for tid in sorted(set(old) | set(new)):
            before = old.get(tid)
            after = new.get(tid)
            before_nodes = list(before.nodes) if before else []
            after_nodes = list(after.nodes) if after else []
            if before_nodes == after_nodes:
                continue
            changed.append(
                {
                    "task": tid,
                    "baseline_route": before_nodes,
                    "repaired_route": after_nodes,
                    "baseline_t_end": round(before.t_end, 3) if before else None,
                    "repaired_t_end": round(after.t_end, 3) if after else None,
                }
            )
        return changed


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_relative(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate)
