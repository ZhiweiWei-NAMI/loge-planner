"""Validator-gated scoped-replanning benchmark with real PDDL + real A*.

Entry point for the scoped-repair / splice-validation / bounded scope-expansion
procedure.  Every task order comes from a real PDDL solve (unified_planning +
fast-downward) and every leg is routed by the time-varying A* in
``src/offline_pipeline/edge_planner.py``; nothing is resampled, shifted or
skipped -- every reported case is a real solve.

Three modes:

* ``--scenarios S1 ... S5 --seeds ... --cases-per-seed N`` builds
  **deterministically generated benchmark instances** (a fixed
  ``(scenario, seed)`` pair reproduces the identical graph, schedule and alert
  set) whose scale grows from S1 (small) to S5 (large), synthesises ``N``
  alerts per pair and runs the real scoped repair on each of them.  Pass
  ``--schedule-dir`` to read real schedule files (graph + snapshot + alerts)
  instead of the generator.
* ``--demo`` runs the bundled smoke fixtures in ``examples/replanning/`` and
  checks each case against its declared expectation.
* ``--graph/--snapshot/--alert`` runs user-supplied external inputs.  An alert
  file may carry its own ``graph``/``snapshot`` paths (resolved relative to the
  alert file) so a single ``--alert`` is enough.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.validation.scoped_replanner import (
    ReplanResult,
    ScopedReplanner,
    load_json,
    resolve_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = PROJECT_ROOT / "examples" / "replanning"
DEFAULT_DOMAIN = PROJECT_ROOT / "config" / "pddl" / "tccn_domain.pddl"

#: fixture -> declared expectation (verified by --demo)
DEMO_CASES = {
    "demo_success": "splice_accepted",
    "demo_detour": "splice_accepted",
    "demo_expand": "splice_accepted",
    "demo_global": "global_recovered",
    "demo_infeasible": "infeasible",
}

# --------------------------------------------------------------------- generator
#: benchmark scale per scenario label: S1 is the smallest instance, S5 the largest.
#: The numbers only set the instance *size*; every case is still solved for real.
SCENARIO_SPECS: Dict[str, Dict[str, int]] = {
    "S1": {"rows": 3, "cols": 4, "drones": 4, "tasks": 8, "workflows": 2},
    "S2": {"rows": 4, "cols": 5, "drones": 6, "tasks": 12, "workflows": 3},
    "S3": {"rows": 5, "cols": 6, "drones": 8, "tasks": 18, "workflows": 4},
    "S4": {"rows": 6, "cols": 7, "drones": 10, "tasks": 24, "workflows": 5},
    "S5": {"rows": 7, "cols": 8, "drones": 12, "tasks": 30, "workflows": 6},
}

GRID_SPACING_M = 1000.0
BASE_VELOCITY_MPS = 15.0
NODE_CAPACITY = 3
EDGE_CAPACITY = 2
OP_TYPES = ("inspect", "compute", "communicate")
SERVICE_CHOICES = (30.0, 45.0, 60.0, 90.0)
#: alert classes drawn by the generator (weights below)
ALERT_CLASSES = ("corridor_blocked", "corridor_lost", "battery_insufficient")
ALERT_WEIGHTS = (0.45, 0.35, 0.20)
#: fraction of generated cases that run with rho_max=0.1 (global-recovery path)
GLOBAL_RECOVERY_FRACTION = 0.2
#: fraction of corridor_lost cases that isolate the anchor node (real infeasibility)
NODE_ISOLATION_FRACTION = 0.2

GENERATOR_NOTE = (
    "deterministically generated benchmark instances: the same (scenario, seed) "
    "pair reproduces an identical graph, schedule and alert set; pass "
    "--schedule-dir to evaluate real schedule files instead"
)


def _grid_id(r: int, c: int) -> str:
    return f"wp_{r}_{c}"


def build_graph(scenario: str) -> Dict[str, Any]:
    """Grid airspace graph for ``scenario`` (deterministic, no randomness)."""
    spec = SCENARIO_SPECS[scenario]
    rows, cols = int(spec["rows"]), int(spec["cols"])

    nodes: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            nodes.append(
                {
                    "id": _grid_id(r, c),
                    "type": "waypoint",
                    "x": round(c * GRID_SPACING_M, 3),
                    "y": round(r * GRID_SPACING_M, 3),
                    "z": 100.0,
                    "capacity": NODE_CAPACITY,
                }
            )

    edges: List[Dict[str, Any]] = []

    def add_edge(u: str, v: str) -> None:
        edges.append(
            {
                "u": u,
                "v": v,
                "base_velocity": BASE_VELOCITY_MPS,
                "rho_jam": 0.5,
                "capacity": EDGE_CAPACITY,
            }
        )

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                add_edge(_grid_id(r, c), _grid_id(r, c + 1))
                add_edge(_grid_id(r, c + 1), _grid_id(r, c))
            if r + 1 < rows:
                add_edge(_grid_id(r, c), _grid_id(r + 1, c))
                add_edge(_grid_id(r + 1, c), _grid_id(r, c))
            # checkerboard diagonal shortcuts: real detour alternatives for A*
            if (r + c) % 2 == 0 and r + 1 < rows and c + 1 < cols:
                add_edge(_grid_id(r, c), _grid_id(r + 1, c + 1))
                add_edge(_grid_id(r + 1, c + 1), _grid_id(r, c))

    return {
        "name": f"generated_{scenario}_grid_{rows}x{cols}",
        "note": f"{scenario} benchmark airspace ({GENERATOR_NOTE}).",
        "bounds": {
            "x_min": 0.0,
            "x_max": round((cols - 1) * GRID_SPACING_M, 3),
            "y_min": 0.0,
            "y_max": round((rows - 1) * GRID_SPACING_M, 3),
            "z_min": 0.0,
            "z_max": 200.0,
        },
        "nodes": nodes,
        "edges": edges,
    }


def build_snapshot(scenario: str, seed: int, graph: Dict[str, Any]) -> Dict[str, Any]:
    """Committed schedule for ``(scenario, seed)`` -- same seed, same schedule."""
    spec = SCENARIO_SPECS[scenario]
    rows, cols = int(spec["rows"]), int(spec["cols"])
    drones, n_tasks, n_wf = int(spec["drones"]), int(spec["tasks"]), int(spec["workflows"])
    rng = random.Random(f"snapshot|{scenario}|{seed}")

    node_ids = [str(n["id"]) for n in graph["nodes"]]
    homes = rng.sample(node_ids, drones)
    agents = {f"drone_{i:02d}": {"node": homes[i]} for i in range(drones)}
    agent_ids = sorted(agents)

    edge_time = GRID_SPACING_M / BASE_VELOCITY_MPS
    diameter_t = (rows + cols) * edge_time

    # contiguous workflow blocks, each split into contiguous per-agent sub-blocks
    base, extra = divmod(n_tasks, n_wf)
    sizes = [base + (1 if k < extra else 0) for k in range(n_wf)]

    tasks: List[Dict[str, Any]] = []
    deps: List[List[str]] = []
    position = 0
    queue_pos: Counter = Counter()
    for k, size in enumerate(sizes):
        if size <= 0:
            continue
        wf = f"wf{k + 1}"
        block = list(range(position, position + size))
        position += size
        n_agents_used = max(1, min(drones, size // 2 or 1))
        chosen = rng.sample(agent_ids, n_agents_used)
        per_agent = max(1, math.ceil(size / n_agents_used))
        previous: Optional[str] = None
        previous_agent: Optional[str] = None
        for offset, index in enumerate(block):
            agent = chosen[min(offset // per_agent, n_agents_used - 1)]
            task_id = f"T{index:02d}"
            service_s = rng.choice(SERVICE_CHOICES)
            release = round(rng.uniform(0.0, 240.0), 1)
            slot = queue_pos[agent]
            queue_pos[agent] += 1
            # deadline follows the agent's queue position: satisfiable for the
            # committed plan, still finite so real detours can violate it.
            deadline = round(release + (slot + 1.5) * diameter_t + 600.0 + 2.0 * service_s, 1)
            tasks.append(
                {
                    "task_id": task_id,
                    "workflow_id": wf,
                    "agent_id": agent,
                    "node": rng.choice(node_ids),
                    "op_type": rng.choice(OP_TYPES),
                    "release": release,
                    "deadline": deadline,
                    "service_s": service_s,
                }
            )
            if previous is not None and previous_agent == agent:
                deps.append([previous, task_id])
            previous, previous_agent = task_id, agent

    return {
        "name": f"generated_{scenario}_seed{seed}_schedule",
        "note": f"{scenario} benchmark schedule, seed {seed} ({GENERATOR_NOTE}).",
        "t0": 0.0,
        "agents": agents,
        "tasks": tasks,
        "dependencies": deps,
    }


def _dijkstra_edges(adj: Dict[str, List[Tuple[str, float]]], start: str, goal: str) -> Set[Tuple[str, str]]:
    """Edges on the shortest ``start -> goal`` path of the base (un-alerted) graph."""
    if start == goal:
        return set()
    dist: Dict[str, float] = {start: 0.0}
    prev: Dict[str, str] = {}
    heap: List[Tuple[float, str]] = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            break
        if cost > dist.get(node, math.inf):
            continue
        for nxt, weight in adj.get(node, ()):
            candidate = cost + weight
            if candidate < dist.get(nxt, math.inf):
                dist[nxt] = candidate
                prev[nxt] = node
                heapq.heappush(heap, (candidate, nxt))
    if goal not in dist:
        return set()
    path: Set[Tuple[str, str]] = set()
    cursor = goal
    while cursor != start:
        parent = prev[cursor]
        path.add((parent, cursor))
        cursor = parent
    return path


def committed_corridors(graph: Dict[str, Any], snapshot: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Corridors the committed schedule flies (base-graph shortest paths)."""
    coords = {str(n["id"]): (float(n["x"]), float(n["y"])) for n in graph["nodes"]}
    adj: Dict[str, List[Tuple[str, float]]] = {}
    for edge in graph["edges"]:
        u, v = str(edge["u"]), str(edge["v"])
        length = math.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1])
        adj.setdefault(u, []).append((v, length))

    queues: Dict[str, List[str]] = {}
    for task in snapshot["tasks"]:
        queues.setdefault(str(task["agent_id"]), []).append(str(task["node"]))

    used: Set[Tuple[str, str]] = set()
    for agent, goals in queues.items():
        cursor = str(snapshot["agents"][agent]["node"])
        for goal in goals:
            used |= _dijkstra_edges(adj, cursor, goal)
            cursor = goal
    return sorted(used)


def generate_alerts(
    scenario: str,
    seed: int,
    graph: Dict[str, Any],
    snapshot: Dict[str, Any],
    count: int,
) -> List[Dict[str, Any]]:
    """``count`` alerts for ``(scenario, seed)`` -- reproducible from the seed."""
    rng = random.Random(f"alerts|{scenario}|{seed}")
    corridors = committed_corridors(graph, snapshot)
    fallback = [(str(e["u"]), str(e["v"])) for e in graph["edges"]]
    pool = corridors or fallback
    incident: Dict[str, Set[Tuple[str, str]]] = {}
    for edge in graph["edges"]:
        incident.setdefault(str(edge["u"]), set()).add((str(edge["u"]), str(edge["v"])))

    alerts: List[Dict[str, Any]] = []
    for index in range(count):
        task = rng.choice(snapshot["tasks"])
        category = rng.choices(ALERT_CLASSES, weights=ALERT_WEIGHTS)[0]
        blocked: List[List[str]] = []
        if category == "corridor_blocked":
            blocked = [list(rng.choice(pool))]
        elif category == "corridor_lost":
            node_edges = sorted(incident.get(str(task["node"]), set()))
            if node_edges and rng.random() < NODE_ISOLATION_FRACTION:
                blocked = [list(pair) for pair in node_edges]  # isolates the node
            else:
                picks = rng.sample(pool, k=min(2, len(pool)))
                blocked = [list(pair) for pair in picks]

        alert: Dict[str, Any] = {
            "case_id": f"{scenario}_s{seed}_c{index:02d}",
            "category": category,
            "severity": "high" if blocked else "medium",
            "window": [0.0, 600.0],
            "workflow_id": str(task["workflow_id"]),
            "node": str(task["node"]),
            "blocked_edges": blocked,
            "directional": False,
            "generated": True,
        }
        if rng.random() < GLOBAL_RECOVERY_FRACTION:
            alert["rho_max"] = 0.1
        alerts.append(alert)
    return alerts


def load_schedule_cases(
    schedule_dir: Path, scenario: str, seed: int, count: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """Read real schedule inputs: ``<dir>/<scenario>_seed<seed>.json``.

    The file carries ``graph`` / ``snapshot`` (inline object or path relative to
    the file) and an ``alerts`` list (objects or paths).
    """
    path = Path(schedule_dir) / f"{scenario}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"schedule file not found: {path} (expected <schedule-dir>/{scenario}_seed{seed}.json)"
        )
    payload = load_json(path)

    def _materialise(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        return load_json(resolve_relative(path.parent, str(value)))

    if "graph" not in payload or "snapshot" not in payload:
        raise ValueError(f"{path}: 'graph' and 'snapshot' are required")
    graph = _materialise(payload["graph"])
    snapshot = _materialise(payload["snapshot"])
    alerts = [_materialise(item) for item in (payload.get("alerts") or [])]
    if len(alerts) < count:
        raise ValueError(
            f"{path}: {len(alerts)} alert(s) available but --cases-per-seed={count}"
        )
    return [(graph, snapshot, alert) for alert in alerts[:count]]


# ------------------------------------------------------------------- case runner
def _overrides(alert: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Per-alert overrides win over CLI defaults (external input stays explicit)."""
    return {
        "k_max": int(alert.get("k_max", args.k_max)),
        "b_rep_s": float(alert.get("b_rep_s", args.b_rep)),
        "rho_max": float(alert.get("rho_max", args.rho_max)),
        "c_max": int(alert.get("c_max", args.c_max)),
        "pddl_timeout": float(alert.get("pddl_timeout", args.pddl_timeout)),
    }


def run_case(
    graph: Dict[str, Any],
    snapshot: Dict[str, Any],
    alert: Dict[str, Any],
    args: argparse.Namespace,
    artifacts_dir: Optional[Path],
) -> ReplanResult:
    """Baseline on the un-alerted graph, then repair on the alerted graph."""
    overrides = _overrides(alert, args)
    case_id = str(alert.get("case_id") or "case")

    baseline_solver = ScopedReplanner(
        graph=graph,
        snapshot=snapshot,
        alert={"case_id": f"{case_id}_baseline"},
        domain_file=args.domain,
        artifacts_dir=artifacts_dir,
        **overrides,
    )
    baseline_steps, baseline_report, _backend = baseline_solver.baseline()
    baseline_end = {s.task_id: s.t_end for s in baseline_steps if s.task_id}
    if not baseline_report.valid:
        raise RuntimeError(
            f"{case_id}: committed baseline plan is invalid before the alert: "
            f"{baseline_report.reasons}"
        )

    replanner = ScopedReplanner(
        graph=graph,
        snapshot=snapshot,
        alert=alert,
        domain_file=args.domain,
        artifacts_dir=artifacts_dir,
        **overrides,
    )
    replanner.apply_alert()
    result = replanner.replan(baseline_steps, baseline_end)
    result.notes.insert(
        0,
        f"baseline: {len(baseline_steps)} steps, backend={_backend}, "
        f"wall={result.timings_ms['total']:.0f}ms total",
    )
    return result


def _fmt_route(nodes: List[str]) -> str:
    return "->".join(nodes) if nodes else "(none)"


def print_case(result: ReplanResult) -> None:
    print(f"\n=== {result.case_id} ===")
    print(f"status           : {result.status}")
    print(f"backend chain    : {', '.join(result.backend_chain) or 'none'}")
    print(
        f"scope            : {len(result.scope_initial)} -> {len(result.scope_final)} tasks "
        f"({result.scope_initial} -> {result.scope_final}), expansion_rounds={result.expansion_rounds}"
    )
    val = result.validation
    print(
        "validation       : "
        f"valid={val['valid']} state={val['c_state']} time={val['c_time']} "
        f"dep={val['c_dep']} res={val['c_res']} rsv={val['c_rsv']}"
    )
    if val["reasons"]:
        for reason in val["reasons"][:4]:
            print(f"  reason         : {reason}")
    timings = result.timings_ms
    print(
        f"timings          : total={timings['total']:.0f}ms pddl={timings['pddl']:.0f}ms "
        f"astar={timings['astar']:.0f}ms ({int(timings['astar_calls'])} calls)"
    )
    print(f"blocked edges    : {result.blocked_edges or 'none'}")
    print("solves           :")
    for solve in result.solves:
        state = "ok" if solve["success"] else f"FAILED ({solve['error']})"
        print(
            f"  - {solve['stage']:<10} backend={solve['backend']:<13} "
            f"actions={solve['n_actions']:<3} wall={solve['wall_ms']:.0f}ms scope={solve['scope_tasks']} {state}"
        )
    if result.changed_routes:
        print("changed routes   :")
        for change in result.changed_routes:
            print(f"  - {change['task']}: {_fmt_route(change['baseline_route'])}")
            print(f"      => {_fmt_route(change['repaired_route'])}")
    else:
        print("changed routes   : none")
    if result.repaired_plan and result.status == "infeasible":
        print("plan (unchanged) : NOT a repair -- this is the pre-alert committed plan")
    elif result.repaired_plan:
        print("plan (repaired)  :")
        for step in result.repaired_plan:
            print(
                f"  {step['agent']:<8} {step['action']:<18} task={str(step['task']):<4} "
                f"[{step['t_start']:8.1f},{step['t_end']:8.1f}] {_fmt_route(step['nodes'])}"
            )
    for note in result.notes:
        print(f"note             : {note}")


# ------------------------------------------------------------------- aggregation
def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Real aggregate metrics over real case outcomes (nothing imputed)."""
    total = len(rows)
    statuses = Counter(str(row["status"]) for row in rows)
    backends: Counter = Counter()
    solves = 0
    failed_solves = 0
    for row in rows:
        for backend, count in row["backends"].items():
            backends[backend] += count
        solves += int(row["solves_total"])
        failed_solves += int(row["solves_failed"])

    latencies = [float(row["repair_ms"]) for row in rows]
    case_walls = [float(row["case_wall_ms"]) for row in rows]

    def rate(value: int) -> float:
        return (value / total) if total else 0.0

    accepted = statuses.get("splice_accepted", 0)
    global_recovered = statuses.get("global_recovered", 0)
    infeasible = statuses.get("infeasible", 0)
    expanded = sum(1 for row in rows if int(row["expansion_rounds"]) > 0)
    other = {
        status: count
        for status, count in statuses.items()
        if status not in {"splice_accepted", "global_recovered", "infeasible"}
    }
    return {
        "cases": total,
        "status_counts": dict(sorted(statuses.items())),
        "accepted": accepted,
        "accept_rate": round(rate(accepted), 4),
        "expanded": expanded,
        "expansion_rate": round(rate(expanded), 4),
        "global_recovered": global_recovered,
        "global_recovery_rate": round(rate(global_recovered), 4),
        "infeasible": infeasible,
        "infeasible_rate": round(rate(infeasible), 4),
        "other_statuses": other,
        "repaired_rate": round(rate(accepted + global_recovered), 4),
        "repair_latency_ms": {
            "mean": round(sum(latencies) / total, 3) if total else 0.0,
            "median": round(_percentile(latencies, 0.5), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "case_wall_ms": {
            "mean": round(sum(case_walls) / total, 3) if total else 0.0,
            "max": round(max(case_walls), 3) if case_walls else 0.0,
        },
        "pddl_ms_total": round(sum(float(row["pddl_ms"]) for row in rows), 3),
        "astar_calls_total": int(sum(int(row["astar_calls"]) for row in rows)),
        "solver_solves": solves,
        "solver_failed_solves": failed_solves,
        "solver_backends": dict(sorted(backends.items())),
    }


def _print_aggregate_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'scenario':<9}{'seed':>6}{'cases':>7}{'accept%':>9}{'expand%':>9}"
        f"{'global%':>9}{'infeas%':>9}{'err':>5}{'mean_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:<9}{row['seed']:>6}{row['cases']:>7}"
            f"{100.0 * row['accept_rate']:>9.1f}{100.0 * row['expansion_rate']:>9.1f}"
            f"{100.0 * row['global_recovery_rate']:>9.1f}{100.0 * row['infeasible_rate']:>9.1f}"
            f"{row['errors']:>5}{row['repair_latency_ms']['mean']:>10.0f}"
        )


def _print_overall(agg: Dict[str, Any], solver_logs: int, failures: Sequence[str]) -> None:
    lat = agg["repair_latency_ms"]
    print("\n=== overall ===")
    print(
        f"cases={agg['cases']} accepted={agg['accepted']} ({100.0 * agg['accept_rate']:.1f}%) "
        f"expanded={agg['expanded']} ({100.0 * agg['expansion_rate']:.1f}%) "
        f"global_recovered={agg['global_recovered']} ({100.0 * agg['global_recovery_rate']:.1f}%) "
        f"infeasible={agg['infeasible']} ({100.0 * agg['infeasible_rate']:.1f}%)"
    )
    if agg["other_statuses"]:
        print(f"other statuses   : {agg['other_statuses']}")
    print(
        f"repair latency   : mean={lat['mean']:.0f}ms median={lat['median']:.0f}ms "
        f"p95={lat['p95']:.0f}ms min={lat['min']:.0f}ms max={lat['max']:.0f}ms "
        f"(real wall clock, replan only)"
    )
    print(
        f"case wall clock  : mean={agg['case_wall_ms']['mean']:.0f}ms "
        f"max={agg['case_wall_ms']['max']:.0f}ms (baseline + replan)"
    )
    print(
        f"solver           : solves={agg['solver_solves']} failed={agg['solver_failed_solves']} "
        f"backends={agg['solver_backends']} pddl={agg['pddl_ms_total']:.0f}ms "
        f"astar_calls={agg['astar_calls_total']} solver_logs={solver_logs}"
    )
    for failure in failures:
        print(f"FAILED: {failure}")


# --------------------------------------------------------------------- batch mode
def _batch_row(
    result: ReplanResult,
    scenario: str,
    seed: int,
    case_index: int,
    alert: Dict[str, Any],
    case_wall_ms: float,
) -> Dict[str, Any]:
    backends = Counter(str(solve["backend"]) for solve in result.solves)
    return {
        "case_id": result.case_id,
        "scenario": scenario,
        "seed": seed,
        "case_index": case_index,
        "category": alert.get("category"),
        "status": result.status,
        "expansion_rounds": result.expansion_rounds,
        "global_recovery": result.global_recovery,
        "scope_initial": len(result.scope_initial),
        "scope_final": len(result.scope_final),
        "blocked_edges": result.blocked_edges,
        "rho_max": alert.get("rho_max"),
        "repair_ms": float(result.timings_ms["total"]),
        "pddl_ms": float(result.timings_ms["pddl"]),
        "astar_calls": int(result.timings_ms["astar_calls"]),
        "solves_total": len(result.solves),
        "solves_failed": sum(1 for solve in result.solves if not solve["success"]),
        "backends": dict(sorted(backends.items())),
        "case_wall_ms": round(case_wall_ms, 3),
    }


def run_batch(args: argparse.Namespace) -> int:
    rows: List[Dict[str, Any]] = []
    case_payloads: List[Dict[str, Any]] = []
    failures: List[str] = []
    batch_begin = time.perf_counter()

    if not args.schedule_dir:
        print(f"instances        : {GENERATOR_NOTE}")
    else:
        print(f"instances        : real schedule files from {args.schedule_dir}")

    for scenario in args.scenarios:
        for seed in args.seeds:
            try:
                if args.schedule_dir:
                    cases = load_schedule_cases(args.schedule_dir, scenario, seed, args.cases_per_seed)
                else:
                    graph = build_graph(scenario)
                    snapshot = build_snapshot(scenario, seed, graph)
                    alerts = generate_alerts(scenario, seed, graph, snapshot, args.cases_per_seed)
                    cases = [(graph, snapshot, alert) for alert in alerts]
            except Exception as exc:  # a broken instance must never look like a success
                failures.append(f"{scenario}/seed{seed}: {exc}")
                print(f"=== {scenario}/seed{seed} ===\nstatus           : error\nerror            : {exc}")
                continue

            for index, (graph, snapshot, alert) in enumerate(cases):
                case_begin = time.perf_counter()
                try:
                    result = run_case(graph, snapshot, alert, args, args.artifacts_dir)
                except Exception as exc:  # a broken case must never look like a success
                    failures.append(f"{alert.get('case_id', 'case')}: {exc}")
                    print(f"\n=== {alert.get('case_id', 'case')} ===")
                    print("status           : error")
                    print(f"error            : {exc}")
                    continue
                case_wall_ms = (time.perf_counter() - case_begin) * 1000.0
                rows.append(
                    _batch_row(result, scenario, seed, index, alert, case_wall_ms)
                )
                payload = dict(result.__dict__)
                payload.update({"scenario": scenario, "seed": seed, "category": alert.get("category")})
                case_payloads.append(payload)
                if not args.quiet:
                    print_case(result)

    # per (scenario, seed)
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["seed"]), []).append(row)
    per_seed: List[Dict[str, Any]] = []
    for (scenario, seed), group in sorted(grouped.items()):
        entry = aggregate_rows(group)
        entry.update({"scenario": scenario, "seed": seed})
        entry["errors"] = sum(
            1 for failure in failures if failure.startswith(f"{scenario}_s{seed}")
        )
        per_seed.append(entry)

    per_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario in args.scenarios:
        group = [row for row in rows if row["scenario"] == scenario]
        if group:
            per_scenario[scenario] = aggregate_rows(group)

    overall = aggregate_rows(rows)
    solver_logs = (
        len(list(Path(args.artifacts_dir).glob("*.solver.log")))
        if args.artifacts_dir is not None and Path(args.artifacts_dir).is_dir()
        else 0
    )

    _print_aggregate_table("batch summary (per scenario/seed)", per_seed)
    _print_overall(overall, solver_logs, failures)
    print(f"batch wall clock : {(time.perf_counter() - batch_begin):.1f}s")

    if args.output_json is not None:
        payload = {
            "mode": "batch",
            "instance_source": "schedule_dir" if args.schedule_dir else "deterministic_generator",
            "instance_note": GENERATOR_NOTE,
            "scenario_specs": {name: SCENARIO_SPECS[name] for name in args.scenarios},
            "scenarios": list(args.scenarios),
            "seeds": list(args.seeds),
            "cases_per_seed": args.cases_per_seed,
            "config": {
                "k_max": args.k_max,
                "b_rep_s": args.b_rep,
                "rho_max": args.rho_max,
                "c_max": args.c_max,
                "pddl_timeout": args.pddl_timeout,
                "domain": str(args.domain),
                "artifacts_dir": str(args.artifacts_dir) if args.artifacts_dir else None,
            },
            "aggregate": overall,
            "per_scenario": per_scenario,
            "per_scenario_seed": {f"{e['scenario']}/seed{e['seed']}": e for e in per_seed},
            "solver_log_files": solver_logs,
            "batch_wall_s": round(time.perf_counter() - batch_begin, 3),
            "case_table": rows,
            "cases": case_payloads,
            "failures": failures,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    return 1 if failures else 0


# -------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validator-gated scoped replanning benchmark (real PDDL + real A*)."
    )
    parser.add_argument("--demo", action="store_true", help="Run the bundled examples/replanning fixtures.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIO_SPECS),
        help="Batch mode: scenario labels to generate (S1 small .. S5 large).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Batch mode: integer seeds; a fixed seed reproduces the identical instance.",
    )
    parser.add_argument(
        "--cases-per-seed",
        type=int,
        default=5,
        help="Batch mode: number of alerts (and real repairs) per (scenario, seed).",
    )
    parser.add_argument(
        "--schedule-dir",
        type=Path,
        default=None,
        help="Batch mode: read real schedule files <dir>/<scenario>_seed<seed>.json "
        "instead of generating instances.",
    )
    parser.add_argument("--graph", type=Path, help="Airspace graph JSON.")
    parser.add_argument("--snapshot", type=Path, help="Committed schedule snapshot JSON.")
    parser.add_argument("--alert", type=Path, action="append", default=[], help="Alert JSON (repeatable).")
    parser.add_argument("--domain", type=Path, default=DEFAULT_DOMAIN, help="PDDL domain file.")
    parser.add_argument("--k-max", type=int, default=3, help="Max scope-expansion rounds.")
    parser.add_argument("--b-rep", type=float, default=30.0, help="Repair wall-clock budget (s).")
    parser.add_argument("--rho-max", type=float, default=0.6, help="Max scope ratio before global recovery.")
    parser.add_argument("--c-max", type=int, default=8, help="Max conflicting tasks per round.")
    parser.add_argument("--pddl-timeout", type=float, default=30.0, help="Per-solve PDDL timeout (s).")
    parser.add_argument("--output-json", type=Path, default=None, help="Write full results as JSON.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Keep PDDL problems and solver logs here.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    args = parser.parse_args(argv)

    if args.cases_per_seed <= 0:
        parser.error("--cases-per-seed must be a positive integer")
    if args.k_max < 0:
        parser.error("--k-max must be >= 0")
    if args.pddl_timeout <= 0:
        parser.error("--pddl-timeout must be > 0")

    if args.scenarios:
        if args.demo or args.alert or args.graph or args.snapshot:
            parser.error("--scenarios cannot be combined with --demo/--graph/--snapshot/--alert")
        if args.seeds is None:
            args.seeds = [42]
        if not args.seeds:
            parser.error("--seeds must list at least one integer seed")
        if args.schedule_dir is not None and not args.schedule_dir.is_dir():
            parser.error(f"--schedule-dir does not exist: {args.schedule_dir}")
        return run_batch(args)

    if args.schedule_dir is not None:
        parser.error("--schedule-dir requires --scenarios")

    cases: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    if args.demo:
        graph = load_json(FIXTURE_DIR / "graph_small.json")
        snapshot = load_json(FIXTURE_DIR / "snapshot_small.json")
        for name in DEMO_CASES:
            cases.append((graph, snapshot, load_json(FIXTURE_DIR / "alerts" / f"{name}.json")))
    else:
        if not args.alert:
            parser.error("either --demo, --scenarios or at least one --alert is required")
        for alert_path in args.alert:
            alert = load_json(alert_path)
            graph_path = resolve_relative(alert_path.parent, alert["graph"]) if alert.get("graph") else args.graph
            snapshot_path = (
                resolve_relative(alert_path.parent, alert["snapshot"]) if alert.get("snapshot") else args.snapshot
            )
            if graph_path is None or snapshot_path is None:
                parser.error(f"{alert_path}: provide --graph/--snapshot or embed paths in the alert file")
            cases.append((load_json(graph_path), load_json(snapshot_path), alert))

    results: List[ReplanResult] = []
    failures: List[str] = []
    for graph, snapshot, alert in cases:
        try:
            result = run_case(graph, snapshot, alert, args, args.artifacts_dir)
        except Exception as exc:  # a broken case must never look like a success
            failures.append(f"{alert.get('case_id', 'case')}: {exc}")
            print(f"\n=== {alert.get('case_id', 'case')} ===")
            print(f"status           : error")
            print(f"error            : {exc}")
            continue
        results.append(result)
        if not args.quiet:
            print_case(result)
        expected = alert.get("expect_status")
        if expected and result.status != expected:
            failures.append(
                f"{result.case_id}: expected status {expected!r} but got {result.status!r}"
            )

    print("\n=== summary ===")
    for result in results:
        print(
            f"{result.case_id:<16} status={result.status:<17} "
            f"backend={','.join(result.backend_chain):<24} "
            f"scope={len(result.scope_initial)}->{len(result.scope_final)} "
            f"rounds={result.expansion_rounds} total={result.timings_ms['total']:.0f}ms"
        )
    for failure in failures:
        print(f"MISMATCH: {failure}")

    if args.output_json is not None:
        payload = {
            "mode": "demo" if args.demo else "external",
            "cases": [result.__dict__ for result in results],
            "failures": failures,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
