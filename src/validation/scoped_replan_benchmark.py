"""Validator-gated semantic-alert-driven scoped replanning benchmark (Algorithm 2).

Entry point for reproducing scoped repair, splice validation (V_splice), bounded
scope expansion, and global-recovery fallback statistics across scenarios S1-S5.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


SCENARIO_ALIASES = {
    "S1": "small_comm_compute",
    "S2": "small_logistics_inspection",
    "S3": "medium_logistics_heavy",
    "S4": "medium_inspection_heavy",
    "S5": "large_mixed",
}


@dataclass
class SemanticAlert:
    category: str
    severity: str
    workflow_id: int
    resource_node: int
    window: Tuple[float, float]


@dataclass
class SpliceValidationResult:
    valid: bool
    c_state: bool
    c_time: bool
    c_res: bool
    c_dep: bool
    c_rsv: bool
    conflict_tasks: List[int]


@dataclass
class RepairCaseOutcome:
    scenario: str
    seed: int
    case_id: int
    initial_scope_size: int
    final_scope_size: int
    total_active_tasks: int
    affected_workflows: int
    expansion_rounds: int
    splice_accepted: bool
    global_recovery: bool
    plan_preservation_ratio: float
    repair_latency_ms: float


def build_synthetic_active_schedule(
    scenario_code: str, rng: np.random.Generator
) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    scale_map = {"S1": 60, "S2": 120, "S3": 240, "S4": 280, "S5": 420}
    n_tasks = scale_map.get(scenario_code, 120)
    n_workflows = max(6, n_tasks // 5)
    n_agents = {"S1": 30, "S2": 30, "S3": 80, "S4": 80, "S5": 150}.get(scenario_code, 30)
    n_nodes = 40

    tasks: List[Dict] = []
    edges: List[Tuple[int, int]] = []
    for tid in range(n_tasks):
        wf = tid % n_workflows
        step = tid // n_workflows
        t_s = float(step * 35.0 + rng.uniform(0.0, 12.0))
        dur = float(rng.uniform(14.0, 28.0))
        node = int(rng.integers(0, n_nodes))
        cap = int(rng.integers(5, 9))
        tasks.append(
            {
                "task_id": tid,
                "workflow_id": wf,
                "agent_id": int(wf % n_agents),
                "node": node,
                "node_capacity": cap,
                "t_start": t_s,
                "t_end": t_s + dur,
                "deadline": t_s + dur + float(rng.uniform(45.0, 120.0)),
                "energy_cost": float(rng.uniform(3.0, 9.0)),
            }
        )
        if tid >= n_workflows:
            edges.append((tid - n_workflows, tid))
    return tasks, edges


def descendants(seed_tasks: Set[int], dep_edges: List[Tuple[int, int]]) -> Set[int]:
    adj: Dict[int, List[int]] = {}
    for u, v in dep_edges:
        adj.setdefault(u, []).append(v)
    visited: Set[int] = set()
    stack = list(seed_tasks)
    while stack:
        curr = stack.pop()
        for nxt in adj.get(curr, []):
            if nxt not in visited and nxt not in seed_tasks:
                visited.add(nxt)
                stack.append(nxt)
    return visited


def evaluate_splice_validator(
    repaired_tasks: Dict[int, Dict],
    frozen_tasks: Dict[int, Dict],
    dep_edges: List[Tuple[int, int]],
) -> SpliceValidationResult:
    combined = {**frozen_tasks, **repaired_tasks}
    conflicts: Set[int] = set()

    c_state = True
    c_time = True
    c_res = True
    c_dep = True
    c_rsv = True

    for tid, t in repaired_tasks.items():
        if t["t_end"] > t["deadline"]:
            c_time = False
            conflicts.add(tid)
        if t["energy_cost"] > 85.0:
            c_res = False
            conflicts.add(tid)

    for u, v in dep_edges:
        if u in combined and v in combined:
            if combined[u]["t_end"] > combined[v]["t_start"] + 1e-6:
                c_dep = False
                if v in frozen_tasks:
                    conflicts.add(v)
                if u in frozen_tasks:
                    conflicts.add(u)

    for r_tid, r_task in repaired_tasks.items():
        concurrent = [r_tid]
        for f_tid, f_task in frozen_tasks.items():
            if f_task["node"] == r_task["node"]:
                overlap = min(r_task["t_end"], f_task["t_end"]) - max(r_task["t_start"], f_task["t_start"])
                if overlap > 0.0:
                    concurrent.append(f_tid)
        if len(concurrent) > r_task["node_capacity"]:
            c_rsv = False
            for f_tid in concurrent[1:]:
                conflicts.add(f_tid)

    valid = c_state and c_time and c_res and c_dep and c_rsv
    return SpliceValidationResult(
        valid=valid,
        c_state=c_state,
        c_time=c_time,
        c_res=c_res,
        c_dep=c_dep,
        c_rsv=c_rsv,
        conflict_tasks=sorted(conflicts),
    )


def run_scoped_repair_case(
    scenario_code: str,
    seed: int,
    case_id: int,
    rng: np.random.Generator,
    k_max: int = 3,
    b_rep_s: float = 2.0,
    rho_max: float = 0.35,
    c_max: int = 5,
) -> RepairCaseOutcome:
    t_start = time.perf_counter()
    tasks, dep_edges = build_synthetic_active_schedule(scenario_code, rng)
    task_map = {t["task_id"]: dict(t) for t in tasks}
    n_total = len(tasks)

    anchor = tasks[int(rng.integers(0, max(1, n_total // 2)))]
    alert = SemanticAlert(
        category=str(rng.choice(["strong_wind_high", "charging_congested_high", "fog_low_vis", "battery_insufficient"])),
        severity="high",
        workflow_id=anchor["workflow_id"],
        resource_node=anchor["node"],
        window=(anchor["t_start"], anchor["t_end"] + 15.0),
    )

    a0 = {anchor["task_id"]}
    t_down = descendants(a0, dep_edges)
    t_agent = {
        tid
        for tid, t in task_map.items()
        if t["agent_id"] == anchor["agent_id"] and t["t_start"] >= anchor["t_start"]
    }
    overlapping = [
        tid
        for tid, t in task_map.items()
        if t["node"] == alert.resource_node
        and min(t["t_end"], alert.window[1]) > max(t["t_start"], alert.window[0])
    ]
    t_res = set(overlapping) if len(overlapping) >= anchor["node_capacity"] else set()
    s_rep: Set[int] = set(a0 | t_down | t_agent | t_res)
    initial_scope_size = len(s_rep)

    splice_accepted = False
    global_recovery = False
    rounds_used = 0

    for k in range(k_max + 1):
        rounds_used = k
        s_fix = set(task_map.keys()) - s_rep
        frozen = {tid: task_map[tid] for tid in s_fix}

        repaired: Dict[int, Dict] = {}
        delay_shift = float(rng.uniform(6.0, 16.0)) if k == 0 else float(rng.uniform(3.0, 8.0))
        for tid in sorted(s_rep):
            updated = dict(task_map[tid])
            updated["t_start"] += delay_shift
            updated["t_end"] += delay_shift
            if k > 0:
                updated["node"] = (updated["node"] + 1) % 40
            repaired[tid] = updated

        for u, v in dep_edges:
            if u in repaired and v in repaired and repaired[v]["t_start"] < repaired[u]["t_end"]:
                dur = repaired[v]["t_end"] - repaired[v]["t_start"]
                repaired[v]["t_start"] = repaired[u]["t_end"] + 1.0
                repaired[v]["t_end"] = repaired[v]["t_start"] + dur

        val = evaluate_splice_validator(repaired, frozen, dep_edges)
        if val.valid:
            splice_accepted = True
            break

        s_rep |= set(val.conflict_tasks)
        elapsed_s = time.perf_counter() - t_start
        if (len(s_rep) / max(1, n_total) > rho_max) or (elapsed_s > b_rep_s) or (len(val.conflict_tasks) > c_max):
            global_recovery = True
            break

    if not splice_accepted and not global_recovery:
        global_recovery = True

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    affected_wfs = len({task_map[tid]["workflow_id"] for tid in s_rep})
    preserved_ratio = 1.0 - (len(s_rep) / max(1, n_total))

    return RepairCaseOutcome(
        scenario=scenario_code,
        seed=seed,
        case_id=case_id,
        initial_scope_size=initial_scope_size,
        final_scope_size=len(s_rep),
        total_active_tasks=n_total,
        affected_workflows=affected_wfs,
        expansion_rounds=rounds_used,
        splice_accepted=splice_accepted,
        global_recovery=global_recovery,
        plan_preservation_ratio=preserved_ratio,
        repair_latency_ms=elapsed_ms,
    )


def summarize_outcomes(outcomes: List[RepairCaseOutcome]) -> Dict[str, float | int]:
    n = max(1, len(outcomes))
    first_round = sum(1 for o in outcomes if o.splice_accepted and o.expansion_rounds == 0)
    expanded_ok = sum(1 for o in outcomes if o.splice_accepted and o.expansion_rounds > 0)
    accepted = sum(1 for o in outcomes if o.splice_accepted)
    global_rec = sum(1 for o in outcomes if o.global_recovery)
    return {
        "total_cases": len(outcomes),
        "first_round_accepted": first_round,
        "first_round_accept_rate": round(100.0 * first_round / n, 2),
        "expanded_accepted": expanded_ok,
        "final_splice_accept_rate": round(100.0 * accepted / n, 2),
        "global_recovery_count": global_rec,
        "global_recovery_rate": round(100.0 * global_rec / n, 2),
        "avg_expansion_rounds": round(float(np.mean([o.expansion_rounds for o in outcomes])), 4),
        "avg_affected_workflows": round(float(np.mean([o.affected_workflows for o in outcomes])), 3),
        "avg_plan_preservation_percent": round(100.0 * float(np.mean([o.plan_preservation_ratio for o in outcomes])), 2),
        "avg_repair_latency_ms": round(float(np.mean([o.repair_latency_ms for o in outcomes])), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validator-gated scoped replanning benchmark (Algorithm 2).")
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--cases-per-seed", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--b-rep", type=float, default=2.0)
    parser.add_argument("--rho-max", type=float, default=0.35)
    parser.add_argument("--c-max", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Run a quick 25-case smoke test.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cases_per_seed = 1 if args.smoke else args.cases_per_seed
    outcomes: List[RepairCaseOutcome] = []
    for s_idx, sc in enumerate(args.scenarios):
        for seed in args.seeds:
            rng = np.random.default_rng(seed * 1009 + s_idx * 97)
            for c_id in range(cases_per_seed):
                outcomes.append(
                    run_scoped_repair_case(
                        scenario_code=sc,
                        seed=seed,
                        case_id=c_id,
                        rng=rng,
                        k_max=args.k_max,
                        b_rep_s=args.b_rep,
                        rho_max=args.rho_max,
                        c_max=args.c_max,
                    )
                )

    summary = summarize_outcomes(outcomes)
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"summary": summary, "cases": [asdict(o) for o in outcomes]}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
