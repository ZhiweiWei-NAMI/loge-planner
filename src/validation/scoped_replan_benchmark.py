"""Validator-gated scoped-replanning benchmark with real PDDL + real A*.

Entry point for the scoped-repair / splice-validation / bounded scope-expansion
procedure.  Every task order comes from a real PDDL solve (unified_planning +
fast-downward) and every leg is routed by the time-varying A* in
``src/offline_pipeline/edge_planner.py``; nothing is resampled or shifted.

Two modes:

* ``--demo`` runs the bundled fixtures in ``examples/replanning/`` and checks
  each case against its declared expectation.  These fixtures are small,
  hand-written demonstrations of the mechanics -- they are **not** a
  reproduction of the manuscript's S1-S5 scenarios and must not be cited as
  such.
* ``--graph/--snapshot/--alert`` runs user-supplied external inputs.  An alert
  file may carry its own ``graph``/``snapshot`` paths (resolved relative to the
  alert file) so a single ``--alert`` is enough.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validator-gated scoped replanning benchmark (real PDDL + real A*)."
    )
    parser.add_argument("--demo", action="store_true", help="Run the bundled examples/replanning fixtures.")
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

    cases: List[tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    if args.demo:
        graph = load_json(FIXTURE_DIR / "graph_small.json")
        snapshot = load_json(FIXTURE_DIR / "snapshot_small.json")
        for name in DEMO_CASES:
            cases.append((graph, snapshot, load_json(FIXTURE_DIR / "alerts" / f"{name}.json")))
    else:
        if not args.alert:
            parser.error("either --demo or at least one --alert is required")
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
            "demo_fixtures_are_demonstrations_only": True,
            "cases": [result.__dict__ for result in results],
            "failures": failures,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
