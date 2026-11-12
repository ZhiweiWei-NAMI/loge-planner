# Scoped-replanning demo fixtures (demonstration only)

These files are a **small, hand-written demonstration** of the scoped-repair
mechanics wired to a real PDDL solver and a real A* router. They are **not** a
reproduction of the manuscript's S1-S5 scenarios, they carry no statistical
meaning, and nothing measured on them should be reported as a paper result.

Everything that consumes them is real:

* task ordering comes from `unified_planning` + Fast Downward
  (`src/pddl_solver/pddl_solver.py`, one solve per drone);
* every `movetoloc` leg is routed by the time-varying A* in
  `src/offline_pipeline/edge_planner.py` over the graph in `graph_small.json`;
* the splice validator re-simulates the merged plan and can reject it.

## Files

| file | content |
| --- | --- |
| `graph_small.json` | 3x4 waypoint grid (12 nodes, 34 directed corridors, 1000 m spacing, 15 m/s, node/edge capacity 2). |
| `snapshot_small.json` | Committed schedule: 4 drones, 8 tasks, 5 workflow dependencies (`predecessor -> successor`). |
| `alerts/demo_success.json` | No corridor loss; small scope should splice on round 0. |
| `alerts/demo_detour.json` | Corridor `wp_0_1 <-> wp_1_1` closed; A* must detour `wp_0_0 -> wp_1_0 -> wp_1_1`. |
| `alerts/demo_expand.json` | Closure of `wp_2_3 <-> wp_1_3` breaks a frozen `drone_b` route; round 0 is rejected, scope expands, round 1 validates. |
| `alerts/demo_global.json` | Same closure with `rho_max=0.1`, so scope growth is forbidden and a real global PDDL recovery runs. |
| `alerts/demo_infeasible.json` | All corridors of `wp_1_3` closed; both the scope solve and the global solve must fail. |

## Schemas

`snapshot_small.json`

```json
{
  "t0": 0.0,
  "agents": {"drone_a": {"node": "wp_0_0"}},
  "tasks": [{"task_id": "T0", "workflow_id": "wf1", "agent_id": "drone_a",
             "node": "wp_1_1", "op_type": "inspect",
             "release": 0.0, "deadline": 600.0, "service_s": 60.0}],
  "dependencies": [["T0", "T1"]]
}
```

`op_type` is one of `compute` / `communicate` / `inspect` (mapped onto the
`assistcompute` / `assistcommunicate` / `inspect` PDDL actions of
`config/pddl/tccn_domain.pddl`). `dependencies` are `[predecessor, successor]`
pairs. All node / drone identifiers must already be valid lowercase PDDL names.

`alerts/*.json`

```json
{
  "case_id": "demo_detour",
  "category": "corridor_blocked",
  "severity": "high",
  "window": [0.0, 300.0],
  "workflow_id": "wf1",
  "node": "wp_1_1",
  "blocked_edges": [["wp_0_1", "wp_1_1"]],
  "directional": false,
  "expect_status": "splice_accepted",
  "k_max": 3, "rho_max": 0.6, "b_rep_s": 30.0, "c_max": 8, "pddl_timeout": 30.0
}
```

`blocked_edges` closes each listed corridor in **both** directions by default;
set `"directional": true` to close exactly the listed directions. The effective
closed set is what the A* router, the PDDL problem and the splice validator all
use, and it is echoed in the run output.

Only `case_id` is required in an alert; `k_max` / `rho_max` / `b_rep_s` /
`c_max` / `pddl_timeout` are optional per-case overrides. An alert may also
carry `graph` and `snapshot` paths (resolved relative to the alert file) so a
single `--alert` argument is enough.

## Run

```bash
./scripts/06_run_scoped_replanning.sh --demo
```

`--demo` exits non-zero if any case misses the `expect_status` declared in its
alert file. See `docs/pddl_verification.md` for the recorded outputs.
