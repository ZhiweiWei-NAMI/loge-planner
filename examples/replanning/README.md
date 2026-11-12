# Scoped-replanning benchmark

Two ways to run the scoped-repair / splice-validation / bounded scope-expansion
procedure. Both go through the same real stack:

* task ordering comes from `unified_planning` + Fast Downward
  (`src/pddl_solver/pddl_solver.py`, one solve per drone);
* every `movetoloc` leg is routed by the time-varying A* in
  `src/offline_pipeline/edge_planner.py` over the graph in use;
* the splice validator re-simulates the merged plan and can reject it;
* a rejected repair expands the scope and, at the bound, runs a real global
  PDDL recovery. When that also fails the case is reported as `infeasible`.

| mode | command | what it is |
| --- | --- | --- |
| smoke fixtures | `--demo` | The 5 hand-written cases in this directory, each checked against the `expect_status` declared in its alert. Fast sanity check of the mechanics. |
| batch evaluation | `--scenarios ... --seeds ... --cases-per-seed N` | Deterministically generated benchmark instances. A fixed `(scenario, seed)` pair reproduces the identical graph, schedule and alert set; scale grows from S1 (small) to S5 (large). |
| real schedules | `--scenarios ... --schedule-dir DIR` | Same batch driver, but graph + snapshot + alerts are read from real schedule files instead of the generator. |
| external inputs | `--graph/--snapshot/--alert` | One-off run on your own graph, snapshot and alert files. |

## Run

```bash
# smoke fixtures
./scripts/06_run_scoped_replanning.sh --demo

# batch: 5 scenarios x 5 seeds x 20 generated alerts per pair
./scripts/06_run_scoped_replanning.sh \
    --scenarios S1 S2 S3 S4 S5 --seeds 42 43 44 45 46 --cases-per-seed 20

# batch: quick check, keep every PDDL problem and solver log
./scripts/06_run_scoped_replanning.sh \
    --scenarios S1 S2 --seeds 42 --cases-per-seed 2 \
    --artifacts-dir tmp/batch --output-json tmp/batch/summary.json
```

`--demo` exits non-zero if any case misses its declared status; batch mode exits
non-zero if any case raises (a broken case never counts as a success). Add
`--quiet` to suppress per-case output and keep only the summary tables.

## Generated instances (`--scenarios`)

Scale per label — the numbers set instance size only, every case is solved for
real:

| label | grid | drones | tasks | workflows |
| --- | --- | --- | --- | --- |
| S1 | 3x4 (12 nodes) | 4 | 8 | 2 |
| S2 | 4x5 (20 nodes) | 6 | 12 | 3 |
| S3 | 5x6 (30 nodes) | 8 | 18 | 4 |
| S4 | 6x7 (42 nodes) | 10 | 24 | 5 |
| S5 | 7x8 (56 nodes) | 12 | 30 | 6 |

Grid spacing 1000 m, 15 m/s, node capacity 3, corridor capacity 2, plus
checkerboard diagonal shortcuts so detours exist. The `seed` drives drone homes,
task nodes, operation types, service times, releases, workflow/dependency
structure and the sampled alerts; the graph topology depends only on the
scenario label. Deadlines follow each drone's queue position, so the committed
plan is satisfiable while real detours can still violate them.

Alerts are drawn from the corridors the committed schedule actually flies
(base-graph shortest paths), in three classes:

| category | blocked corridors |
| --- | --- |
| `corridor_blocked` | 1 committed corridor |
| `corridor_lost` | 2 committed corridors, or (20% of the time) every corridor of the anchor node, which is a genuinely infeasible instance |
| `battery_insufficient` | none — pure scope-repair case |

20% of the cases run with `rho_max=0.1`, which forces the bounded-expansion path
to fall through to the real global recovery.

## Reported metrics

`--output-json` writes an aggregate payload (`aggregate`, `per_scenario`,
`per_scenario_seed`, `case_table`, full `cases`, `failures`) plus these
quantities, all measured on real runs:

| field | meaning |
| --- | --- |
| `accept_rate` | share of cases repaired inside the scope (`splice_accepted`) |
| `expansion_rate` | share of cases whose scope actually expanded (`expansion_rounds > 0`) |
| `global_recovery_rate` | share of cases solved by the global PDDL recovery |
| `infeasible_rate` | share of cases where both scope and global solve failed |
| `repair_latency_ms` | real wall clock of the repair (mean / median / p95 / min / max) |
| `case_wall_ms` | real wall clock including the pre-alert baseline solve |
| `solver_solves` / `solver_failed_solves` / `solver_backends` | real solver calls, failures and backends (Fast Downward) |
| `pddl_ms_total` / `astar_calls_total` | time inside the PDDL solver and number of A* route queries |

## Real schedule files (`--schedule-dir`)

One JSON file per `(scenario, seed)` named `<dir>/<scenario>_seed<seed>.json`:

```json
{
  "graph": "graphs/g1.json",
  "snapshot": "snapshots/s1.json",
  "alerts": ["alerts/a1.json", "alerts/a2.json"]
}
```

`graph` / `snapshot` / each alert may be an inline object or a path relative to
the schedule file. At least `--cases-per-seed` alerts must be present.

## Fixtures in this directory

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
single `--alert` argument is enough. `expect_status` is only enforced in
`--demo` / `--alert` mode.

## Artifacts

With `--artifacts-dir DIR` every real solver call leaves
`<case>_<stage>_<agent>.problem.pddl` and `<case>_<stage>_<agent>.solver.log`
behind, and the batch summary counts them (`solver_log_files`). See
`docs/pddl_verification.md` for recorded outputs.
