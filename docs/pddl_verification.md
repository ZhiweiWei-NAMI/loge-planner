# PDDL / A* verification record

Commands actually executed and the outputs they produced, for the scoped
replanning path (`src/validation/scoped_replanner.py`,
`src/validation/scoped_replan_benchmark.py`) and for the Stage-3 PDDL path
(`src/offline_pipeline/stage3_plan_with_pddl.py`,
`src/pddl_solver/pddl_solver.py`).

All commands assume:

```bash
export PROJECT_ROOT=/path/to/SCIS_auto_planner   # repository root
cd "$PROJECT_ROOT"
```

| item | value |
| --- | --- |
| Python | 3.10.14 |
| `unified-planning` (distribution version) | 1.2.0 |
| `unified_planning.__version__` (upstream string, not synced) | 1.1.0.1.post1 |
| planner backend reported by the solver | `Fast Downward` |
| domain | `config/pddl/tccn_domain.pddl` (unchanged) |
| repository version | `v1.1.0` (unchanged) |

Wall-clock numbers below are single-run observations on this machine; they vary
a few percent between runs and are demo-scale.

> The bundled fixtures in `examples/replanning/` are a **hand-written
> demonstration** of the mechanics; manuscript-scale batches use
> `--scenarios S1 … S5 --seeds … --cases-per-seed N` (see `README.md`).

## 1. Call chain

```
scoped_replan_benchmark.run_case
  └─ ScopedReplanner.baseline()            # committed plan, un-alerted graph
       ├─ PDDLSolver.solve(...)  x N drones  -> fast-downward action order
       └─ EdgePlanner.plan_route(...) x legs -> timed A* routes
  └─ ScopedReplanner.replan()              # after the alert
       ├─ apply_alert()                    # corridors closed both directions
       ├─ _seed_scope()                    # A0 + descendants + same-agent queues
       ├─ _solve_agents("scope_rK")        # real PDDL solve per in-scope drone
       ├─ _schedule_scope()                # real A* per leg, dependency-aware timing
       ├─ _validate()                      # c_state / c_time / c_dep / c_res / c_rsv
       ├─ _expand_scope()  (bounded by k_max, rho_max, b_rep_s, c_max)
       └─ _solve_agents("global")          # real global PDDL recovery, or explicit failure
```

Two properties make the solver input match the router's graph:

* the PDDL location universe is the **whole airspace graph**, and the distance
  matrix contains **only existing directed corridors**, with
  `distance_threshold = max(corridor length) + 1`;
* `PDDLSolver._build_connectivity_graph` emits exactly those directed pairs — no
  minimum spanning tree, no bidirectional symmetrisation. Both were removed
  because they invent corridors (or reverse a one-way corridor), which is how a
  PDDL plan ends up flying a route the A* router cannot execute.

A closed corridor is closed in **both** directions by default
(`ScopedReplanner.apply_alert`); `"directional": true` in an alert keeps only the
listed directions. Without this default another drone could traverse the closed
corridor backwards and the splice validator would still report the plan valid.

Node occupancy for capacity / reservation checks starts at **arrival**, not at
the service start, so a drone waiting for a release time or a dependency still
holds the node.

## 2. Demo run (5 cases, all real solves)

```bash
cd "$PROJECT_ROOT"
./scripts/06_run_scoped_replanning.sh --demo --artifacts-dir tmp/scoped_replan_artifacts
```

Observed summary (exit code `0`; `--demo` returns non-zero when a case misses
its declared `expect_status`):

```
=== summary ===
demo_success     status=splice_accepted   backend=Fast Downward            scope=4->4 rounds=0 total=720ms
demo_detour      status=splice_accepted   backend=Fast Downward            scope=4->4 rounds=0 total=674ms
demo_expand      status=splice_accepted   backend=Fast Downward            scope=4->6 rounds=1 total=1953ms
demo_global      status=global_recovered  backend=Fast Downward            scope=4->4 rounds=0 total=2098ms
demo_infeasible  status=infeasible        backend=Fast Downward            scope=2->2 rounds=0 total=1558ms
```

`--artifacts-dir` kept one `*.problem.pddl` and one `*.solver.log` per real solve
(baseline + every scope round + global), e.g.
`demo_detour_baseline_baseline_drone_a.problem.pddl`.

### 2.1 Small successful repair — `demo_success`

```
status           : splice_accepted
scope            : 4 -> 4 tasks (['T0', 'T1', 'T4', 'T5'] -> ['T0', 'T1', 'T4', 'T5']), expansion_rounds=0
validation       : valid=True state=True time=True dep=True res=True rsv=True
solves           :
  - scope_r0[drone_a] backend=Fast Downward actions=5   wall=309ms scope=['T0', 'T1'] ok
  - scope_r0[drone_c] backend=Fast Downward actions=8   wall=324ms scope=['T4', 'T5'] ok
```

Scope closure is `T0 -> T1` (same workflow) plus `T1 -> T5` (cross-workflow
dependency), which pulls `drone_c`'s queue in: 4 of 8 tasks.

### 2.2 A* obstacle detour — `demo_detour`

Corridor `wp_0_1 <-> wp_1_1` closed in both directions; the repaired leg is
re-routed by A* and the change is reported:

```
blocked edges    : [['wp_0_1', 'wp_1_1'], ['wp_1_1', 'wp_0_1']]
changed routes   :
  - T0: wp_0_0->wp_0_1->wp_1_1
      => wp_0_0->wp_1_0->wp_1_1
```

No other drone's committed route used that corridor in either direction, so
round 0 validates.

### 2.3 Rejection then scope expansion — `demo_expand`

```
scope            : 4 -> 6 tasks (['T0', 'T1', 'T4', 'T5'] -> ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']), expansion_rounds=1
blocked edges    : [['wp_1_3', 'wp_2_3'], ['wp_2_3', 'wp_1_3']]
solves           :
  - scope_r0[drone_a] ... ok
  - scope_r0[drone_c] ... ok
  - scope_r1[drone_a] ... ok
  - scope_r1[drone_b] ... ok
  - scope_r1[drone_c] ... ok
changed routes   :
  - T2: wp_2_3->wp_1_3->wp_1_2->wp_0_2
      => wp_2_3->wp_2_2->wp_1_2->wp_0_2
```

Round 0 is rejected by the validator because a **frozen** `drone_b` route used
the closed corridor; the conflict task `T2` grows the scope to `drone_b`'s queue
and round 1 re-solves and validates.

### 2.4 Global recovery — `demo_global`

Same closure, alert sets `rho_max=0.1` so the scope may not grow:

```
status           : global_recovered
scope            : 4 -> 4 tasks ..., expansion_rounds=0
solves           :
  - scope_r0[drone_a] ... ok
  - scope_r0[drone_c] ... ok
  - global[drone_a] ... ok
  - global[drone_b] ... ok
  - global[drone_c] ... ok
  - global[drone_d] ... ok
```

### 2.5 Infeasible case is reported as a failure — `demo_infeasible`

All corridors of `wp_1_3` (task `T7`) are closed:

```
status           : infeasible
validation       : valid=False
  reason         : global PDDL solve failed: drone_d: No solution found. Status: PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
solves           :
  - scope_r0[drone_d] backend=Fast Downward actions=0 wall=261ms FAILED (No solution found. Status: PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY)
  - global[drone_a] ... ok
  - global[drone_b] ... ok
  - global[drone_c] ... ok
  - global[drone_d] backend=Fast Downward actions=0 wall=254ms FAILED (No solution found. Status: PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY)
plan (unchanged) : NOT a repair -- this is the pre-alert committed plan
```

The solver failure is propagated verbatim (`success=False` plus the
unified_planning status string), the validator reports `valid=False`, and the
printed plan is labelled as the unchanged pre-alert plan. Nothing is presented
as a successful repair.

## 3. External input, copy-paste runnable

```bash
cd /tmp
"$PROJECT_ROOT/scripts/06_run_scoped_replanning.sh" \
  --graph "$PROJECT_ROOT/examples/replanning/graph_small.json" \
  --snapshot "$PROJECT_ROOT/examples/replanning/snapshot_small.json" \
  --alert "$PROJECT_ROOT/examples/replanning/alerts/demo_detour.json" \
  --output-json /tmp/scoped_out.json
```

Observed: exit `0`, `status: splice_accepted`, `backend chain: Fast Downward`,
the same `T0` detour as §2.2, and `/tmp/scoped_out.json` written. It runs from an
unrelated working directory because the wrapper exports `PYTHONPATH` from its own
location.

Schema for external snapshots / graphs / alerts: `examples/replanning/README.md`.

## 4. Stage 3 on a real generated topology

Real Stage-1 output,
`workflow_decouple/data/multi_seed_runs/small_comm_compute/run_seed_42`
(170 nodes, 850 directed corridors). A one-cluster assignment is placed in the
output root so the run has a cluster to plan:

```bash
cd "$PROJECT_ROOT"
mkdir -p tmp/stage3_topo_check/in/small_comm_compute/run_seed_42 \
         tmp/stage3_topo_check/out/small_comm_compute/run_seed_42
R=workflow_decouple/data/multi_seed_runs/small_comm_compute/run_seed_42
for f in scenario.yaml workflow_init_info.csv checkpoint_clusters_rule_kmeans.csv; do
  ln -sf "$PWD/$R/$f" tmp/stage3_topo_check/in/small_comm_compute/run_seed_42/$f
done
printf 'cluster_group,assigned_agents\n1,agent_communication_only_1;agent_communication_only_2\n' \
  > tmp/stage3_topo_check/out/small_comm_compute/run_seed_42/cluster_assignment_rule_kmeans.csv
python3 - <<'PY'
import json, logging
from pathlib import Path
logging.disable(logging.INFO)
import src.offline_pipeline.stage3_plan_with_pddl as st3
from src.simulation.environment.topology_manager import TopologyManager

run = Path("workflow_decouple/data/multi_seed_runs/small_comm_compute/run_seed_42")
out = st3.run_stage3_planning(
    seed=42, cluster_method="rule_kmeans",
    input_root=Path("tmp/stage3_topo_check/in/small_comm_compute"),
    output_root=Path("tmp/stage3_topo_check/out/small_comm_compute"),
    topology_file=run / "graph_topology.json", export_edge_routes=True)

plan = json.loads(out.read_text())
topo = TopologyManager(bounds={"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1})
topo.load_from_file(str(run / "graph_topology.json"))
legs, bad = 0, []
for agent, payload in plan["agents"].items():
    for step in payload["result"]["solution"]:
        if step["action_name"] == "movetoloc" and len(step["parameters"]) >= 3:
            u, v = step["parameters"][1], step["parameters"][2]
            u = u[5:] if u.startswith("topo_") else u
            v = v[5:] if v.startswith("topo_") else v
            legs += 1
            if (u, v) not in topo.edges:
                bad.append((agent, u, v))
print("plan:", out.name, "| agents:", len(plan["agents"]),
      "| movetoloc legs:", legs, "| legs outside real corridors:", len(bad))
PY
```

Observed:

```
✓ Built connectivity graph with 850 directed connections
plan: pddl_plan_rule_kmeans.json | agents: 30 | movetoloc legs: 10 | legs outside real corridors: 0
```

The solver's connectivity is exactly the 850 directed corridors of the real
graph, and every planned leg is an existing corridor.

### 4.1 Disconnected goal on the same real graph

`waypoint_104` is a goal of the assigned cluster and is **not** a drone start
node; all its incident corridors are removed, so it is unreachable. The same
`run_stage3_planning` call then refuses to write a plan:

```
Stage 3 PDDL failure: cluster 1: PDDL planning failed (No solution found. Status: PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY); drones=['agent_communication_only_1', 'agent_communication_only_2'], goals=8
RuntimeError: Stage 3 PDDL planning failed for 1 cluster(s); refusing to emit a plan file. First failure: cluster 1: PDDL planning failed (No solution found. Status: PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY); drones=[...], goals=8
plan file written: False
```

(An earlier attempt isolated `waypoint_26`, which is the start node of
`agent_communication_only_1`; that problem is legitimately solvable, because a
drone already parked at a node does not need a corridor to serve it. The cut
above avoids that case.)

### 4.2 Missing topology is an explicit failure

No random grid topology is generated any more; Stage 1
(`src/offline_pipeline/stage1_precluster_from_config.py`) owns generation and
writes `graph_topology.json`.

```
FileNotFoundError: Stage 3 requires a topology graph, but none was found (looked for scenario.environment.graph_topology_file and .../run_seed_42/graph_topology.json). Generate it explicitly in Stage 1 (src.offline_pipeline.stage1_precluster_from_config) or pass --topology-file.
```

### 4.3 Cluster-level goal satisfaction

`run_stage3_planning` no longer treats "a candidate drone got no steps" as a
failure — handing the whole cluster to one drone is legitimate. Instead the
returned plan must contain an action for **every** goal of the cluster; otherwise
the cluster is recorded as failed and the run raises before writing the plan
file.

## 5. `pddl_solver.py` cleanup

The earlier draft added a `method='heuristic'` branch (`_solve_heuristic`, ~80
lines of greedy nearest-goal assignment) plus an `importlib.util.find_spec` check
*after* `unified_planning` had already been imported unconditionally. Both are
gone:

* `unified_planning` is imported unconditionally — a missing dependency raises
  `ImportError` at import time instead of silently degrading;
* there is no heuristic path, so `PDDLSolver` returns a plan that fast-downward
  produced, or `{"success": False, ...}`;
* `_build_connectivity_graph` now preserves the caller's directed edges exactly.

```bash
python3 -c "import src.pddl_solver.pddl_solver as m; print('import ok')"
# import ok
```

## 6. Stage-3 failure propagation (unit level)

`solve_pddl_for_cluster` previously returned `{drone: []}` when the PDDL solve
failed, which made a failed cluster indistinguishable from a cluster with
nothing to do. It now raises. Verified with real solver failures (no mocks):

```bash
cd "$PROJECT_ROOT"
python3 - <<'PY'
from pathlib import Path
from src.offline_pipeline.stage3_plan_with_pddl import solve_pddl_for_cluster
from src.simulation.environment.topology_manager import TopologyManager

topo = TopologyManager(bounds={"x_min": 0, "x_max": 1000, "y_min": 0, "y_max": 1000, "z_min": 0, "z_max": 10})
topo.add_node("wp_0", 0.0, 0.0, 1.0, "waypoint", capacity=2)
topo.add_node("wp_1", 50.0, 0.0, 1.0, "waypoint", capacity=2)
topo.add_edge("wp_0", "wp_1"); topo.add_edge("wp_1", "wp_0")
ckps = [{"workflow_id": "wf1", "operation_type": "inspect", "x": 50.0, "y": 0.0, "z": 1.0,
         "cluster_group": 0, "start_time": 0.0}]
solve_pddl_for_cluster(0, ["drone_a"], {"drone_a": (0.0, 0.0, 1.0)}, ckps,
                       Path("config/pddl/tccn_domain.pddl"), topo, pddl_timeout=1e-9)
PY
```

Observed:

```
RuntimeError: cluster 0: PDDL planning failed (No solution found. Status: PlanGenerationResultStatus.TIMEOUT); drones=['drone_a'], goals=1
```

and the second path (no drone can be snapped to the topology):

```
RuntimeError: cluster 1: no candidate drone could be snapped to a topology node (drones=['drone_a'])
```

Upstream callers were checked for swallowing: `plot/run_experiments.py` returns
`False` on non-zero exit, `scripts/run_all_experiments.py` logs and returns
`False`, `scripts/03_run_pddl_planning.sh` prints `⚠ Stage3 failed` and
continues. Because the module now exits non-zero instead of writing an empty plan
file, those callers can no longer mistake a failed cluster for a successful one.

## 7. Remaining limitations

1. `src/baselines/graph_mapf_priority.py` still auto-generates a grid topology
   when `graph_topology.json` is missing (that baseline owns its own topology
   setup and was outside the scope of this fix).
2. The demo fixtures use uniform 15 m/s corridors on a 12-node grid, so the
   timings in §2 are demo-scale.
3. Wall-clock values are single-run samples; re-running changes them by a few
   percent.
