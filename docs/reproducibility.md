# Reproducibility Notes — LoGe Planner `v1.1.0`

This document records what this package can and cannot reproduce, what was actually
executed while preparing the release, and where each published number comes from.
It is deliberately narrower than a "fully reproducible" claim: several components
(LLM agents, trained weights, ns-3 reruns) are outside what is shipped here.

## 1. License status

No `LICENSE`, `COPYING`, or `NOTICE` file exists in this repository, and no source
file carries a copyright header. **Authorization is pending: the license is undecided
(TBD by the author).** This is a release-metadata gap, not a runtime defect — the code
runs. Nothing in this package should be read as granting a license, and no license
(e.g. MIT) is implied or assigned here. A license must be selected by the author
before public redistribution.

Version string: `v1.1.0` (unchanged; no tag or version bump is made by this pass).

## 2. What was executed vs. what is only shipped

| Item | Status | Measured evidence |
| --- | --- | --- |
| PDDL domain parse + solve (`config/pddl/tccn_domain.pddl`, Fast Downward via `unified-planning 1.2.0`) | **Executed** | 6 problem files solved (`SOLVED_SATISFICING`), plan lengths 1/1/1/16/22/37 actions |
| Stage 2 assignment (`stage2_plan_from_clusters`) | **Executed** | `small_logistics_inspection`, seed 42, `htt_recursive` → `cluster_assignment_htt_recursive.csv` |
| Stage 3 grounding (`stage3_plan_with_pddl`) | **Executed** | same run: 30 agents, 170-node/850-edge topology, `pddl_plan_htt_recursive.json`, 2.7–3.1 s wall clock — both with `airfogsim` installed and with the import blocked |
| Stage 3 monolithic PDDL (`stage3_plan_monolithic_pddl`) | **Executed** | 30 agents / 70 goals, 0.60 s, with `airfogsim` import blocked |
| Scoped-replanning demo (`scripts/06_run_scoped_replanning.sh --demo`) | **Executed** | 5/5 fixtures as declared: 3 × `splice_accepted` (one requires a round-1 scope expansion), 1 × `global_recovered`, 1 × `infeasible`; all real Fast Downward solves, exit code non-zero on any mismatch |
| Stage 1 HtT clustering (`htt_recursive`) | **Executed** | 11 clusters, 21 tree nodes, max depth 5–6 |
| CLI entry points (`--help`) | **Executed** | stage1/2/3, monolithic, plan_validation_gate, scoped_replan_benchmark, terminal_monitor_eval, `run_offline_simulation.py` — all exit 0 |
| Dependency set in `requirements.txt` / `requirements-llm.txt` | **Import-level check only** | every core package imported successfully; the full pipeline was *not* run end-to-end |
| LLM decomposition (`scripts/08_run_llm_decomposition.sh`) | **Not executed** | no API call was made; only the `--dry-run` / `--from-json` code paths are exercised by fixtures |
| LoRA SFT + terminal-monitor LLM (`scripts/08_train_terminal_lora.sh`, `07_...`) | **Not executed** | no model was downloaded or trained; `--validate-only` exists but was not used for published numbers |
| ns-3 benchmark | **Shipped as recorded run outputs, not rerun in this pass** | regenerable from `ns3_comm_sim/inputs/` + `ns3_comm_sim/results/raw/` given an ns-3.40 build (§6) |

Any number not listed in the "Executed" column is *not* asserted by this package.

**Stage-3 verification (single fixture, not a paper reproduction).** On a real
generated topology — 170 nodes / 850 directed corridors — the recorded run built
30 agents and 10 `movetoloc` legs with **0 legs outside real corridors**; an isolated
cluster goal (`waypoint_104`) is reported `UNSOLVABLE_INCOMPLETELY` and **no plan file is written**;
a missing topology raises `FileNotFoundError` with an actionable message; node
occupancy for capacity/reservation checks starts at **arrival**, so a drone waiting
for a release time still occupies the node; `blocked_edges` are bidirectional by
default and `"directional": true` keeps only the one-way closure. Details and the
exact commands: `docs/pddl_verification.md`. Timings are single-run samples on this
one fixture and are not a stability claim.

## 3. "HtT" clustering is a heuristic rule procedure, not an LLM

The published rule artifacts are **heuristic rules mined from RL bandit weight
libraries**, not PDDL rules and not language-model output:

- `analysis/rules/rule_kmeans_config_*.yaml` — per-scenario weight/temperature
  configuration consumed by `RuleBasedWeightSelector`.
- `analysis/rules/cluster_rules_*.json`, `cluster_assignment_rules.json` —
  threshold rules with statistics, e.g. `spatial_spread >= 253.2204`,
  `t = 6.14`, `p = 1.1e-09`, mined by `analysis/cluster_rule_mining.py` /
  `scripts/summarize_rules_from_weights.py` from weight libraries.
- At runtime the rules only steer a deterministic weighted KMeans over the 26-D
  checkpoint features (`_apply_htt_rules_postprocessing`,
  `recursive_htt_clusterer`). No language model participates.

The `llm` clustering method (`--cluster-method llm`) is a **separate** path that
calls an external model; it is never used by `htt_recursive`, and none of its outputs
are published.

Consequence for reproduction: tracking these JSON/YAML files is **necessary but not
sufficient**. They were derived from RL weight libraries
(`workflow_decouple/data/multi_seed_runs/*/bandit_results/weight_library.json`) and
from generated checkpoint data, none of which is published. Re-mining the rules via
`scripts/run_htt_full_pipeline.sh` will not bit-reproduce the committed thresholds,
because the upstream weights are not shipped.

## 4. PDDL: canonical domain and standalone execution

- Canonical domain: `config/pddl/tccn_domain.pddl` (tracked, `:requirements :strips :typing`).
- A byte-identical legacy copy exists at `data/pddl/domain/tccn_domain.pddl`; `data/`
  is excluded from the release, so a clean checkout only carries the canonical file.
- `src/pddl_solver/pddl_solver.py` resolves the domain relative to the repository
  root and fails loudly when it is missing (no silent fallback).

Measured solves with the tracked domain (Fast Downward, `unified-planning 1.2.0`):

| Problem | Size | Status | Plan length |
| --- | --- | --- | --- |
| `1759808441_pddl_problem.pddl` | 0.6 KB | SOLVED_SATISFICING | 1 |
| `1759805996_pddl_problem.pddl` | 0.4 KB | SOLVED_SATISFICING | 1 |
| `1759805775_pddl_problem.pddl` | 0.4 KB | SOLVED_SATISFICING | 1 |
| `1759803885_pddl_problem.pddl` | 4.2 KB | SOLVED_SATISFICING | 22 |
| `1759802007_pddl_problem.pddl` | 6.2 KB | SOLVED_SATISFICING | 16 |
| `1759801209_pddl_problem.pddl` | 7.7 KB | SOLVED_SATISFICING | 37 |

Can the PDDL stack run without the external `airfogsim` simulator? Yes for the whole
offline planning path, verified with an import blocker that raises on any `airfogsim`
import:

| Module | Imports/runs without `airfogsim` | Note |
| --- | --- | --- |
| `src.pddl_solver.pddl_solver` | yes | pure `unified_planning` + `networkx` |
| `src.offline_pipeline.stage2_plan_from_clusters` | yes | ran end-to-end |
| `src.offline_pipeline.stage3_plan_with_pddl` | yes | ran end-to-end, plan file written (2.7–3.1 s) |
| `src.offline_pipeline.stage3_plan_monolithic_pddl` | yes | ran end-to-end, 0.60 s |
| `src.baselines.monolithic_global_pddl` | yes | imports the stage-3 module |
| `src.validation.scoped_replanner` | yes | demo benchmark passed |
| `src.simulation.environment` weather/UFO/event components | **no (by design)** | `src/simulation/environment/__init__.py` uses PEP 562 lazy imports; requesting `WeatherController` without `airfogsim` raises an explicit `ImportError` naming the missing package and the PDDL-only path |

The environment package previously re-exported `weather_controller` eagerly, which
dragged `airfogsim` into every stage-3 import. It now imports each component on first
attribute access, so `TopologyManager`/`GlobalMonitor` (which never needed
`airfogsim`) are usable in a planning-only environment. No mock or fallback object is
introduced: the dependency error is raised, not hidden.

## 5. Entry points: required inputs

All data below is **generated locally by this package**; it is not the original
manuscript dataset. No historical result files are published (see §7).

| Entry point | Needs | Produces |
| --- | --- | --- |
| `scripts/00_train_triplet.sh` | generated checkpoint data; GPU optional | `triplet_learning_results/siamese_encoder.pth` (weights, not shipped) |
| `scripts/01_generate_data.sh` | `airfogsim`, `workflow_decouple/config/*.yaml` | `workflow_decouple/data/multi_seed_runs/<scenario>/run_seed_*/` |
| `scripts/02_run_clustering.sh` | stage-1 outputs; `kmeans`/`hungarian`/`htt_recursive` need no weights, `triplet` needs the trained encoder | `checkpoint_clusters_<method>.csv` |
| `scripts/03_run_pddl_planning.sh` | cluster CSVs + `config/pddl/tccn_domain.pddl` | `data/offline_plans/<scenario>_<method>/run_seed_*/pddl_plan_*.json` |
| `scripts/04_run_simulation.sh` | plan JSONs + init data + `airfogsim` | `data/results/Offline-<method>_*.json` |
| `scripts/05_plot_results.sh` | `data/results/` | `plot/aggregated_results.csv`, figures |
| `scripts/06_run_scoped_replanning.sh` | bundled `examples/replanning/` fixtures or your own graph/snapshot/alert JSON | benchmark report; exit code non-zero when a case misses its declared expectation |
| `scripts/07_run_terminal_monitor_eval.sh` | window CSV; optional LoRA adapter (`requirements-llm.txt`) | evaluation JSON |
| `scripts/08_run_llm_decomposition.sh` | Stage 1 run dir + API endpoint/key, or `--from-json` / `--dry-run` | `checkpoint_clusters_llm.csv` for stage 2/3 |
| `scripts/08_prepare_terminal_sft.sh` | window CSV (long format, optional `split` column); `--split` defaults to `auto` = train-only when a split column exists, hard error otherwise | SFT chat JSONL |
| `scripts/08_train_terminal_lora.sh` | SFT JSONL + `requirements-llm.txt`; downloads Qwen2.5-0.5B once | LoRA adapter directory |
| `scripts/run_htt_full_pipeline.sh` | stage-1 outputs, `workflow_decouple/src/rl` | bandit policies, `weight_library.json`, rule YAML/JSON, RL curves |

`scripts/run_htt_full_pipeline.sh` lives in the root `scripts/` directory (not under
`workflow_decouple/`). It trains the RL bandit first, so a clean checkout cannot
reproduce the committed rule files without regenerating the checkpoint data.

## 6. ns-3 communication benchmark: traceability

Published (all under `ns3_comm_sim/`, core baseline only):

| Path | Content | Size |
| --- | --- | --- |
| `results/core_baseline_explanation.md` | how to read the architecture comparison, including modeling assumptions | 8 KB |
| `results/parsed/core_baseline_summary.csv` | machine-readable summary | 3 KB |
| `results/parsed/core_baseline__{cloud,cloud_edge,cloud_edge_terminal}.csv` | per-architecture parsed metrics | ~5 KB |
| `results/tables/core_baseline_summary.csv` | summary table copy | 3 KB |
| `results/raw/core_baseline/<run>/{transactions.csv,flow_summary.csv,run_manifest.json}` | per-run control-plane records; manifests list the configs used, with repository-relative paths | ~60 KB |
| `inputs/traces/core_baseline.csv`, `inputs/events/core_baseline.csv`, `inputs/regions/core_baseline.csv`, `inputs/packet_profiles/control_messages.json` | the generated UAV traces/events that drive the runs | ~100 KB |

Not published: the smoke-test run, duplicate per-run JSON, `results/figures/`,
`results/raw/**/snr_samples.csv` (~9 MB), `results/raw/**/netanim.xml`,
`flow_monitor.xml`, and all `*.log` files (`ns3_run.log`, `**/build.log`,
`**/configure.log`) which the global `*.log` rule excludes, plus
`parsed/ns3_install_check.json` (host paths).

**Traceability check performed:** re-running
`python3 ns3_comm_sim/analysis/parse_results.py --raw-dir ns3_comm_sim/results/raw/core_baseline/core_baseline__cloud`
regenerates the committed `results/parsed/core_baseline__cloud.csv`
**byte-identically**. Without `snr_samples.csv` only four fields differ
(`run_name`, `scenario`, `architecture` derived from the directory name, and
`mean_snr_db`); all delay/success/timeout metrics are reproducible from
`transactions.csv` + `flow_summary.csv` alone. The published `mean_snr_db` values
therefore cannot be re-derived from this repository — they require the omitted SNR
dumps, which is why the SNR column is documented as non-recomputable rather than
reproducible.

**Modeling assumption, not a measurement.** The processing delays used by the ns-3
benchmark are fixed constants, not measured LLM inference latency:

| Parameter | Value | Source |
| --- | --- | --- |
| `localProcessingMs` | 8 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1261` |
| `edgeProcessingMs` | 20 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1262` |
| `cloudProcessingMs` | 60 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1263` |
| `cloud_backhaul_delay_ms` | 12 ms | `ns3_comm_sim/configs/radio/wifi_5ghz_fixed.json` |

No LLM was run in or timed by this benchmark. Transaction streams are synthetic,
generated from `configs/scenarios/core_baseline.json` (12 UAVs, 120 s, seed
20260413). The runner is `scripts/run_core_baseline.py`; it was **not** rerun during
this documentation pass, so the committed outputs are the archived run of record.

## 7. Data boundary

| Kept | Excluded |
| --- | --- |
| source code, shell entry points, runnable YAML/JSON configs, PDDL domain | `data/` (generated plans, init snapshots, results) |
| `analysis/rules/*` heuristic rule configs | `workflow_decouple/data/` (checkpoint data, RL weight libraries) |
| `examples/` demonstration fixtures (labelled as fixtures) | model weights (`*.pth`, `*.pt`, LoRA adapters, `model_cache/`) |
| `ns3_comm_sim` configs, inputs, core-baseline evidence (as listed in §6) | `reproducibility/`, `paper_latex/`, `data/replan_prediction/` — historical numbers without generating code; publishing them would present unreproducible evidence |
| root `README.md`, `docs/`, `requirements.txt`, `requirements-llm.txt` | `ns3_install/` (5 GB local ns-3 build), figures/PDFs, logs, caches |

## 8. Known gaps

1. **Authorization pending** — no license selected (§1); release metadata only.
2. Stage 3 is verified on a single fixture only (170 nodes / 850 directed corridors,
   30 agents, 10 `movetoloc` legs, 0 off-corridor legs; isolated goal → `UNSOLVABLE_INCOMPLETELY`
   with no plan written; missing graph → `FileNotFoundError`). This is not a
   manuscript-scenario reproduction, and `src/baselines/graph_mapf_priority.py` still
   auto-generates a grid topology when the graph file is missing (that baseline owns
   its own topology setup).
3. LLM paths are wired but unexecuted here: no API call, no model download, no
   adapter training. All `examples/llm/` inputs are fixtures.
4. **SFT split hygiene is enforced by default**: `prepare_terminal_sft --split`
   defaults to `auto` — a CSV with a `split` column exports only `split=train`, and a
   CSV without one is a hard error unless `--split all` is passed, which declares the
   file training-only and prints a leakage warning. Use `--split train` for real data
   and a separate evaluation set; `--split all` must never be paired with evaluation
   on the same CSV.
5. **Scheduler latency is labelled, never silently "measured"**. Every evaluation
   report carries `service_time_source`, one of `measured_llm_inference` (the adapter
   really ran), `replay_user_supplied_measured` (user-provided, not verified by this
   code), `replay_fixture_not_measured` (example/fixture replay file), or
   `assumed_constant_not_measured` (2.8 s default). The older
   `replay_measured_latency` label no longer exists.
6. `config/preprocessing.yaml` is the default of
   `workflow_decouple/src/run_multi_seed_pipeline.py` but only exists as
   `workflow_decouple/config/preprocessing.yaml`.
7. `src/replan_prediction/` is referenced by earlier planning notes but is absent
   from this checkout; no module imports it.
