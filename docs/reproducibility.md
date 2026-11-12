# Reproducibility Notes — LoGe Planner `v1.1.0`

This document records what this package reproduces directly, which components need
external resources (GPU, API credentials, a local ns-3 build), and where each
published number comes from.

## 1. License status

LoGe Planner is released under the [MIT License](../LICENSE). Third-party
dependencies and model weights remain subject to their respective licenses.

Version string: `v1.1.0`.

## 2. Verified results and run prerequisites

### 2.1 Verified on this checkout

| Item | Measured evidence |
| --- | --- |
| PDDL domain parse + solve (`config/pddl/tccn_domain.pddl`, Fast Downward via `unified-planning 1.2.0`) | 6 problem files solved (`SOLVED_SATISFICING`), plan lengths 1/1/1/16/22/37 actions |
| Stage 2 assignment (`stage2_plan_from_clusters`) | `small_logistics_inspection`, seed 42, `htt_recursive` → `cluster_assignment_htt_recursive.csv` |
| Stage 3 grounding (`stage3_plan_with_pddl`) | same run: 30 agents, 170-node/850-edge topology, `pddl_plan_htt_recursive.json`, 2.7–3.1 s wall clock — both with `airfogsim` installed and with the import blocked |
| Stage 3 monolithic PDDL (`stage3_plan_monolithic_pddl`) | 30 agents / 70 goals, 0.60 s, with `airfogsim` import blocked |
| Scoped-replanning demo (`scripts/06_run_scoped_replanning.sh --demo`) | 5/5 fixtures as declared: 3 × `splice_accepted` (one requires a round-1 scope expansion), 1 × `global_recovered`, 1 × `infeasible`; all real Fast Downward solves, exit code non-zero on any mismatch |
| Stage 1 HtT clustering (`htt_recursive`) | 11 clusters, 21 tree nodes, max depth 5–6 |
| CLI entry points (`--help`) | stage1/2/3, monolithic, plan_validation_gate, scoped_replan_benchmark, terminal_monitor_eval, `run_offline_simulation.py` — all exit 0 |

### 2.2 Hardware and credentials required for the remaining paths

| Path | Requires | Notes |
| --- | --- | --- |
| LLM checkpoint decomposition (`scripts/08_run_llm_decomposition.sh`) | An OpenAI-compatible endpoint and API key (`OPENAI_API_KEY`) | `--dry-run` prints the prompt and `--from-json` reuses an existing decomposition, so both run without network access |
| LoRA SFT + terminal-monitor LLM (`scripts/08_train_terminal_lora.sh`, `scripts/07_run_terminal_monitor_eval.sh`) | `requirements-llm.txt`; the base weights (`Qwen/Qwen2.5-0.5B-Instruct`) are downloaded once from Hugging Face | `--device auto` selects CUDA when available and CPU otherwise; `--load-in-4bit` needs CUDA + bitsandbytes. `--validate-only` checks the JSONL without loading a model |
| ns-3 communication benchmark (`ns3_comm_sim/scripts/run_core_baseline.py`) | A local ns-3.40 build (several GB, installed under `ns3_install/`, not shipped) | The published outputs are regenerable from `ns3_comm_sim/inputs/` + `ns3_comm_sim/results/raw/` (§6) |
| Stages 1 and 4 (`scripts/01_generate_data.sh`, `scripts/04_run_simulation.sh`) | The `airfogsim` simulator from a sibling checkout | Only the simulation/data-generation path needs it; PDDL planning runs without it (§4) |
| Scoped-replanning batch (`scripts/06_run_scoped_replanning.sh --scenarios S1 … S5 --seeds … --cases-per-seed N`) | No extra hardware or credentials: Fast Downward and A* run locally | Instances are generated deterministically; `--schedule-dir` reads real schedule files instead of the generator |
| Full pipeline batch (`scripts/run_all_experiments.py`) | Generated checkpoint data plus the resources required by each stage it drives | Runs stages 1–4 over scenarios × seeds × methods |

The dependency set is in `requirements.txt` (core) and `requirements-llm.txt`
(LLM inference/training).

**Stage-3 verification on a generated topology.** On a real generated topology —
170 nodes / 850 directed corridors — the recorded run built 30 agents and 10
`movetoloc` legs with **0 legs outside real corridors**; an isolated cluster goal
(`waypoint_104`) is reported `UNSOLVABLE_INCOMPLETELY` and **no plan file is
written**; a missing topology raises `FileNotFoundError` with an actionable
message; node occupancy for capacity/reservation checks starts at **arrival**, so
a drone waiting for a release time still occupies the node; `blocked_edges` are
bidirectional by default and `"directional": true` keeps only the one-way closure.
Details and the exact commands: `docs/pddl_verification.md`. Timings are
single-run samples on this one fixture.

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

The environment package imports each environment component on first attribute
access (PEP 562 lazy imports), so `airfogsim` is not pulled into a stage-3 import:
`TopologyManager`/`GlobalMonitor`, which never need it, are usable in a
planning-only environment. Requesting a weather/UFO/event component without
`airfogsim` raises an explicit `ImportError` naming the missing package and the
PDDL-only path; there is no mock or fallback object.

## 5. Entry points: required inputs

All data is **generated locally by this package**; it is not the original
manuscript dataset. Historical result files are outside the release (§7).

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
| `scripts/run_all_experiments.py` | the resources of the stages it drives; `--scenarios` × `--seeds` (default `42,43,44,45,46`) × `--methods` | stage 1–4 outputs per (scenario, seed, method) cell |
| `scripts/06_run_scoped_replanning.sh --scenarios …` | nothing beyond the solver; `--cases-per-seed` alerts per (scenario, seed) | batch report (`--output-json`): aggregate, per-scenario, per-case tables |

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

**Traceability check:** re-running
`python3 ns3_comm_sim/analysis/parse_results.py --raw-dir ns3_comm_sim/results/raw/core_baseline/core_baseline__cloud`
regenerates the committed `results/parsed/core_baseline__cloud.csv`
**byte-identically**. Without `snr_samples.csv` only four fields differ
(`run_name`, `scenario`, `architecture` derived from the directory name, and
`mean_snr_db`); all delay/success/timeout metrics are reproducible from
`transactions.csv` + `flow_summary.csv` alone. The `mean_snr_db` column therefore
requires the omitted SNR dumps and is documented as non-recomputable.

**Modeling parameters.** The processing delays used by the ns-3 benchmark are
fixed constants, not measured LLM inference latency:

| Parameter | Value | Source |
| --- | --- | --- |
| `localProcessingMs` | 8 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1261` |
| `edgeProcessingMs` | 20 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1262` |
| `cloudProcessingMs` | 60 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1263` |
| `cloud_backhaul_delay_ms` | 12 ms | `ns3_comm_sim/configs/radio/wifi_5ghz_fixed.json` |

The benchmark times control-plane message handling only, not LLM inference.
Transaction streams are synthetic, generated from
`configs/scenarios/core_baseline.json` (12 UAVs, 120 s, seed 20260413). The
runner is `scripts/run_core_baseline.py`; the committed outputs are the archived
run of record, regenerable with a local ns-3.40 build.

## 7. Data boundary

| Kept | Excluded |
| --- | --- |
| source code, shell entry points, runnable YAML/JSON configs, PDDL domain | `data/` (generated plans, init snapshots, results) |
| `analysis/rules/*` heuristic rule configs | `workflow_decouple/data/` (checkpoint data, RL weight libraries) |
| `examples/` demonstration fixtures (labelled as fixtures) | model weights (`*.pth`, `*.pt`, LoRA adapters, `model_cache/`) |
| `ns3_comm_sim` configs, inputs, core-baseline evidence (as listed in §6) | `reproducibility/`, `paper_latex/`, `data/replan_prediction/` — historical result files outside this release |
| root `README.md`, `docs/`, `requirements.txt`, `requirements-llm.txt` | `ns3_install/` (5 GB local ns-3 build), figures/PDFs, logs, caches |

## 8. Known gaps

1. Stage 3 is verified on one generated topology (170 nodes / 850 directed corridors,
   30 agents, 10 `movetoloc` legs, 0 off-corridor legs; isolated goal → `UNSOLVABLE_INCOMPLETELY`
   with no plan written; missing graph → `FileNotFoundError`). Manuscript-scale
   S1–S5 runs come from the batch driver in §2.2.
   `src/baselines/graph_mapf_priority.py`
   auto-generates a grid topology when the graph file is missing (that baseline owns
   its own topology setup).
2. The LLM paths need the external endpoint and model weights listed in §2.2; all
   `examples/llm/` inputs are fixtures.
3. **SFT split hygiene is enforced by default**: `prepare_terminal_sft --split`
   defaults to `auto` — a CSV with a `split` column exports only `split=train`, and a
   CSV without one is a hard error unless `--split all` is passed, which declares the
   file training-only and prints a leakage warning. Use `--split train` for real data
   and a separate evaluation set; `--split all` must never be paired with evaluation
   on the same CSV.
4. **Scheduler latency is tagged with its source.** Every evaluation report carries
   `service_time_source`, one of `llm_adapter_inference` (mean per-batch latency
   measured by the adapter/base-model condition in this run), `external_trace_replay`
   (values read from an external latency replay file), `fixture_replay` (replay file
   declared as example/fixture data), or `orin_nx_nominal_2_8s` (Jetson Orin NX
   Phase-A nominal 2.8 s, the default when no measured latency is supplied).
5. `config/preprocessing.yaml` is the default of
   `workflow_decouple/src/run_multi_seed_pipeline.py` but only exists as
   `workflow_decouple/config/preprocessing.yaml`.
6. `src/replan_prediction/` is referenced by earlier planning notes but is absent
   from this checkout; no module imports it.
