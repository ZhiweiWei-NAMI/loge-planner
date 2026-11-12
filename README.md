# LoGe Planner (`v1.1.0`)

LoGe Planner (`v1.1.0`, corresponding to the revised SCIS manuscript) is a multi-tier planning and runtime-recovery codebase for low-altitude multi-UAV workflow scheduling. It converts scenario configurations into workflow checkpoints, decomposes checkpoints into causality-preserving macro-tasks, grounds symbolic PDDL plans onto reservation-constrained airspace graphs, and evaluates closed-loop anomaly monitoring and validator-gated scoped replanning.

This package contains source code, PDDL domain specifications, runnable configurations, a small set of demonstration fixtures, and the traceable ns-3 core-baseline evidence. Generated datasets, trained weights, LLM credentials, and manuscript artifacts are excluded — see [docs/reproducibility.md](docs/reproducibility.md) for the exact boundary and for what was and was not executed.

## Installation

Tested interpreter: CPython 3.10 on Linux. Dependency requirements are in [`requirements.txt`](requirements.txt) (dependency requirements only — not a validated environment lock).

```bash
pip install -r requirements.txt
# Optional: local Qwen2.5 + PEFT/LoRA inference and training
pip install -r requirements-llm.txt
# External, not on PyPI: the UAV simulator used by src/simulation/**
pip install -e ../airfogsim-project
```

`airfogsim` is only needed for simulation/data generation. The PDDL solver, the offline stage 2/3 planning path, and the scoped-replanning demo import and run without it (the environment package uses lazy imports so that a missing `airfogsim` surfaces only when a weather/UFO/event component is actually requested).

## Code Layout

```text
.
├── config/
│   ├── pddl/tccn_domain.pddl  # Canonical STRIPS/typed PDDL domain specification
│   └── *.yaml                 # Simulator component, topology, and event configs
├── src/
│   ├── offline_pipeline/      # Stage 1-3 decomposition, assignment, and PDDL+A* grounding
│   ├── baselines/             # Greedy, Priority Graph-MAPF, and Monolithic Global PDDL baselines
│   ├── pddl_solver/           # PDDL problem instantiation and solver wrapper
│   ├── simulation/            # Agent, workflow, task, component, and environment models
│   ├── config/                # Config loading and object factory utilities
│   ├── agent_llm/             # LLM checkpoint decomposition, LoRA SFT, terminal-monitor LLM
│   └── validation/            # Parser gate, scoped replanning, terminal monitor evaluation
├── workflow_decouple/src/     # Checkpoint extraction, 26-D feature construction, RL/triplet/HtT rules
├── workflow_decouple/config/  # Scenario generation and preprocessing configs
├── analysis/rules/            # Scenario-specific HtT rule and recursive-budget configs (S1-S5)
├── examples/                  # Demonstration fixtures (replanning graph/snapshot/alerts, LLM samples)
├── plot/                      # Experiment orchestration and result aggregation code
├── scripts/                   # Shell/Python pipeline entry points (00-07)
├── ns3_comm_sim/              # ns-3.40 city-canyon control-plane communication benchmark
├── docs/reproducibility.md    # Execution status, evidence provenance, modeling assumptions
└── run_offline_simulation.py  # Closed-loop plan replay entry point
```

## Planning Schemes

- `htt_recursive`: LoGe-Planner's main decomposition path using scenario rules in `analysis/rules/`, causal anchors, and recursive partitioning.
- `kmeans`: Static 26-D checkpoint feature clustering matched to the macro-task cluster budget.
- `triplet`: Metric-learning checkpoint embeddings (`DML-KMeans`) followed by budget-matched clustering.
- `hungarian` / `graph_mapf_priority`: Direct assignment and priority reservation-aware routing baselines.
- `monolithic`: Undecomposed global PDDL solving (`config/pddl/tccn_domain.pddl`).

The `HtT` rules are **heuristic rules mined from RL bandit weight libraries** (feature thresholds such as `spatial_spread >= 253.22`), applied to a deterministic weighted KMeans. They are neither PDDL rules nor LLM output; no language model participates in clustering.

## Entry Points

### 1. Offline Pipeline: Decomposition, PDDL Grounding, and Replay (Stages 1-4)

```bash
bash scripts/01_generate_data.sh     # needs airfogsim; generates checkpoint data
bash scripts/02_run_clustering.sh    # kmeans/triplet/hungarian/htt_recursive
bash scripts/03_run_pddl_planning.sh # needs config/pddl/tccn_domain.pddl
bash scripts/04_run_simulation.sh    # needs airfogsim
bash scripts/05_plot_results.sh      # python3 -m plot.aggregate_results
```

Single-instance entry points:

```bash
python3 -m src.offline_pipeline.stage1_precluster_from_config --help
python3 -m src.offline_pipeline.stage2_plan_from_clusters --help
python3 -m src.offline_pipeline.stage3_plan_with_pddl --help
python3 -m src.offline_pipeline.stage3_plan_monolithic_pddl --help
python3 run_offline_simulation.py --help
```

Inputs are generated locally by this package; they are **not** the original manuscript dataset. `triplet` additionally needs a trained encoder from `scripts/00_train_triplet.sh`.

### 2. Validator-Gated Scoped Replanning & Parser Gate

Runs real PDDL (Fast Downward) plus the time-varying A* router on the bundled fixtures:

```bash
bash scripts/06_run_scoped_replanning.sh            # == --demo
bash scripts/06_run_scoped_replanning.sh --demo --artifacts-dir tmp/artifacts
bash scripts/06_run_scoped_replanning.sh \
    --graph examples/replanning/graph_small.json \
    --snapshot examples/replanning/snapshot_small.json \
    --alert examples/replanning/alerts/demo_detour.json
python3 -m src.validation.plan_validation_gate --help
```

The fixtures are hand-written demonstrations (`examples/replanning/README.md`); they carry no statistical meaning and must not be reported as paper results.

### 3. LLM Agents and Terminal Semantic Monitoring

Every file in `examples/llm/` is a **fixture**, not measured data (`examples/llm/README.md`). No API call and no model run were performed on the machine that prepared this release.

```bash
# (a) Cloud macro-task decomposition (OpenAI-compatible endpoint)
export OPENAI_API_KEY=...            # never commit the key
bash scripts/08_run_llm_decomposition.sh \
    --run-dir data/offline_init/large_mixed/run_seed_42 \
    --model gpt-4o-mini --base-url https://api.openai.com/v1 \
    --min-macro-tasks 8 --max-macro-tasks 60 --cluster-method llm --inplace
# --dry-run prints the prompt without any network call; --from-json reuses an
# existing decomposition JSON (see examples/llm/decomposition_example.json).
python3 -m src.offline_pipeline.stage2_plan_from_clusters --seed 42 --cluster-method llm
python3 -m src.offline_pipeline.stage3_plan_with_pddl     --seed 42 --cluster-method llm

# (b) Terminal-monitor SFT data -> LoRA training
# --split auto (default) exports split=train only when a split column exists,
# and refuses to guess otherwise; use --split train explicitly for real data.
bash scripts/08_prepare_terminal_sft.sh \
    --windows-csv examples/llm/terminal_windows_example.csv --split train \
    --output-jsonl /tmp/terminal_sft.jsonl
bash scripts/08_train_terminal_lora.sh --train-jsonl /tmp/terminal_sft.jsonl \
    --output-dir /tmp/qwen_terminal_lora --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 3 --batch-size 4 --lr 2e-4
# --validate-only checks the JSONL without loading or downloading a model

# (c) Terminal-monitor evaluation with a real adapter
bash scripts/07_run_terminal_monitor_eval.sh \
    --data-csv data/terminal/windows.csv --adapter-dir /tmp/qwen_terminal_lora \
    --include-base-model --output-json data/terminal/eval.json
```

Two limits that matter when reading any number from this path:

- **Split hygiene is leak-safe by default.** `prepare_terminal_sft --split` defaults to `auto`: a CSV with a `split` column exports only `split=train`; a CSV without one is a hard error unless you pass `--split all`, which declares the file **training-only** and prints a leakage warning. Never train on `all` and evaluate on the same CSV — use `--split train` plus a separate evaluation set.
- **Scheduler latency is never silently "measured".** Every evaluation report carries `service_time_source`, one of: `measured_llm_inference` (the adapter really ran), `replay_user_supplied_measured` (user-provided values, **not verified here**), `replay_fixture_not_measured` (example/fixture replay file), or `assumed_constant_not_measured` (the 2.8 s default). Only the first is a measurement made by this code.

### 4. ns-3.40 Communication Benchmark

```bash
python3 ns3_comm_sim/scripts/run_core_baseline.py    # needs a local ns-3.40 build
python3 ns3_comm_sim/analysis/parse_results.py --raw-dir ns3_comm_sim/results/raw/core_baseline/core_baseline__cloud
```

Published evidence under `ns3_comm_sim/results/`: the explanation document, the summary table, per-architecture parsed metrics, and the per-run `transactions.csv` / `flow_summary.csv` / `run_manifest.json` records that regenerate them. `mean_snr_db` additionally requires the omitted `snr_samples.csv` dumps.

**The ns-3 processing delays are fixed modeling assumptions, not measurements**: `localProcessingMs = 8`, `edgeProcessingMs = 20`, `cloudProcessingMs = 60` (`ns3_comm_sim/ns3/uav_comm_benchmark.cc`), plus a 12 ms cloud backhaul. No LLM latency is measured anywhere in this benchmark.

## What Is Measured Here

These are single-fixture verifications on generated inputs, **not a paper reproduction**: no manuscript scenario set (S1–S5) is regenerated here.

| Claim | Status |
| --- | --- |
| PDDL domain parses and solves (Fast Downward) | measured; plan lengths reported in [docs/reproducibility.md](docs/reproducibility.md) |
| Stage 2/3 offline grounding | single-fixture verification on a real generated topology (170 nodes / 850 directed corridors): 30 agents, 10 `movetoloc` legs, 0 legs outside real corridors; an isolated goal yields `UNSOLVABLE_INCOMPLETELY` with no plan file written; a missing graph raises `FileNotFoundError`. Runs with or without `airfogsim`. Details: [docs/pddl_verification.md](docs/pddl_verification.md) |
| Scoped-replanning demo (5 fixtures) | measured, all real Fast Downward solves: 3 × `splice_accepted` (one needs a round-1 scope expansion), 1 × `global_recovered`, 1 × `infeasible`; exit code non-zero on any mismatch |
| ns-3 architecture comparison | archived run of record; regenerable from the published inputs/raw records |
| LLM decomposition, LoRA adapter, terminal-monitor LLM | implementation provided; real API/model execution not performed here; weights not distributed. All `examples/llm/` inputs are fixtures |
| Scheduler service time | labelled per run: `measured_llm_inference` / `replay_user_supplied_measured` (unverified) / `replay_fixture_not_measured` / `assumed_constant_not_measured` (2.8 s default) |

## Data Boundary

Tracked: source code, shell entry points, runnable YAML/JSON configs, the PDDL domain, `analysis/rules/*` heuristic rule configs, `examples/` fixtures, `docs/`, and the ns-3 core-baseline evidence set (explanation, summary, parsed metrics, per-run transaction/flow/manifest records, benchmark inputs).

Excluded: `data/`, `workflow_decouple/data/`, `reproducibility/`, `paper_latex/`, model weights and adapters, ns-3 SNR/netanim dumps, figures, logs, and the local ns-3 build. Historical result files without generating code are intentionally not published.

## License

**Undecided (TBD by the author).** This repository contains no `LICENSE`, `COPYING`, or `NOTICE` file and no copyright headers; nothing here grants a license. A license must be selected and added before redistribution.
