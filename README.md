# LoGe Planner

LoGe Planner is an offline planning codebase for low-altitude multi-agent workflow scheduling. It converts scenario configurations into workflow checkpoints, decomposes checkpoints with several clustering or assignment schemes, generates executable plans, and replays those plans in the AirFogSim-based simulator.

This public package contains code and runnable configuration only. Experiment outputs, plotting data, model weights, logs, API keys, and paper/revision artifacts are intentionally excluded by `.gitignore`.

## Code Layout

```text
.
├── src/
│   ├── offline_pipeline/      # Stage 1-3 offline planning pipeline
│   ├── baselines/             # Greedy, graph, and monolithic baselines
│   ├── pddl_solver/           # PDDL parameter conversion and solver wrapper
│   ├── simulation/            # Agent, workflow, task, component, and environment models
│   ├── config/                # Config loading and object factory utilities
│   └── validation/            # Plan validation checks
├── workflow_decouple/src/     # Checkpoint extraction, feature construction, RL/triplet tooling
├── workflow_decouple/config/  # Scenario generation and preprocessing configs
├── analysis/                  # Rule application and rule-mining utilities
├── analysis/rules/            # Scenario-specific rule-weight configs for HtT variants
├── plot/                      # Experiment orchestration and result aggregation code
├── scripts/                   # Shell/Python pipeline entry points
├── ns3_comm_sim/              # Optional ns-3 communication benchmark code
├── config/                    # Simulator component and topology configs
└── run_offline_simulation.py  # Offline plan replay entry point
```

## Planning Schemes

`htt_recursive` is the main LoGe decomposition path. It uses scenario rules from `analysis/rules/`, recursively chooses feature weights, clusters checkpoints, assigns clusters to agents, and generates cluster-level plans.

`kmeans` uses standard checkpoint feature clustering before the same cluster-to-agent and planning stages.

`triplet` uses learned checkpoint embeddings from triplet training and then clusters in the embedding space.

`hungarian` is a workflow-level greedy assignment baseline. It keeps a workflow together and emits direct sequential plans.

`monolithic` builds one global PDDL planning problem without decomposition.

## Pipeline

1. Stage 1 extracts workflow checkpoints and writes `checkpoint_clusters_<method>.csv`.
2. Stage 2 assigns each cluster to one or more agents and writes `cluster_assignment_<method>.csv`.
3. Stage 3 converts assignments into `pddl_plan_<method>.json`.
4. `run_offline_simulation.py` replays a generated plan in the simulator.
5. `plot/aggregate_results.py` aggregates generated result files when local experiment outputs exist.

The standard scripted flow is:

```bash
bash scripts/01_generate_data.sh
bash scripts/02_run_clustering.sh
bash scripts/03_run_pddl_planning.sh
bash scripts/04_run_simulation.sh
python -m plot.aggregate_results
```

For a single controlled run:

```bash
python -m src.offline_pipeline.stage1_precluster_from_config --help
python -m src.offline_pipeline.stage3_plan_with_pddl --help
python run_offline_simulation.py --help
```

## Data Boundary

The repository is configured as default-deny for open-source export. Only source code, executable scripts, runnable configuration, and this README are included. The following are excluded:

- `data/`, `workflow_decouple/data/`, `reproducibility/`, runtime logs, and plot outputs
- model weights, checkpoints, learned adapters, prediction JSON files, and caches
- paper drafts, review artifacts, generated figures, and local third-party installs
- online-model configs and archived internal rule/prompt paths

If you run the pipeline locally, generated files will stay outside the Git upload set unless explicitly force-added.
