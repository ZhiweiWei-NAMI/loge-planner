# LoGe Planner (`v1.1.0`)

LoGe Planner (`v1.1.0`, corresponding to the revised SCIS manuscript) is a multi-tier planning and runtime-recovery codebase for low-altitude multi-UAV workflow scheduling. It converts scenario configurations into workflow checkpoints, decomposes checkpoints into causality-preserving macro-tasks, grounds symbolic PDDL plans onto reservation-constrained airspace graphs, and evaluates closed-loop anomaly monitoring and validator-gated scoped replanning.

This public package contains source code, PDDL domain specifications, and runnable configurations. Raw simulation outputs, model weights, caches, API keys, and manuscript artifacts are excluded by `.gitignore`.

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
│   └── validation/            # Parser gate, scoped replanning, and terminal monitor evaluation
├── workflow_decouple/src/     # Checkpoint extraction, 26-D feature construction, RL/triplet/HtT rules
├── workflow_decouple/config/  # Scenario generation and preprocessing configs
├── analysis/rules/            # Scenario-specific HtT rule and recursive-budget configs (S1-S5)
├── plot/                      # Experiment orchestration and result aggregation code
├── scripts/                   # Shell/Python pipeline entry points (00-07)
├── ns3_comm_sim/              # ns-3.40 city-canyon control-plane communication benchmark
└── run_offline_simulation.py  # Closed-loop plan replay entry point
```

## Planning Schemes

- `htt_recursive`: LoGe-Planner's main decomposition path using scenario rules in `analysis/rules/`, causal anchors, and recursive partitioning.
- `kmeans`: Static 26-D checkpoint feature clustering matched to the macro-task cluster budget.
- `triplet`: Metric-learning checkpoint embeddings (`DML-KMeans`) followed by budget-matched clustering.
- `hungarian` / `graph_mapf_priority`: Direct assignment and priority reservation-aware routing baselines.
- `monolithic`: Undecomposed global PDDL solving (`config/pddl/tccn_domain.pddl`).

## Reproducible Entry Points

### 1. End-to-End Decomposition, PDDL Grounding, and Simulation (Stages 1-4)

```bash
bash scripts/01_generate_data.sh
bash scripts/02_run_clustering.sh
bash scripts/03_run_pddl_planning.sh
bash scripts/04_run_simulation.sh
python3 -m plot.aggregate_results
```

Single-instance CLI entry points (automatically loading `config/pddl/tccn_domain.pddl`):

```bash
python3 -m src.offline_pipeline.stage1_precluster_from_config --help
python3 -m src.offline_pipeline.stage3_plan_with_pddl --help
python3 run_offline_simulation.py --help
```

### 2. Validator-Gated Scoped Replanning & Parser Gate (Section 3.3.2 / Algorithm 2)

```bash
bash scripts/06_run_scoped_replanning.sh --scenarios S1 S2 S3 S4 S5 --seeds 42 43 44 45 46 --cases-per-seed 20
python3 -m src.validation.plan_validation_gate --help
```

### 3. Terminal Semantic Monitoring & Observation Degradation ($L_0$--$L_3$, Section 4.1.2 & 4.5.7)

```bash
bash scripts/07_run_terminal_monitor_eval.sh --levels L0 L1 L2 L3 --samples-per-class 80
# Optional: attach a local PEFT/LoRA Qwen2.5-0.5B adapter checkpoint directory
bash scripts/07_run_terminal_monitor_eval.sh --levels L0 L1 L2 L3 --adapter-dir /path/to/lora_adapter
```

### 4. ns-3.40 Communication Robustness Benchmark (Section 4.5.4)

```bash
python3 ns3_comm_sim/scripts/run_core_baseline.py
python3 ns3_comm_sim/analysis/parse_results.py
```

## Data Boundary

Only source code, executable scripts, PDDL domain files (`config/pddl/tccn_domain.pddl`), runnable configuration files, and this README are tracked for open-source export (`v1.1.0`). Generated datasets (`data/`, `workflow_decouple/data/`), model weights/adapters, runtime logs, and manuscript files (`paper_latex/`, `reproducibility/`) remain excluded by `.gitignore`.
