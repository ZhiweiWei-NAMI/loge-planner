#!/bin/bash
# Script 1: Generate scenario data for all scenarios and seeds
# This creates the initial scenario configurations and checkpoint data

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"

echo "=========================================="
echo "Step 1: Generate Scenario Data"
echo "=========================================="

# Configuration
SCENARIOS=(
    "small_logistics_inspection"
    "small_comm_compute"
    "medium_logistics_heavy"
    "medium_inspection_heavy"
    "large_mixed"
)

# Seeds configuration: start seed and number of seeds
START_SEED="${START_SEED:-42}"
N_SEEDS="${N_SEEDS:-5}"

echo "Project root: $PROJECT_ROOT"
echo "Scenarios: ${SCENARIOS[*]}"
echo "Seeds: $START_SEED to $((START_SEED + N_SEEDS - 1))"
echo ""

# Set PYTHONPATH - include workflow_decouple/src for module imports
export PYTHONPATH="$PROJECT_ROOT:$WORKFLOW_DECOUPLE/src:$PROJECT_ROOT/../src:${PYTHONPATH:-}"

# Generate data for each scenario
for scenario in "${SCENARIOS[@]}"; do
    echo "Generating data for scenario: $scenario"
    
    # Run from workflow_decouple directory where the module is located
    cd "$WORKFLOW_DECOUPLE"
    # 只生成仿真数据和预处理，不训练模型
    # --only-clustering: 只执行仿真和预处理聚类，不合并和训练
    python -m src.run_multi_seed_pipeline \
        --scenario-name "$scenario" \
        --start-seed "$START_SEED" \
        --n-seeds "$N_SEEDS" \
        --only-clustering \
        --output-dir "data/multi_seed_runs"
    
    echo "✓ Completed: $scenario"
    echo ""
done

echo "=========================================="
echo "Data generation complete!"
echo "Output: workflow_decouple/data/multi_seed_runs/"
echo "=========================================="
