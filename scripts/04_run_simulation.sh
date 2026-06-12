#!/bin/bash
# Script 4: Replay offline simulation for all methods
# Executes the simulator using generated offline plans

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"

echo "=========================================="
echo "Step 4: Replay Offline Simulation"
echo "=========================================="

SCENARIOS=(
    "small_logistics_inspection"
    "small_comm_compute"
    "medium_logistics_heavy"
    "medium_inspection_heavy"
    "large_mixed"
)

METHODS=(
    "kmeans"
    "triplet"
    "hungarian"
    "htt_recursive"
)

# Seeds configuration
START_SEED="${START_SEED:-42}"
N_SEEDS="${N_SEEDS:-5}"

# Generate seed array
SEED_ARRAY=()
for ((i=0; i<N_SEEDS; i++)); do
    SEED_ARRAY+=($((START_SEED + i)))
done

echo "Scenarios: ${SCENARIOS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEED_ARRAY[*]}"
echo ""

export PYTHONPATH="$PROJECT_ROOT:$WORKFLOW_DECOUPLE/src:$PROJECT_ROOT/../src:${PYTHONPATH:-}"

DATA_ROOT="$WORKFLOW_DECOUPLE/data/multi_seed_runs"
PLANS_ROOT="$PROJECT_ROOT/data/offline_plans"
RESULTS_ROOT="$PROJECT_ROOT/data/results"

mkdir -p "$RESULTS_ROOT"

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        INIT_ROOT="$DATA_ROOT/$scenario"
        
        for method in "${METHODS[@]}"; do
            PLAN_ROOT="$PLANS_ROOT/${scenario}_${method}"
            PLAN_FILE="$PLAN_ROOT/run_seed_$seed/pddl_plan_${method}.json"
            
            # Check if result already exists
            RESULT_PATTERN="Offline-${method}_*_workflow_summary.json"
            EXISTING=$(find "$RESULTS_ROOT" -name "$RESULT_PATTERN" -newer "$PLAN_FILE" 2>/dev/null | head -1)
            
            if [ ! -f "$PLAN_FILE" ]; then
                echo "⚠ Skipping: plan file not found for $scenario/seed_$seed/$method"
                continue
            fi
            
            echo "Simulating: $scenario/seed_$seed/$method"
            
            cd "$PROJECT_ROOT"
            python run_offline_simulation.py \
                --method "$method" \
                --seed "$seed" \
                --scenario "$scenario" \
                --init-root "$INIT_ROOT" \
                --plan-root "$PLAN_ROOT" \
                || echo "⚠ Simulation failed for $scenario/seed_$seed/$method"
            
            echo "✓ Done: $scenario/seed_$seed/$method"
        done
    done
done

echo "=========================================="
echo "Simulation complete!"
echo "Output: data/results/"
echo "=========================================="
