#!/bin/bash
# Script 3: Run Stage 3 PDDL planning for all methods
# Generates PDDL plans from cluster assignments

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"

echo "=========================================="
echo "Step 3: Run PDDL Planning (Stage 3)"
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

for scenario in "${SCENARIOS[@]}"; do
    # input-root 是场景级目录，stage2/3 脚本内部会拼接 run_seed_$seed
    SCENARIO_INPUT_ROOT="$DATA_ROOT/$scenario"
    
    for seed in "${SEED_ARRAY[@]}"; do
        RUN_DIR="$SCENARIO_INPUT_ROOT/run_seed_$seed"
        
        if [ ! -d "$RUN_DIR" ]; then
            echo "⚠ Skipping: run directory not found for $scenario/seed_$seed"
            continue
        fi
        
        for method in "${METHODS[@]}"; do
            CLUSTER_CSV="$RUN_DIR/checkpoint_clusters_${method}.csv"
            # output-root 也是场景级目录，脚本内部会拼接 run_seed_$seed
            METHOD_OUTPUT_ROOT="$PLANS_ROOT/${scenario}_${method}"
            ASSIGNMENT_FILE="$METHOD_OUTPUT_ROOT/run_seed_$seed/cluster_assignment_${method}.csv"
            PLAN_FILE="$METHOD_OUTPUT_ROOT/run_seed_$seed/pddl_plan_${method}.json"
            
            if [ ! -f "$CLUSTER_CSV" ]; then
                echo "⚠ Skipping: cluster CSV not found for $scenario/seed_$seed/$method"
                continue
            fi
            
            if [ -f "$PLAN_FILE" ]; then
                echo "✓ Exists: $scenario/seed_$seed/$method"
                continue
            fi
            
            mkdir -p "$METHOD_OUTPUT_ROOT"
            
            cd "$PROJECT_ROOT"
            
            # Stage 2: 生成 cluster_assignment (如果不存在)
            if [ ! -f "$ASSIGNMENT_FILE" ]; then
                echo "Stage2: $scenario/seed_$seed/$method"
                python -m src.offline_pipeline.stage2_plan_from_clusters \
                    --seed "$seed" \
                    --cluster-method "$method" \
                    --input-root "$SCENARIO_INPUT_ROOT" \
                    --output-root "$METHOD_OUTPUT_ROOT" \
                    || { echo "⚠ Stage2 failed for $scenario/seed_$seed/$method"; continue; }
            fi
            
            # Stage 3: 生成 PDDL plan
            echo "Stage3: $scenario/seed_$seed/$method"
            python -m src.offline_pipeline.stage3_plan_with_pddl \
                --seed "$seed" \
                --cluster-method "$method" \
                --input-root "$SCENARIO_INPUT_ROOT" \
                --output-root "$METHOD_OUTPUT_ROOT" \
                || { echo "⚠ Stage3 failed for $scenario/seed_$seed/$method"; continue; }
            
            echo "✓ Done: $scenario/seed_$seed/$method"
        done
    done
done

echo "=========================================="
echo "Stage 3 PDDL planning complete!"
echo "Output: data/offline_plans/"
echo "=========================================="
