#!/bin/bash
# Script 2: Run Stage 1 clustering for all methods
# Applies different clustering algorithms to checkpoint data

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"

echo "=========================================="
echo "Step 2: Run Stage 1 Clustering"
echo "=========================================="

# Configuration
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

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        RUN_DIR="$DATA_ROOT/$scenario/run_seed_$seed"
        # Try multiple input sources in order of preference
        INPUT_CSV=""
        for candidate in "checkpoint_clusters_kmeans.csv" "workflow_checkpoint_clustering.csv" "checkpoints_precluster.csv" "checkpoint_init.csv"; do
            if [ -f "$RUN_DIR/$candidate" ]; then
                INPUT_CSV="$RUN_DIR/$candidate"
                break
            fi
        done
        
        if [ -z "$INPUT_CSV" ]; then
            echo "⚠ Skipping: no input CSV found for $scenario/seed_$seed"
            continue
        fi
        
        echo "  Using input: $(basename $INPUT_CSV)"
        
        for method in "${METHODS[@]}"; do
            OUTPUT_CSV="$RUN_DIR/checkpoint_clusters_${method}.csv"
            
            if [ -f "$OUTPUT_CSV" ]; then
                echo "✓ Exists: $scenario/seed_$seed/$method"
                continue
            fi
            
            echo "Running: $scenario/seed_$seed/$method"
            
            # 对于已存在kmeans结果的情况，直接调用Python脚本处理各方法
            case $method in
                kmeans)
                    # KMeans 结果通常已存在（由 run_multi_seed_pipeline 生成）
                    echo "  KMeans: copying existing result or skipping"
                    if [ ! -f "$OUTPUT_CSV" ] && [ -f "$RUN_DIR/workflow_checkpoint_clustering.csv" ]; then
                        cp "$RUN_DIR/workflow_checkpoint_clustering.csv" "$OUTPUT_CSV"
                    fi
                    ;;
                triplet)
                    # Triplet 需要已训练的模型
                    TRIPLET_MODEL="$DATA_ROOT/triplet_learning_results/siamese_encoder.pth"
                    if [ ! -f "$TRIPLET_MODEL" ]; then
                        echo "⚠ Triplet model not found at $TRIPLET_MODEL"
                        echo "  Run 00_train_triplet.sh first."
                        exit 1
                    else
                        cd "$PROJECT_ROOT"
                        python -c "
import pandas as pd
import sys
sys.path.insert(0, 'workflow_decouple/src')
from ml_training.deep_learning import SiameseEncoder
from ml_training.checkpoint_extractor import CheckpointExtractor
import torch
import numpy as np
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('$INPUT_CSV')
n_clusters = df['cluster_group'].nunique()

# Load model checkpoint
ckpt = torch.load('$TRIPLET_MODEL', map_location='cpu', weights_only=False)
model = SiameseEncoder(input_dim=ckpt['input_dim'], hidden_dims=[64,128,64], output_dim=ckpt['output_dim'])
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Extract features and get embeddings
extractor = CheckpointExtractor()
features = extractor.prepare_features_for_clustering(df)
with torch.no_grad():
    embeddings = model.encoder(torch.tensor(features, dtype=torch.float32)).numpy()

# Cluster on embeddings
kmeans = KMeans(n_clusters=n_clusters, random_state=$seed, n_init=10)
labels = kmeans.fit_predict(embeddings)
df['cluster_group'] = labels
df.to_csv('$OUTPUT_CSV', index=False)
print(f'Triplet clustering: {len(df)} -> {n_clusters} clusters')
"
                    fi
                    ;;
                hungarian)
                    cd "$PROJECT_ROOT"
                    python -m src.baselines.hungarian_greedy \
                        --input "$INPUT_CSV" \
                        --output "$OUTPUT_CSV"
                    ;;
                htt_recursive)
                    # HtT 使用 checkpoint_clusters_htt_recursive.csv（优先）或 rule_kmeans
                    HTT_CSV="$RUN_DIR/checkpoint_clusters_htt_recursive.csv"
                    RULE_CSV="$RUN_DIR/checkpoint_clusters_rule_kmeans.csv"
                    if [ -f "$HTT_CSV" ]; then
                        cp "$HTT_CSV" "$OUTPUT_CSV"
                    elif [ -f "$RULE_CSV" ]; then
                        cp "$RULE_CSV" "$OUTPUT_CSV"
                    else
                        # 动态生成 htt_recursive
                        echo "  Generating htt_recursive via apply_rule_kmeans.py..."
                        cd "$PROJECT_ROOT"
                        python scripts/apply_rule_kmeans.py \
                            --data-root "$DATA_ROOT" \
                            --rules-dir "$PROJECT_ROOT/analysis/rules" \
                            --scenarios "$scenario" \
                            --start-seed "$seed" \
                            --n-seeds 1 2>/dev/null
                        if [ -f "$HTT_CSV" ]; then
                            cp "$HTT_CSV" "$OUTPUT_CSV"
                        else
                            echo "⚠ htt_recursive generation failed"
                            exit 1
                        fi
                    fi
                    ;;
            esac
            
            echo "✓ Done: $scenario/seed_$seed/$method"
        done
    done
done

echo "=========================================="
echo "Stage 1 clustering complete!"
echo "=========================================="
