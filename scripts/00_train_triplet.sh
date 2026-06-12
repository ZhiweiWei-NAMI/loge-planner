#!/bin/bash
# Script 0: Train Triplet model (run once before experiments)
# This trains the Triplet Learning model using merged data
# The trained model is used by the 'triplet' baseline method
#
# 数据说明:
#   - merged_dataset/: large_mixed 场景的 seed 42-51 合并数据
#   - all_scenarios_merged/: 所有场景合并数据
#   - 训练使用 merged_dataset 或 all_scenarios_merged

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"
DATA_ROOT="$WORKFLOW_DECOUPLE/data/multi_seed_runs"

echo "=========================================="
echo "Step 0: Train Triplet Learning Model"
echo "=========================================="

# Configuration
TRAIN_RUNS="${TRAIN_RUNS:-42,43,44,45,46,47,48}"
VAL_RUNS="${VAL_RUNS:-49,50}"
EPOCHS="${EPOCHS:-100}"
# 使用 all_scenarios_merged 或 merged_dataset
DATA_DIR="${DATA_DIR:-$DATA_ROOT/all_scenarios_merged}"

echo "Project root: $PROJECT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Train runs: $TRAIN_RUNS"
echo "Val runs: $VAL_RUNS"
echo "Epochs: $EPOCHS"
echo ""

export PYTHONPATH="$PROJECT_ROOT:$WORKFLOW_DECOUPLE/src:$PROJECT_ROOT/../src:${PYTHONPATH:-}"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Available directories:"
    ls -la "$DATA_ROOT/" | grep "^d"
    exit 1
fi

# Check if model already exists
MODEL_PATH="$DATA_ROOT/triplet_learning_results/siamese_encoder.pth"
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Triplet model already exists: $MODEL_PATH"
    echo "  To retrain, delete the file first."
    exit 0
fi

# Train Triplet model directly using train_triplet_learning.py
cd "$WORKFLOW_DECOUPLE/src"
echo "Training Triplet Learning model..."

python -m ml_training.train_triplet_learning \
    --data-dir "$DATA_DIR" \
    --save-dir "$DATA_ROOT/triplet_learning_results" \
    --train-runs "$TRAIN_RUNS" \
    --val-runs "$VAL_RUNS" \
    --epochs "$EPOCHS"

echo "=========================================="
echo "Triplet training complete!"
echo "Model: $DATA_ROOT/triplet_learning_results/"
echo "=========================================="
