#!/bin/bash
# HtT (Hierarchical Task Thinking) 完整流程
# 
# 步骤:
# 1. RL Bandit 训练 -> 获取权重策略
# 2. 递归权重收集 -> 生成 weight_library.json
# 3. 权重规则总结 -> 生成各场景 rule_kmeans_config.yaml
# 4. 应用规则生成 rule_kmeans 聚类结果
# 5. 绘制 RL 训练收敛曲线

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"
DATA_ROOT="$WORKFLOW_DECOUPLE/data/multi_seed_runs"
RULES_DIR="$PROJECT_ROOT/analysis/rules"

# 场景列表
SCENARIOS=(
    "small_logistics_inspection"
    "small_comm_compute"
    "medium_logistics_heavy"
    "medium_inspection_heavy"
    "large_mixed"
)

# Seeds 配置
START_SEED="${START_SEED:-42}"
N_SEEDS="${N_SEEDS:-5}"
TRAIN_RUNS="42,43,44,45,46,47,48"
VAL_RUNS="49,50"

# RL 训练参数
N_EPOCHS="${N_EPOCHS:-50}"
N_CLUSTERS="${N_CLUSTERS:-30}"

echo "=========================================="
echo "HtT Full Pipeline"
echo "=========================================="
echo "Scenarios: ${SCENARIOS[*]}"
echo "Seeds: $START_SEED to $((START_SEED + N_SEEDS - 1))"
echo "RL Epochs: $N_EPOCHS"
echo ""

export PYTHONPATH="$PROJECT_ROOT:$WORKFLOW_DECOUPLE/src:${PYTHONPATH:-}"

# ========================================
# Step 1: RL Bandit 训练
# ========================================
echo ""
echo "=========================================="
echo "Step 1: RL Bandit Training"
echo "=========================================="

for scenario in "${SCENARIOS[@]}"; do
    BANDIT_DIR="$DATA_ROOT/$scenario/bandit_results"
    
    if [ -f "$BANDIT_DIR/best_policy.pth" ]; then
        echo "✓ Bandit model exists for $scenario"
        continue
    fi
    
    echo "Training Bandit for: $scenario"
    mkdir -p "$BANDIT_DIR"
    
    cd "$WORKFLOW_DECOUPLE/src"
    python -m rl.train_bandit \
        --data-root "$DATA_ROOT" \
        --scenario-name "$scenario" \
        --train-runs "$TRAIN_RUNS" \
        --val-runs "$VAL_RUNS" \
        --n-epochs "$N_EPOCHS" \
        --n-clusters "$N_CLUSTERS" \
        --output-dir "$BANDIT_DIR" \
        2>&1 | tee "$BANDIT_DIR/training.log"
    
    echo "✓ Done: $scenario"
done

# ========================================
# Step 2: 递归权重收集
# ========================================
echo ""
echo "=========================================="
echo "Step 2: Recursive Weight Collection"
echo "=========================================="

for scenario in "${SCENARIOS[@]}"; do
    BANDIT_DIR="$DATA_ROOT/$scenario/bandit_results"
    WEIGHT_LIB="$BANDIT_DIR/weight_library.json"
    
    if [ -f "$WEIGHT_LIB" ]; then
        echo "✓ Weight library exists for $scenario"
        continue
    fi
    
    if [ ! -f "$BANDIT_DIR/best_policy.pth" ]; then
        echo "⚠ Skipping $scenario: no trained policy"
        continue
    fi
    
    echo "Collecting weights for: $scenario"
    
    cd "$WORKFLOW_DECOUPLE/src"
    python -m rl.collect_recursive_weights \
        --policy "$BANDIT_DIR/best_policy.pth" \
        --encoder "$BANDIT_DIR/encoder.pth" \
        --data-root "$DATA_ROOT" \
        --scenario-name "$scenario" \
        --data-runs "$TRAIN_RUNS" \
        --max-depth 5 \
        --min-samples 30 \
        --output "$WEIGHT_LIB" \
        2>&1 | tee "$BANDIT_DIR/weight_collection.log"
    
    echo "✓ Done: $scenario"
done

# ========================================
# Step 3: 权重规则总结 (Python 脚本)
# ========================================
echo ""
echo "=========================================="
echo "Step 3: Rule Summarization From Weights"
echo "=========================================="

cd "$PROJECT_ROOT"
python scripts/summarize_rules_from_weights.py \
    --data-root "$DATA_ROOT" \
    --scenarios "${SCENARIOS[@]}" \
    --output-dir "$RULES_DIR"

# ========================================
# Step 4: 生成 rule_kmeans 聚类结果
# ========================================
echo ""
echo "=========================================="
echo "Step 4: Generate rule_kmeans Clustering"
echo "=========================================="

cd "$PROJECT_ROOT"
python scripts/apply_rule_kmeans.py \
    --data-root "$DATA_ROOT" \
    --rules-dir "$RULES_DIR" \
    --scenarios "${SCENARIOS[@]}" \
    --start-seed "$START_SEED" \
    --n-seeds "$N_SEEDS"

# ========================================
# Step 5: 绘制 RL 训练收敛曲线
# ========================================
echo ""
echo "=========================================="
echo "Step 5: Plot RL Training Curves"
echo "=========================================="

cd "$PROJECT_ROOT"
python scripts/plot_rl_training.py \
    --data-root "$DATA_ROOT" \
    --scenarios "${SCENARIOS[@]}" \
    --output-dir "$PROJECT_ROOT/plot/figures"

echo ""
echo "=========================================="
echo "HtT Pipeline Complete!"
echo "=========================================="
echo "Results:"
echo "  - Bandit models: $DATA_ROOT/*/bandit_results/"
echo "  - Weight libraries: $DATA_ROOT/*/bandit_results/weight_library.json"
echo "  - Rule configs: $RULES_DIR/rule_kmeans_config_*.yaml"
echo "  - Clustering: $DATA_ROOT/*/run_seed_*/checkpoint_clusters_htt_recursive.csv"
echo "  - Training curves: $PROJECT_ROOT/plot/figures/rl_training_*.pdf"
