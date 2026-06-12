#!/usr/bin/env python3
"""
应用规则生成 rule_kmeans 聚类结果

读取 rule_kmeans_config.yaml，为每个 (scenario, seed) 生成聚类结果。
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rule_config(config_path: Path) -> dict:
    """加载规则配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_condition(condition: dict, context: dict) -> bool:
    """评估单个条件"""
    feature = condition.get("feature")
    op = condition.get("op")
    value = condition.get("value")
    
    if feature not in context:
        return False
    
    ctx_value = context[feature]
    
    if op == ">":
        return ctx_value > value
    elif op == ">=":
        return ctx_value >= value
    elif op == "<":
        return ctx_value < value
    elif op == "<=":
        return ctx_value <= value
    elif op == "==":
        return ctx_value == value
    else:
        return False


def select_rule(rules: list, context: dict) -> dict:
    """根据上下文选择最高优先级的匹配规则"""
    # 按优先级排序（降序）
    sorted_rules = sorted(rules, key=lambda r: r.get("priority", 0), reverse=True)
    
    for rule in sorted_rules:
        # 检查是否是默认规则
        if rule.get("default", False):
            continue  # 默认规则最后考虑
        
        # 检查所有条件
        conditions = rule.get("if", [])
        if not conditions:
            # 无条件规则，直接匹配
            return rule
        
        all_match = all(evaluate_condition(cond, context) for cond in conditions)
        if all_match:
            return rule
    
    # 返回默认规则
    for rule in sorted_rules:
        if rule.get("default", False):
            return rule
    
    # 如果没有默认规则，返回最后一个
    return sorted_rules[-1] if sorted_rules else None


def compute_context(df: pd.DataFrame) -> dict:
    """从数据计算场景上下文特征"""
    context = {}
    
    # 样本数量
    context["n_samples"] = len(df)
    
    # 空间方差
    if "avg_position_x" in df.columns:
        positions = df[["avg_position_x", "avg_position_y"]].values
    elif "location" in df.columns:
        import ast
        positions = []
        for loc in df["location"]:
            if isinstance(loc, str):
                try:
                    pos = ast.literal_eval(loc)
                    positions.append(pos[:2])
                except:
                    positions.append([0, 0])
            else:
                positions.append([0, 0])
        positions = np.array(positions)
    else:
        positions = np.zeros((len(df), 2))
    
    if len(positions) > 0:
        # 归一化后的方差
        pos_range = positions.max(axis=0) - positions.min(axis=0)
        pos_range = np.maximum(pos_range, 1e-6)
        normalized_pos = (positions - positions.min(axis=0)) / pos_range
        context["spatial_variance"] = float(np.mean(np.var(normalized_pos, axis=0)))
    else:
        context["spatial_variance"] = 0
    
    # 任务多样性
    if "operation_type" in df.columns:
        op_types = df["operation_type"].unique()
        context["task_diversity"] = len(op_types) / 8.0  # 假设最多8种操作类型
    elif "task_types" in df.columns:
        # task_types 可能是字符串列表
        all_types = set()
        for tt in df["task_types"]:
            if isinstance(tt, str):
                try:
                    types = eval(tt) if tt.startswith("[") else [tt]
                    all_types.update(types)
                except:
                    all_types.add(tt)
        context["task_diversity"] = min(len(all_types) / 8.0, 1.0)
    else:
        context["task_diversity"] = 0.5
    
    # 时间密度（如果有时间信息）
    if "start_time" in df.columns and "end_time" in df.columns:
        durations = df["end_time"] - df["start_time"]
        context["temporal_density"] = float(1.0 / (1 + np.std(durations)))
    else:
        context["temporal_density"] = 0.5
    
    return context


def extract_features(df: pd.DataFrame, feature_names: list) -> np.ndarray:
    """从 DataFrame 提取特征矩阵"""
    n_features = len(feature_names)
    features = np.zeros((len(df), n_features))
    
    for i, row in df.iterrows():
        feat_vec = np.zeros(n_features)
        
        # 位置特征
        if "location" in df.columns:
            try:
                import ast
                loc = ast.literal_eval(row["location"]) if isinstance(row["location"], str) else row["location"]
                feat_vec[0:3] = loc[:3] if len(loc) >= 3 else list(loc) + [0]*(3-len(loc))
            except:
                pass
        elif "avg_position_x" in df.columns:
            feat_vec[0] = row.get("avg_position_x", 0)
            feat_vec[1] = row.get("avg_position_y", 0)
            feat_vec[2] = row.get("avg_position_z", 0)
        
        # 操作类型 one-hot
        op_type = row.get("operation_type", "")
        op_map = {"pickup": 3, "transport": 4, "delivery": 5, "inspection": 6,
                  "compute": 7, "communication": 8, "completion": 9}
        if op_type in op_map:
            feat_vec[op_map[op_type]] = 1.0
        
        # 需求特征
        reqs = row.get("requirements", "")
        if isinstance(reqs, str) and reqs:
            if "mobility" in reqs:
                feat_vec[11] = 1.0
            if "logistics" in reqs:
                feat_vec[12] = 1.0
            if "inspection" in reqs:
                feat_vec[13] = 1.0
            if "communication" in reqs:
                feat_vec[14] = 1.0
            if "compute" in reqs:
                feat_vec[15] = 1.0
        
        # 资源特征
        feat_vec[16] = row.get("payload_weight", 0) or 0
        feat_vec[17] = row.get("compute_gflops", 0) or 0
        feat_vec[18] = row.get("cpu_intensity", 0) or 0
        feat_vec[19] = row.get("memory_intensity", 0) or 0
        feat_vec[20] = row.get("io_intensity", 0) or 0
        
        # QoS 特征
        feat_vec[21] = row.get("comm_bandwidth_mb", 0) or 0
        feat_vec[22] = row.get("max_latency_s", 0) or 0
        feat_vec[23] = row.get("quality_threshold", 0) or 0
        feat_vec[24] = row.get("inspection_interval", 0) or 0
        feat_vec[25] = row.get("n_inspection_points", 0) or 0
        
        features[df.index.get_loc(i)] = feat_vec
    
    return features


def apply_weighted_kmeans(features: np.ndarray, weights: list, n_clusters: int, 
                          random_state: int = 42) -> np.ndarray:
    """应用加权 KMeans"""
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 加权
    weights = np.array(weights)
    weights = np.clip(weights, 1e-5, 1.0)
    features_weighted = features_scaled * np.sqrt(weights)
    
    # 聚类
    n_clusters = min(n_clusters, len(features))
    if n_clusters <= 1:
        return np.zeros(len(features), dtype=int)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_weighted)
    
    return labels


def process_run(run_dir: Path, config: dict, seed: int) -> bool:
    """处理单个 run 目录"""
    # 查找输入文件
    input_candidates = [
        "checkpoint_clusters_kmeans.csv",
        "workflow_checkpoint_clustering.csv",
        "checkpoints_precluster.csv"
    ]
    
    input_csv = None
    for candidate in input_candidates:
        path = run_dir / candidate
        if path.exists():
            input_csv = path
            break
    
    if input_csv is None:
        logger.warning(f"No input CSV found in {run_dir}")
        return False
    
    # 输出文件
    output_csv = run_dir / "checkpoint_clusters_htt_recursive.csv"
    
    # 读取数据
    df = pd.read_csv(input_csv)
    
    if len(df) == 0:
        logger.warning(f"Empty DataFrame: {input_csv}")
        return False
    
    # 计算上下文
    context = compute_context(df)
    logger.info(f"  Context: n_samples={context['n_samples']}, "
                f"spatial_var={context['spatial_variance']:.3f}, "
                f"task_div={context['task_diversity']:.3f}")
    
    # 选择规则
    rules = config.get("rules", [])
    selected_rule = select_rule(rules, context)
    
    if selected_rule is None:
        logger.error(f"No matching rule found")
        return False
    
    logger.info(f"  Selected rule: {selected_rule['id']}")
    
    # 获取权重和簇数
    then_clause = selected_rule.get("then", {})
    weights = then_clause.get("weights", [0.7] * 26)
    n_clusters = then_clause.get("n_clusters_per_run", 30)
    
    # 提取特征
    feature_names = config.get("feature_names", [f"f{i}" for i in range(26)])
    features = extract_features(df, feature_names)
    
    # 应用加权 KMeans
    labels = apply_weighted_kmeans(features, weights, n_clusters, random_state=seed)
    
    # 保存结果
    df["cluster_group"] = labels
    df.to_csv(output_csv, index=False)
    
    logger.info(f"  ✓ Saved: {output_csv.name} ({len(df)} -> {len(np.unique(labels))} clusters)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='应用规则生成 rule_kmeans 聚类')
    parser.add_argument('--data-root', type=str, required=True,
                       help='multi_seed_runs 根目录')
    parser.add_argument('--rules-dir', type=str, required=True,
                       help='规则配置目录')
    parser.add_argument('--scenarios', type=str, nargs='+', required=True,
                       help='场景列表')
    parser.add_argument('--start-seed', type=int, default=42,
                       help='起始 seed')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='seed 数量')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    rules_dir = Path(args.rules_dir)
    
    print("="*60)
    print("📊 Apply Rule-KMeans Clustering")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Rules dir: {rules_dir}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.n_seeds - 1}")
    print()
    
    total_success = 0
    total_skip = 0
    
    for scenario in args.scenarios:
        print(f"\nProcessing: {scenario}")
        
        # 加载规则配置
        config_path = rules_dir / f"rule_kmeans_config_{scenario}.yaml"
        if not config_path.exists():
            # 尝试使用默认配置
            config_path = rules_dir / "rule_kmeans_config_default.yaml"
        
        if not config_path.exists():
            logger.warning(f"No rule config found for {scenario}")
            continue
        
        config = load_rule_config(config_path)
        print(f"  Using config: {config_path.name}")
        
        for seed in range(args.start_seed, args.start_seed + args.n_seeds):
            run_dir = data_root / scenario / f"run_seed_{seed}"
            
            if not run_dir.exists():
                logger.debug(f"  Skip: {run_dir} not found")
                total_skip += 1
                continue
            
            output_csv = run_dir / "checkpoint_clusters_htt_recursive.csv"
            if output_csv.exists():
                print(f"  ✓ Exists: seed_{seed}")
                total_success += 1
                continue
            
            print(f"  Processing: seed_{seed}")
            if process_run(run_dir, config, seed):
                total_success += 1
            else:
                total_skip += 1
    
    print()
    print("="*60)
    print(f"✓ Complete! Success: {total_success}, Skipped: {total_skip}")
    print("="*60)


if __name__ == "__main__":
    main()
