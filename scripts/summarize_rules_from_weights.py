#!/usr/bin/env python3
"""
Rule summarizer from RL weight libraries.

从 RL 训练的 weight_library.json 中提取规则，生成人类可读的
rule_kmeans_config.yaml 文件。

工作流程:
1. 读取 weight_library.json (包含递归收集的权重和场景特征)
2. 分析权重模式，识别主导特征组
3. 生成条件规则 (if-then)
4. 输出 YAML 配置文件
"""

import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


# 特征名称和分组
FEATURE_NAMES = [
    "location_x", "location_y", "location_z",  # spatial: 0-2
    "op_pickup", "op_transport", "op_delivery", "op_inspection",  # op_type: 3-6
    "op_compute", "op_communication", "op_completion", "op_empty",  # op_type: 7-10
    "req_mobility", "req_logistics", "req_inspection", "req_communication", "req_compute",  # req: 11-15
    "payload_weight", "compute_gflops", "cpu_intensity", "memory_intensity", "io_intensity",  # resource: 16-20
    "comm_bandwidth_mb", "max_latency_s", "quality_threshold", "inspection_interval", "n_inspection_points"  # qos: 21-25
]

WEIGHT_MAPPING = {
    "spatial": [0, 1, 2],
    "operation_type": [3, 4, 5, 6, 7, 8, 9, 10],
    "requirement": [11, 12, 13, 14, 15],
    "resource": [16, 17, 18, 19, 20],
    "qos": [21, 22, 23, 24, 25]
}

FEATURES_META = [
    {"name": "spatial_variance", "source": "S_ctx.geo_var"},
    {"name": "temporal_density", "source": "S_ctx.time_var"},
    {"name": "task_diversity", "source": "S_ctx.task_diversity"},
    {"name": "n_samples", "source": "S_ctx.n_samples"}
]


def analyze_weights(weights: List[float]) -> Dict[str, float]:
    """分析权重，计算各组的平均权重"""
    weights = np.array(weights)
    group_means = {}
    for group_name, indices in WEIGHT_MAPPING.items():
        group_means[group_name] = float(np.mean(weights[indices]))
    return group_means


def identify_dominant_groups(weights: List[float], threshold: float = 0.7) -> List[str]:
    """识别权重高于阈值的主导特征组"""
    group_means = analyze_weights(weights)
    dominant = [name for name, mean in group_means.items() if mean >= threshold]
    return dominant


def generate_rule_from_weight_library(weight_lib: Dict, scenario: str) -> Dict:
    """从 weight_library 生成规则配置"""
    
    rules = []
    
    # 如果有递归收集的权重树
    if "weight_tree" in weight_lib:
        tree = weight_lib["weight_tree"]
        rules.extend(extract_rules_from_tree(tree, priority=10))
    
    # 如果有全局平均权重
    if "global_weights" in weight_lib:
        global_w = weight_lib["global_weights"]
        rules.append({
            "id": "R_global_average",
            "priority": 5,
            "if": [],  # 无条件
            "then": {
                "weights": [round(w, 2) for w in global_w],
                "n_clusters_per_run": 30
            },
            "notes": "全局平均权重，从RL策略学习得到"
        })
    
    # 如果没有任何权重信息，使用默认规则
    if not rules:
        rules = generate_scenario_specific_default_rules(scenario)
    
    # 添加默认回退规则
    rules.append({
        "id": "R_balanced_default",
        "priority": 0,
        "default": True,
        "then": {
            "weights": [0.7] * 26,
            "n_clusters_per_run": 30
        },
        "notes": "默认平衡规则，适用于一般场景"
    })
    
    return {
        "metadata": {
            "scenario": scenario,
            "version": 1,
            "source": "rl_weight_summarizer",
            "n_features": 26
        },
        "features": FEATURES_META,
        "weight_mapping": WEIGHT_MAPPING,
        "feature_names": FEATURE_NAMES,
        "rules": rules
    }


def extract_rules_from_tree(tree: Dict, priority: int = 10, path: List[str] = None) -> List[Dict]:
    """从权重树中递归提取规则"""
    if path is None:
        path = []
    
    rules = []
    
    if "weights" in tree:
        weights = tree["weights"]
        condition = tree.get("condition", {})
        
        # 构建条件
        if_conditions = []
        if "spatial_variance" in condition:
            op, val = condition["spatial_variance"]
            if_conditions.append({
                "feature": "spatial_variance",
                "op": op,
                "value": val
            })
        if "task_diversity" in condition:
            op, val = condition["task_diversity"]
            if_conditions.append({
                "feature": "task_diversity",
                "op": op,
                "value": val
            })
        
        # 识别主导特征组
        dominant = identify_dominant_groups(weights)
        notes = f"主导特征组: {', '.join(dominant)}" if dominant else "均衡权重"
        
        rule_id = f"R_{'_'.join(path)}" if path else f"R_node_{priority}"
        rules.append({
            "id": rule_id,
            "priority": priority,
            "if": if_conditions,
            "then": {
                "weights": [round(w, 2) for w in weights],
                "n_clusters_per_run": tree.get("n_clusters", 30)
            },
            "notes": notes
        })
    
    # 递归处理子节点
    for key, child in tree.items():
        if isinstance(child, dict) and key not in ["weights", "condition", "n_clusters"]:
            child_rules = extract_rules_from_tree(
                child, 
                priority=priority - 1, 
                path=path + [key]
            )
            rules.extend(child_rules)
    
    return rules


def generate_scenario_specific_default_rules(scenario: str) -> List[Dict]:
    """
    根据场景名称和任务结构生成可解释的启发式规则。
    """
    
    rules = []
    
    # 基于场景名称的语义分析
    if "logistics" in scenario:
        # 物流场景：强调空间位置和运输类型
        rules.append({
            "id": "R_logistics_spatial",
            "priority": 10,
            "if": [
                {"feature": "spatial_variance", "op": ">", "value": 0.15}
            ],
            "then": {
                "weights": [
                    0.85, 0.85, 0.7,  # spatial (高权重)
                    0.9, 0.9, 0.9, 0.4, 0.3, 0.3, 0.3, 0.2,  # op_type (物流相关高)
                    0.7, 0.9, 0.4, 0.4, 0.3,  # requirement
                    0.8, 0.3, 0.3, 0.3, 0.3,  # resource (payload高)
                    0.5, 0.6, 0.4, 0.4, 0.4   # qos
                ],
                "n_clusters_per_run": 30
            },
            "notes": "物流场景：空间分散时，强调位置和运输任务类型"
        })
        rules.append({
            "id": "R_logistics_compact",
            "priority": 8,
            "if": [
                {"feature": "spatial_variance", "op": "<=", "value": 0.15}
            ],
            "then": {
                "weights": [
                    0.6, 0.6, 0.5,  # spatial (较低)
                    0.85, 0.85, 0.85, 0.4, 0.3, 0.3, 0.3, 0.2,  # op_type
                    0.6, 0.8, 0.4, 0.4, 0.3,  # requirement
                    0.75, 0.3, 0.3, 0.3, 0.3,  # resource
                    0.5, 0.55, 0.4, 0.4, 0.4   # qos
                ],
                "n_clusters_per_run": 25
            },
            "notes": "物流场景：空间紧凑时，更多按任务类型聚类"
        })
    
    if "inspection" in scenario:
        # 巡检场景：强调巡检相关特征
        rules.append({
            "id": "R_inspection_heavy",
            "priority": 10,
            "if": [
                {"feature": "task_diversity", "op": "<", "value": 0.3}
            ],
            "then": {
                "weights": [
                    0.7, 0.7, 0.6,  # spatial
                    0.4, 0.4, 0.4, 0.95, 0.3, 0.3, 0.3, 0.2,  # op_type (inspection高)
                    0.5, 0.5, 0.95, 0.4, 0.3,  # requirement (inspection高)
                    0.4, 0.3, 0.3, 0.3, 0.3,  # resource
                    0.5, 0.5, 0.9, 0.9, 0.9   # qos (质量相关高)
                ],
                "n_clusters_per_run": 35
            },
            "notes": "巡检为主场景：强调巡检任务和质量要求"
        })
    
    if "comm" in scenario or "compute" in scenario:
        # 通信/计算场景：强调计算和通信特征
        rules.append({
            "id": "R_compute_comm",
            "priority": 10,
            "if": [],
            "then": {
                "weights": [
                    0.5, 0.5, 0.4,  # spatial (较低)
                    0.3, 0.3, 0.3, 0.4, 0.9, 0.9, 0.3, 0.2,  # op_type (compute/comm高)
                    0.4, 0.4, 0.4, 0.9, 0.9,  # requirement (comm/compute高)
                    0.5, 0.9, 0.9, 0.8, 0.8,  # resource (计算资源高)
                    0.9, 0.9, 0.5, 0.4, 0.4   # qos (延迟和带宽高)
                ],
                "n_clusters_per_run": 25
            },
            "notes": "通信/计算场景：强调计算和通信资源需求"
        })
    
    if "mixed" in scenario or "large" in scenario:
        # 混合/大规模场景：均衡但稍强调空间
        rules.append({
            "id": "R_mixed_balanced",
            "priority": 10,
            "if": [
                {"feature": "n_samples", "op": ">", "value": 200}
            ],
            "then": {
                "weights": [
                    0.8, 0.8, 0.7,  # spatial
                    0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.4, 0.3,  # op_type
                    0.6, 0.7, 0.7, 0.6, 0.6,  # requirement
                    0.6, 0.5, 0.5, 0.5, 0.5,  # resource
                    0.6, 0.6, 0.6, 0.6, 0.6   # qos
                ],
                "n_clusters_per_run": 40
            },
            "notes": "大规模混合场景：均衡权重，增加簇数"
        })
        rules.append({
            "id": "R_mixed_small",
            "priority": 8,
            "if": [
                {"feature": "n_samples", "op": "<=", "value": 200}
            ],
            "then": {
                "weights": [
                    0.75, 0.75, 0.65,  # spatial
                    0.65, 0.65, 0.65, 0.65, 0.55, 0.55, 0.4, 0.3,  # op_type
                    0.55, 0.65, 0.65, 0.55, 0.55,  # requirement
                    0.55, 0.45, 0.45, 0.45, 0.45,  # resource
                    0.55, 0.55, 0.55, 0.55, 0.55   # qos
                ],
                "n_clusters_per_run": 30
            },
            "notes": "中小规模混合场景：均衡权重"
        })
    
    # 如果没有匹配的场景规则，生成通用规则
    if not rules:
        rules.append({
            "id": "R_generic_spatial",
            "priority": 10,
            "if": [
                {"feature": "spatial_variance", "op": ">", "value": 0.1}
            ],
            "then": {
                "weights": [0.8, 0.8, 0.7] + [0.6] * 8 + [0.5] * 5 + [0.5] * 5 + [0.5] * 5,
                "n_clusters_per_run": 30
            },
            "notes": "通用规则：空间分散时强调位置"
        })
    
    return rules


def summarize_rules_for_scenario(scenario: str, data_root: Path, output_dir: Path):
    """为单个场景生成规则配置"""
    
    weight_lib_path = data_root / scenario / "bandit_results" / "weight_library.json"
    
    # 尝试读取 weight_library
    weight_lib = {}
    if weight_lib_path.exists():
        with open(weight_lib_path, 'r') as f:
            weight_lib = json.load(f)
        print(f"  ✓ Loaded weight_library for {scenario}")
    else:
        print(f"  No weight_library for {scenario}, using scenario-derived heuristic rules")
    
    # 生成规则配置
    config = generate_rule_from_weight_library(weight_lib, scenario)
    
    # 保存 YAML
    output_path = output_dir / f"rule_kmeans_config_{scenario}.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  ✓ Saved: {output_path}")
    print(f"    Rules: {len(config['rules'])}")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Rule summarizer from RL weight libraries')
    parser.add_argument('--data-root', type=str, required=True,
                       help='multi_seed_runs 根目录')
    parser.add_argument('--scenarios', type=str, nargs='+', required=True,
                       help='场景列表')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='规则输出目录')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Rule Summarizer From Weights")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Scenarios: {args.scenarios}")
    print()
    
    for scenario in args.scenarios:
        print(f"Processing: {scenario}")
        summarize_rules_for_scenario(scenario, data_root, output_dir)
        print()
    
    print("="*60)
    print("✓ Rule summarization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
