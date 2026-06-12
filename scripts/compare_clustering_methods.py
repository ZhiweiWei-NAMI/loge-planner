"""
对比不同聚类方法的离线计划质量

比较指标：
1. 聚类纯度 (Cluster Purity) - 同簇内任务的协同性
2. 平均簇内距离 (Intra-cluster Distance) - 空间紧凑性
3. 预估完成时间 (Estimated Completion Time) - 基于PDDL计划的动作数
4. Agent负载均衡 (Load Balance) - 任务分配的均匀程度
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import argparse


def load_clustering_result(cluster_file: Path) -> pd.DataFrame:
    """加载聚类结果"""
    return pd.read_csv(cluster_file)


def load_pddl_plan(plan_file: Path) -> Dict[str, Any]:
    """加载PDDL计划"""
    with open(plan_file, 'r') as f:
        return json.load(f)


def compute_cluster_purity(cluster_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算聚类纯度指标
    
    Returns:
        type_purity: 同簇内任务类型的纯度（主要类型占比）
        workflow_purity: 同簇内workflow的纯度
    """
    # 兼容不同列名
    cluster_col = 'cluster_group' if 'cluster_group' in cluster_df.columns else 'cluster_id'
    cluster_ids = cluster_df[cluster_col].unique()
    
    type_purities = []
    workflow_purities = []
    
    for cid in cluster_ids:
        cluster_mask = cluster_df[cluster_col] == cid
        cluster_data = cluster_df[cluster_mask]
        
        if len(cluster_data) == 0:
            continue
        
        # 任务类型纯度
        if 'operation_type' in cluster_data.columns:
            type_counts = cluster_data['operation_type'].value_counts()
            type_purity = type_counts.iloc[0] / len(cluster_data) if len(type_counts) > 0 else 1.0
            type_purities.append(type_purity)
        
        # Workflow纯度
        if 'workflow_id' in cluster_data.columns:
            wf_counts = cluster_data['workflow_id'].value_counts()
            wf_purity = wf_counts.iloc[0] / len(cluster_data) if len(wf_counts) > 0 else 1.0
            workflow_purities.append(wf_purity)
    
    return {
        'mean_type_purity': np.mean(type_purities) if type_purities else 0.0,
        'mean_workflow_purity': np.mean(workflow_purities) if workflow_purities else 0.0,
    }


def compute_spatial_compactness(cluster_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算空间紧凑性
    
    Returns:
        mean_intra_distance: 平均簇内距离
        max_intra_distance: 最大簇内距离
    """
    import ast
    
    cluster_col = 'cluster_group' if 'cluster_group' in cluster_df.columns else 'cluster_id'
    cluster_ids = cluster_df[cluster_col].unique()
    intra_distances = []
    
    for cid in cluster_ids:
        cluster_mask = cluster_df[cluster_col] == cid
        cluster_data = cluster_df[cluster_mask]
        
        if len(cluster_data) < 2:
            continue
        
        # 解析位置
        locations = []
        for loc_str in cluster_data['location']:
            try:
                loc = ast.literal_eval(str(loc_str))
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    locations.append([float(loc[0]), float(loc[1])])
            except:
                pass
        
        if len(locations) < 2:
            continue
        
        locations = np.array(locations)
        
        # 计算成对距离
        dists = []
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                dists.append(np.linalg.norm(locations[i] - locations[j]))
        
        if dists:
            intra_distances.append(np.mean(dists))
    
    return {
        'mean_intra_distance': np.mean(intra_distances) if intra_distances else 0.0,
        'std_intra_distance': np.std(intra_distances) if intra_distances else 0.0,
    }


def compute_plan_metrics(plan: Dict[str, Any]) -> Dict[str, float]:
    """
    从PDDL计划计算执行效率指标
    
    Returns:
        total_actions: 总动作数
        avg_actions_per_agent: 平均每个agent的动作数
        max_actions_per_agent: 最大agent动作数（影响整体完成时间）
        load_balance: 负载均衡度（0-1，1为完全均衡）
    """
    # 兼容不同的计划格式
    agents_plans = plan.get('agents_plans', plan.get('agents', {}))
    
    if not agents_plans:
        return {
            'total_actions': 0,
            'avg_actions_per_agent': 0,
            'max_actions_per_agent': 0,
            'load_balance': 0,
            'active_agents': 0,
        }
    
    action_counts = []
    for agent_id, agent_plan in agents_plans.items():
        # 尝试多种格式: actions, result.solution
        actions = agent_plan.get('actions', [])
        if not actions and 'result' in agent_plan:
            result = agent_plan['result']
            if isinstance(result, dict):
                actions = result.get('solution', [])
        action_counts.append(len(actions))
    
    total_actions = sum(action_counts)
    avg_actions = np.mean(action_counts) if action_counts else 0
    max_actions = max(action_counts) if action_counts else 0
    
    # 负载均衡度：1 - (std / mean)，归一化到0-1
    if avg_actions > 0:
        load_balance = 1.0 - min(np.std(action_counts) / avg_actions, 1.0)
    else:
        load_balance = 0
    
    return {
        'total_actions': total_actions,
        'avg_actions_per_agent': avg_actions,
        'max_actions_per_agent': max_actions,
        'load_balance': load_balance,
        'active_agents': len([c for c in action_counts if c > 0]),
    }


def compare_methods(
    kmeans_cluster_file: Path,
    htt_cluster_file: Path,
    kmeans_plan_file: Path,
    htt_plan_file: Path,
) -> Dict[str, Any]:
    """对比两种方法"""
    
    results = {}
    
    # 加载数据
    kmeans_clusters = load_clustering_result(kmeans_cluster_file)
    htt_clusters = load_clustering_result(htt_cluster_file)
    kmeans_plan = load_pddl_plan(kmeans_plan_file)
    htt_plan = load_pddl_plan(htt_plan_file)
    
    # 聚类纯度
    kmeans_purity = compute_cluster_purity(kmeans_clusters)
    htt_purity = compute_cluster_purity(htt_clusters)
    
    results['cluster_purity'] = {
        'kmeans': kmeans_purity,
        'htt': htt_purity,
        'improvement': {
            'type_purity': htt_purity['mean_type_purity'] - kmeans_purity['mean_type_purity'],
            'workflow_purity': htt_purity['mean_workflow_purity'] - kmeans_purity['mean_workflow_purity'],
        }
    }
    
    # 空间紧凑性
    kmeans_spatial = compute_spatial_compactness(kmeans_clusters)
    htt_spatial = compute_spatial_compactness(htt_clusters)
    
    results['spatial_compactness'] = {
        'kmeans': kmeans_spatial,
        'htt': htt_spatial,
        'improvement': {
            'mean_intra_distance': kmeans_spatial['mean_intra_distance'] - htt_spatial['mean_intra_distance'],
        }
    }
    
    # 计划效率
    kmeans_plan_metrics = compute_plan_metrics(kmeans_plan)
    htt_plan_metrics = compute_plan_metrics(htt_plan)
    
    results['plan_efficiency'] = {
        'kmeans': kmeans_plan_metrics,
        'htt': htt_plan_metrics,
        'improvement': {
            'total_actions': kmeans_plan_metrics['total_actions'] - htt_plan_metrics['total_actions'],
            'max_actions': kmeans_plan_metrics['max_actions_per_agent'] - htt_plan_metrics['max_actions_per_agent'],
            'load_balance': htt_plan_metrics['load_balance'] - kmeans_plan_metrics['load_balance'],
        }
    }
    
    return results


def print_comparison(results: Dict[str, Any]):
    """打印对比结果"""
    print("\n" + "="*70)
    print("📊 KMeans vs HtT 聚类方法对比")
    print("="*70)
    
    # 聚类纯度
    print("\n🎯 1. 聚类纯度 (Cluster Purity)")
    print("-" * 50)
    purity = results['cluster_purity']
    print(f"   {'指标':<25} {'KMeans':<12} {'HtT':<12} {'提升':<12}")
    print(f"   {'类型纯度':<25} {purity['kmeans']['mean_type_purity']:.4f}       {purity['htt']['mean_type_purity']:.4f}       {purity['improvement']['type_purity']:+.4f}")
    print(f"   {'Workflow纯度':<25} {purity['kmeans']['mean_workflow_purity']:.4f}       {purity['htt']['mean_workflow_purity']:.4f}       {purity['improvement']['workflow_purity']:+.4f}")
    
    # 空间紧凑性
    print("\n📍 2. 空间紧凑性 (Spatial Compactness)")
    print("-" * 50)
    spatial = results['spatial_compactness']
    print(f"   {'指标':<25} {'KMeans':<12} {'HtT':<12} {'提升':<12}")
    print(f"   {'平均簇内距离':<25} {spatial['kmeans']['mean_intra_distance']:.2f}       {spatial['htt']['mean_intra_distance']:.2f}       {spatial['improvement']['mean_intra_distance']:+.2f}")
    
    # 计划效率
    print("\n⚡ 3. 计划效率 (Plan Efficiency)")
    print("-" * 50)
    plan = results['plan_efficiency']
    print(f"   {'指标':<25} {'KMeans':<12} {'HtT':<12} {'提升':<12}")
    print(f"   {'总动作数':<25} {plan['kmeans']['total_actions']:<12} {plan['htt']['total_actions']:<12} {plan['improvement']['total_actions']:+d}")
    print(f"   {'最大Agent动作数':<25} {plan['kmeans']['max_actions_per_agent']:<12} {plan['htt']['max_actions_per_agent']:<12} {plan['improvement']['max_actions']:+.0f}")
    print(f"   {'负载均衡度':<25} {plan['kmeans']['load_balance']:.4f}       {plan['htt']['load_balance']:.4f}       {plan['improvement']['load_balance']:+.4f}")
    print(f"   {'活跃Agent数':<25} {plan['kmeans']['active_agents']:<12} {plan['htt']['active_agents']:<12}")
    
    # 总结
    print("\n" + "="*70)
    print("📋 总结")
    print("="*70)
    
    wins = 0
    if purity['improvement']['type_purity'] > 0:
        wins += 1
        print("   ✅ HtT 类型纯度更高 → 同簇任务更协同")
    else:
        print("   ❌ KMeans 类型纯度更高")
    
    if spatial['improvement']['mean_intra_distance'] > 0:
        wins += 1
        print("   ✅ HtT 空间更紧凑 → Agent移动距离更短")
    else:
        print("   ❌ KMeans 空间更紧凑")
    
    if plan['improvement']['total_actions'] > 0:
        wins += 1
        print("   ✅ HtT 总动作数更少 → 执行效率更高")
    else:
        print("   ❌ KMeans 总动作数更少")
    
    if plan['improvement']['max_actions'] > 0:
        wins += 1
        print("   ✅ HtT 最大Agent负载更低 → 预期完成时间更短")
    else:
        print("   ❌ KMeans 最大Agent负载更低")
    
    print(f"\n   HtT 在 {wins}/4 项指标上优于 KMeans")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="对比 KMeans vs HtT 聚类方法")
    parser.add_argument("--seed", type=int, required=True, help="Seed number")
    parser.add_argument("--kmeans-root", type=str, required=True, help="KMeans output root")
    parser.add_argument("--htt-root", type=str, required=True, help="HtT output root")
    parser.add_argument("--cluster-root", type=str, required=True, help="Cluster files root (Stage1 output)")
    args = parser.parse_args()
    
    cluster_root = Path(args.cluster_root) / f"run_seed_{args.seed}"
    kmeans_root = Path(args.kmeans_root) / f"run_seed_{args.seed}"
    htt_root = Path(args.htt_root) / f"run_seed_{args.seed}"
    
    results = compare_methods(
        kmeans_cluster_file=cluster_root / "checkpoint_clusters_kmeans.csv",
        htt_cluster_file=cluster_root / "checkpoint_clusters_rule_kmeans.csv",
        kmeans_plan_file=kmeans_root / "pddl_plan_kmeans.json",
        htt_plan_file=htt_root / "pddl_plan_rule_kmeans.json",
    )
    
    print_comparison(results)
    
    # 保存结果
    output_file = Path(args.htt_root) / f"comparison_seed_{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 详细结果已保存: {output_file}")


if __name__ == "__main__":
    main()
