"""
Rule Applicator - 应用挖掘的规则进行Checkpoint到Cluster的分配

核心功能：
1. 分析高性能clusters满足哪些规则
2. 根据场景配置动态调整规则权重
3. 提供 predict(checkpoint, clusters, rules) -> best_cluster 函数
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
RULES_FILE = Path(__file__).parent / 'rules' / 'cluster_assignment_rules.json'


# =============================================================================
# Part 1: 分析高性能Clusters满足的规则
# =============================================================================

def analyze_rule_satisfaction_by_performance():
    """
    分析不同性能等级的clusters满足各规则的比例
    """
    from cluster_rule_mining import align_data_sources, compute_synergy_metrics, design_new_metrics
    
    print("="*70)
    print("高性能 Clusters 规则满足度分析")
    print("="*70)
    
    # 加载规则
    with open(RULES_FILE, 'r') as f:
        rules_data = json.load(f)
    rules = rules_data['rules']
    
    # 加载数据
    aligned_df = align_data_sources()
    df = compute_synergy_metrics(aligned_df)
    df = design_new_metrics(df)
    
    # 按completion_rate分组
    percentiles = df['completion_rate'].quantile([0.25, 0.5, 0.75])
    
    def get_performance_tier(rate):
        if rate >= percentiles[0.75]:
            return 'Top 25%'
        elif rate >= percentiles[0.5]:
            return 'Top 50%'
        elif rate >= percentiles[0.25]:
            return 'Bottom 50%'
        else:
            return 'Bottom 25%'
    
    df['performance_tier'] = df['completion_rate'].apply(get_performance_tier)
    
    # 计算每个tier的规则满足率
    print("\n各性能层级的规则满足比例:")
    print("-"*70)
    
    tier_rule_satisfaction = {}
    
    for tier in ['Top 25%', 'Top 50%', 'Bottom 50%', 'Bottom 25%']:
        tier_df = df[df['performance_tier'] == tier]
        if len(tier_df) == 0:
            continue
        
        tier_rule_satisfaction[tier] = {}
        
        for rule in rules:
            feat = rule['feature_name']
            if feat not in df.columns:
                continue
            
            threshold = rule.get('threshold')
            operator = rule.get('operator')
            
            if threshold is None:
                # 相关性规则 - 使用中位数
                threshold = df[feat].median()
                operator = '>=' if operator == 'maximize' else '<='
            
            if operator in ['>=', 'maximize']:
                satisfied = (tier_df[feat] >= threshold).mean()
            else:
                satisfied = (tier_df[feat] <= threshold).mean()
            
            tier_rule_satisfaction[tier][rule['rule_id']] = satisfied
    
    # 输出表格
    print(f"\n{'Rule ID':<10} {'Feature':<25} {'Top 25%':>10} {'Top 50%':>10} {'Bot 50%':>10} {'Bot 25%':>10}")
    print("-"*70)
    
    for rule in rules:
        rid = rule['rule_id']
        feat = rule['feature_name'][:24]
        
        top25 = tier_rule_satisfaction.get('Top 25%', {}).get(rid, 0)
        top50 = tier_rule_satisfaction.get('Top 50%', {}).get(rid, 0)
        bot50 = tier_rule_satisfaction.get('Bottom 50%', {}).get(rid, 0)
        bot25 = tier_rule_satisfaction.get('Bottom 25%', {}).get(rid, 0)
        
        print(f"{rid:<10} {feat:<25} {top25:>9.1%} {top50:>10.1%} {bot50:>10.1%} {bot25:>10.1%}")
    
    # 关键发现
    print("\n" + "="*70)
    print("关键发现 - Top 25% vs Bottom 25% 差异最大的规则:")
    print("-"*70)
    
    diffs = []
    for rule in rules:
        rid = rule['rule_id']
        top25 = tier_rule_satisfaction.get('Top 25%', {}).get(rid, 0)
        bot25 = tier_rule_satisfaction.get('Bottom 25%', {}).get(rid, 0)
        diffs.append((rid, rule['feature_name'], top25 - bot25, top25, bot25))
    
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for rid, feat, diff, top, bot in diffs[:5]:
        print(f"  {rid}: {feat}")
        print(f"       Top 25%: {top:.1%}  |  Bottom 25%: {bot:.1%}  |  Δ = {diff:+.1%}")
    
    return tier_rule_satisfaction


# =============================================================================
# Part 2: 场景感知的规则权重调整
# =============================================================================

@dataclass
class ScenarioConfig:
    """场景配置"""
    name: str
    n_agents: int
    n_workflows: int
    dominant_task_type: str  # 'logistics', 'inspection', 'compute', 'mixed'
    spatial_area_km2: float
    time_horizon_s: float
    
    @classmethod
    def from_scenario_name(cls, scenario_name: str) -> 'ScenarioConfig':
        """从场景名称推断配置"""
        configs = {
            'large_mixed': cls('large_mixed', 50, 150, 'mixed', 25.0, 1800),
            'medium_inspection_heavy': cls('medium_inspection_heavy', 40, 80, 'inspection', 16.0, 1800),
            'medium_logistics_heavy': cls('medium_logistics_heavy', 40, 80, 'logistics', 16.0, 1800),
            'small_comm_compute': cls('small_comm_compute', 20, 30, 'compute', 9.0, 1200),
            'small_logistics_inspection': cls('small_logistics_inspection', 20, 30, 'mixed', 9.0, 1200),
        }
        return configs.get(scenario_name, cls(scenario_name, 30, 50, 'mixed', 16.0, 1800))


class ScenarioAwareRuleWeightAdjuster:
    """
    根据场景配置动态调整规则权重
    
    核心思想：
    - 大规模场景：更重视cluster size限制（避免规划超时）
    - 物流密集：更重视spatial compactness（减少飞行距离）
    - 检测密集：更重视type homogeneity（检测任务需要连续性）
    - 计算密集：更重视cluster density（资源利用率）
    """
    
    # 基础权重（从规则挖掘结果）
    BASE_WEIGHTS = {
        'n_checkpoints': -0.534,
        'spatial_spread': -1.0,
        'mean_pairwise_dist': -1.0,
        'type_homogeneity': 0.966,
        'type_entropy': -0.62,
        'cluster_density': 1.0,
        'synergy_score': 1.0,
        'payload_mean': 1.0,
        'type_synergy': 0.619,
    }
    
    # 场景调整因子
    SCENARIO_ADJUSTMENTS = {
        'large_mixed': {
            'n_checkpoints': 1.5,      # 大场景更需要限制cluster size
            'spatial_spread': 1.2,     # 空间分布更重要
            'type_homogeneity': 0.8,   # 混合场景允许更多多样性
        },
        'medium_inspection_heavy': {
            'type_homogeneity': 1.5,   # 检测任务需要连续性
            'spatial_spread': 1.3,     # 路径连续性
            'mean_pairwise_dist': 1.2,
        },
        'medium_logistics_heavy': {
            'spatial_spread': 1.5,     # 物流最关心距离
            'payload_mean': 1.3,       # 载荷匹配
            'cluster_density': 1.2,
        },
        'small_comm_compute': {
            'cluster_density': 1.5,    # 计算任务关心密度
            'type_homogeneity': 1.2,
            'n_checkpoints': 0.8,      # 小场景cluster size不是瓶颈
        },
        'small_logistics_inspection': {
            'type_synergy': 1.3,       # 混合任务需要协同
            'spatial_spread': 1.2,
        },
    }
    
    def __init__(self, scenario_config: ScenarioConfig):
        self.config = scenario_config
        self.weights = self._compute_adjusted_weights()
    
    def _compute_adjusted_weights(self) -> Dict[str, float]:
        """计算场景调整后的权重"""
        weights = self.BASE_WEIGHTS.copy()
        
        # 应用场景特定调整
        adjustments = self.SCENARIO_ADJUSTMENTS.get(self.config.name, {})
        for feat, factor in adjustments.items():
            if feat in weights:
                weights[feat] *= factor
        
        # 基于场景规模的动态调整
        scale_factor = self.config.n_workflows / 80  # 80 as baseline
        
        if scale_factor > 1.2:  # 大规模场景
            weights['n_checkpoints'] *= 1.3
            weights['spatial_spread'] *= 1.1
        elif scale_factor < 0.5:  # 小规模场景
            weights['n_checkpoints'] *= 0.7
            weights['cluster_density'] *= 1.2
        
        # 归一化权重
        max_abs = max(abs(v) for v in weights.values())
        weights = {k: v / max_abs for k, v in weights.items()}
        
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights


# =============================================================================
# Part 3: Checkpoint到Cluster分配函数
# =============================================================================

class ClusterAssignmentPredictor:
    """
    核心分配器：predict(checkpoint, clusters, rules) -> best_cluster
    """
    
    def __init__(self, rules_file: Path = RULES_FILE, scenario_name: str = None):
        """
        初始化预测器
        
        Args:
            rules_file: 规则JSON文件路径
            scenario_name: 场景名称（用于权重调整）
        """
        with open(rules_file, 'r') as f:
            rules_data = json.load(f)
        
        self.rules = rules_data['rules']
        self.scoring_function = rules_data['scoring_function']
        self.normalization = self.scoring_function['normalization']
        
        # 场景感知权重
        if scenario_name:
            config = ScenarioConfig.from_scenario_name(scenario_name)
            adjuster = ScenarioAwareRuleWeightAdjuster(config)
            self.weights = adjuster.get_weights()
        else:
            self.weights = self.scoring_function['weights']
    
    def compute_cluster_features(self, cluster_checkpoints: List[Dict]) -> Dict[str, float]:
        """
        计算一个cluster的特征
        
        Args:
            cluster_checkpoints: cluster中现有checkpoints的列表
                每个checkpoint: {'location': (x,y,z), 'workflow_type': str, 'payload_weight': float, ...}
        
        Returns:
            特征字典
        """
        n = len(cluster_checkpoints)
        if n == 0:
            return self._empty_features()
        
        # 提取位置
        locations = np.array([cp['location'] for cp in cluster_checkpoints])
        xs, ys, zs = locations[:, 0], locations[:, 1], locations[:, 2]
        
        # 空间统计
        spatial_spread = np.sqrt(np.var(xs) + np.var(ys) + np.var(zs))
        
        if n > 1:
            from scipy.spatial.distance import pdist
            pairwise_dists = pdist(locations)
            mean_pairwise_dist = np.mean(pairwise_dists)
            max_pairwise_dist = np.max(pairwise_dists)
        else:
            mean_pairwise_dist = 0
            max_pairwise_dist = 0
        
        # 任务类型统计
        types = [cp.get('workflow_type', 'Unknown') for cp in cluster_checkpoints]
        type_counts = pd.Series(types).value_counts()
        type_homogeneity = type_counts.iloc[0] / n if len(type_counts) > 0 else 1.0
        
        from scipy.stats import entropy
        type_entropy = entropy(type_counts.values) if len(type_counts) > 1 else 0
        
        # Payload统计
        payloads = [cp.get('payload_weight', 0) for cp in cluster_checkpoints]
        payload_mean = np.mean(payloads)
        
        # 密度
        cluster_density = n / max(spatial_spread ** 2, 1.0)
        
        # Synergy score
        spatial_compactness = 1.0 / (1.0 + spatial_spread / 1000.0)
        synergy_score = (spatial_compactness + type_homogeneity) / 2.0
        
        return {
            'n_checkpoints': n,
            'spatial_spread': spatial_spread,
            'mean_pairwise_dist': mean_pairwise_dist,
            'max_pairwise_dist': max_pairwise_dist,
            'type_homogeneity': type_homogeneity,
            'type_entropy': type_entropy,
            'type_synergy': type_homogeneity,
            'payload_mean': payload_mean,
            'cluster_density': cluster_density,
            'synergy_score': synergy_score,
            'centroid': (np.mean(xs), np.mean(ys), np.mean(zs)),
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """空cluster的特征"""
        return {
            'n_checkpoints': 0,
            'spatial_spread': 0,
            'mean_pairwise_dist': 0,
            'max_pairwise_dist': 0,
            'type_homogeneity': 1.0,
            'type_entropy': 0,
            'type_synergy': 1.0,
            'payload_mean': 0,
            'cluster_density': 0,
            'synergy_score': 1.0,
            'centroid': (0, 0, 0),
        }
    
    def compute_assignment_score(self, 
                                   checkpoint: Dict, 
                                   cluster_checkpoints: List[Dict]) -> float:
        """
        计算将checkpoint加入cluster后的质量分数
        
        Args:
            checkpoint: 待分配的checkpoint {'location': (x,y,z), 'workflow_type': str, ...}
            cluster_checkpoints: cluster现有的checkpoints列表
        
        Returns:
            分数（越高越好）
        """
        # 模拟将checkpoint加入cluster后的状态
        new_cluster = cluster_checkpoints + [checkpoint]
        features_after = self.compute_cluster_features(new_cluster)
        
        # 计算归一化分数
        score = 0.0
        for feat, weight in self.weights.items():
            if feat not in features_after or feat not in self.normalization:
                continue
            
            value = features_after[feat]
            norm = self.normalization[feat]
            
            # Z-score归一化
            std = norm['std'] if norm['std'] > 0 else 1.0
            normalized = (value - norm['mean']) / std
            
            score += weight * normalized
        
        # 添加距离惩罚（新checkpoint与centroid的距离）
        if cluster_checkpoints:
            centroid = features_after['centroid']
            ckpt_loc = np.array(checkpoint['location'])
            centroid_arr = np.array(centroid)
            distance = np.linalg.norm(ckpt_loc - centroid_arr)
            
            # 距离惩罚（归一化）
            distance_penalty = distance / 1000.0  # 1000m为基准
            score -= 0.3 * distance_penalty  # 距离权重
        
        return score
    
    def predict(self, 
                checkpoint: Dict, 
                clusters: Dict[int, List[Dict]],
                allow_new_cluster: bool = True) -> Tuple[int, float, Dict]:
        """
        预测checkpoint应该加入哪个cluster
        
        Args:
            checkpoint: 待分配的checkpoint
                {'location': (x, y, z), 'workflow_type': str, 'payload_weight': float, ...}
            clusters: 现有clusters {cluster_id: [checkpoint1, checkpoint2, ...]}
            allow_new_cluster: 是否允许创建新cluster
        
        Returns:
            (best_cluster_id, score, details)
        """
        scores = {}
        details = {}
        
        for cluster_id, cluster_checkpoints in clusters.items():
            score = self.compute_assignment_score(checkpoint, cluster_checkpoints)
            scores[cluster_id] = score
            
            # 检查规则满足情况
            new_cluster = cluster_checkpoints + [checkpoint]
            features = self.compute_cluster_features(new_cluster)
            
            rules_satisfied = self._check_rules(features)
            
            details[cluster_id] = {
                'score': score,
                'features': {k: v for k, v in features.items() if k != 'centroid'},
                'rules_satisfied': rules_satisfied,
                'satisfaction_rate': sum(rules_satisfied.values()) / len(rules_satisfied)
            }
        
        if not scores:
            # 没有现有cluster，需要创建新的
            return -1, 0.0, {'message': 'No existing clusters, create new'}
        
        # 选择分数最高的cluster
        best_cluster_id = max(scores, key=scores.get)
        best_score = scores[best_cluster_id]
        
        # 检查是否应该创建新cluster（如果满足率太低）
        if allow_new_cluster:
            best_satisfaction = details[best_cluster_id]['satisfaction_rate']
            if best_satisfaction < 0.5 and best_score < 0:
                return -1, best_score, {
                    'message': 'Low satisfaction rate, recommend new cluster',
                    'best_existing': best_cluster_id,
                    'best_existing_score': best_score,
                    'satisfaction_rate': best_satisfaction
                }
        
        return best_cluster_id, best_score, details[best_cluster_id]
    
    def _check_rules(self, features: Dict) -> Dict[str, bool]:
        """检查特征是否满足各规则"""
        satisfied = {}
        
        for rule in self.rules:
            rid = rule['rule_id']
            feat = rule['feature_name']
            
            if feat not in features:
                satisfied[rid] = True  # 无法检查的规则默认通过
                continue
            
            value = features[feat]
            threshold = rule.get('threshold')
            operator = rule.get('operator')
            
            if threshold is None:
                # 相关性规则，使用normalization中的mean作为参考
                if feat in self.normalization:
                    threshold = self.normalization[feat]['mean']
                else:
                    satisfied[rid] = True
                    continue
            
            if operator in ['>=', 'maximize']:
                satisfied[rid] = value >= threshold
            elif operator in ['<=', 'minimize']:
                satisfied[rid] = value <= threshold
            else:
                satisfied[rid] = True
        
        return satisfied


# =============================================================================
# 示例用法
# =============================================================================

def demo_usage():
    """演示如何使用分配器"""
    print("\n" + "="*70)
    print("Demo: Checkpoint-to-Cluster Assignment")
    print("="*70)
    
    # 创建预测器（指定场景）
    predictor = ClusterAssignmentPredictor(scenario_name='large_mixed')
    
    print(f"\n场景: large_mixed")
    print(f"调整后权重:")
    for feat, w in sorted(predictor.weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {feat:25s}: {w:+.3f}")
    
    # 模拟现有clusters
    clusters = {
        0: [
            {'location': (1000, 2000, 50), 'workflow_type': 'LogisticsWorkflow', 'payload_weight': 1.5},
            {'location': (1050, 2020, 50), 'workflow_type': 'LogisticsWorkflow', 'payload_weight': 1.2},
            {'location': (980, 1980, 50), 'workflow_type': 'LogisticsWorkflow', 'payload_weight': 1.8},
        ],
        1: [
            {'location': (3000, 3500, 100), 'workflow_type': 'InspectionWorkflow', 'payload_weight': 0},
            {'location': (3200, 3600, 120), 'workflow_type': 'InspectionWorkflow', 'payload_weight': 0},
        ],
        2: [
            {'location': (500, 500, 50), 'workflow_type': 'LogisticsWorkflow', 'payload_weight': 2.0},
            {'location': (4500, 4500, 50), 'workflow_type': 'ComputationWorkflow', 'payload_weight': 0},
        ],
    }
    
    # 待分配的checkpoint
    new_checkpoint = {
        'location': (1100, 2050, 50),
        'workflow_type': 'LogisticsWorkflow',
        'payload_weight': 1.3,
    }
    
    print(f"\n待分配Checkpoint: {new_checkpoint}")
    print(f"\n现有Clusters:")
    for cid, cps in clusters.items():
        types = [cp['workflow_type'].replace('Workflow', '') for cp in cps]
        print(f"  Cluster {cid}: {len(cps)} checkpoints, types={set(types)}")
    
    # 预测
    best_id, score, details = predictor.predict(new_checkpoint, clusters)
    
    print(f"\n预测结果:")
    print(f"  推荐Cluster: {best_id}")
    print(f"  分数: {score:.4f}")
    print(f"  规则满足率: {details.get('satisfaction_rate', 'N/A'):.1%}")
    
    if 'rules_satisfied' in details:
        satisfied = sum(details['rules_satisfied'].values())
        total = len(details['rules_satisfied'])
        print(f"  满足规则: {satisfied}/{total}")
        
        # 显示未满足的规则
        not_satisfied = [k for k, v in details['rules_satisfied'].items() if not v]
        if not_satisfied:
            print(f"  未满足规则: {not_satisfied}")


if __name__ == '__main__':
    # 1. 分析高性能clusters满足的规则
    tier_satisfaction = analyze_rule_satisfaction_by_performance()
    
    # 2. 演示使用
    demo_usage()
