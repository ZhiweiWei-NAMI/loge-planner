"""
Rule-Based Weight Selector for HtT-guided Clustering

根据场景上下文特征选择KMeans的特征权重和簇数，
替代在线推理，实现完全离线的规则驱动聚类。
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class RuleBasedWeightSelector:
    """规则驱动的权重选择器"""
    
    def __init__(self, config_path: str):
        """
        初始化选择器
        
        Args:
            config_path: 规则配置文件路径 (YAML)
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Rule config not found: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.rules = self.config.get('rules', [])
        self.weight_mapping = self.config.get('weight_mapping', {})
        
        logger.info(
            "RuleBasedWeightSelector initialized: %d rules loaded from %s",
            len(self.rules), self.config_path
        )
    
    def summarize_context(
        self, 
        checkpoint_df: pd.DataFrame, 
        feature_matrix: np.ndarray
    ) -> Dict:
        """
        计算场景上下文特征 S_ctx
        
        Args:
            checkpoint_df: checkpoint DataFrame
            feature_matrix: 特征矩阵 (N, D)
        
        Returns:
            上下文字典 S_ctx
        """
        n_checkpoints = len(checkpoint_df)
        
        # 空间特征
        locations = []
        for loc_str in checkpoint_df['location']:
            try:
                import ast
                loc = ast.literal_eval(str(loc_str))
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    locations.append([float(loc[0]), float(loc[1])])
            except:
                pass
        
        locations = np.array(locations) if locations else np.zeros((0, 2))
        
        if len(locations) > 1:
            spatial_variance = np.var(locations, axis=0).sum()
            mean_pairwise_dist = np.mean(
                [np.linalg.norm(locations[i] - locations[j]) 
                 for i in range(min(100, len(locations))) 
                 for j in range(i+1, min(100, len(locations)))]
            ) if len(locations) > 1 else 0.0
        else:
            spatial_variance = 0.0
            mean_pairwise_dist = 0.0
        
        # 时间特征
        if 'start_time' in checkpoint_df.columns:
            start_times = checkpoint_df['start_time'].dropna().values
            if len(start_times) > 1:
                time_range = start_times.max() - start_times.min()
                temporal_density = len(start_times) / max(time_range, 1.0) if time_range > 0 else 0.0
            else:
                temporal_density = 0.0
        else:
            temporal_density = 0.0
        
        # 任务异质性
        if 'operation_type' in checkpoint_df.columns:
            op_types = checkpoint_df['operation_type'].value_counts(normalize=True)
            priority_entropy = -np.sum(op_types * np.log(op_types + 1e-10))
        else:
            priority_entropy = 0.0
        
        # 计算任务多样性（基于 operation_type 的唯一值数量）
        if 'operation_type' in checkpoint_df.columns:
            unique_types = checkpoint_df['operation_type'].nunique()
            total_possible = 8  # pickup/transport/delivery/inspection/compute/communication/completion/empty
            task_diversity = unique_types / total_possible
        else:
            task_diversity = 0.0

        # 巡检强度（用于识别 Inspection-heavy 节点）
        inspection_ratio = 0.0
        inspection_move_ratio = 0.0
        if 'operation_type' in checkpoint_df.columns and len(checkpoint_df) > 0:
            op_series = checkpoint_df['operation_type'].astype(str)
            inspection_mask = op_series == 'inspection'
            move_inspection_mask = op_series.isin(['inspection', 'move_to_inspection'])
            inspection_ratio = float(inspection_mask.mean())
            inspection_move_ratio = float(move_inspection_mask.mean())
        
        ctx = {
            'n_checkpoints': n_checkpoints,
            'n_samples': n_checkpoints,
            'spatial_variance': spatial_variance,
            'mean_pairwise_dist': mean_pairwise_dist,
            'temporal_density': temporal_density,
            'priority_entropy': priority_entropy,
            'task_diversity': task_diversity,
            'inspection_ratio': inspection_ratio,
            'inspection_move_ratio': inspection_move_ratio,
        }
        
        logger.debug("Context summary: %s", ctx)
        return ctx
    
    def select_weights_and_k(self, ctx: Dict) -> Tuple[np.ndarray, int]:
        """
        根据上下文选择权重和簇数
        
        Args:
            ctx: 上下文字典
        
        Returns:
            (weights, n_clusters_per_run)
        """
        # 按优先级排序规则
        sorted_rules = sorted(
            self.rules, 
            key=lambda r: r.get('priority', 0), 
            reverse=True
        )
        
        # 匹配规则
        matched_rule = None
        for rule in sorted_rules:
            if self._match_rule(rule, ctx):
                matched_rule = rule
                break
        
        # 如果没有匹配，使用默认规则
        if matched_rule is None:
            default_rules = [r for r in self.rules if r.get('default', False)]
            if default_rules:
                matched_rule = default_rules[0]
            else:
                # 明确默认规则，避免无规则场景静默失败。
                logger.warning("No rule matched and no default rule, using built-in default rule")
                return np.ones(26) / 26, 30
        
        # 提取权重和簇数
        then_clause = matched_rule.get('then', {})
        weights_compact = then_clause.get('weights', [0.5, 0.5, 0.0])
        n_clusters = then_clause.get('n_clusters_per_run', 30)

        rule_id = matched_rule.get('id', 'unknown')
        logger.info(
            "Matched rule '%s': weights=%s, n_clusters=%d",
            rule_id, weights_compact, n_clusters
        )

        # 展开/解析权重到向量
        weights = self._expand_weights(weights_compact)

        return weights, n_clusters

    def _match_rule(self, rule: Dict, ctx: Dict) -> bool:
        """
        检查规则是否匹配上下文

        Args:
            rule: 规则字典
            ctx: 上下文字典

        Returns:
            是否匹配
        """
        conditions = rule.get('if', [])
        if not conditions:
            # 没有条件的规则（如default）总是匹配
            return True

        for cond in conditions:
            feature_name = cond.get('feature')
            op = cond.get('op')
            value = cond.get('value')

            if feature_name not in ctx:
                return False

            ctx_value = ctx[feature_name]

            if op == '>':
                if not (ctx_value > value):
                    return False
            elif op == '>=':
                if not (ctx_value >= value):
                    return False
            elif op == '<':
                if not (ctx_value < value):
                    return False
            elif op == '<=':
                if not (ctx_value <= value):
                    return False
            elif op == '==':
                if not (ctx_value == value):
                    return False
            else:
                logger.warning("Unknown operator: %s", op)
                return False

        return True

    def _expand_weights(self, weights_compact: List[float]) -> np.ndarray:
        """
        将紧凑权重展开到26维，或直接使用已经是26维的权重

        Args:
            weights_compact: 可能是 [w_spatial, w_temporal, w_task_type] 或完整的 26 维向量

        Returns:
            26维权重向量
        """
        # 如果已经是 26 维，直接使用完整权重。
        if len(weights_compact) == 26:
            return np.array(weights_compact, dtype=np.float32)

        # 如果权重数量本身已经较多（>=20），视为完整向量直接返回
        if len(weights_compact) >= 20:
            w = np.array(weights_compact, dtype=np.float32)
            # 归一化到与原始维度一致的尺度
            w = w / (w.sum() + 1e-10) * len(w)
            return w

        # 否则按旧的 3 维 compact 格式展开
        # 默认维度分配
        # spatial: 0-9 (10维)
        # temporal: 10-15 (6维)
        # task_type: 16-25 (10维)

        weights = np.zeros(26, dtype=np.float32)

        if len(weights_compact) >= 1:
            weights[0:10] = weights_compact[0] / 10
        if len(weights_compact) >= 2:
            weights[10:16] = weights_compact[1] / 6
        if len(weights_compact) >= 3:
            weights[16:26] = weights_compact[2] / 10

        # 归一化
        weights = weights / (weights.sum() + 1e-10)

        return weights
