"""
Recursive HtT Clusterer - 论文描述的递归分解树在线实现

基于 Algorithm 1 (HtT-Guided Recursive Workflow Decoupling):
1. 从根节点开始，递归二分任务集合
2. 在每个节点根据当前上下文 S_ctx 查询规则库 L 选择权重 w*
3. 使用选定的权重进行 weighted spectral bisection
4. 递归处理子节点
5. 终止条件：样本数过少、深度过大、或方差过低（匹配 Leaf Condition）
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """递归分解树节点"""
    node_id: int
    depth: int
    n_samples: int
    indices: np.ndarray  # 原始数据中的索引
    weights: np.ndarray  # 选择的权重
    is_leaf: bool
    cluster_label: int = -1  # 最终的cluster标签（仅叶节点有效）
    children: List['TreeNode'] = None
    context: Dict = None


class RecursiveHtTClusterer:
    """
    递归HtT分解树聚类器
    
    实现论文中的 Algorithm 1: HtT-Guided Recursive Workflow Decoupling
    
    分层规则设计（基于RL数据分析）：
    - 浅层（depth 0-1）：按任务类型/需求分，分离不同类型的workflow
    - 中层（depth 2-3）：按空间位置分，聚合空间上接近的任务
    - 深层（depth 4+）：按workflow完整性分，保持同一workflow的任务在一起
    """
    
    def __init__(
        self,
        weight_selector,
        max_depth: int = 6,
        min_samples: int = 10,
        min_variance: float = 0.01,
        bisection_method: str = 'spectral',  # 'spectral' or 'kmeans'
        max_leaf_size: int | None = None,    # 叶子cluster允许的最大样本数，None表示不限制
        spatial_var_threshold: float | None = None,  # 空间方差阈值，超过则继续拆分
        max_workflows_per_cluster: int | None = None,  # 每个簇允许的最大workflow数，超过则继续拆分
        causal_anchor_lambda: float = 0.0,
    ):
        """
        初始化递归聚类器
        
        Args:
            weight_selector: RuleBasedWeightSelector 实例
            max_depth: 最大递归深度（安全上限）
            min_samples: 最小样本数（叶节点条件）
            min_variance: 最小方差（叶节点条件）
            bisection_method: 二分方法 ('spectral' 或 'kmeans')
            max_leaf_size: 叶子cluster允许的最大样本数，超过则继续拆分
        """
        self.weight_selector = weight_selector
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_variance = min_variance
        self.bisection_method = bisection_method
        self.max_leaf_size = max_leaf_size
        self.spatial_var_threshold = spatial_var_threshold
        self.max_workflows_per_cluster = max_workflows_per_cluster
        self.causal_anchor_lambda = float(causal_anchor_lambda)
        
        self.node_counter = 0
        self.leaf_counter = 0
        self.nodes = []
        
        logger.info(
            "RecursiveHtTClusterer initialized: max_depth=%d, min_samples=%d, "
            "min_variance=%.4f, bisection=%s, max_leaf_size=%s, spatial_var_threshold=%s, "
            "max_workflows_per_cluster=%s, causal_anchor_lambda=%.3f",
            max_depth, min_samples, min_variance, bisection_method, str(max_leaf_size),
            str(spatial_var_threshold), str(max_workflows_per_cluster), self.causal_anchor_lambda
        )
    
    def _should_stop(
        self,
        depth: int,
        n_samples: int,
        feature_subset: np.ndarray,
        checkpoint_df: pd.DataFrame,
    ) -> bool:
        """
        检查是否应该停止递归（Leaf Condition）
        
        Returns:
            True if should stop (create leaf node)
        """
        # 条件0：无法再二分
        if n_samples < 2:
            return True

        # 条件1：样本数过大时强制继续拆分（优先级最高）
        if self.max_leaf_size is not None and n_samples > self.max_leaf_size:
            logger.debug(
                "Continue: n_samples (%d) > max_leaf_size (%d)",
                n_samples,
                self.max_leaf_size,
            )
            return False

        # 条件2：空间方差过低时继续拆分（仅在未到达最大深度时）
        if self.spatial_var_threshold is not None and depth < self.max_depth and feature_subset.shape[1] >= 3:
            # 仅使用前三个维度作为空间坐标 (x,y,z)
            spatial = feature_subset[:, :3]
            spatial_var = float(np.var(spatial, axis=0).sum())
            if spatial_var < self.spatial_var_threshold and n_samples >= self.min_samples:
                logger.debug(
                    "Continue: spatial_var (%.2f) < threshold (%.2f) at depth=%d",
                    spatial_var, self.spatial_var_threshold, depth
                )
                return False

        # 条件3：workflow数量过多时继续拆分（仅在未到达最大深度时）
        if self.max_workflows_per_cluster is not None and depth < self.max_depth:
            if 'workflow_id' in checkpoint_df.columns:
                n_workflows = int(pd.Series(checkpoint_df['workflow_id']).nunique())
                if n_workflows > self.max_workflows_per_cluster and n_samples >= self.min_samples:
                    logger.debug(
                        "Continue: n_workflows (%d) > max_workflows_per_cluster (%d) at depth=%d",
                        n_workflows, self.max_workflows_per_cluster, depth
                    )
                    return False

        # 条件4：达到最大深度（安全上限）
        if depth >= self.max_depth:
            logger.debug("Stop: max_depth reached (%d) with n_samples=%d", depth, n_samples)
            return True
        
        # 条件5：样本数过少
        if n_samples < self.min_samples:
            logger.debug("Stop: n_samples too small (%d < %d)", n_samples, self.min_samples)
            return True
        
        # 条件6：特征方差过低（内部已经足够同质）
        total_variance = np.var(feature_subset)
        if total_variance < self.min_variance:
            logger.debug("Stop: variance too low (%.6f < %.6f)", total_variance, self.min_variance)
            return True
        
        return False
    
    def _compute_affinity_matrix(
        self,
        feature_subset: np.ndarray,
        weights: np.ndarray,
        gamma: float = 1.0,
        checkpoint_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        计算加权亲和矩阵 A，其中 A_ij = exp(-gamma * d_w(t_i, t_j))
        
        论文公式：d_w(t_i, t_j) = sqrt(sum_d w_d * (f_{i,d} - f_{j,d})^2)
        """
        n = len(feature_subset)
        
        # 加权特征
        weighted_features = feature_subset * np.sqrt(weights)
        
        # 计算距离矩阵
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(weighted_features, weighted_features, metric='euclidean')
        
        # 转换为亲和矩阵
        affinity = np.exp(-gamma * dist_matrix)

        # Optional causal anchor for revision experiments: boost edges between
        # consecutive checkpoints from the same workflow.
        if (
            self.causal_anchor_lambda > 0
            and checkpoint_df is not None
            and "workflow_id" in checkpoint_df.columns
            and "checkpoint_idx" in checkpoint_df.columns
            and len(checkpoint_df) == affinity.shape[0]
        ):
            sorted_df = checkpoint_df.reset_index(drop=True)
            for _workflow_id, group in sorted_df.groupby("workflow_id", sort=False):
                ordered = group.sort_values("checkpoint_idx").index.tolist()
                for left, right in zip(ordered, ordered[1:]):
                    affinity[left, right] += self.causal_anchor_lambda
                    affinity[right, left] += self.causal_anchor_lambda
        
        return affinity
    
    def _bisect_by_workflow(
        self,
        checkpoint_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        weights: np.ndarray,
        depth: int = 0
    ) -> np.ndarray:
        """
        以workflow为单位进行二分聚类，保证同一workflow的checkpoints不被拆分。

        简化版策略（不再做checkpoint级备用拆分）：
        1. 将checkpoints按workflow_id聚合，计算每个workflow的质心特征
        2. 对workflow质心进行二分聚类
        3. 将每个workflow的所有checkpoints分配到相应的cluster

        叶子是否继续拆分由 _should_stop 控制（含max_cluster_size逻辑）。
        """
        if self.causal_anchor_lambda > 0.0:
            labels = self._bisect(
                feature_matrix,
                weights,
                depth,
                checkpoint_df=checkpoint_df,
            )
            return self._fix_workflow_integrity(checkpoint_df, labels)

        workflow_ids = checkpoint_df['workflow_id'].values
        unique_workflows = np.unique(workflow_ids)
        n_workflows = len(unique_workflows)
        
        # 如果只有1个workflow，无法按workflow二分，全部给同一个label
        if n_workflows <= 1:
            return np.zeros(len(checkpoint_df), dtype=int)
        
        # 计算每个workflow的质心特征
        workflow_centroids = []
        workflow_id_list = []
        for wf_id in unique_workflows:
            mask = (workflow_ids == wf_id)
            wf_features = feature_matrix[mask]
            centroid = np.mean(wf_features, axis=0)
            workflow_centroids.append(centroid)
            workflow_id_list.append(wf_id)
        
        workflow_centroids = np.array(workflow_centroids)
        
        # 对workflow质心进行二分聚类
        workflow_labels = self._bisect(workflow_centroids, weights, depth)
        
        # 将每个workflow的所有checkpoints分配到相应的cluster
        checkpoint_labels = np.zeros(len(checkpoint_df), dtype=int)
        for i, wf_id in enumerate(workflow_id_list):
            mask = (workflow_ids == wf_id)
            checkpoint_labels[mask] = workflow_labels[i]
        
        return checkpoint_labels

    def _fix_workflow_integrity(self, checkpoint_df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        """
        100%修复workflow完整性：将被拆分的workflow的所有checkpoints移到多数派所在的cluster
        
        这确保同一个workflow的checkpoints一定在同一个cluster中，不允许任何拆分
        """
        if 'workflow_id' not in checkpoint_df.columns:
            return labels
        
        labels = labels.copy()
        workflow_ids = checkpoint_df['workflow_id'].values
        
        # 统计每个workflow在每个cluster中的数量
        from collections import Counter
        
        for wf_id in np.unique(workflow_ids):
            mask = (workflow_ids == wf_id)
            wf_labels = labels[mask]
            counts = Counter(wf_labels)
            
            if len(counts) > 1:
                # workflow被拆分了，100%合并到多数派cluster
                majority_cluster = counts.most_common(1)[0][0]
                labels[mask] = majority_cluster
        
        return labels

    def _get_depth_based_weights(self, depth: int, base_weights: np.ndarray) -> np.ndarray:
        """
        根据深度调整权重，实现分层规则
        
        浅层（depth 0-1）：强调任务类型/需求
        中层（depth 2-3）：强调空间位置
        深层（depth 4+）：空间+资源平衡
        """
        weights = base_weights.copy()
        
        # 特征索引分组
        spatial_idx = [0, 1, 2]  # location_x, y, z
        operation_idx = list(range(3, 11))  # op_pickup...op_empty
        requirement_idx = list(range(11, 16))  # req_mobility...req_compute
        resource_idx = list(range(16, 23))  # payload_weight...max_latency_s
        
        if depth <= 1:
            # 浅层：强调操作类型和需求，按任务逻辑分离
            weights[spatial_idx] *= 0.6
            weights[operation_idx] *= 1.4
            weights[requirement_idx] *= 1.3
            logger.debug("Depth %d: Emphasizing operation/requirement (task logic)", depth)
        elif depth <= 3:
            # 中层：强调空间位置，聚合空间上接近的任务
            weights[spatial_idx] *= 1.5
            weights[operation_idx] *= 0.7
            weights[requirement_idx] *= 0.8
            logger.debug("Depth %d: Emphasizing spatial (location clustering)", depth)
        else:
            # 深层：空间+资源平衡，保持workflow完整性
            weights[spatial_idx] *= 1.2
            weights[resource_idx] *= 1.1
            logger.debug("Depth %d: Balanced spatial+resource (workflow integrity)", depth)
        
        # 归一化
        weights = weights / (np.sum(weights) + 1e-10) * len(weights)
        
        return weights

    def _bisect(
        self,
        feature_subset: np.ndarray,
        weights: np.ndarray,
        depth: int = 0,
        checkpoint_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        使用选定的权重进行二分（spectral bisection 或 weighted kmeans）
        
        Returns:
            labels: 二分标签 (0 或 1)
        """
        n_samples = len(feature_subset)
        
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int)
        
        # 根据深度调整权重
        adjusted_weights = self._get_depth_based_weights(depth, weights)
        
        # 加权特征
        weighted_features = feature_subset * np.sqrt(adjusted_weights)
        
        if self.bisection_method == 'spectral':
            try:
                # Spectral Clustering with precomputed affinity
                affinity = self._compute_affinity_matrix(
                    feature_subset,
                    adjusted_weights,
                    checkpoint_df=checkpoint_df,
                )
                sc = SpectralClustering(
                    n_clusters=2,
                    affinity='precomputed',
                    random_state=42,
                    assign_labels='kmeans'
                )
                labels = sc.fit_predict(affinity)
            except Exception as e:
                logger.warning("Spectral clustering failed: %s, falling back to KMeans", e)
                km = KMeans(n_clusters=2, random_state=42, n_init=5)
                labels = km.fit_predict(weighted_features)
        else:
            # Weighted KMeans
            km = KMeans(n_clusters=2, random_state=42, n_init=5)
            labels = km.fit_predict(weighted_features)
        
        return labels
    
    def _build_tree_recursive(
        self,
        checkpoint_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        indices: np.ndarray,
        depth: int
    ) -> TreeNode:
        """
        递归构建分解树
        
        Args:
            checkpoint_df: 当前子集的checkpoint数据
            feature_matrix: 当前子集的特征矩阵
            indices: 在原始数据中的索引
            depth: 当前深度
        
        Returns:
            TreeNode: 当前节点
        """
        node_id = self.node_counter
        self.node_counter += 1
        n_samples = len(checkpoint_df)
        
        # 计算当前上下文 S_ctx
        ctx = self.weight_selector.summarize_context(checkpoint_df, feature_matrix)
        ctx = dict(ctx) if ctx is not None else {}
        ctx.setdefault('n_checkpoints', n_samples)
        ctx['depth'] = depth
        if 'workflow_id' in checkpoint_df.columns:
            ctx['n_workflows'] = int(pd.Series(checkpoint_df['workflow_id']).nunique())
        else:
            ctx['n_workflows'] = 1
        
        # 检查是否应该停止（Leaf Condition）
        should_stop = self._should_stop(depth, n_samples, feature_matrix, checkpoint_df)
        
        # 查询规则库获取权重 w*
        weights, _ = self.weight_selector.select_weights_and_k(ctx)
        
        if should_stop:
            # 创建叶节点
            cluster_label = self.leaf_counter
            self.leaf_counter += 1
            
            node = TreeNode(
                node_id=node_id,
                depth=depth,
                n_samples=n_samples,
                indices=indices,
                weights=weights,
                is_leaf=True,
                cluster_label=cluster_label,
                children=[],
                context=ctx
            )
            
            logger.debug(
                "Leaf node %d: depth=%d, n_samples=%d, cluster_label=%d",
                node_id, depth, n_samples, cluster_label
            )
            
            self.nodes.append(node)
            return node
        
        # Checkpoint-level spectral bisection; causal_anchor_lambda controls
        # whether the affinity matrix receives same-workflow adjacency boost.
        # Both lambda=0 and lambda>0 go through the same path so the only
        # ablation variable is the affinity augmentation.
        # No post-hoc _fix_workflow_integrity: the anchor must work at the
        # affinity level, otherwise the ablation comparison is meaningless.
        labels = self._bisect(
            feature_matrix,
            weights,
            depth,
            checkpoint_df=checkpoint_df,
        )
        
        # 递归处理子节点
        children = []
        for child_label in [0, 1]:
            mask = (labels == child_label)
            if mask.sum() == 0:
                continue
            
            child_checkpoint_df = checkpoint_df[mask].reset_index(drop=True)
            child_feature_matrix = feature_matrix[mask]
            child_indices = indices[mask]
            
            # 避免无限递归：如果子节点和父节点一样大，强制停止
            if len(child_checkpoint_df) == n_samples:
                # 无法有效分割，创建叶节点
                cluster_label = self.leaf_counter
                self.leaf_counter += 1
                
                child_node = TreeNode(
                    node_id=self.node_counter,
                    depth=depth + 1,
                    n_samples=len(child_checkpoint_df),
                    indices=child_indices,
                    weights=weights,
                    is_leaf=True,
                    cluster_label=cluster_label,
                    children=[],
                    context=ctx
                )
                self.node_counter += 1
                self.nodes.append(child_node)
                children.append(child_node)
            else:
                child_node = self._build_tree_recursive(
                    child_checkpoint_df,
                    child_feature_matrix,
                    child_indices,
                    depth + 1
                )
                children.append(child_node)
        
        # 创建内部节点
        node = TreeNode(
            node_id=node_id,
            depth=depth,
            n_samples=n_samples,
            indices=indices,
            weights=weights,
            is_leaf=False,
            cluster_label=-1,
            children=children,
            context=ctx
        )
        
        logger.debug(
            "Internal node %d: depth=%d, n_samples=%d, n_children=%d",
            node_id, depth, n_samples, len(children)
        )
        
        self.nodes.append(node)
        return node
    
    def _collect_leaf_labels(self, root: TreeNode) -> Dict[int, int]:
        """
        收集所有叶节点的标签映射
        
        Returns:
            Dict[original_index, cluster_label]
        """
        index_to_label = {}
        
        def traverse(node: TreeNode):
            if node.is_leaf:
                for idx in node.indices:
                    index_to_label[idx] = node.cluster_label
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(root)
        return index_to_label
    
    def fit_predict(
        self,
        checkpoint_df: pd.DataFrame,
        feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        执行递归HtT分解并返回聚类标签
        
        Args:
            checkpoint_df: checkpoint DataFrame
            feature_matrix: 特征矩阵 (N, D)
        
        Returns:
            labels: 聚类标签数组
        """
        # 重置状态
        self.node_counter = 0
        self.leaf_counter = 0
        self.nodes = []
        
        n_samples = len(checkpoint_df)
        indices = np.arange(n_samples)
        
        logger.info("Starting recursive HtT decomposition: %d samples", n_samples)
        
        # 构建递归分解树
        root = self._build_tree_recursive(
            checkpoint_df,
            feature_matrix,
            indices,
            depth=0
        )
        
        # 收集叶节点标签
        index_to_label = self._collect_leaf_labels(root)
        
        # 构建标签数组
        labels = np.array([index_to_label.get(i, -1) for i in range(n_samples)])
        
        n_clusters = self.leaf_counter
        logger.info(
            "Recursive HtT decomposition complete: %d clusters, %d nodes, max_depth=%d",
            n_clusters, len(self.nodes), self.max_depth
        )
        
        return labels
    
    def get_tree_stats(self) -> Dict:
        """获取分解树统计信息"""
        leaf_nodes = [n for n in self.nodes if n.is_leaf]
        internal_nodes = [n for n in self.nodes if not n.is_leaf]
        
        return {
            'total_nodes': len(self.nodes),
            'leaf_nodes': len(leaf_nodes),
            'internal_nodes': len(internal_nodes),
            'n_clusters': self.leaf_counter,
            'leaf_sizes': [n.n_samples for n in leaf_nodes],
            'max_depth_reached': max(n.depth for n in self.nodes) if self.nodes else 0
        }
