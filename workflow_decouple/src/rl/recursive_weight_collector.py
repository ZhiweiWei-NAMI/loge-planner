"""
递归权重收集器 - 构建解耦树并收集权重库

基于训练好的Bandit策略，递归分裂数据集并记录每个子集的最优权重配置。
输出weight_library.json供HtT规则引擎使用。
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict

# 添加父目录到路径
parent_path = Path(__file__).parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

from ml_training.initial_data_loader import InitialDataLoader
from ml_training.checkpoint_extractor import CheckpointExtractor
from rl.observation_encoder import ObservationEncoder
from rl.policy import MixedActionPolicy
from rl.weighted_kmeans import kmeans_with_weights

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """解耦树节点"""
    id: str
    depth: int
    n_samples: int
    context: Dict
    weights: List[float]
    n_clusters: int
    children: List[str]
    is_leaf: bool = False


class RecursiveWeightCollector:
    """递归权重收集器"""
    
    def __init__(
        self,
        policy: MixedActionPolicy,
        encoder: ObservationEncoder,
        split_strategy: str = 'binary',
        device: str = None
    ):
        """
        初始化收集器
        
        Args:
            policy: 训练好的策略网络
            encoder: 观测编码器
            split_strategy: 分裂策略 ('binary' | 'kmeans')
            device: 计算设备
        """
        self.policy = policy
        self.encoder = encoder
        self.split_strategy = split_strategy
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.policy.to(self.device)
        self.policy.eval()
        
        self.nodes = []
        self.node_counter = 0
        
        logger.info(f"RecursiveWeightCollector initialized, strategy={split_strategy}")
    
    def _compute_context(self, checkpoint_df: pd.DataFrame, feature_matrix: np.ndarray) -> Dict:
        """
        计算数据子集的上下文特征
        
        Args:
            checkpoint_df: checkpoint数据
            feature_matrix: 特征矩阵 (n_samples, 26)
        
        Returns:
            context: 上下文字典
        """
        context = {}
        
        # 几何方差（前9维：3D位置 x 3个点）
        geo_features = feature_matrix[:, :9]
        context['geo_var'] = float(np.var(geo_features))
        context['geo_mean'] = float(np.mean(geo_features))
        
        # 时间方差（接下来8维：时间相关特征）
        time_features = feature_matrix[:, 9:17]
        context['time_var'] = float(np.var(time_features))
        context['time_mean'] = float(np.mean(time_features))
        
        # 任务特征方差（剩余维度）
        task_features = feature_matrix[:, 17:]
        context['task_var'] = float(np.var(task_features))
        
        # 任务类型多样性
        if 'operation_type' in checkpoint_df.columns:
            op_counts = checkpoint_df['operation_type'].value_counts()
            context['task_diversity'] = float(1.0 - (op_counts.max() / len(checkpoint_df)))
        
        context['n_samples'] = len(checkpoint_df)
        
        return context
    
    def _should_stop_splitting(
        self,
        depth: int,
        n_samples: int,
        feature_matrix: np.ndarray,
        max_depth: int,
        min_samples: int,
        min_variance: float
    ) -> bool:
        """
        判断是否停止分裂
        
        Returns:
            True if should stop, False otherwise
        """
        # 条件1：达到最大深度
        if depth >= max_depth:
            return True
        
        # 条件2：样本数过少
        if n_samples < min_samples:
            return True
        
        # 条件3：特征方差过低
        total_var = float(np.var(feature_matrix))
        if total_var < min_variance:
            return True
        
        return False
    
    def _split_by_policy(
        self,
        checkpoint_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        data_dir: Path
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        使用策略网络预测权重并分裂
        
        Returns:
            weights: 权重向量
            n_clusters: 簇数
            labels: 聚类标签
        """
        # 编码观测
        obs = self.encoder.encode(data_dir).to(self.device)
        
        # 策略采样（deterministic=True）
        with torch.no_grad():
            weights, n_clusters, _ = self.policy.sample(obs, deterministic=True)
        
        weights_np = weights.cpu().numpy()
        
        # Binary split：强制k=2
        if self.split_strategy == 'binary':
            k = 2
        else:
            k = int(n_clusters.item())
        
        # 执行聚类：将当前子集视为单一 run，run_ids 全 0
        run_ids = np.zeros(len(feature_matrix), dtype=int)
        labels = kmeans_with_weights(
            features=feature_matrix,
            run_ids=run_ids,
            weights=weights_np,
            n_clusters_per_run=k,
            random_state=42,
            n_init=10,
        )
        
        return weights_np, k, labels

    def _collect_recursive(
        self,
        checkpoint_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        data_dir: Path,
        depth: int,
        max_depth: int,
        min_samples: int,
        min_variance: float
    ) -> TreeNode:
        """
        递归收集权重（内部递归函数）

        Returns:
            node: 当前节点
        """
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        n_samples = len(checkpoint_df)

        # 计算上下文
        context = self._compute_context(checkpoint_df, feature_matrix)

        # 检查停止条件
        should_stop = self._should_stop_splitting(
            depth, n_samples, feature_matrix, max_depth, min_samples, min_variance
        )

        # 获取权重
        weights_np, k, labels = self._split_by_policy(checkpoint_df, feature_matrix, data_dir)

        # 创建节点
        node = TreeNode(
            id=node_id,
            depth=depth,
            n_samples=n_samples,
            context=context,
            weights=weights_np.tolist(),
            n_clusters=k,
            children=[],
            is_leaf=should_stop
        )

        logger.info(
            f"Node {node_id}: depth={depth}, n={n_samples}, weights={weights_np}, k={k}, leaf={should_stop}"
        )

        if not should_stop and k >= 2:
            # 递归分裂子节点
            for cluster_id in range(k):
                mask = (labels == cluster_id)
                child_checkpoint_df = checkpoint_df[mask].reset_index(drop=True)
                child_feature_matrix = feature_matrix[mask]

                if len(child_checkpoint_df) >= min_samples:
                    child_node = self._collect_recursive(
                        child_checkpoint_df,
                        child_feature_matrix,
                        data_dir,
                        depth + 1,
                        max_depth,
                        min_samples,
                        min_variance
                    )
                    node.children.append(child_node.id)
                    self.nodes.append(child_node)

        return node

    def collect_weights(
        self,
        data_dirs: List[Path],
        max_depth: int = 5,
        min_samples: int = 50,
        min_variance: float = 0.05
    ) -> Dict:
        """
        递归收集权重配置

        Args:
            data_dirs: 数据目录列表
            max_depth: 最大深度
            min_samples: 最小样本数
            min_variance: 最小方差

        Returns:
            weight_library: 权重库字典
        """
        self.nodes = []
        self.node_counter = 0

        # 加载所有数据
        all_checkpoint_dfs = []
        all_feature_matrices = []

        logger.info(f"Loading data from {len(data_dirs)} directories...")

        for data_dir in data_dirs:
            loader = InitialDataLoader(data_dir)
            workflow_df, agent_df = loader.load_all()
            workflows_features = loader.get_all_workflows_with_agents()

            # 使用与 HTTDatasetBuilder 一致的 CheckpointExtractor API
            extractor = CheckpointExtractor()
            checkpoint_df = extractor.extract_all_checkpoints(workflows_features)
            feature_matrix = extractor.prepare_features_for_clustering(checkpoint_df)

            all_checkpoint_dfs.append(checkpoint_df)
            all_feature_matrices.append(feature_matrix)

        # 合并数据
        merged_checkpoint_df = pd.concat(all_checkpoint_dfs, ignore_index=True)
        merged_feature_matrix = np.vstack(all_feature_matrices)

        logger.info(
            f"Merged data: {len(merged_checkpoint_df)} checkpoints, "
            f"feature shape={merged_feature_matrix.shape}"
        )

        # 递归构建树
        root_node = self._collect_recursive(
            merged_checkpoint_df,
            merged_feature_matrix,
            data_dirs[0],  # 使用第一个目录计算观测（简化）
            depth=0,
            max_depth=max_depth,
            min_samples=min_samples,
            min_variance=min_variance
        )

        self.nodes.insert(0, root_node)

        # 构造输出
        weight_library = {
            'metadata': {
                'n_data_dirs': len(data_dirs),
                'total_samples': len(merged_checkpoint_df),
                'max_depth': max_depth,
                'min_samples': min_samples,
                'min_variance': min_variance,
                'split_strategy': self.split_strategy,
                'total_nodes': len(self.nodes)
            },
            'nodes': [asdict(node) for node in self.nodes]
        }

        logger.info(f"Collected {len(self.nodes)} nodes in weight library")

        return weight_library

