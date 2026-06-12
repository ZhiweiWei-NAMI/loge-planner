"""
层次聚类工具：使用与Oracle相同的方法
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def load_oracle_best_thresholds(data_dir: Path) -> dict:
    """
    从各个run读取Oracle的最优阈值
    
    Args:
        data_dir: 数据目录（merged_dataset或包含多个run_seed_*的目录）
        
    Returns:
        dict: {run_id: best_threshold}
    """
    thresholds = {}
    
    # 检查是否是merged_dataset目录
    if data_dir.name == 'merged_dataset':
        parent_dir = data_dir.parent
    else:
        parent_dir = data_dir
    
    # 查找所有run_seed_*目录
    run_dirs = sorted(parent_dir.glob('run_seed_*'))
    
    if len(run_dirs) == 0:
        logger.warning(f"未找到run_seed_*目录，使用默认阈值0.3")
        return {1: 0.3}  # 默认
    
    for run_dir in run_dirs:
        threshold_file = run_dir / 'threshold_comparison.json'
        if threshold_file.exists():
            with open(threshold_file, 'r') as f:
                data = json.load(f)
                best_threshold = data.get('best_threshold', 0.3)
                
                # 从目录名提取run_id
                run_id = int(run_dir.name.split('_')[-1])
                thresholds[run_id] = best_threshold
                logger.debug(f"Run {run_id}: 最优阈值 = {best_threshold}")
    
    # 如果没有找到任何阈值，使用全局默认值
    if not thresholds:
        logger.warning("未找到任何threshold_comparison.json，使用默认阈值0.3")
        return {1: 0.3}
    
    logger.info(f"加载了{len(thresholds)}个runs的Oracle阈值")
    return thresholds


def hierarchical_clustering_by_threshold(
    features: np.ndarray,
    threshold: float,
    method: str = 'ward',
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    层次聚类（使用阈值切分）
    
    Args:
        features: [n_samples, n_features] 特征矩阵
        threshold: 切分阈值
        method: 链接方法 ('ward', 'average', 'complete', 'single')
        metric: 距离度量 ('euclidean', 'cosine', etc.)
        
    Returns:
        labels: [n_samples] 聚类标签
    """
    if len(features) == 0:
        return np.array([], dtype=int)
    
    if len(features) == 1:
        return np.array([0], dtype=int)
    
    # 计算距离矩阵
    if method == 'ward':
        # Ward方法必须使用euclidean距离
        distances = pdist(features, metric='euclidean')
    else:
        distances = pdist(features, metric=metric)
    
    # 层次聚类
    linkage_matrix = linkage(distances, method=method)
    
    # 使用阈值切分
    labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    
    # fcluster返回的标签从1开始，转换为从0开始
    labels = labels - 1
    
    return labels


def hierarchical_clustering_by_run(
    checkpoint_df: pd.DataFrame,
    features: np.ndarray,
    thresholds: dict,
    method: str = 'ward'
) -> np.ndarray:
    """
    按run分组进行层次聚类
    
    Args:
        checkpoint_df: checkpoint DataFrame（必须包含source_run_id列）
        features: [n_samples, n_features] 特征矩阵
        thresholds: {run_id: threshold} 每个run的阈值
        method: 层次聚类方法
        
    Returns:
        labels: [n_samples] 聚类标签
    """
    if 'source_run_id' not in checkpoint_df.columns:
        raise ValueError("checkpoint_df必须包含source_run_id列")
    
    labels = np.zeros(len(checkpoint_df), dtype=int)
    run_ids = checkpoint_df['source_run_id'].unique()
    
    for run_id in sorted(run_ids):
        run_mask = checkpoint_df['source_run_id'] == run_id
        run_features = features[run_mask]
        
        # 获取该run的阈值
        threshold = thresholds.get(run_id, 0.3)  # 默认0.3
        
        # 对该run进行层次聚类
        run_labels = hierarchical_clustering_by_threshold(
            run_features,
            threshold=threshold,
            method=method
        )
        
        # 保存该run的labels
        labels[run_mask] = run_labels
        
        logger.debug(f"Run {run_id}: threshold={threshold}, {len(np.unique(run_labels))} clusters")
    
    return labels


def get_cluster_statistics(labels: np.ndarray) -> dict:
    """
    获取聚类统计信息
    
    Args:
        labels: 聚类标签
        
    Returns:
        统计信息字典
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    
    return {
        'n_clusters': n_clusters,
        'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
        'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        'cluster_sizes': cluster_sizes
    }
