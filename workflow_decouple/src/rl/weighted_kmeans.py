"""
加权KMeans聚类接口

通过对特征维度加权，影响KMeans的距离计算。
工程实现：X_weighted = X ⊙ sqrt(w)，然后调用sklearn KMeans

重要：必须先标准化特征，否则大尺度特征会主导距离计算，导致权重失效
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def kmeans_with_weights(
    features: np.ndarray,
    run_ids: np.ndarray,
    weights: np.ndarray,
    n_clusters_per_run: int,
    random_state: int = 42,
    n_init: int = 10,
    w_min: float = 1e-5
) -> np.ndarray:
    """
    按run分组执行加权KMeans聚类
    
    Args:
        features: checkpoint特征矩阵 [N, d]
        run_ids: 每个checkpoint对应的run ID [N]
        weights: 维度权重 [d]，范围[0, 1]
        n_clusters_per_run: 每个run的簇数（动态）
        random_state: 随机种子
        n_init: KMeans初始化次数
        w_min: 权重下界，防止退化为0
        
    Returns:
        labels: 全局聚类标签 [N]
    """
    # 参数校验
    assert features.shape[1] == len(weights), \
        f"特征维度({features.shape[1]})与权重维度({len(weights)})不匹配"
    assert len(features) == len(run_ids), \
        f"特征数量({len(features)})与run_ids数量({len(run_ids)})不匹配"
    
    # **关键修复**：先标准化特征，确保所有维度在同一尺度
    # 否则大尺度特征（如distance=1000+）会完全主导距离计算，导致权重失效
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 权重安全处理：限制在 [w_min, 1]
    weights_safe = np.clip(weights, w_min, 1.0)
    
    # 加权特征：X_w = X ⊙ sqrt(w)
    # sqrt是因为距离计算用欧式距离：d^2 = sum(w_i * (x_i - y_i)^2)
    features_weighted = features_scaled * np.sqrt(weights_safe)
    
    # 按run分组聚类
    labels = np.zeros(len(features), dtype=int)
    unique_runs = np.unique(run_ids)
    
    cluster_offset = 0  # 全局cluster标签偏移
    
    for run_id in sorted(unique_runs):
        run_mask = (run_ids == run_id)
        run_features = features_weighted[run_mask]
        
        # 对该run使用KMeans聚类
        n_samples = len(run_features)
        
        # 检测并处理重复点
        unique_features, unique_indices, inverse_indices = np.unique(
            run_features, axis=0, return_index=True, return_inverse=True
        )
        n_unique = len(unique_features)
        
        # 根据唯一样本数调整簇数
        n_clusters_actual = min(n_clusters_per_run, n_samples, n_unique)
        
        if n_clusters_actual == 0:
            logger.warning(f"Run {run_id} 无样本，跳过")
            continue
        
        if n_unique < n_samples:
            logger.debug(f"Run {run_id}: 检测到 {n_samples - n_unique} 个重复样本，"
                        f"唯一样本数={n_unique}, 调整簇数为 {n_clusters_actual}")
        
        if n_clusters_actual == 1:
            # 只有1个样本或强制1个簇，直接赋值
            run_labels = np.zeros(n_samples, dtype=int)
        else:
            # 在唯一样本上进行聚类
            kmeans = KMeans(
                n_clusters=n_clusters_actual,
                random_state=random_state,
                n_init=n_init,
                algorithm='lloyd'  # 明确指定算法
            )
            unique_labels = kmeans.fit_predict(unique_features)
            
            # 将标签映射回原始样本
            run_labels = unique_labels[inverse_indices]
        
        # 加上偏移量，确保不同run的cluster标签不重复
        labels[run_mask] = run_labels + cluster_offset
        cluster_offset += n_clusters_actual
    
    return labels


def compute_weighted_distance_matrix(
    features: np.ndarray,
    weights: np.ndarray,
    w_min: float = 1e-5
) -> np.ndarray:
    """
    计算加权距离矩阵（用于层次聚类或可视化）
    
    Args:
        features: [N, d]
        weights: [d]
        w_min: 权重下界
        
    Returns:
        distance_matrix: [N, N]
    """
    from scipy.spatial.distance import pdist, squareform
    
    weights_safe = np.clip(weights, w_min, 1.0)
    features_weighted = features * np.sqrt(weights_safe)
    
    # 计算成对距离
    distances = pdist(features_weighted, metric='euclidean')
    distance_matrix = squareform(distances)
    
    return distance_matrix
