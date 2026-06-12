"""
端到端训练器：使用Synergy Loss优化Siamese Network
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import copy
from tqdm import tqdm
from sklearn.cluster import KMeans

from .siamese_encoder import SiameseEncoder, TripletLoss
from ..hierarchical_clustering import (
    load_oracle_best_thresholds,
    get_cluster_statistics
)

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Triplet数据集：(anchor, positive, negative)"""
    
    def __init__(self, features: np.ndarray, triplets: list):
        """
        Args:
            features: [n_samples, n_features]
            triplets: [(anchor_idx, pos_idx, neg_idx), ...]
        """
        self.features = torch.FloatTensor(features)
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        return (
            self.features[anchor_idx],
            self.features[pos_idx],
            self.features[neg_idx]
        )


class CheckpointDataset(Dataset):
    """Checkpoint特征数据集"""
    
    def __init__(self, features: np.ndarray):
        """
        Args:
            features: [n_samples, n_features] numpy array
        """
        self.features = torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class ClusteringTrainer:
    """
    Triplet Network训练器
    
    使用Triplet Loss训练Siamese Encoder，
    学习能够产生高协同增益的embedding表示。
    评估时使用KMeans聚类（每个run固定100个cluster）。
    """
    
    def __init__(
        self,
        encoder: SiameseEncoder,
        synergy_evaluator,
        checkpoint_df: pd.DataFrame,
        features: np.ndarray,
        data_dir: Path = None,
        n_clusters: int = 15,  # 未使用，保留以保持API兼容
        train_runs: list = None,
        val_runs: list = None,
        device: str = None
    ):
        """
        Args:
            encoder: SiameseEncoder模型
            synergy_evaluator: CheckpointSynergyEvaluator实例
            checkpoint_df: checkpoint DataFrame
            features: checkpoint特征矩阵 [n_samples, n_features]
            data_dir: 数据目录（用于加载Oracle阈值）
            n_clusters: 未使用，保留以保持API兼容
            device: 计算设备
        """
        self.encoder = encoder
        self.synergy_evaluator = synergy_evaluator
        self.checkpoint_df = checkpoint_df
        self.features = features
        self.data_dir = data_dir
        
        # 加载Oracle最优阈值（仅用于记录，实际评估使用KMeans）
        if data_dir:
            try:
                self.oracle_thresholds = load_oracle_best_thresholds(data_dir)
                logger.info(f"加载Oracle阈值（参考）: {self.oracle_thresholds}")
            except Exception as e:
                logger.warning(f"加载Oracle阈值失败: {e}，使用默认阈值")
                self.oracle_thresholds = {1: 0.5}
        else:
            # 使用默认阈值
            self.oracle_thresholds = {1: 0.5}
            logger.warning("未指定data_dir，使用默认阈值0.5")
        
        # 固定每个run的cluster数量（用于KMeans）
        self.n_clusters_per_run = 30
        logger.info(f"评估策略: KMeans, 每个run固定 {self.n_clusters_per_run} 个cluster")
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.encoder.to(self.device)
        
        # 设置train/val runs
        self.train_runs = train_runs if train_runs is not None else []
        self.val_runs = val_runs if val_runs is not None else []
        
        # 训练历史 (类似RL的history格式)
        self.history = {
            'epoch': [],
            'train_mean_reward': [],
            'train_std_reward': [],
            'val_mean_reward': [],
            'val_std_reward': [],
            'loss': [],
            'learning_rate': [],
            'baseline': []  # 基线均值
        }
        
        logger.info(f"初始化ClusteringTrainer:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Encoder: {sum(p.numel() for p in encoder.parameters())} 参数")
        logger.info(f"  Features: {features.shape}")

    def train_triplet(
        self,
        triplets: list,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        margin: float = 0.5,
        weight_decay: float = 1e-5,
        eval_every: int = 5,
        early_stopping_patience: int = 20,
        save_dir: Path = None
    ):
        """
        使用Triplet Loss训练模型（基于Oracle协同增益率构造triplet样本）
        
        Args:
            triplets: [(anchor_idx, pos_idx, neg_idx), ...] 三元组列表
            n_epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            margin: Triplet loss margin
            weight_decay: L2正则化系数
            eval_every: 每N个epoch评估一次synergy
            early_stopping_patience: 早停patience
            save_dir: 模型保存目录
        """
        logger.info("="*70)
        logger.info("开始Triplet Network训练（基于Oracle协同增益率）")
        logger.info("="*70)
        
        # 数据加载器
        dataset = TripletDataset(self.features, triplets)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 创建Triplet Loss（这是训练中实际使用的损失函数）
        criterion = TripletLoss(margin=margin)
        
        # 优化器
        optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器（更aggressive）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 监控synergy最大化
            factor=0.5,  # 每次减半
            patience=5,  # 5个eval周期不提升就降低（原10，太宽松）
            min_lr=1e-5,  # 最小学习率
            verbose=True
        )
        
        # 早停
        best_synergy = -float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 训练历史 (RL-compatible格式)
        triplet_history = {
            'epoch': [],
            'loss': [],
            'train_mean_reward': [],
            'train_std_reward': [],
            'val_mean_reward': [],
            'val_std_reward': [],
            'learning_rate': [],
            'baseline': []
        }
        
        # 训练循环
        for epoch in range(n_epochs):
            self.encoder.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for anchor, positive, negative in dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # 前向传播
                emb_anchor = self.encoder(anchor)
                emb_positive = self.encoder(positive)
                emb_negative = self.encoder(negative)
                
                # 计算Triplet Loss
                triplet_loss = criterion(emb_anchor, emb_positive, emb_negative)
                
                # 添加正则化项以改善聚类效果
                # 1. 簇内紧凑：鼓励正样本对更近
                # dist_ap = F.pairwise_distance(emb_anchor, emb_positive)
                # compact_loss = dist_ap.mean()
                
                # # 2. 簇间分离：鼓励负样本对更远
                # dist_an = F.pairwise_distance(emb_anchor, emb_negative)
                # separate_loss = torch.clamp(margin - dist_an, min=0).mean()
                
                # # 组合损失：triplet(主要) + compact(0.1) + separate(0.5)
                # # 较小的compact权重以缓解全局收缩，较大的separate权重以强化簇间边界
                # loss = triplet_loss + 0.1 * compact_loss + 0.5 * separate_loss
                
                # 反向传播
                optimizer.zero_grad()
                triplet_loss.backward()
                optimizer.step()
                
                epoch_loss += triplet_loss.item()
                n_batches += 1
            
            # Epoch结束
            avg_loss = epoch_loss / max(n_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 每N个epoch评估synergy
            train_mean = None
            train_std = None
            val_mean = None
            val_std = None
            if epoch % eval_every == 0 or epoch == n_epochs - 1:
                # 评估训练集
                train_result = self.evaluate_split(self.train_runs, 'train')
                train_mean = train_result['mean_reward']
                train_std = train_result['std_reward']
                train_rewards = train_result['run_rewards']
                
                # 评估验证集
                val_result = self.evaluate_split(self.val_runs, 'val')
                val_mean = val_result['mean_reward']
                val_std = val_result['std_reward']
                val_rewards = val_result['run_rewards']
                
                logger.info(f"Epoch {epoch}/{n_epochs}:")
                logger.info(f"  Loss: {avg_loss:.4f}")
                logger.info(f"  Train Reward (per run): {train_mean:.4f} ± {train_std:.4f}")
                logger.info(f"    Runs: {', '.join(f'{r:.3f}' for r in train_rewards)}")
                logger.info(f"  Val Reward (per run): {val_mean:.4f} ± {val_std:.4f}")
                logger.info(f"    Runs: {', '.join(f'{r:.3f}' for r in val_rewards)}")
                logger.info(f"  LR: {current_lr:.6f}")
                
                # 早停检查（基于val mean reward）
                if val_mean > best_synergy:
                    best_synergy = val_mean
                    patience_counter = 0
                    # 使用深拷贝保存最佳模型权重（修复浅拷贝bug）
                    best_model_state = copy.deepcopy(self.encoder.state_dict())
                    
                    if save_dir:
                        save_path = save_dir / 'best_triplet_model.pth'
                        self.encoder.save(save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    logger.info(f"Best Val Reward: {best_synergy:.4f}")
                    break
            
            triplet_history['epoch'].append(epoch)
            triplet_history['loss'].append(avg_loss)
            triplet_history['train_mean_reward'].append(train_mean)
            triplet_history['train_std_reward'].append(train_std)
            triplet_history['val_mean_reward'].append(val_mean)
            triplet_history['val_std_reward'].append(val_std)
            triplet_history['learning_rate'].append(current_lr)
            triplet_history['baseline'].append(0.0)  # reserved baseline metric
        
        # 恢复最佳模型
        if best_model_state:
            self.encoder.load_state_dict(best_model_state)
        
        logger.info("="*70)
        logger.info("Triplet训练完成！")
        logger.info(f"最佳Synergy Ratio: {best_synergy:.4f}")
        logger.info("="*70)
        
        # 更新self.history以便save_results使用
        self.history = triplet_history
        
        # 如果指定了save_dir，自动保存结果
        if save_dir:
            self.save_results(save_dir)
        
        return triplet_history
    
    def evaluate_split(self, run_ids: list, split_name: str = ''):
        """
        评估指定runs的聚类性能，返回per-run的synergy ratio
        
        Args:
            run_ids: run ID列表
            split_name: 数据集名称（用于日志）
            
        Returns:
            dict: {
                'mean_reward': float,  # per-run平均synergy ratio
                'std_reward': float,   # per-run标准差
                'run_rewards': list    # 每个run的synergy ratio
            }
        """
        if not run_ids:
            return {'mean_reward': 0.0, 'std_reward': 0.0, 'run_rewards': []}
        
        self.encoder.eval()
        
        # 只评估指定的runs
        mask = self.checkpoint_df['source_run_id'].isin(run_ids)
        checkpoint_df_filtered = self.checkpoint_df[mask].copy()
        features_filtered = self.features[mask]
        
        # 获取embeddings
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_filtered).to(self.device)
            embeddings = self.encoder(features_tensor).cpu().numpy()
        
        # 按run分组聚类
        labels = np.zeros(len(checkpoint_df_filtered), dtype=int)
        cluster_offset = 0
        
        for run_id in sorted(run_ids):
            run_mask = checkpoint_df_filtered['source_run_id'] == run_id
            run_embeddings = embeddings[run_mask]
            
            n_samples = len(run_embeddings)
            n_clusters_actual = min(self.n_clusters_per_run, n_samples)
            
            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
            run_labels = kmeans.fit_predict(run_embeddings)
            
            labels[run_mask] = run_labels + cluster_offset
            cluster_offset += n_clusters_actual
        
        # 评估synergy（按run分组）
        result = self.synergy_evaluator.evaluate_clustering_by_run(
            checkpoint_df=checkpoint_df_filtered,
            labels=labels,
            save_result=False,
            verbose=False
        )
        
        # 提取per-run的synergy ratio
        run_rewards = []
        for run_result in result.get('run_results', []):
            run_rewards.append(run_result['sum_synergy_ratio'])
        
        mean_reward = np.mean(run_rewards)
        std_reward = np.std(run_rewards)
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'run_rewards': run_rewards
        }
    
    def evaluate(self):
        """
        评估当前模型（使用KMeans聚类 + Synergy计算）
        
        Returns:
            dict: 包含synergy评估结果
        """
        self.encoder.eval()
        
        # 获取embeddings
        with torch.no_grad():
            features_tensor = torch.FloatTensor(self.features).to(self.device)
            embeddings = self.encoder(features_tensor).cpu().numpy()
        
        # 改用KMeans固定cluster数量（避免阈值不匹配问题）
        # 按run分组聚类，每个run固定n_clusters_per_run个cluster
        labels = np.zeros(len(self.checkpoint_df), dtype=int)
        run_ids = self.checkpoint_df['source_run_id'].unique()
        
        cluster_offset = 0  # 全局cluster标签偏移
        for run_id in sorted(run_ids):
            run_mask = self.checkpoint_df['source_run_id'] == run_id
            run_embeddings = embeddings[run_mask]
            
            # 对该run使用KMeans聚类
            n_samples = len(run_embeddings)
            n_clusters_actual = min(self.n_clusters_per_run, n_samples)  # 避免cluster数超过样本数
            
            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
            run_labels = kmeans.fit_predict(run_embeddings)
            
            # 加上偏移量，确保不同run的cluster标签不重复
            labels[run_mask] = run_labels + cluster_offset
            cluster_offset += n_clusters_actual
        
        # 输出聚类统计
        stats = get_cluster_statistics(labels)
        logger.info(f"KMeans聚类结果: {stats['n_clusters']} clusters, "
                   f"avg_size={stats['avg_cluster_size']:.1f}")
        
        # 按run分组评估synergy（简洁模式）
        result = self.synergy_evaluator.evaluate_clustering_by_run(
            checkpoint_df=self.checkpoint_df,
            labels=labels,
            save_result=False,
            verbose=False
        )
        
        return result
    
    def save_results(self, save_dir: Path):
        """保存训练结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = save_dir / 'siamese_encoder.pth'
        self.encoder.save(model_path)
        
        # 保存训练历史
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"结果已保存到: {save_dir}")
        
        return {
            'model_path': model_path,
            'history_path': history_path
        }
