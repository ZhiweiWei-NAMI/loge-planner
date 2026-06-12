"""
上下文Bandit训练器

单步REINFORCE训练流程：
1. 遍历所有数据目录（每个是独立问题）
2. 对每个问题：观测→采样动作→评估奖励
3. 用本epoch收集的样本更新策略
"""

import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
from tqdm import tqdm
import copy

# 添加父目录到路径
parent_path = Path(__file__).parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

from ml_training.initial_data_loader import InitialDataLoader
from ml_training.checkpoint_extractor import CheckpointExtractor
from ml_training.synergy_evaluator import CheckpointSynergyEvaluator

from rl.observation_encoder import ObservationEncoder
from rl.weighted_kmeans import kmeans_with_weights
from rl.policy import MixedActionPolicy

logger = logging.getLogger(__name__)


class BanditTrainer:
    """
    上下文Bandit训练器（单步REINFORCE）
    """
    
    def __init__(
        self,
        policy: MixedActionPolicy,
        encoder: ObservationEncoder,
        library_hash: str = "",
        n_clusters_fixed: Optional[int] = None,
        learning_rate: float = 1e-3,
        baseline_momentum: float = 0.9,
        entropy_coef: float = 0.01,
        cluster_purity_coef: float = 0.5,
        synergy_coef: float = 0.5,
        device: str = None
    ):
        """
        Args:
            policy: 策略网络（输出weights和n_clusters）
            encoder: 观测编码器
            library_hash: 指标库哈希（用于定位预计算特征文件）
            n_clusters_fixed: 固定簇数（如果指定，则不采样n_clusters）
            learning_rate: 学习率
            baseline_momentum: 基线（移动平均）动量
            entropy_coef: 熵正则化系数（鼓励探索）
            cluster_purity_coef: 簇纯度奖励系数（默认0.5）
            synergy_coef: 协同度奖励系数（默认0.5）
            device: 计算设备
        """
        self.policy = policy
        self.encoder = encoder
        self.library_hash = library_hash
        self.n_clusters_fixed = n_clusters_fixed
        self.baseline_momentum = baseline_momentum
        self.entropy_coef = entropy_coef
        self.cluster_purity_coef = cluster_purity_coef
        self.synergy_coef = synergy_coef
        
        # 设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.policy.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 基线（移动平均奖励）
        self.baseline = None
        
        # 训练历史
        self.history = {
            'epoch': [],
            'mean_reward': [],
            'std_reward': [],
            'baseline': [],
            'policy_loss': [],
            'entropy': []
        }
        
        # 初始化信息通过train_bandit.py输出，这里不重复
    
    def _compute_cluster_purity(self, checkpoint_df: pd.DataFrame, labels: np.ndarray) -> float:
        """
        计算簇纯度（operation_type同质性）
        
        Purity定义：每个簇中最常见operation_type的占比，然后对所有簇加权平均
        
        Args:
            checkpoint_df: checkpoint数据
            labels: 聚类标签
            
        Returns:
            avg_purity: 平均簇纯度 [0, 1]，越高越好
        """
        if 'operation_type' not in checkpoint_df.columns:
            logger.warning("checkpoint_df中无operation_type列，簇纯度设为0")
            return 0.0
        
        df = checkpoint_df.copy()
        df['cluster'] = labels
        
        cluster_purities = []
        cluster_sizes = []
        
        for cluster_id in np.unique(labels):
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            if cluster_size == 0:
                continue
            
            # 计算最常见operation_type的占比
            op_counts = cluster_data['operation_type'].value_counts()
            max_count = op_counts.max() if len(op_counts) > 0 else 0
            purity = max_count / cluster_size
            
            cluster_purities.append(purity)
            cluster_sizes.append(cluster_size)
        
        if len(cluster_purities) == 0:
            return 0.0
        
        # 按簇大小加权平均
        total_size = sum(cluster_sizes)
        weighted_purity = sum(p * s for p, s in zip(cluster_purities, cluster_sizes)) / total_size
        
        return weighted_purity
    
    def _compute_weight_diversity(self, weights: np.ndarray) -> float:
        """
        计算权重多样性（基于熵）
        
        鼓励权重不均匀分布。如果所有权重≈0.5，则多样性低。
        
        Args:
            weights: 特征权重 [d]
            
        Returns:
            diversity: 权重熵的归一化值 [0, 1]，越高表示权重越不均匀
        """
        # 计算权重的标准差（简单但有效的多样性度量）
        weight_std = np.std(weights)
        
        # 归一化：理论最大std ≈ 0.289（当一半权重=0，一半=1时）
        # 实际中我们用0.2作为期望的std
        diversity_score = min(weight_std / 0.2, 1.0)
        
        return diversity_score
    
    def collect_episode(
        self,
        data_dir: Path,
        deterministic: bool = False,
        silent: bool = True
    ) -> Dict:
        """
        收集单个episode（单个问题的1次交互）
        
        Args:
            data_dir: 数据目录
            deterministic: 是否确定性采样
            silent: 是否静默模式（不输出中间日志）
            
        Returns:
            episode_data: {obs, weights, n_clusters, log_prob, reward, checkpoint_df, feature_matrix}
        """
        # 临时禁用logger（静默模式）
        if silent:
            import logging
            loggers_to_silence = [
                'ml_training.initial_data_loader',
                'ml_training.checkpoint_extractor',
                'ml_training.synergy_evaluator',
                'compute_synergy_metric'
            ]
            original_levels = {}
            for logger_name in loggers_to_silence:
                logger_obj = logging.getLogger(logger_name)
                original_levels[logger_name] = logger_obj.level
                logger_obj.setLevel(logging.WARNING)
        
        try:
            # 1. 编码观测
            obs = self.encoder.encode(data_dir).to(self.device)

            # 2. 采样动作（权重 + 簇数）
            with torch.no_grad():
                weights, n_clusters_sampled, log_prob = self.policy.sample(obs, deterministic=deterministic)

            weights_np = weights.cpu().numpy()

            # 使用固定簇数（如果指定）
            if self.n_clusters_fixed is not None:
                n_clusters = self.n_clusters_fixed
            else:
                n_clusters = n_clusters_sampled
            
            # 3. 准备聚类数据
            loader = InitialDataLoader(data_dir)
            workflow_df, agent_df = loader.load_all()
            workflows_features = loader.get_all_workflows_with_agents()
            
            # 从目录名提取run_id并添加到features（单run情况）
            import re
            run_id_match = re.search(r'run_seed_(\d+)', str(data_dir))
            if run_id_match:
                run_id = int(run_id_match.group(1))
                for wf_feat in workflows_features:
                    if 'source_run_id' not in wf_feat:
                        wf_feat['source_run_id'] = run_id
            
            # 3.5 提取/加载特征矩阵
            extractor = CheckpointExtractor()
            checkpoint_df = extractor.extract_all_checkpoints(workflows_features)
            
            target_dim = int(weights_np.shape[0])
            base_dim = 26  # 基础特征维度
            
            if target_dim == base_dim:
                # Baseline：只用基础特征，不加载预计算文件
                feature_matrix = extractor.prepare_features_for_clustering(checkpoint_df)
                if feature_matrix.shape[1] != base_dim:
                    raise AssertionError(
                        f"基础特征维度({feature_matrix.shape[1]})不等于期望的 {base_dim}"
                    )
            elif target_dim > base_dim:
                raise ValueError(
                    "Extended-feature RL training is not part of the public package. "
                    f"Expected base feature dimension {base_dim}, got {target_dim}."
                )
            else:
                raise AssertionError(
                    f"权重维度({target_dim})小于基础特征维度({base_dim})，无法聚类"
                )
            
            # 校验特征矩阵与 checkpoint_df 行数一致
            if feature_matrix.shape[0] != len(checkpoint_df):
                raise AssertionError(
                    f"特征矩阵行数({feature_matrix.shape[0]})与checkpoint数量({len(checkpoint_df)})不匹配"
                )
            
            # 提取 run_ids
            run_ids = checkpoint_df['source_run_id'].values
            
            # 4. 执行加权KMeans
            labels = kmeans_with_weights(
                features=feature_matrix,
                run_ids=run_ids,
                weights=weights_np,
                n_clusters_per_run=n_clusters,
                random_state=42,
                n_init=10
            )
            
            # 5. 评估奖励（多目标：Synergy + Cluster Purity）
            evaluator = CheckpointSynergyEvaluator(data_dir)
            result = evaluator.evaluate_clustering_by_run(
                checkpoint_df=checkpoint_df,
                labels=labels,
                save_result=False,
                verbose=False
            )
            
            synergy_reward = result['total_sum_synergy_ratio']
            
            # 计算簇纯度（operation_type同质性）
            cluster_purity = self._compute_cluster_purity(checkpoint_df, labels)
            
            # 计算权重多样性（熵，惩罚均匀权重）
            weight_diversity = self._compute_weight_diversity(weights_np)
            
            # 组合奖励: reward = α*purity + β*synergy + γ*diversity
            # diversity是奖励（不是惩罚），因为我们希望权重有差异
            reward = (synergy_reward )  # 添加10%的权重多样性奖励
            
            return {
                'obs': obs,
                'weights': weights,
                'n_clusters': n_clusters,
                'log_prob': log_prob,
                'reward': reward,
                'checkpoint_df': checkpoint_df,
                'feature_matrix': feature_matrix,
                'labels': labels
            }
        finally:
            # 恢复logger级别
            if silent:
                for logger_name, level in original_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
    
    def train_epoch(
        self,
        training_dirs: List[Path],
        progress_bar: bool = True
    ) -> Dict:
        """
        训练一个epoch（遍历所有训练数据目录）
        
        Args:
            training_dirs: 训练数据目录列表
            progress_bar: 是否显示进度条
            
        Returns:
            epoch_stats: 统计信息
        """
        self.policy.train()
        
        # 收集所有episodes
        episodes = []
        rewards = []
        
        if progress_bar:
            iterator = tqdm(training_dirs, desc="Collecting", ncols=100, 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            iterator = training_dirs
        
        # 删除try-except鲁棒性，让错误直接暴露
        for data_dir in iterator:
            episode = self.collect_episode(data_dir, deterministic=False, silent=True)
            episodes.append(episode)
            reward = episode['reward']
            rewards.append(reward)
            
            # 实时显示reward
            if progress_bar:
                iterator.set_postfix({'reward': f'{reward:.2f}', 'mean': f'{np.mean(rewards):.2f}'})
        
        # 更新基线（移动平均）
        mean_reward = np.mean(rewards)
        if self.baseline is None:
            self.baseline = mean_reward
        else:
            self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * mean_reward
        
        # 准备训练数据
        obs_batch = torch.stack([ep['obs'] for ep in episodes]).to(self.device)
        weights_batch = torch.stack([ep['weights'] for ep in episodes]).to(self.device)
        n_clusters_list = [ep['n_clusters'] for ep in episodes]
        cluster_indices = torch.tensor(
            [self.policy.get_cluster_idx(n) for n in n_clusters_list],
            dtype=torch.long
        ).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # 计算优势（奖励 - 基线）
        advantages = rewards_tensor - self.baseline
        
        # 评估动作（权重 + 簇数）
        log_probs, entropy = self.policy.evaluate_actions(
            obs=obs_batch,
            weights=weights_batch,
            cluster_indices=cluster_indices
        )
        
        # REINFORCE损失：-E[advantage * log_prob]
        policy_loss = -(advantages * log_probs).mean()
        
        # 添加熵正则化（鼓励探索）
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # 总损失
        total_loss = policy_loss + entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 计算权重统计（用于监控）
        weights_np = weights_batch.detach().cpu().numpy()
        weight_mean = weights_np.mean()
        weight_std = weights_np.std()
        weight_min = weights_np.min()
        weight_max = weights_np.max()
        
        # 统计
        epoch_stats = {
            'mean_reward': mean_reward,
            'std_reward': np.std(rewards),
            'baseline': self.baseline,
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
            'n_episodes': len(episodes),
            'weight_mean': weight_mean,
            'weight_std': weight_std,
            'weight_min': weight_min,
            'weight_max': weight_max
        }
        
        return epoch_stats
    
    def train(
        self,
        training_dirs: List[Path],
        n_epochs: int = 100,
        eval_every: int = 10,
        validation_dirs: Optional[List[Path]] = None,
        early_stopping_patience: int = 20,
        save_dir: Optional[Path] = None
    ):
        """
        完整训练流程
        
        Args:
            training_dirs: 训练数据目录列表
            n_epochs: 训练轮数
            eval_every: 每N个epoch评估一次
            validation_dirs: 验证数据目录列表（可选）
            early_stopping_patience: 早停patience
            save_dir: 模型保存目录
        """
        print("="*80)
        print(f"🚀 开始训练: {len(training_dirs)} train / {len(validation_dirs) if validation_dirs else 0} val | {n_epochs} epochs")
        print("="*80)
        
        best_val_reward = -float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # 训练一个epoch
            print(f"\n📊 Epoch {epoch+1}/{n_epochs}")
            stats = self.train_epoch(training_dirs, progress_bar=True)
            
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['mean_reward'].append(stats['mean_reward'])
            self.history['std_reward'].append(stats['std_reward'])
            self.history['baseline'].append(stats['baseline'])
            self.history['policy_loss'].append(stats['policy_loss'])
            self.history['entropy'].append(stats['entropy'])
            
            # 打印统计
            print(f"  Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
            print(f"  Loss: {stats['policy_loss']:.6f} | Entropy: {stats['entropy']:.4f}")
            print(f"  Baseline: {stats['baseline']:.4f}")
            print(f"  Weights: mean={stats['weight_mean']:.4f}, std={stats['weight_std']:.4f}, range=[{stats['weight_min']:.4f}, {stats['weight_max']:.4f}]")
            
            # 定期评估
            if epoch % eval_every == 0 or epoch == n_epochs - 1:
                # 验证集评估
                if validation_dirs:
                    print(f"   ├─ Validating...", end=' ', flush=True)
                    val_reward = self.evaluate(validation_dirs, verbose=False)
                    print(f"Val Reward: {val_reward:.4f}")
                    
                    # 早停检查
                    if val_reward > best_val_reward:
                        best_val_reward = val_reward
                        patience_counter = 0
                        best_model_state = copy.deepcopy(self.policy.state_dict())
                        
                        if save_dir:
                            save_path = save_dir / 'best_policy.pth'
                            torch.save(best_model_state, save_path)
                            print(f"   └─ ✅ 新最佳模型! (patience reset)")
                    else:
                        patience_counter += 1
                        print(f"   └─ Patience: {patience_counter}/{early_stopping_patience}")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                        break
        
        # 恢复最佳模型
        if best_model_state:
            self.policy.load_state_dict(best_model_state)
            print(f"\n🎯 恢复最佳模型 (Val Reward: {best_val_reward:.4f})")
        
        print("\n" + "="*80)
        print("✅ 训练完成!")
        print("="*80)
        
        # 保存结果
        if save_dir:
            self.save_results(save_dir)
    
    def evaluate(
        self,
        eval_dirs: List[Path],
        verbose: bool = True
    ) -> float:
        """
        评估策略（确定性采样）
        
        Args:
            eval_dirs: 评估数据目录列表
            verbose: 是否输出详细信息
            
        Returns:
            mean_reward: 平均奖励
        """
        self.policy.eval()
        
        # 删除try-except鲁棒性
        rewards = []
        for data_dir in eval_dirs:
            episode = self.collect_episode(data_dir, deterministic=True, silent=True)
            rewards.append(episode['reward'])
        
        mean_reward = np.mean(rewards)
        
        return mean_reward
    
    def save_results(self, save_dir: Path):
        """保存训练结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存策略
        policy_path = save_dir / 'policy.pth'
        torch.save(self.policy.state_dict(), policy_path)
        
        # 保存训练历史
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 保存编码器统计（静默）
        if self.encoder.stats is not None:
            stats_path = save_dir / 'encoder_stats.pth'
            torch.save(self.encoder.stats, stats_path)
