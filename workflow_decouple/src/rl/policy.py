"""
混合动作策略网络

输出：
1. Beta分布参数 (alpha, beta) 用于采样权重 w∈[0,1]^D
2. Categorical分布 logits 用于采样簇数 k∈{20,30,...,100}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MixedActionPolicy(nn.Module):
    """
    混合动作策略网络
    
    输入：观测向量 o 
    输出：(alpha, beta, cluster_logits)
    """
    
    def __init__(
        self,
        obs_dim: int = 120,
        weight_dim: int = 26,
        hidden_dims: list = [256, 256],
        cluster_choices: list = None,
        dropout: float = 0.1
    ):
        """
        Args:
            obs_dim: 观测维度
            weight_dim: 权重维度（特征维度）
            hidden_dims: 隐藏层维度列表
            cluster_choices: 簇数候选列表，默认[10,20,30,40,50]
            dropout: Dropout比例
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.weight_dim = weight_dim
        
        if cluster_choices is None:
            cluster_choices = [10, 20, 30, 40, 50]  # 离散选择空间
        self.cluster_choices = cluster_choices
        self.n_cluster_choices = len(cluster_choices)
        
        # 共享backbone
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        
        # 权重分支：输出 (alpha, beta)
        self.weight_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, weight_dim * 2)  # alpha和beta
        )
        
        # 簇数分支：输出 logits
        self.cluster_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_cluster_choices)
        )
        
        # 初始化权重，让模型偏好扩展特征（如果 weight_dim > 26）
        self._initialize_with_extended_feature_prior()
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: [batch, obs_dim] 或 [obs_dim]
            
        Returns:
            alpha: [batch, weight_dim] 或 [weight_dim]
            beta: [batch, weight_dim] 或 [weight_dim]
            cluster_logits: [batch, n_cluster_choices] 或 [n_cluster_choices]
        """
        # 确保obs是2维
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 共享特征
        features = self.backbone(obs)
        
        # 权重分支
        weight_params = self.weight_head(features)
        alpha_raw, beta_raw = torch.chunk(weight_params, 2, dim=-1)
        
        # 确保alpha, beta > 0，使用softplus + 小偏移
        alpha = F.softplus(alpha_raw) + 0.1
        beta = F.softplus(beta_raw) + 0.1
        
        # 簇数分支
        cluster_logits = self.cluster_head(features)
        
        if squeeze_output:
            alpha = alpha.squeeze(0)
            beta = beta.squeeze(0)
            cluster_logits = cluster_logits.squeeze(0)
        
        return alpha, beta, cluster_logits
    
    def _initialize_with_extended_feature_prior(self):
        """
        初始化策略，让模型在初始时更偏好扩展特征

        策略：
        - 基础特征（前26维）：正常初始化，alpha ≈ beta → 初始权重均值 ≈ 0.5
        - 扩展特征（26维之后）：bias 初始化让 alpha > beta → 初始权重均值 ≈ 0.65-0.7

        这样既偏好扩展特征，又保留足够的学习空间（不是完全固定）
        """
        # BASE_FEATURE_DIM: 26维基础特征 (3位置 + 8op_type + 5requirements + 10数值)
        BASE_FEATURE_DIM = 26
        if self.weight_dim <= BASE_FEATURE_DIM:
            # 没有扩展特征，使用默认初始化
            return
        
        # 获取 weight_head 的最后一层
        last_layer = self.weight_head[-1]
        if not isinstance(last_layer, nn.Linear):
            logger.warning("weight_head 最后一层不是 Linear，跳过扩展特征初始化")
            return

        with torch.no_grad():
            # 调整 bias 使得扩展特征的 alpha 初始值更高
            # weight_params = [alpha_1, ..., alpha_D, beta_1, ..., beta_D]
            D = self.weight_dim
            
            # 基础特征（前26维）：保持默认初始化
            # alpha_bias[0:26] 和 beta_bias[0:26] 保持不变
            
            # 扩展特征（26维之后）：提高 alpha 的 bias
            # 让初始 alpha ≈ 2.5, beta ≈ 1.5 → 均值 = 2.5/(2.5+1.5) = 0.625
            # softplus(x) + 0.1 ≈ 2.5 → x ≈ 2.2
            extended_alpha_bias = 2.2  # 对应 alpha ≈ 2.5
            extended_beta_bias = 1.0   # 对应 beta ≈ 1.5

            # 设置扩展特征的 bias
            last_layer.bias[BASE_FEATURE_DIM:D] = extended_alpha_bias
            last_layer.bias[D+BASE_FEATURE_DIM:] = extended_beta_bias

        n_extended_features = self.weight_dim - BASE_FEATURE_DIM
        logger.info(f"✓ 初始化策略: 前{BASE_FEATURE_DIM}维均衡初始化，后{n_extended_features}维扩展特征偏好初始化")
        logger.info("  预期初始权重: 基础特征≈0.5, 扩展特征≈0.62")
    
    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        采样动作
        
        Args:
            obs: [obs_dim] 单个观测
            deterministic: 是否确定性采样（用均值）
            
        Returns:
            weights: [weight_dim] 权重向量
            n_clusters: int 簇数
            log_prob: scalar 总对数概率
        """
        alpha, beta, cluster_logits = self.forward(obs)
        
        # 采样权重
        if deterministic:
            # 使用Beta分布的均值
            weights = alpha / (alpha + beta)
        else:
            beta_dist = Beta(alpha, beta)
            weights = beta_dist.rsample()  # 使用rsample支持梯度
        
        # 采样簇数
        cluster_dist = Categorical(logits=cluster_logits)
        if deterministic:
            cluster_idx = torch.argmax(cluster_logits)
        else:
            cluster_idx = cluster_dist.sample()
        
        n_clusters = self.cluster_choices[cluster_idx.item()]
        
        # 计算总对数概率
        if deterministic:
            log_prob = torch.tensor(0.0)
        else:
            log_prob_weights = beta_dist.log_prob(weights).sum()  # 各维度独立
            log_prob_cluster = cluster_dist.log_prob(cluster_idx)
            log_prob = log_prob_weights + log_prob_cluster
        
        return weights, n_clusters, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        weights: torch.Tensor,
        cluster_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率（用于训练）
        
        Args:
            obs: [batch, obs_dim]
            weights: [batch, weight_dim]
            cluster_indices: [batch] 簇数索引
            
        Returns:
            log_probs: [batch] 总对数概率
            entropy: [batch] 总熵
        """
        alpha, beta, cluster_logits = self.forward(obs)
        
        # 权重的对数概率
        beta_dist = Beta(alpha, beta)
        log_prob_weights = beta_dist.log_prob(weights).sum(dim=-1)  # [batch]
        
        # 簇数的对数概率
        cluster_dist = Categorical(logits=cluster_logits)
        log_prob_cluster = cluster_dist.log_prob(cluster_indices)  # [batch]
        
        # 总对数概率
        log_probs = log_prob_weights + log_prob_cluster
        
        # 总熵（用于探索鼓励）
        entropy_weights = beta_dist.entropy().sum(dim=-1)  # [batch]
        entropy_cluster = cluster_dist.entropy()  # [batch]
        entropy = entropy_weights + entropy_cluster
        
        return log_probs, entropy
    
    def get_cluster_idx(self, n_clusters: int) -> int:
        """获取簇数对应的索引"""
        return self.cluster_choices.index(n_clusters)


def test_policy():
    """测试策略网络"""
    print("测试MixedActionPolicy...")
    
    policy = MixedActionPolicy(
        obs_dim=120,
        weight_dim=26,
        hidden_dims=[256, 256],
        cluster_choices=[10, 20, 30, 40, 50]
    )
    
    # 测试forward
    obs = torch.randn(4, 120)
    alpha, beta, cluster_logits = policy(obs)
    print(f"Alpha shape: {alpha.shape}, range: [{alpha.min():.2f}, {alpha.max():.2f}]")
    print(f"Beta shape: {beta.shape}, range: [{beta.min():.2f}, {beta.max():.2f}]")
    print(f"Cluster logits shape: {cluster_logits.shape}")
    
    # 测试sample
    obs_single = torch.randn(120)
    weights, n_clusters, log_prob = policy.sample(obs_single)
    print(f"\nSample结果:")
    print(f"  Weights shape: {weights.shape}, range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"  N_clusters: {n_clusters}")
    print(f"  Log_prob: {log_prob:.4f}")
    
    # 测试evaluate_actions
    obs_batch = torch.randn(8, 120)
    weights_batch = torch.rand(8, 26)
    cluster_indices_batch = torch.randint(0, 9, (8,))
    log_probs, entropy = policy.evaluate_actions(obs_batch, weights_batch, cluster_indices_batch)
    print(f"\nEvaluate actions:")
    print(f"  Log_probs shape: {log_probs.shape}, mean: {log_probs.mean():.4f}")
    print(f"  Entropy shape: {entropy.shape}, mean: {entropy.mean():.4f}")
    
    print("\n✓ 策略网络测试通过")


if __name__ == '__main__':
    test_policy()
