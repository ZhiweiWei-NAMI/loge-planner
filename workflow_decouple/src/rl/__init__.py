"""
RL模块 - 深度RL/上下文Bandit方案用于学习KMeans聚类权重

主要组件：
- observation_encoder: 将workflow初始信息编码为观测向量
- weighted_kmeans: 加权KMeans聚类接口
- policy: 混合动作策略（权重+簇数）
- bandit_trainer: 训练器
"""

__version__ = "0.1.0"
