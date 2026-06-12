"""
Deep Learning模块：基于Siamese Networks的Checkpoint聚类

使用MLP作为encoder，学习checkpoint的embedding表示，
然后通过聚类和synergy评估进行端到端优化。
"""

from .siamese_encoder import SiameseEncoder, TripletLoss, SiameseContrastiveLoss
from .clustering_trainer import ClusteringTrainer
from .synergy_pair_generator import SynergyPairGenerator

__all__ = [
    'SiameseEncoder',
    'TripletLoss',
    'SiameseContrastiveLoss',
    'ClusteringTrainer', 
    'SynergyPairGenerator',
]
