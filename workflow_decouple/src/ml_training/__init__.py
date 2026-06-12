"""
ML训练模块 - 基于初始信息的聚类算法训练

正确的流程：
1. 初始信息：workflow_init_info.csv + agent初始状态（从agent_snapshots.csv第一次出现时提取）
2. 从workflow定义中提取checkpoints（非预测）
3. 聚类算法：仅使用初始信息对checkpoint进行聚类
4. 评估：聚类结果 + 运行时日志（实际checkpoint执行） → synergy ratio
5. 对比：不同算法 → 不同synergy ratio → 选最优

@author: TCCN Auto Planner
"""

from .initial_data_loader import InitialDataLoader
from .checkpoint_extractor import CheckpointExtractor

__all__ = [
    'InitialDataLoader',
    'CheckpointExtractor'
]
