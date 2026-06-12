"""
实验配置管理器

用于管理不同规划方法的实验配置和执行
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PlanningMethod(Enum):
    """规划方法枚举"""
    RANDOM = "Random"  # 随机分配
    HD_PLANNER = "HD-Planner"  # 分层解耦规划器（当前方法）
    MONOLITHIC_PDDL = "Monolithic-PDDL"  # 单体PDDL规划
    GREEDY = "Greedy"  # 贪心调度


@dataclass
class ExperimentConfig:
    """实验配置"""
    method: PlanningMethod
    enable_online_model: bool
    planning_mode: str  # 'hierarchical', 'monolithic', 'greedy'
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'enable_online_model': self.enable_online_model,
            'planning_mode': self.planning_mode,
            'description': self.description
        }


class ExperimentManager:
    """实验管理器"""
    
    # 预定义的实验配置
    EXPERIMENT_CONFIGS = {
        PlanningMethod.RANDOM: ExperimentConfig(
            method=PlanningMethod.RANDOM,
            enable_online_model=False,
            planning_mode='none',
            description='随机分配工作流到智能体'
        ),
        PlanningMethod.HD_PLANNER: ExperimentConfig(
            method=PlanningMethod.HD_PLANNER,
            enable_online_model=False,
            planning_mode='none',
            description='分层解耦规划：规则聚类 + 分组PDDL求解'
        ),
        PlanningMethod.MONOLITHIC_PDDL: ExperimentConfig(
            method=PlanningMethod.MONOLITHIC_PDDL,
            enable_online_model=False,
            planning_mode='none',
            description='单体PDDL规划：直接构建完整PDDL并求解'
        ),
        PlanningMethod.GREEDY: ExperimentConfig(
            method=PlanningMethod.GREEDY,
            enable_online_model=False,
            planning_mode='none',
            description='贪心调度：针对每个工作流选择一个智能体'
        )
    }
    
    @classmethod
    def get_config(cls, method: PlanningMethod) -> ExperimentConfig:
        """获取指定方法的配置
        
        Args:
            method: 规划方法
            
        Returns:
            实验配置
        """
        return cls.EXPERIMENT_CONFIGS[method]
    
    @classmethod
    def get_all_methods(cls):
        """获取所有规划方法"""
        return list(cls.EXPERIMENT_CONFIGS.keys())
    
    @classmethod
    def print_config(cls, method: PlanningMethod):
        """打印配置信息"""
        config = cls.get_config(method)
        logger.info(f"\n{'='*60}")
        logger.info(f"实验配置: {config.method.value}")
        logger.info(f"{'='*60}")
        logger.info(f"启用在线模型: {config.enable_online_model}")
        logger.info(f"规划模式: {config.planning_mode}")
        logger.info(f"描述: {config.description}")
        logger.info(f"{'='*60}\n")
