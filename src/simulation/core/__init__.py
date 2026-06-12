"""
核心框架模块

基于 AirFogSim 架构设计文档实现的核心组件：
- Agent: 统一的智能体基类（使用 AirFogSim 的 Agent 基类和元类系统）
- Component: 组件抽象和性能指标计算（基于 AirFogSim Component）
- Task: 任务执行和状态管理（基于 AirFogSim Task，增强工作流集成）
- Workflow: 工作流状态机和事件触发（基于 AirFogSim Workflow 和元类系统）
- Event: 事件系统和通配符支持（基于 AirFogSim EventRegistry）
- State: 状态管理和模板系统（独立实现）

@author: TCCN Auto Planner
"""

from .agent import DroneAgent
from .component import Component, EnergyComponent
from .task import Task
from .workflow import BaseWorkflow
from .data_collector import DataCollector

__all__ = [
    'DroneAgent',
    'Component',
    'EnergyComponent',
    'Task',
    'BaseWorkflow',
    'DataCollector'
]
