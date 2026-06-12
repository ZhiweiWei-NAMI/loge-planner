#!/usr/bin/env python3
"""
多业务规划问题自主生成和子问题交替优化求解系统

本模块实现了一个完整的多业务规划系统，包括：
1. 问题生成引擎 (Problem Generation Engine)：随机生成包含多种业务的复杂规划问题
2. 规划执行引擎 (Planning Execution Engine)：标准化的规划求解接口
3. 交叉迭代协调器 (Cross-Iterative Coordinator)：执行核心的优化循环

文件结构：
├── main.py                 # 主程序入口
├── config.yaml             # 配置文件，用于问题生成和优化策略
├── problem_generator.py    # 问题生成引擎
├── planner_executor.py     # 规划执行引擎
├── coordinator.py          # 交叉迭代协调器
└── domain_definitions.py   # 从 low_altitude_domains.py 演变而来
"""

__version__ = "1.0.0"
__author__ = "TCCN Auto Planner Team"

from .pddl_solver import PDDLSolver

__all__ = [
    "PDDLSolver"
]
