"""
工作流实现

基于 AirFogSim 架构设计文档的 Workflow 设计。
使用 AirFogSim 的 Workflow 基类和元类系统，支持属性模板继承。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../src'))

from airfogsim.core.workflow import Workflow
from airfogsim.core.enums import TriggerOperator
from typing import Dict, List, Optional, Any, Union, Callable
import logging

logger = logging.getLogger(__name__)

class BaseWorkflow(Workflow):
    """
    基础工作流类
    
    基于 AirFogSim Workflow 基类实现，使用元类系统管理属性模板。
    支持：
    1. 属性模板系统（通过 WorkflowMeta 元类）
    2. 状态机和事件触发
    3. 任务建议机制
    4. 通配符事件监听
    """
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """初始化子类时注册属性模板"""
        super().__init_subclass__(**kwargs)
        
        # 注册基础属性模板
        cls._register_base_property_templates()
    
    @classmethod
    def _register_base_property_templates(cls):
        """注册基础属性模板"""
        
        # 工作流配置属性
        cls.register_property_template('workflow_type', value_type=str, required=True,
                                     description="工作流类型")
        cls.register_property_template('timeout', value_type=float, required=False,
                                     description="工作流超时时间（秒）")
        cls.register_property_template('max_retries', value_type=int, required=False,
                                     description="最大重试次数")
        
        # 事件监听配置
        cls.register_property_template('listen_to_all_agents', value_type=bool, required=False,
                                     description="是否监听所有智能体的事件")
        cls.register_property_template('event_filter', value_type=dict, required=False,
                                     description="事件过滤条件")
