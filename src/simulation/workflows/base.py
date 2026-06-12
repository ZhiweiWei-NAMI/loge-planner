"""
基础工作流类

严格按照设计文档实现的工作流基类。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

from airfogsim.core.workflow import Workflow
from airfogsim.core.enums import TriggerOperator
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class BaseWorkflow(Workflow):
    """基础工作流类"""
    
    def __init__(self, env, workflow_id: str, owner_agent, properties: Optional[Dict[str, Any]] = None):
        """初始化基础工作流"""
        # 提取超时配置 - 子类可以重写此方法来设置默认超时
        timeout = self._get_workflow_timeout(properties)

        # 调用父类构造函数，传入正确的参数顺序
        # super().__init__(env, workflow_id, owner_agent, timeout=timeout, properties=properties or {})
        super().__init__(env, name=workflow_id, owner=owner_agent, timeout=timeout, properties=properties or {}, 
        initial_status=properties.get('initial_status', 'idle'))
        # 工作流基础属性
        self.workflow_type = self.__class__.__name__
        self.current_payload_id = properties.get('payload_id') if properties else None
        self.priority = properties.get('priority', 'normal') if properties else 'normal'

        # 状态机初始化
        self._setup_base_transitions()

        logger.info(f"创建 {self.workflow_type}: {workflow_id}, 超时: {timeout}秒" if timeout else f"创建 {self.workflow_type}: {workflow_id}")

    def _setup_transitions(self):
        """设置状态转换（重写父类方法）"""
        pass

    def _get_workflow_timeout(self, properties: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """获取工作流超时时间 - 子类可以重写此方法设置默认超时"""
        if properties:
            return properties.get('timeout')
        return None

    def _setup_base_transitions(self):
        """设置基础状态转换"""
        # 添加通用的失败转换（如果需要额外的失败条件）
        pass
    
    def get_description(self) -> Dict[str, Any]:
        """获取工作流的详细描述信息（子类应该重写此方法）"""
        return {
            'workflow_type': self.workflow_type,
            'workflow_id': self.id,
            'current_state': self.status_machine.current_status,
            'progress': self.get_progress(),
            'checkpoints': {},
            'properties': self.properties
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """获取工作流进度信息（子类应该重写此方法以提供具体进度）"""
        return {'current_state': self.status_machine.current_status}

    def get_current_suggested_task(self) -> Optional[Dict[str, Any]]:
        """获取当前建议的任务（子类必须实现）"""
        raise NotImplementedError("子类必须实现 get_current_suggested_task 方法")
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        return {
            'id': self.id,
            'type': self.workflow_type,
            'owner_agent': self.owner.id if self.owner else None,
            'current_state': self.status_machine.current_status,
            'status': self.status,
            'priority': self.priority,
            'payload_id': self.current_payload_id,
            'properties': self.properties
        }
    
    def _create_task_suggestion(self, component: str, task_class: str, task_name: str, 
                              target_state: Dict[str, Any], properties: Dict[str, Any]) -> Dict[str, Any]:
        """创建任务建议的辅助方法"""
        return {
            'component': component,
            'task_class': task_class,
            'task_name': task_name,
            'workflow_id': self.id,  # 关键：设置工作流ID
            'target_state': target_state,
            'properties': properties
        }
    
    def _is_delivery_location(self, position: List[float]) -> bool:
        """检查位置是否为交付地点（辅助方法）"""
        delivery_location = self.properties.get('delivery_location')
        if not delivery_location:
            return False
        
        # 简单的距离检查（容差5米）
        distance = sum((a - b) ** 2 for a, b in zip(position, delivery_location)) ** 0.5
        return distance <= 5.0
    
    def _is_pickup_location(self, position: List[float]) -> bool:
        """检查位置是否为取件地点（辅助方法）"""
        pickup_location = self.properties.get('pickup_location')
        if not pickup_location:
            return False
        
        # 简单的距离检查（容差5米）
        distance = sum((a - b) ** 2 for a, b in zip(position, pickup_location)) ** 0.5
        return distance <= 5.0
    
    def _is_at_waypoint(self, position: List[float], waypoint: List[float], tolerance: float = 5.0) -> bool:
        """检查是否到达航点（辅助方法）"""
        if not position or not waypoint:
            return False
        
        distance = sum((a - b) ** 2 for a, b in zip(position, waypoint)) ** 0.5
        return distance <= tolerance
