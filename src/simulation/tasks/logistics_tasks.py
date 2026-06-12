"""
物流任务实现

实现物流配送相关的任务：PickupTask、TransportTask、DeliveryTask

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from airfogsim.core.task import Task
from airfogsim.core.enums import TaskStatus
from typing import Dict, List, Optional, Any, Tuple
import math
from airfogsim.utils.logging_config import get_logger

logger = get_logger(__name__)

class PickupTask(Task):
    """
    Pick up package or item at the current location.

    This task picks up a package or item at the agent's current position.
    The agent should be positioned at the pickup location before starting this task.

    Required parameters:
    - target_state: Dict containing 'position' key with pickup coordinates [x, y, z]

    Required properties:
    - payload_id: Unique identifier for the package/item to pick up, extract from the workflow information
    - payload_weight: Weight of the package in kg, extract from the workflow information

    Example usage:
    properties = {'payload_id': 'package_001', 'payload_weight': 2.5}
    """

    # AirFogSim 要求的类属性 - 必须与 DroneAgent 的状态模板匹配
    PRODUCED_STATES = ['current_payload_weight', 'payload_ids', 'attached_items', 'delivery_status']
    NECESSARY_METRICS = ['pickup_processing_time']

    def __init__(self, env, agent, component_name, task_name,
                 workflow_id=None, target_state=None, properties=None):
        """
        初始化取件任务
        
        Args:
            env: 仿真环境
            agent: 执行任务的智能体
            component_name: 组件名称
            task_name: 任务名称
            workflow_id: 工作流ID
            target_state: 目标状态
            properties: 任务属性
        """
        super().__init__(env, agent, component_name, task_name, workflow_id, target_state, properties)
        
        # 从target_state和properties获取任务参数
        props = properties or {}
        self.pickup_location = target_state.get('position', (0, 0, 0))
        self.payload_id = props.get('payload_id', 'unknown')
        self.payload_weight = props.get('payload_weight', 1.0)
        
        # 任务状态
        self.pickup_completed = False
        self.pickup_start_time = None
        self.estimated_pickup_time = 2.0  # 默认取件时间2秒
    
    def estimate_remaining_time(self, performance_metrics: Dict) -> float:
        """估算剩余时间"""
        # 获取取件处理时间
        pickup_processing_time = performance_metrics.get('pickup_processing_time', 30.0)  # 秒

        if self.pickup_completed:
            return 0.0

        # 检查是否在取件位置附近
        current_position = self.agent.get_state('position')
        if not self._is_at_location(current_position, self.pickup_location):
            return float('inf')  # 不在位置附近，无法估算

        # 使用component提供的处理时间
        if self.pickup_start_time is None:
            return pickup_processing_time
        else:
            elapsed_time = self.env.now - self.pickup_start_time
            return max(0.0, pickup_processing_time - elapsed_time)
    
    def _update_task_state(self, performance_metrics: Dict):
        """更新任务进度和内部状态"""
        # 检查智能体是否在取件位置附近
        current_position = self.agent.get_state('position')
        if not self._is_at_location(current_position, self.pickup_location):
            return  # 不在位置附近，无法执行取件
        
        if self.pickup_completed:
            return  # 取件已完成

        # 检查载重能力
        current_payload = self.agent.get_state('current_payload_weight') or 0.0
        payload_capacity = self.agent.get_state('payload_capacity') or 5.0
        
        if current_payload + self.payload_weight > payload_capacity:
            return  # 载重超限，无法取件
        
        # 开始取件过程
        if self.pickup_start_time is None:
            self.pickup_start_time = self.env.now
        
        # 获取取件处理时间
        pickup_processing_time = performance_metrics.get('pickup_processing_time', 30.0)  # 秒

        # 检查是否完成取件
        elapsed_time = self.env.now - self.pickup_start_time
        if elapsed_time >= pickup_processing_time and not self.pickup_completed:
            self._complete_pickup()
    
    def _complete_pickup(self):
        """完成取件操作"""
        # 更新智能体状态
        current_payload = self.agent.get_state('current_payload_weight') or 0.0
        new_payload_weight = current_payload + self.payload_weight
        self.agent.set_state('current_payload_weight', new_payload_weight)
        
        # 更新载荷列表
        payload_ids = self.agent.get_state('payload_ids') or []
        payload_ids.append(self.payload_id)
        self.agent.set_state('payload_ids', payload_ids)
        
        attached_items = self.agent.get_state('attached_items') or []
        attached_items.append({
            'id': self.payload_id,
            'weight': self.payload_weight,
            'pickup_time': self.env.now,
            'pickup_location': self.pickup_location
        })
        self.agent.set_state('attached_items', attached_items)
        
        # 更新配送状态
        self.agent.set_state('delivery_status', 'transporting')
        
        self.pickup_completed = True
    
    def _get_task_specific_state_repr(self) -> Dict:
        """返回任务特定状态的表示"""
        current_position = self.agent.get_state('position')
        at_pickup_location = self._is_at_location(current_position, self.pickup_location)
        
        return {
            'current_payload_weight': self.agent.get_state('current_payload_weight', 0.0),
            'payload_ids': self.agent.get_state('payload_ids', []),
            'attached_items': self.agent.get_state('attached_items', []),
            'delivery_status': self.agent.get_state('delivery_status', 'idle')
        }
    
    def _is_at_location(self, current_pos: Tuple, target_pos: Tuple, threshold: float = 5.0) -> bool:
        """检查是否在目标位置附近"""
        if not current_pos or not target_pos:
            return False
        
        distance = sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)) ** 0.5
        return distance <= threshold
    
    def complete(self):
        super().complete()
        self.result.update({
            'pickup_position': self.pickup_location,
            'payload_id': self.payload_id,
            'payload_weight': self.payload_weight
        })
        
class DeliveryTask(Task):
    """
    Deliver package to destination

    Required parameters:
    - target_state: Dict containing 'position' key with delivery coordinates [x, y, z]

    Required properties:
    - payload_id: Unique identifier for the package/item to pick up, extract from the workflow information
    - payload_weight: Weight of the package in kg, extract from the workflow information
    """

    # AirFogSim 要求的类属性 - 必须与 DroneAgent 的状态模板匹配
    PRODUCED_STATES = ['payload_ids', 'current_payload_weight', 'attached_items', 'delivery_status']
    NECESSARY_METRICS = ['handover_processing_time']

    def __init__(self, env, agent, component_name, task_name,
                 workflow_id=None, target_state=None, properties=None):
        """
        初始化交付任务
        
        Args:
            env: 仿真环境
            agent: 执行任务的智能体
            component_name: 组件名称
            task_name: 任务名称
            workflow_id: 工作流ID
            target_state: 目标状态
            properties: 任务属性
        """
        super().__init__(env, agent, component_name, task_name, workflow_id, target_state, properties)
        
        # 从target_state和properties获取任务参数
        props = properties or {}
        self.delivery_location = target_state.get('position', (0, 0, 0))
        self.payload_id = props.get('payload_id', 'unknown')
        self.payload_weight = props.get('payload_weight', 1.0)
        
        # 任务状态
        self.delivery_completed = False
        self.delivery_start_time = None
        self.estimated_delivery_time = 3.0  # 默认交付时间3秒
    
    def estimate_remaining_time(self, performance_metrics: Dict) -> float:
        """估算剩余时间"""
        # 获取交接处理时间
        handover_processing_time = performance_metrics.get('handover_processing_time', 40.0)  # 秒

        if self.delivery_completed:
            return 0.0

        # 检查是否在交付位置附近
        current_position = self.agent.get_state('position')
        if not self._is_at_location(current_position, self.delivery_location):
            return float('inf')  # 不在位置附近，无法估算

        # 检查是否有要交付的货物
        payload_ids = self.agent.get_state('payload_ids') or []
        if self.payload_id not in payload_ids:
            return float('inf')  # 没有要交付的货物

        # 使用component提供的处理时间
        if self.delivery_start_time is None:
            return handover_processing_time
        else:
            elapsed_time = self.env.now - self.delivery_start_time
            return max(0.0, handover_processing_time - elapsed_time)
    
    def _update_task_state(self, performance_metrics: Dict):
        """更新任务进度和内部状态"""
        # 检查智能体是否在交付位置附近
        current_position = self.agent.get_state('position')
        if not self._is_at_location(current_position, self.delivery_location):
            return  # 不在位置附近，无法执行交付
        
        # 检查是否有要交付的货物
        payload_ids = self.agent.get_state('payload_ids') or []
        if self.payload_id not in payload_ids:
            return  # 没有要交付的货物
        
        # 开始交付过程
        if self.delivery_start_time is None:
            self.delivery_start_time = self.env.now
        
        # 获取交接处理时间
        handover_processing_time = performance_metrics.get('handover_processing_time', 40.0)  # 秒

        # 检查是否完成交付
        elapsed_time = self.env.now - self.delivery_start_time
        if elapsed_time >= handover_processing_time and not self.delivery_completed:
            self._complete_delivery()
    
    def _complete_delivery(self):
        """完成交付操作"""
        # 从智能体移除货物
        payload_ids = self.agent.get_state('payload_ids') or []
        if self.payload_id in payload_ids:
            payload_ids.remove(self.payload_id)
            self.agent.set_state('payload_ids', payload_ids)
        
        # 更新载重
        attached_items = self.agent.get_state('attached_items') or []
        delivered_item = None
        for item in attached_items:
            if item.get('id') == self.payload_id:
                delivered_item = item
                break
        
        if delivered_item:
            current_weight = self.agent.get_state('current_payload_weight') or 0.0
            new_weight = max(0.0, current_weight - delivered_item.get('weight', 0.0))
            self.agent.set_state('current_payload_weight', new_weight)
            
            # 移除物品
            attached_items = [item for item in attached_items if item.get('id') != self.payload_id]
            self.agent.set_state('attached_items', attached_items)
        
        # 更新配送状态
        remaining_payloads = self.agent.get_state('payload_ids') or []
        if not remaining_payloads:
            self.agent.set_state('delivery_status', 'completed')
        else:
            self.agent.set_state('delivery_status', 'transporting')
        
        self.delivery_completed = True
    
    def _get_task_specific_state_repr(self) -> Dict:
        """返回任务特定状态的表示"""
        current_position = self.agent.get_state('position')
        at_delivery_location = self._is_at_location(current_position, self.delivery_location)
        
        return {
            'current_payload_weight': self.agent.get_state('current_payload_weight', 0.0),
            'payload_ids': self.agent.get_state('payload_ids', []),
            'attached_items': self.agent.get_state('attached_items', []),
            'delivery_status': self.agent.get_state('delivery_status', 'idle')
        }
    
    def _is_at_location(self, current_pos: Tuple, target_pos: Tuple, threshold: float = 5.0) -> bool:
        """检查是否在目标位置附近"""
        if not current_pos or not target_pos:
            return False
        
        distance = sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)) ** 0.5
        return distance <= threshold

    def complete(self):
        super().complete()
        self.result.update({
            'delivery_position': self.delivery_location,
            'payload_id': self.payload_id,
            'payload_weight': self.payload_weight
        })