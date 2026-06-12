"""
物流配送工作流

严格按照设计文档实现的物流配送工作流。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

from airfogsim.core.enums import TriggerOperator
from .base import BaseWorkflow
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class LogisticsWorkflow(BaseWorkflow):
    """物流配送工作流"""
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """初始化子类时注册属性模板"""
        super().__init_subclass__(**kwargs)
        cls._register_logistics_property_templates()
    
    @classmethod
    def _register_logistics_property_templates(cls):
        """注册物流工作流属性模板"""
        cls.register_property_template('pickup_location', value_type=(list, tuple), required=True,
                                     description="取件地点坐标")
        cls.register_property_template('delivery_location', value_type=(list, tuple), required=True,
                                     description="交付地点坐标")
        cls.register_property_template('payload_weight', value_type=float, required=False,
                                     description="货物重量（kg）")
    
    def _get_workflow_timeout(self, properties: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """获取物流工作流的超时时间"""
        if properties:
            # 优先使用显式的timeout配置
            if 'timeout' in properties:
                return properties['timeout']
            # 否则使用delivery_timeout作为超时时间
            return properties.get('delivery_timeout', 1800.0)  # 默认30分钟
        return 1800.0  # 默认30分钟

    def __init__(self, env, workflow_id: str, owner_agent, properties: Optional[Dict[str, Any]] = None):
        """初始化物流工作流"""
        # BaseWorkflow会自动调用_get_workflow_timeout()来获取超时时间
        super().__init__(env, workflow_id, owner_agent, properties)

        # 物流特定属性
        self.pickup_location = self.properties.get('pickup_location')
        self.delivery_location = self.properties.get('delivery_location')
        self.payload_weight = self.properties.get('payload_weight', 1.0)
        self.payload_id = self.properties.get('payload_id')
        self.delivery_timeout = self.properties.get('delivery_timeout', 1800.0)  # 默认30分钟

        # 物品状态追踪
        self.is_picked_up = False
        self.is_delivered = False
        self.pickup_agent_id = None
        self.delivery_agent_ids = []

        # 设置初始状态
        self.status_machine.set_start_transition('picking_up')

        logger.info(f"创建物流工作流: {workflow_id}, 物品ID: {self.payload_id}")

    def _setup_transitions(self):
        """设置状态转换（重写父类方法）"""
        # 调用基础转换设置
        super()._setup_transitions()
        # 设置物流特定转换
        self._setup_logistics_transitions()
    
    def _setup_logistics_transitions(self):
        """设置物流工作流状态转换"""
        # 核心逻辑：
        # 1. picking_up -> transporting (物品被取走)
        # 2. transporting -> delivering (开始交付过程)
        # 3. transporting -> completed (直接完成交付)
        # 4. delivering -> completed (通过delivering完成交付)

        # 1) 取货任务完成 -> 进入运输状态
        self.status_machine.add_transition(
            'picking_up', 'transporting',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_pickup_task_completed(event_data)
            },
            callback=lambda context: self._on_pickup_completed(context)
        )

        # 2) 运输任务完成 -> 进入交付状态
        self.status_machine.add_transition(
            'transporting', 'delivering',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_transport_task_completed(event_data)
            },
            callback=lambda context: self._on_delivery_started(context)
        )

        # 3) 交付任务完成 -> 从运输状态直接完成
        self.status_machine.add_transition(
            'transporting', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_delivery_task_completed(event_data)
            },
            callback=lambda context: self._on_delivery_completed(context)
        )

        # 4) 交付任务完成 -> 从交付状态完成
        self.status_machine.add_transition(
            'delivering', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_delivery_task_completed(event_data)
            },
            callback=lambda context: self._on_delivery_completed(context)
        )
    
    def _pos_to_list(self, pos) -> List[float]:
        """Convert a position (dict or list) to a list of coordinates."""
        if isinstance(pos, dict):
            return [float(pos.get('x', 0)), float(pos.get('y', 0)), float(pos.get('z', 0))]
        elif isinstance(pos, (list, tuple)):
            return [float(c) for c in pos]
        return pos

    def _is_pickup_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的取货任务完成"""
        if not isinstance(event_data, dict):
            return False
        result = event_data.get('result', {})
        # 确保result不是None
        if not isinstance(result, dict):
            return False
        # 1. 检查是否是自身的workflow
        workflow_id = result.get('workflow_id')
        if workflow_id != self.id:
            return False

        pickup_position = result.get('pickup_position')
        payload_id = result.get('payload_id')
        payload_weight = result.get('payload_weight')
        # 2. 检查是否是取货任务类型
        task_class = result.get('task_class', '')
        return task_class == 'PickupTask' and self._is_at_location(pickup_position, self.pickup_location) and payload_id == self.payload_id and payload_weight == self.payload_weight
    
    def _is_at_location(self, current_pos: List[float], target_pos: List[float], tolerance: float = 5.0) -> bool:
        """检查是否在目标位置附近"""
        # Convert both positions to list format
        current_pos = self._pos_to_list(current_pos)
        target_pos = self._pos_to_list(target_pos)
        if not current_pos or not target_pos:
            return False
        
        distance = sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)) ** 0.5
        return distance <= tolerance
    
    def _is_transport_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的运输任务完成"""
        if not isinstance(event_data, dict):
            return False
        result = event_data.get('result', {})
        # 确保result不是None
        if not isinstance(result, dict):
            return False

        # 1. 检查是否是自身的workflow
        workflow_id = result.get('workflow_id')
        if workflow_id != self.id:
            return False

        # 2. 检查是否是运输任务类型
        task_class = result.get('task_class', '')
        return task_class == 'MoveToTask'

    def _is_delivery_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的交付任务完成"""
        if not isinstance(event_data, dict):
            return False
        result = event_data.get('result', {})
        # 确保result不是None
        if not isinstance(result, dict):
            return False

        delivery_position = result.get('delivery_position')
        payload_id = result.get('payload_id')
        payload_weight = result.get('payload_weight')

        # 1. 检查是否是自身的workflow
        workflow_id = result.get('workflow_id')
        if workflow_id != self.id:
            return False

        # 2. 检查是否是交付任务类型
        task_class = result.get('task_class', '')
        return task_class == 'DeliveryTask' and self._is_at_location(delivery_position, self.delivery_location) and payload_id == self.payload_id and payload_weight == self.payload_weight
    
    def _on_pickup_completed(self, context: Dict[str, Any]):
        """物品被取走回调"""
        event_data = context.get('event_value', {})
        agent_id = event_data.get('agent_id')

        self.is_picked_up = True
        self.pickup_agent_id = agent_id

        logger.info(f"物流工作流 {self.id}: 物品 {self.payload_id} 被 {agent_id} 取走")

    def _on_delivery_started(self, context: Dict[str, Any]):
        """开始交付过程回调"""
        event_data = context.get('event_value', {})
        agent_id = event_data.get('agent_id')

        logger.info(f"物流工作流 {self.id}: 开始交付物品 {self.payload_id}，执行者: {agent_id}")

    def _on_delivery_completed(self, context: Dict[str, Any]):
        """物品交付完成回调"""
        event_data = context.get('event_value', {})
        agent_id = event_data.get('agent_id')

        self.is_delivered = True
        self.delivery_agent_ids.append(agent_id)

        logger.info(f"物流工作流 {self.id}: 物品 {self.payload_id} 交付完成，执行者: {agent_id}")
    
    def get_description(self) -> Dict[str, Any]:
        """获取物流工作流的详细描述信息"""
        return {
            'workflow_type': 'logistics',
            'workflow_id': self.id,
            'payload_id': self.payload_id,
            'current_state': self.status_machine.current_status,
            'progress': self.get_progress(),
            'checkpoints': self._get_checkpoint_descriptions(),
            'properties': {
                'pickup_location': self.properties.get('pickup_location'),
                'delivery_location': self.properties.get('delivery_location'),
                'payload_weight': self.properties.get('payload_weight'),
                'priority': self.priority
            }
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """获取物流工作流的进度信息"""
        current_state = self.status_machine.current_status
        
        # 定义状态对应的进度
        state_progress_map = {
            'picking_up': (0, 4),
            'transporting': (1, 4),
            'delivering': (2, 4),
            'completed': (4, 4),
            'failed': (0, 4)
        }
        
        completed, total = state_progress_map.get(current_state, (0, 4))
        
        return {
            'current_state': current_state,
            'is_picked_up': self.is_picked_up,
            'is_delivered': self.is_delivered,
            'payload_id': self.payload_id,
            'completed': completed,
            'total': total,
            'progress': f'{completed}/{total}'
        }

    def _get_checkpoint_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """获取物流工作流各个checkpoint的详细描述"""
        pickup_loc = self.properties.get('pickup_location', [0, 0, 0])
        delivery_loc = self.properties.get('delivery_location', [0, 0, 0])
        payload_weight = self.properties.get('payload_weight', 1.0)

        checkpoints = {
            'picking_up': {
                'state_name': 'picking_up',
                'description': f'前往坐标({pickup_loc[0]}, {pickup_loc[1]}, {pickup_loc[2]})取货，物品ID: {self.payload_id}，重量: {payload_weight}kg',
                'location': pickup_loc,
                'operation_type': 'pickup',
                'requirements': ['mobility_component', 'logistics_component'],
                'next_states': ['transporting']
            },
            'transporting': {
                'state_name': 'transporting',
                'description': f'运输物品{self.payload_id}从({pickup_loc[0]}, {pickup_loc[1]}, {pickup_loc[2]})到({delivery_loc[0]}, {delivery_loc[1]}, {delivery_loc[2]})',
                'location': delivery_loc,  # 目标位置
                'operation_type': 'transport',
                'requirements': ['logistics_component'],
                'next_states': ['delivering', 'completed']
            },
            'delivering': {
                'state_name': 'delivering',
                'description': f'在坐标({delivery_loc[0]}, {delivery_loc[1]}, {delivery_loc[2]})交付物品{self.payload_id}，重量: {payload_weight}kg',
                'location': delivery_loc,
                'operation_type': 'delivery',
                'requirements': ['logistics_component'],
                'next_states': ['completed']
            },
            'completed': {
                'state_name': 'completed',
                'description': f'物流任务完成，物品{self.payload_id}已成功交付',
                'location': delivery_loc,
                'operation_type': 'completion',
                'requirements': [],
                'next_states': []
            }
        }

        return checkpoints

    def get_current_suggested_task(self) -> Optional[Dict[str, Any]]:
        """根据当前状态建议下一个任务"""
        current_state = self.status_machine.current_status
        
        if current_state == 'picking_up':
            # 检查是否需要先移动到取件位置
            if self._needs_move_to_location(self.pickup_location):
                return self._create_task_suggestion(
                    component='Mobility',
                    task_class='MoveToTask',
                    task_name='移动至取件位置',
                    target_state={'position': self.pickup_location},
                    properties={
                        'movement_type': 'path_following',
                        'target_position': self.pickup_location,
                        'priority': 'high'
                    }
                )
            
            # 建议取件任务
            return self._create_task_suggestion(
                component='Logistics',
                task_class='PickupTask',
                task_name='取件任务',
                target_state={'position': self.pickup_location},
                properties={
                    'payload_id': self.current_payload_id,
                    'pickup_location': self.pickup_location,
                    'payload_weight': self.payload_weight
                }
            )
        
        elif current_state == 'transporting':
            return self._create_task_suggestion(
                component='Mobility',
                task_class='MoveToTask',
                task_name='运输货物至交付点',
                target_state={'position': self.delivery_location},
                properties={
                    'movement_type': 'path_following',
                    'target_position': self.delivery_location,
                    'priority': 'high',
                    'payload_id': self.current_payload_id
                }
            )
        
        elif current_state == 'delivering':
            # 检查是否需要先移动到交付位置
            if self._needs_move_to_location(self.delivery_location):
                return self._create_task_suggestion(
                    component='Mobility',
                    task_class='MoveToTask',
                    task_name='移动至交付位置',
                    target_state={'position': self.delivery_location},
                    properties={
                        'movement_type': 'path_following',
                        'target_position': self.delivery_location,
                        'priority': 'high'
                    }
                )
            
            # 建议交付任务
            return self._create_task_suggestion(
                component='Logistics',
                task_class='DeliveryTask',
                task_name='交付货物',
                target_state={'position': self.delivery_location},
                properties={
                    'payload_id': self.current_payload_id,
                    'delivery_location': self.delivery_location,
                    'target_agent_id': getattr(self, 'target_agent_id', None),
                    'payload_weight': self.payload_weight
                }
            )
        
        return None
    
    def _needs_move_to_location(self, target_location: List[float]) -> bool:
        """检查智能体是否需要移动到目标位置"""
        # 检查智能体当前是否有移动任务正在执行
        if self._has_active_move_task():
            return True
        
        # 检查智能体当前位置是否已在目标位置
        current_position = self.owner.get_state("position", None)
        if current_position is None:
            return True  # 位置未知，需要移动
        
        # 计算距离，如果距离小于阈值则认为已到达
        distance_threshold = 5.0  # 5米阈值
        distance = self._calculate_distance(current_position, target_location)
        return distance > distance_threshold
    
    def _has_active_move_task(self) -> bool:
        """检查是否有活跃的移动任务"""
        # 检查智能体当前是否有MoveToTask正在执行
        for task in self.owner.get_component_tasks("Mobility"):
            if task.__class__.__name__ == 'MoveToTask':
                return True
        return False
    
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点间的欧几里得距离"""
        if len(pos1) != len(pos2):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5
