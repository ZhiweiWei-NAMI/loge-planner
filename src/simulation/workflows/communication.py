"""
通信中继工作流

根据配置参数，管理无人机移动到指定位置并执行数据中继/传输任务。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

from airfogsim.core.enums import TriggerOperator
from .base import BaseWorkflow
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class CommunicationWorkflow(BaseWorkflow):
    """通信中继工作流"""
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """初始化子类时注册属性模板"""
        super().__init_subclass__(**kwargs)
        cls._register_communication_property_templates()
    
    @classmethod
    def _register_communication_property_templates(cls):
        """注册通信工作流属性模板"""
        cls.register_property_template('data_id', value_type=str, required=True,
                                     description="要传输的数据的唯一标识符")
        cls.register_property_template('data_size_mb', value_type=float, required=True,
                                     description="要传输的数据大小（MB）")
        cls.register_property_template('target_location', value_type=list, required=True,
                                     description="执行通信任务的目标位置坐标 [x, y, z]")
        cls.register_property_template('max_end_to_end_latency_s', value_type=float, required=False,
                                     description="允许的最大端到端延迟（秒）")
    
    def __init__(self, env, workflow_id: str, owner_agent, properties: Optional[Dict[str, Any]] = None):
        """初始化通信工作流"""
        # 设置通信特定属性
        self.data_id = properties.get('data_id') if properties else None
        self.data_size_mb = properties.get('data_size_mb') if properties else None
        self.target_location = properties.get('target_location') if properties else None
        self.max_end_to_end_latency_s = properties.get('max_end_to_end_latency_s') if properties else None
        
        # 为工作流设置初始状态，使其首先进入移动阶段
        if properties:
            properties['initial_status'] = 'moving_to_relay_position'

        # 通信状态
        self.transmission_result = None

        super().__init__(env, workflow_id, owner_agent, properties)
        logger.info(f"创建通信工作流: {self.id}, 数据ID: {self.data_id}")

    def _setup_transitions(self):
        """设置状态转换（重写父类方法）"""
        super()._setup_transitions()
        self._setup_communication_transitions()
    
    def _setup_communication_transitions(self):
        """设置通信工作流状态转换"""
        # 1. 从“移动”状态到“中继”状态的转换
        # 当移动任务完成时，工作流进入数据传输阶段
        self.status_machine.add_transition(
            'moving_to_relay_position', 'relaying_data',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': self._is_move_task_completed
            }
        )
        
        # 2. 从“中继”状态到“完成”状态的转换
        # 当通信任务完成时，工作流结束
        self.status_machine.add_transition(
            'relaying_data', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': self._is_communication_task_completed
            }
        )
        # 3. 从“移动”状态到“完成”状态的转换
        self.status_machine.add_transition(
            'moving_to_relay_position', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': self._is_communication_task_completed
            }
        )

    def _is_move_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的移动任务已完成"""
        result = event_data.get('result', {})
        if not isinstance(result, dict):
            return False
        
        # 验证是否是与本工作流相关的 MoveToTask
        return (result.get('workflow_id') == self.id and
                result.get('task_class') == 'MoveToTask') and self._calculate_distance(result.get('target_position'), self.target_location)<=10

    def _is_communication_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的通信任务已完成"""
        result = event_data.get('result', {})
        if not isinstance(result, dict):
            return False

        return (result.get('workflow_id') == self.id and
                result.get('task_class') == 'CommunicationTask') and result.get('data_size_mb')>=self.data_size_mb
    
    def get_description(self) -> Dict[str, Any]:
        """获取通信工作流的详细描述信息"""
        return {
            'workflow_type': 'communication',
            'workflow_id': self.id,
            'data_id': self.data_id,
            'current_state': self.status_machine.current_status,
            'progress': self.get_progress(),
            'checkpoints': self._get_checkpoint_descriptions(),
            'properties': {
                'target_location': self.target_location,
                'data_size_mb': self.data_size_mb,
                'max_end_to_end_latency_s': self.max_end_to_end_latency_s,
                'priority': self.priority
            }
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """获取通信工作流的进度信息"""
        current_state = self.status_machine.current_status
        
        # 定义状态对应的进度
        state_progress_map = {
            'moving_to_relay_position': (1, 3),
            'relaying_data': (2, 3),
            'completed': (3, 3),
            'failed': (0, 3)
        }
        
        completed, total = state_progress_map.get(current_state, (0, 3))
        
        return {
            'current_state': current_state,
            'data_id': self.data_id,
            'data_size_mb': self.data_size_mb,
            'transmission_result': self.transmission_result,
            'completed': completed,
            'total': total,
            'progress': f'{completed}/{total}'
        }

    def _get_checkpoint_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """获取通信工作流各个checkpoint的详细描述"""
        target_loc = self.target_location or [0, 0, 0]
        data_size = self.data_size_mb or 0
        max_latency = self.max_end_to_end_latency_s or 60

        checkpoints = {
            'moving_to_relay_position': {
                'state_name': 'moving_to_relay_position',
                'description': f'前往通信中继位置({target_loc[0]}, {target_loc[1]}, {target_loc[2]})，准备传输数据{self.data_id}',
                'location': target_loc,
                'operation_type': 'movement',
                'requirements': ['mobility_component'],
                'next_states': ['relaying_data']
            },
            'relaying_data': {
                'state_name': 'relaying_data',
                'description': f'在位置({target_loc[0]}, {target_loc[1]}, {target_loc[2]})执行数据中继，传输{data_size}MB数据，最大延迟{max_latency}秒',
                'location': target_loc,
                'operation_type': 'communication',
                'requirements': ['communication_component'],
                'next_states': ['completed']
            },
            'completed': {
                'state_name': 'completed',
                'description': f'通信任务完成，数据{self.data_id}传输成功',
                'location': target_loc,
                'operation_type': 'completion',
                'requirements': [],
                'next_states': []
            }
        }

        return checkpoints

    def get_current_suggested_task(self) -> Optional[Dict[str, Any]]:
        """根据当前状态建议下一个任务"""
        current_state = self.status_machine.current_status
        
        if current_state == 'moving_to_relay_position':
            # 检查是否需要移动
            if self._needs_move_to_location(self.target_location):
                return self._create_task_suggestion(
                    component='Mobility',
                    task_class='MoveToTask',
                    task_name=f'移动至通信位置 {self.id}',
                    target_state={'position': self.target_location},
                    properties={'target_position': self.target_location}
                )
            else:
                return None # 返回None，等待下一个仿真步处理新状态的任务建议

        elif current_state == 'relaying_data':
            # 建议通信任务
            return self._create_task_suggestion(
                component='Communication',
                task_class='CommunicationTask', 
                task_name=f'传输数据 {self.data_id}',
                target_state={'data_size_mb': self.data_size_mb},
                properties={
                    'data_id': self.data_id,
                    'data_size_mb': self.data_size_mb
                }
            )
            
        return None
        
    def _needs_move_to_location(self, target_location: List[float]) -> bool:
        """检查智能体是否需要移动到目标位置"""
        # 检查智能体当前是否有移动任务正在执行
        if self._has_active_move_task():
            return False
        
        current_position = self.owner.get_state("position", None)
        if current_position is None:
            return True  # 位置未知，需要移动
        
        distance_threshold = 5.0  # 5米阈值
        distance = self._calculate_distance(current_position, target_location)
        return distance > distance_threshold
    
    def _has_active_move_task(self) -> bool:
        """检查是否有活跃的移动任务"""
        for task in self.owner.get_component_tasks("Mobility"):
            if task.__class__.__name__ == 'MoveToTask' and not task.is_completed:
                return True
        return False
    
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点间的欧几里得距离"""
        if len(pos1) != len(pos2):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5