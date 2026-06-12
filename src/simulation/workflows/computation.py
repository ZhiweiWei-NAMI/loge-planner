"""
计算任务工作流

严格按照设计文档实现的计算任务工作流。

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

class ComputeWorkflow(BaseWorkflow):
    """计算工作流 - 编排整个计算任务流程，确保无人机先到达指定地点，再开始执行计算"""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """初始化子类时注册属性模板"""
        super().__init_subclass__(**kwargs)
        cls._register_compute_property_templates()

    @classmethod
    def _register_compute_property_templates(cls):
        """注册计算工作流属性模板"""
        cls.register_property_template('compute_location', value_type=list, required=True,
                                     description="计算任务执行的地理位置 [lat, lon, alt]")
        cls.register_property_template('job_id', value_type=str, required=True,
                                     description="计算任务ID")
        cls.register_property_template('required_gflops', value_type=float, required=True,
                                     description="需要完成的总计算量 (GFLOPs)")
        cls.register_property_template('workload_profile', value_type=dict, required=True,
                                     description="工作负载配置文件")
    
    def _get_workflow_timeout(self, properties: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """获取计算工作流的超时时间"""
        if properties:
            # 优先使用显式的timeout配置
            if 'timeout' in properties:
                return properties['timeout']
            # 否则使用deadline作为超时时间
            deadline = properties.get('deadline')
            if deadline:
                return deadline
        return 1800.0  # 默认30分钟

    def __init__(self, env, workflow_id: str, owner_agent, properties: Optional[Dict[str, Any]] = None):
        """初始化计算工作流"""
        # 先设置计算特定属性，避免在父类初始化过程中访问未定义的属性
        self.compute_location = properties.get('compute_location', [0, 0, 0]) if properties else [0, 0, 0]
        self.job_id = properties.get('job_id') if properties else None
        self.required_gflops = properties.get('required_gflops', 1000.0) if properties else 1000.0
        self.workload_profile = properties.get('workload_profile', {}) if properties else {}

        # 设置初始状态为 moving_to_compute_location
        if properties is None:
            properties = {}
        properties['initial_status'] = 'moving_to_compute_location'

        # 计算状态
        self.is_completed = False
        self.compute_result = None
        self.start_time = None
        self.end_time = None
        self.movement_completed = False

        # BaseWorkflow会自动调用_get_workflow_timeout()来获取超时时间
        super().__init__(env, workflow_id, owner_agent, properties)

        logger.info(f"创建计算工作流: {workflow_id}, 任务ID: {self.job_id}, 计算位置: {self.compute_location}")

    def _setup_transitions(self):
        """设置状态转换（重写父类方法）"""
        # 调用基础转换设置
        super()._setup_transitions()
        # 设置计算特定转换
        self._setup_compute_transitions()
    
    def _setup_compute_transitions(self):
        """设置计算工作流状态转换"""
        # 状态转换: moving_to_compute_location -> computing
        self.status_machine.add_transition(
            'moving_to_compute_location', 'computing',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_move_task_completed(event_data)
            },
            callback=lambda context: self._on_movement_completed(context)
        )

        # 状态转换: moving_to_compute_location -> completed
        self.status_machine.add_transition(
            'moving_to_compute_location', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_correct_compute_task_completed(event_data)
            },
            callback=lambda context: self._on_compute_completed(context)
        )
        # 状态转换: computing -> completed
        self.status_machine.add_transition(
            'computing', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_correct_compute_task_completed(event_data)
            },
            callback=lambda context: self._on_compute_completed(context)
        )

    def _is_move_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的移动任务完成"""
        if not isinstance(event_data, dict):
            return False

        # 从result中获取workflow_id和task_class
        result = event_data.get('result', {})
        if not isinstance(result, dict):
            return False

        workflow_id = result.get('workflow_id')
        if workflow_id != self.id:
            return False

        task_class = result.get('task_class')

        # 检查是否是移动任务
        return task_class == 'MoveToTask'

    def _is_correct_compute_task_completed(self, event_data: Dict[str, Any]) -> bool:
        """检查是否是本工作流的计算任务完成"""
        if not isinstance(event_data, dict):
            return False

        # 从result中获取workflow_id和task_class
        result = event_data.get('result', {})
        if not isinstance(result, dict):
            return False

        workflow_id = result.get('workflow_id')
        if workflow_id != self.id:
            return False
        
        completed_gflops = result.get('completed_gflops')

        task_class = result.get('task_class')

        # 检查是否是计算任务
        # 添加None检查和浮点数容差
        if task_class != 'ComputeTask':
            return False
        if completed_gflops is None:
            return False
        return completed_gflops >= self.required_gflops * 0.99  # 1% tolerance for floating point

    def _on_movement_completed(self, context: Dict[str, Any]):
        """移动任务完成回调"""
        event_data = context.get('event_value', {})

        logger.info(f"计算工作流 {self.id} 完成移动到计算位置: {self.compute_location}")

        # 更新移动状态
        self.movement_completed = True

    def _on_compute_completed(self, context: Dict[str, Any]):
        """计算任务完成回调"""
        event_data = context.get('event_value', {})

        logger.info(f"计算工作流 {self.id} 完成计算任务: {self.job_id}")

        # 更新计算状态
        self.is_completed = True
        self.end_time = self.env.now

        # 从result中获取计算结果
        result = event_data.get('result', {})
        if isinstance(result, dict):
            self.compute_result = result.get('compute_result')
    
    def get_description(self) -> Dict[str, Any]:
        """获取计算工作流的详细描述信息"""
        return {
            'workflow_type': 'computation',
            'workflow_id': self.id,
            'job_id': self.job_id,
            'current_state': self.status_machine.current_status,
            'progress': self.get_progress(),
            'checkpoints': self._get_checkpoint_descriptions(),
            'properties': {
                'compute_location': self.compute_location,
                'required_gflops': self.required_gflops,
                'workload_profile': self.workload_profile,
                'priority': self.priority
            }
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """获取计算工作流的进度信息"""
        current_state = self.status_machine.current_status
        
        # 定义状态对应的进度
        state_progress_map = {
            'moving_to_compute_location': (1, 3),
            'computing': (2, 3),
            'completed': (3, 3),
            'failed': (0, 3)
        }
        
        completed, total = state_progress_map.get(current_state, (0, 3))
        
        return {
            'current_state': current_state,
            'job_id': self.job_id,
            'is_completed': self.is_completed,
            'movement_completed': self.movement_completed,
            'completed': completed,
            'total': total,
            'progress': f'{completed}/{total}'
        }

    def _get_checkpoint_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """获取计算工作流各个checkpoint的详细描述"""
        compute_loc = self.compute_location
        required_gflops = self.required_gflops

        # 基于计算量估算执行时间（假设1GHz处理器）
        estimated_compute_time = required_gflops / 1.0  # 简化估算：1秒/GFLOP

        checkpoints = {
            'moving_to_compute_location': {
                'state_name': 'moving_to_compute_location',
                'description': f'前往计算执行位置({compute_loc[0]}, {compute_loc[1]}, {compute_loc[2]})，准备执行计算任务{self.job_id}',
                'location': compute_loc,
                'operation_type': 'movement',
                'requirements': ['mobility_component'],
                'next_states': ['computing']
            },
            'computing': {
                'state_name': 'computing',
                'description': f'在位置({compute_loc[0]}, {compute_loc[1]}, {compute_loc[2]})执行计算任务{self.job_id}，需要{required_gflops}GFLOPS计算量',
                'location': compute_loc,
                'operation_type': 'computation',
                'requirements': ['compute_component'],
                'next_states': ['completed']
            },
            'completed': {
                'state_name': 'completed',
                'description': f'计算任务{self.job_id}执行完成，已完成{required_gflops}GFLOPS计算',
                'location': compute_loc,
                'operation_type': 'completion',
                'requirements': [],
                'next_states': []
            }
        }

        return checkpoints

    def get_current_suggested_task(self) -> Optional[Dict[str, Any]]:
        """根据当前状态建议下一个任务 - 实现先移动后计算的流程"""
        current_state = self.status_machine.current_status

        if current_state == 'moving_to_compute_location':
            # 当状态为 moving_to_compute_location: 建议一个 MoveToTask
            return self._create_task_suggestion(
                component='Mobility',
                task_class='MoveToTask',
                task_name=f"Move to Compute Location for {self.job_id}",
                target_state={
                    'position': self.compute_location
                },
                properties={
                    'movement_type': 'direct',
                    'max_speed': 15.0  # m/s
                }
            )

        elif current_state == 'computing':
            # 当状态为 computing: 建议一个 ComputeTask
            # 记录开始时间
            if self.start_time is None:
                self.start_time = self.env.now

            # 构建并返回计算任务建议，将所有属性传递给ComputeTask
            return self._create_task_suggestion(
                component='Compute',
                task_class='ComputeTask',
                task_name=f"Compute Job {self.job_id}",
                target_state={},
                properties={
                    'job_id': self.job_id,
                    'required_gflops': self.required_gflops,
                    'workload_profile': self.workload_profile
                }
            )

        return None