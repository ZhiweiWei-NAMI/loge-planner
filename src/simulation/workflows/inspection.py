"""
巡检监控工作流

严格按照设计文档实现的巡检监控工作流。

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

class InspectionWorkflow(BaseWorkflow):
    """巡检监控工作流"""
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """初始化子类时注册属性模板"""
        super().__init_subclass__(**kwargs)
        cls._register_inspection_property_templates()
    
    @classmethod
    def _register_inspection_property_templates(cls):
        """注册巡检工作流属性模板"""
        cls.register_property_template('inspection_points', value_type=list, required=True,
                                     description="巡检点坐标列表")
        cls.register_property_template('inspection_mode', value_type=str, required=False,
                                     description="巡检模式: sequential 或 adaptive")
        cls.register_property_template('data_collection_type', value_type=str, required=False,
                                     description="数据采集类型: image, video, sensor_data")
        cls.register_property_template('inspection_altitude', value_type=float, required=False,
                                     description="巡检高度（米）")
        cls.register_property_template('hover_duration', value_type=float, required=False,
                                     description="每个点的悬停时间（秒）")
        cls.register_property_template('detection_targets', value_type=list, required=False,
                                     description="检测目标类型列表")
    
    def _get_workflow_timeout(self, properties: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """获取巡检工作流的超时时间"""
        if properties:
            # 优先使用显式的timeout配置
            if 'timeout' in properties:
                return properties['timeout']
            # 否则根据巡检点数量和每点时间估算超时时间
            inspection_points = properties.get('inspection_points', [])
            hover_duration = properties.get('hover_duration', 30.0)
            # 估算：每个点的悬停时间 + 移动时间(60秒) + 缓冲时间(50%)
            if inspection_points:
                estimated_time = len(inspection_points) * (hover_duration + 60) * 1.5
                return max(estimated_time, 1800.0)  # 最少30分钟
        return 1800.0  # 默认30分钟

    def __init__(self, env, workflow_id: str, owner_agent, properties: Optional[Dict[str, Any]] = None):
        """初始化巡检工作流"""
        # 先设置巡检特定属性，避免在父类初始化过程中访问未定义的属性
        self.inspection_points = properties.get('inspection_points', []) if properties else []
        self.inspection_mode = properties.get('inspection_mode', 'sequential') if properties else 'sequential'
        self.data_collection_type = properties.get('data_collection_type', 'image') if properties else 'image'
        self.inspection_altitude = properties.get('inspection_altitude', 50.0) if properties else 50.0
        self.required_data_size = properties.get('required_data_size', 10) if properties else 10
        self.detection_accuracy_threshold = properties.get('detection_accuracy_threshold', 0.8) if properties else 0.8
        if len(self.inspection_points) > 0:
            properties['initial_status'] = 'inspection_point'

        # 巡检状态
        self.completed_waypoints = set()  # 已完成的巡检点坐标集合
        self.collected_data = []
        self.detected_objects = []

        # BaseWorkflow会自动调用_get_workflow_timeout()来获取超时时间
        super().__init__(env, workflow_id, owner_agent, properties)

        logger.info(f"创建巡检工作流: {workflow_id}, 巡检点数: {len(self.inspection_points)}")

    def _setup_transitions(self):
        """设置状态转换（重写父类方法）"""
        # 调用基础转换设置
        super()._setup_transitions()
        # 设置巡检特定转换
        self._setup_inspection_transitions()
    
    def _setup_inspection_transitions(self):
        """设置巡检工作流状态转换"""
        # 设置统一的巡检状态转换：inspection_point -> inspection_point
        for i in range(len(self.inspection_points)):
            self.status_machine.add_transition(
                'inspection_point', 'inspection_point',
                event_trigger={
                    'source_id': '*',
                    'event_name': 'task_completed',
                    'operator': TriggerOperator.CUSTOM,
                    'target_value': lambda event_data: self._is_inspection_task_completed(event_data, require_all=False)
                },
                callback=self._on_inspection_task_completed
            )
        # 当所有巡检点都完成时转换到completed状态
        self.status_machine.add_transition(
            'inspection_point', 'completed',
            event_trigger={
                'source_id': '*',
                'event_name': 'task_completed',
                'operator': TriggerOperator.CUSTOM,
                'target_value': lambda event_data: self._is_inspection_task_completed(event_data, require_all=True)
            },
            callback=self._on_inspection_task_completed
        )

    
    def _point_to_tuple(self, point) -> tuple:
        """Convert an inspection point (dict or list) to a tuple of coordinates."""
        if isinstance(point, dict):
            return (float(point.get('x', 0)), float(point.get('y', 0)), float(point.get('z', 0)))
        elif isinstance(point, (list, tuple)):
            return tuple(float(c) for c in point)
        return point

    def _is_inspection_task_completed(self, event_data: Dict[str, Any], require_all=False) -> bool:
        """检查是否是巡检任务完成"""
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

        # 检查是否是巡检任务
        if task_class == 'InspectionTask':
            # 检查是否有waypoint信息
            waypoint = result.get('waypoint')
            if waypoint is not None:
                # 计算未完成的巡检点：使用距离容差判断，而不是精确相等
                to_be_inspect_points = [
                    point for point in self.inspection_points
                    if not any(self._is_at_waypoint(self._point_to_tuple(point), completed) for completed in self.completed_waypoints)
                ]
                in_inspection_points = any(self._is_at_waypoint(waypoint, self._point_to_tuple(point)) for point in to_be_inspect_points)
                if not require_all: # 针对单个点的巡检检查
                    if in_inspection_points:
                        return True
                else:
                    return len(to_be_inspect_points) == 0
        return False
    
    def _on_inspection_task_completed(self, context: Dict[str, Any]):
        """巡检任务完成回调"""
        event_data = context.get('event_value', {})
        result = event_data.get('result', {})

        if not isinstance(result, dict):
            return

        # 从result中获取巡检的waypoint
        waypoint = result.get('waypoint')
        if waypoint is None:
            logger.warning(f"巡检工作流 {self.id} 收到的任务结果中没有waypoint信息")
            return

        # 将waypoint转换为tuple以便在set中使用
        waypoint_tuple = self._point_to_tuple(waypoint)

        # 使用距离容差检查这个waypoint是否靠近某个巡检点
        # OfflineTaskLoader 生成的 waypoint 是 cluster center，可能和原始 inspection_points 有误差
        matched_point = None
        for point in self.inspection_points:
            point_tuple = self._point_to_tuple(point)
            if self._is_at_waypoint(waypoint_tuple, point_tuple):
                matched_point = point_tuple
                break

        if matched_point is None:
            logger.warning(f"巡检工作流 {self.id} 收到未知巡检点的完成通知: {waypoint}")
            return

        # 添加到已完成列表（使用匹配到的原始巡检点坐标）
        self.completed_waypoints.add(matched_point)
        logger.info(f"巡检工作流 {self.id} 完成巡检点: {matched_point}, 已完成 {len(self.completed_waypoints)}/{len(self.inspection_points)}")
    
    def get_description(self) -> Dict[str, Any]:
        """获取巡检工作流的详细描述信息"""
        return {
            'workflow_type': 'inspection',
            'workflow_id': self.id,
            'current_state': self.status_machine.current_status,
            'progress': self.get_progress(),
            'checkpoints': self._get_checkpoint_descriptions(),
            'properties': {
                'inspection_points': self.inspection_points,
                'inspection_mode': self.inspection_mode,
                'data_collection_type': self.data_collection_type,
                'inspection_altitude': self.inspection_altitude,
                'detection_targets': self.properties.get('detection_targets', []),
                'priority': self.priority
            }
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """获取巡检工作流的进度信息"""
        current_state = self.status_machine.current_status
        total_points = len(self.inspection_points)
        completed_points = len(self.completed_waypoints)
        
        return {
            'current_state': current_state,
            'completed': completed_points,
            'total': total_points,
            'progress': f'{completed_points}/{total_points}'
        }

    def _get_checkpoint_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """获取巡检工作流各个checkpoint的详细描述"""
        checkpoints = {}

        # 创建统一的巡检状态描述，包含所有巡检点信息
        if self.inspection_points:
            # 构建所有巡检点的描述
            points_desc = []
            for i, point in enumerate(self.inspection_points):
                points_desc.append(f'巡检点{i+1}({point[0]}, {point[1]}, {point[2]})')

            checkpoints['inspection_point'] = {
                'state_name': 'inspection_point',
                'description': f'执行巡检任务，可在以下任意巡检点进行{self.data_collection_type}数据采集: {", ".join(points_desc)}',
                'location': self.inspection_points,  # 包含所有巡检点
                'operation_type': 'inspection',
                'requirements': ['mobility_component', 'inspection_component'],
                'next_states': ['completed']
            }

            # 添加完成状态
            checkpoints['completed'] = {
                'state_name': 'completed',
                'description': f'巡检任务完成，已完成{len(self.inspection_points)}个巡检点的数据采集',
                'location': self.inspection_points[-1],  # 使用最后一个巡检点作为完成位置
                'operation_type': 'completion',
                'requirements': [],
                'next_states': []
            }

        return checkpoints

    def get_current_suggested_task(self) -> Optional[Dict[str, Any]]:
        """根据当前状态建议下一个任务"""
        current_state = self.status_machine.current_status

        if current_state == 'inspection_point':
            # 找到下一个未完成的巡检点
            next_waypoint = self._get_next_unfinished_waypoint()
            if next_waypoint is not None:
                waypoint_index = self.inspection_points.index(next_waypoint)

                # 检查是否需要先移动到巡检点
                if self._needs_move_to_location(next_waypoint):
                    return self._create_task_suggestion(
                        component='Mobility',
                        task_class='MoveToTask',
                        task_name=f'移动至巡检点 {waypoint_index + 1}',
                        target_state={'position': next_waypoint},
                        properties={
                            'target_position': next_waypoint,
                            'priority': 'high'
                        }
                    )

                # 建议巡检任务
                return self._create_task_suggestion(
                    component='Inspection',
                    task_class='InspectionTask',
                    task_name=f'巡检点 {waypoint_index + 1}',
                    target_state={'position': next_waypoint},
                    properties={
                        'waypoint': next_waypoint,
                        'inspection_mode': self.inspection_mode,
                        'inspection_altitude': self.inspection_altitude,
                        'required_data_size': self.required_data_size,
                        'data_collection_type': self.data_collection_type,
                        'detection_accuracy_threshold': self.detection_accuracy_threshold
                    }
                )

        return None

    def _get_next_unfinished_waypoint(self) -> Optional[List[float]]:
        """获取下一个未完成的巡检点"""
        for waypoint in self.inspection_points:
            waypoint_tuple = tuple(waypoint)
            if waypoint_tuple not in self.completed_waypoints:
                return waypoint
        return None
        
    def _needs_move_to_location(self, target_location: List[float]) -> bool:
        """检查智能体是否需要移动到目标位置"""
        # 检查智能体当前是否有移动任务正在执行
        if self._has_active_move_task():
            return False
        
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
    