"""巡检任务实现

实现巡检相关的任务：InspectionTask

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

class InspectionTask(Task):
    """
    Perform inspection at the current location.

    This task performs inspection/surveillance at the agent's current position.
    The agent should be positioned at the inspection point before starting this task.

    Required parameters:
    - target_state: Dict containing 'position' key with inspection coordinates [x, y, z]

    Optional properties:
    - No optional properties, all parameters are provided by the workflow
    """

    # AirFogSim 要求的类属性 - 严格按照设计文档
    # InspectionTask的PRODUCED_STATES - 必须与DroneAgent状态模板匹配
    PRODUCED_STATES = [
        'sensor_status',
        'data_collection_rate'
    ]

    # 严格按照 InspectionComponent 的 PRODUCED_METRICS
    NECESSARY_METRICS = [
        'sensing_capability',    # ratio - 感知能力 0-1
        'detection_accuracy',    # ratio - 检测精度 0-1
        'data_collection_rate'   # MB/s - 数据采集速率
    ]

    def __init__(self, env, agent, component_name, task_name,
                 workflow_id=None, target_state=None, properties=None):
        """
        初始化巡检任务

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

        # 从target_state获取巡检参数
        self.inspection_position = target_state.get('position')

        # 从properties获取任务参数 - 严格使用workflow提供的参数
        props = properties or {}
        self.point_index = props.get('point_index', 0)
        self.waypoint = props.get('waypoint', self.inspection_position)
        self.inspection_mode = props.get('inspection_mode', 'sequential')
        self.inspection_altitude = props.get('inspection_altitude', 50.0)
        self.required_data_size = props.get('required_data_size', 10)  # MB - 需要收集的数据量
        self.data_collection_type = props.get('data_collection_type', 'image')
        self.detection_accuracy_threshold = props.get('detection_accuracy_threshold', 0.8)

        # 巡检状态
        self.inspection_status = 'preparing'
        self.start_time = None
        self.elapsed_time = 0.0
        self.progress = 0.0
        self.collected_data_size = 0.0  # MB - 已收集的数据量
        self.last_update_time = env.now

        # 与DroneAgent状态模板对应的属性
        self.data_collection_rate = 0.0
        self.storage_available = self.agent.get_state('storage_available', 100.0)

        # 初始化巡检状态
        self._initialize_inspection()
    
    def _initialize_inspection(self):
        """初始化巡检状态"""
        # 获取当前位置
        self.current_position = self.agent.get_state('position')
        if not self.current_position:
            self.current_position = (0, 0, 0)
        
        # 设置开始时间
        self.start_time = self.env.now
        
    
    def estimate_remaining_time(self, performance_metrics: Dict) -> float:
        """估算剩余时间 - 基于数据收集进度"""
        # 获取数据采集速率
        data_collection_rate = performance_metrics.get('data_collection_rate', 1.0)  # MB/s
        detection_accuracy = performance_metrics.get('detection_accuracy', 0.5)

        if self.progress >= 1.0:
            return 0.0

        # 检查是否在巡检位置附近,且检测精度满足要求
        if not self._is_at_inspection_position() or detection_accuracy < self.detection_accuracy_threshold:
            return float('inf')

        # 检查存储空间是否足够
        storage_available = self.agent.get_state('storage_available', 100.0)
        if storage_available <= 0:
            return float('inf')  # 存储空间不足

        # 基于数据收集进度估算剩余时间
        remaining_data = self.required_data_size - self.collected_data_size
        if remaining_data <= 0 or data_collection_rate <= 0:
            return 0.0

        return remaining_data / data_collection_rate
    
    def _update_task_state(self, performance_metrics: Dict):
        """更新任务状态 - 基于数据收集进度"""
        current_time = self.env.now
        time_delta = current_time - self.last_update_time

        if self.start_time is None:
            self.start_time = current_time

        self.elapsed_time = current_time - self.start_time

        # 获取必要的性能指标
        detection_accuracy = performance_metrics.get('detection_accuracy', 0.5)
        data_collection_rate = performance_metrics.get('data_collection_rate', 1.0)  # MB/s
        storage_available = self.agent.get_state('storage_available', 100.0)

        # 基于性能指标更新巡检进度
        if self._is_at_inspection_position() and detection_accuracy >= self.detection_accuracy_threshold and storage_available > 0:
            # 更新巡检状态
            self.inspection_status = 'inspecting'

            # 计算本周期收集的数据量
            data_collected_this_cycle = data_collection_rate * time_delta

            # 检查存储空间限制
            max_collectible = min(data_collected_this_cycle, storage_available)
            self.collected_data_size += max_collectible

            # 更新与DroneAgent状态模板对应的属性
            self.data_collection_rate = data_collection_rate
            self.storage_available = max(0.0, storage_available - max_collectible)

            # 基于数据收集量计算巡检进度
            if self.required_data_size > 0:
                self.progress = min(1.0, self.collected_data_size / self.required_data_size)
            else:
                self.progress = 1.0  # 如果不需要收集数据，立即完成

            # 检查是否完成
            if self.progress >= 1.0:
                self.inspection_status = 'completed'
        else:
            # 非巡检状态时的默认值
            self.data_collection_rate = 0.0
            if self.inspection_status != 'completed':
                self.inspection_status = 'waiting'

        # 更新Agent的inspection_power_consumption状态（由Task产生，EnergyComponent监听）
        if self.inspection_status == 'inspecting':
            power_consumption = 50.0  # 巡检时功耗
        else:
            power_consumption = 10.0  # 待机功耗

        self.agent.set_state('inspection_power_consumption', power_consumption)

        self.last_update_time = current_time
    
    # 移除_update_battery_level方法，电池管理由EnergyComponent统一处理
    
    def _get_task_specific_state_repr(self) -> Dict[str, Any]:
        """
        获取任务特定状态表示
        返回与DroneAgent状态模板对应的状态值
        """
        # sensor_status应该是字典类型，包含各种传感器的状态
        # InspectionComponent期望的状态值：'normal', 'degraded', 'failed'
        camera_status = 'normal' if self.inspection_status == 'inspecting' else 'normal'
        sensor_status_dict = {
            'camera': camera_status,
            'gps': 'normal'
        }

        return {
            'sensor_status': sensor_status_dict,
            'data_collection_rate': self.data_collection_rate
        }

    def complete(self):
        """重写complete方法，添加巡检特定的结果数据"""
        if self.status == TaskStatus.COMPLETED:
            return

        # 调用父类的complete方法
        super().complete()

        # 添加巡检特定的结果数据
        inspection_data = {
            'data_size': self.collected_data_size,
            'data_type': self.data_collection_type,
            'collection_rate': self.data_collection_rate,
            'inspection_duration': self.elapsed_time,
            'position': self.inspection_position,
            'altitude': self.inspection_altitude
        }

        # 模拟检测到的对象（基于检测精度）
        detected_objects = []
        detection_accuracy = self.current_metrics.get('detection_accuracy', 0.5)
        if detection_accuracy >= self.detection_accuracy_threshold:
            # 基于检测精度模拟检测到的对象数量
            num_objects = int(detection_accuracy * 5)  # 最多检测到5个对象
            for i in range(num_objects):
                detected_objects.append({
                    'object_id': f'obj_{self.point_index}_{i}',
                    'type': 'unknown',
                    'confidence': detection_accuracy,
                    'position': self.inspection_position
                })

        # 更新result，包含巡检特定数据
        self.result.update({
            'point_index': self.point_index,
            'waypoint': self.waypoint,
            'inspection_data': inspection_data,
            'detected_objects': detected_objects,
            'workflow_id': self.workflow_id,
            'task_class': self.task_class
        })
    
    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """计算两点间的欧几里得距离"""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )
    
    def _is_at_inspection_position(self, threshold: float = 5.0) -> bool:
        """检查是否在巡检位置附近"""
        current_position = self.agent.get_state('position')
        if not current_position or not self.inspection_position:
            return False
        
        # 计算距离
        distance = sum((a - b) ** 2 for a, b in zip(current_position, self.inspection_position)) ** 0.5
        return distance <= threshold
    