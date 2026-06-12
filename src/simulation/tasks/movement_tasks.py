"""
移动任务实现

实现移动相关的任务：MoveToTask

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

class MoveToTask(Task):
    """
    Move agent to specified location.

    This task moves the agent from its current position to a target position.

    Required parameters:
    - target_state: Dict containing the destination's 'position' key with target coordinates [x, y, z]

    Optional properties:
    - speed: Movement speed in m/s (default: 10.0), should be lower than the agent's max_speed
    - recommended_speed: Topo-aware speed limit from offline planning (if set, overrides speed)
    - topo_aware: Boolean flag indicating if topo-aware speed was used

    Example usage:
    target_state = {'position': [100.0, 200.0, 50.0]} # This is the target position
    properties = {'max_speed': 15.0}
    # Or with topo-aware:
    properties = {'recommended_speed': 13.5, 'topo_aware': True}
    """

    # AirFogSim 要求的类属性 - 必须与 DroneAgent 的状态模板匹配
    PRODUCED_STATES = ['position', 'orientation', 'distance_traveled', 'altitude', 'battery_level', 'status', 'moving_status', 'velocity']
    NECESSARY_METRICS = ['max_speed']

    def __init__(self, env, agent, component_name, task_name,
                 workflow_id=None, target_state=None, properties=None):
        """
        初始化移动任务
        
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
        
        # 从target_state获取目标位置
        self.target_position = target_state.get('position') if target_state else (0, 0, 0)
        
        # 从 properties获取任务参数
        props = properties or {}
        self.movement_type = props.get('movement_type', 'direct')
        self.waypoints = props.get('waypoints', [])
        
        # Speed selection priority:
        # 1. recommended_speed from topo-aware planning (considers time-varying corridor limits)
        # 2. explicit speed in properties
        # 3. topology corridor base_velocity (from config)
        # 4. agent max_speed as final default
        from ..environment.topology_manager import TopologyManager
        topo_config = TopologyManager.get_corridor_config()
        topo_max_speed = topo_config.get('base_velocity', 15.0)
        
        self.topo_aware = props.get('topo_aware', False)
        self._topology_manager = props.get('topology_manager', None)  # Runtime topology reference
        self._waypoint_nodes = props.get('waypoint_nodes', [])  # 节点ID列表用于运行时速度查询
        
        if self.topo_aware and 'recommended_speed' in props:
            # Topo-aware mode: use pre-computed speed limit from path planning
            self.speed = float(props['recommended_speed'])
            logger.debug(f"MoveToTask {task_name}: topo-aware speed {self.speed:.2f} m/s")
        elif 'max_speed' in props:
            # Explicit max_speed specified
            self.speed = float(props['max_speed'])
        else:
            # Default: use topology corridor speed limit (not agent max_speed)
            self.speed = topo_max_speed
            logger.debug(f"MoveToTask {task_name}: using topo max speed {self.speed:.2f} m/s")
        
        # 移动状态
        self.current_waypoint_index = 0
        self.total_distance = 0.0
        self.distance_traveled = 0.0
        self.current_position = None
        self.start_position = None
        self.orientation = (0, 0, 0)
        self.progress = 0.0
        self.battery_level = 100.0
        self.last_update_time = env.now
        
        # 计算初始距离和方向
        self._calculate_distance_and_orientation()
    
    def _calculate_distance_and_orientation(self):
        """计算距离和方向"""
        # 获取起始位置
        self.start_position = self.agent.get_state('position')
        if not self.start_position:
            self.start_position = (0, 0, 0)
        
        self.current_position = self.start_position
        
        # 计算总距离
        self.total_distance = self._calculate_distance(self.start_position, self.target_position)
        
        # 计算方向向量（单位向量）
        if self.total_distance > 0:
            dx = self.target_position[0] - self.start_position[0]
            dy = self.target_position[1] - self.start_position[1]
            dz = self.target_position[2] - self.start_position[2]
            self.orientation = (dx / self.total_distance, dy / self.total_distance, dz / self.total_distance)
        else:
            self.orientation = (0, 0, 0)
    
    def estimate_remaining_time(self, performance_metrics: Dict) -> float:
        """估算剩余时间"""
        # 获取当前速度 (TCCN MobilityComponent 提供 'max_speed')
        speed = performance_metrics.get('max_speed', 0)
        if speed <= 0:
            speed = self.speed  # use task's configured speed
        
        if speed <= 0:
            return float('inf')
        
        # 计算剩余距离
        remaining_distance = self.total_distance * (1.0 - self.progress)
        
        # 估算剩余时间
        remaining_time = remaining_distance / speed
        
        return remaining_time
    
    def _get_runtime_speed_limit(self) -> float:
        """获取当前位置和时间的运行时速度限制
        
        使用 waypoints 确定当前所在的拓扑边，查询时变速度。
        
        Returns:
            当前允许的最大速度 (m/s)
        """
        if self._topology_manager is None:
            return self.speed
        
        if not self._waypoint_nodes or len(self._waypoint_nodes) < 2:
            return self.speed  # 无 waypoint_nodes，无法确定边
        
        try:
            current_time = self.env.now
            
            # 根据 current_waypoint_index 确定当前边
            if self.current_waypoint_index >= len(self._waypoint_nodes) - 1:
                return self.speed  # 已到最后一个 waypoint
            
            # 获取当前边的两个节点
            from_node = self._waypoint_nodes[self.current_waypoint_index]
            to_node = self._waypoint_nodes[self.current_waypoint_index + 1]
            
            # 检查边是否存在
            edge_key = (from_node, to_node)
            if edge_key not in self._topology_manager.edges:
                return self.speed
            
            # 查询当前时间这条边的速度限制
            edge_velocity = self._topology_manager.get_edge_velocity(
                from_node, to_node, current_time, current_density=0.0
            )
            
            if edge_velocity <= 0:
                # 边被阻塞
                logger.warning(
                    f"MoveToTask {self.task_name}: edge {from_node}->{to_node} blocked at t={current_time:.0f}s"
                )
                return 0.5  # 最低速度
            
            return min(self.speed, edge_velocity)
            
        except Exception as e:
            logger.debug(f"MoveToTask runtime speed query failed: {e}")
            return self.speed
    
    def _update_task_state(self, performance_metrics: Dict):
        """更新任务进度和内部状态"""
        # 获取基础速度 (TCCN MobilityComponent 提供 'max_speed')
        speed = performance_metrics.get('max_speed', 0)
        if speed <= 0:
            speed = self.speed  # use task's configured speed
        
        # 应用运行时速度限制（考虑 traffic_profiles 的时变阻塞）
        if self._waypoint_nodes and self._topology_manager:
            runtime_speed = self._get_runtime_speed_limit()
            if runtime_speed < speed:
                speed = runtime_speed
        
        # 如果没有获取到当前位置，使用上次记录的位置
        if not self.current_position:
            self.current_position = self.start_position
        
        elapsed_time = self.env.now - self.last_update_time
        # 计算这段时间移动的距离
        distance_moved = speed * elapsed_time
        self.distance_traveled += distance_moved
        
        # 更新电池电量
        self._update_battery_level(distance_moved)
        
        # 根据经过的时间和速度计算新位置
        if self.total_distance > 0 and self.orientation != (0, 0, 0):
            # 计算在每个方向上移动的距离
            dx = self.orientation[0] * distance_moved
            dy = self.orientation[1] * distance_moved
            dz = self.orientation[2] * distance_moved
            
            # 更新当前位置
            current_x, current_y, current_z = self.current_position
            new_x = current_x + dx
            new_y = current_y + dy
            new_z = current_z + dz
            
            # 确保不会越过目标点
            target_x, target_y, target_z = self.target_position
            if ((dx > 0 and new_x > target_x) or (dx < 0 and new_x < target_x) or
                (dy > 0 and new_y > target_y) or (dy < 0 and new_y < target_y) or
                (dz > 0 and new_z > target_z) or (dz < 0 and new_z < target_z)):
                # 已到达或越过目标点，直接设置为目标位置
                self.current_position = self.target_position
                self.progress = 1.0
            else:
                # 正常更新位置
                self.current_position = (new_x, new_y, new_z)
            
            # 更新代理的位置状态，这将触发环境中的碰撞检测
            self.agent.set_state('position', self.current_position)
        
        # 更新进度
        if self.total_distance > 0:
            self.progress = min(1.0, self.distance_traveled / self.total_distance)
        else:
            self.progress = 1.0
        
        # 计算当前位置到目标的实际距离
        current_x, current_y, current_z = self.current_position
        target_x, target_y, target_z = self.target_position
        current_distance = math.sqrt((target_x - current_x)**2 +
                                     (target_y - current_y)**2 +
                                     (target_z - current_z)**2)
        
        # 如果距离目标足够近，完成任务
        if current_distance < 1.0:  # 1.0是容差值
            self.progress = 1.0
            self.current_position = self.target_position
        
        self.last_update_time = self.env.now
    
    def _update_battery_level(self, distance_moved):
        """更新电池电量，根据移动距离消耗电量"""
        # 获取当前电池电量
        battery_level = self.agent.get_state('battery_level', 1.0)  # 默认为满电量
        
        # 计算能量消耗 - 根据移动距离按比例减少电量
        # 假设每单位距离消耗0.005的电量（可根据实际情况调整）
        energy_consumed = distance_moved * 0.005
        
        # 更新电池电量
        new_battery_level = max(0.0, battery_level - energy_consumed)
        
        self.battery_level = new_battery_level
        
        # 如果电量过低，打印警告信息
        if new_battery_level < 20 and battery_level >= 20:
            logger.warning(f"时间 {self.env.now}: 警告! {self.agent.id} 电量低于20%: {new_battery_level:.1f}%")
        elif new_battery_level < 10:
            logger.warning(f"时间 {self.env.now}: 严重警告! {self.agent.id} 电量极低: {new_battery_level:.1f}%")
    
    def _get_task_specific_state_repr(self) -> Dict:
        """返回任务特定状态的表示"""
        
        # 确定移动状态
        moving_status = 'idle'
        if self.progress < 1.0:
            moving_status = 'flying'
        else:
            moving_status = 'hovering'  # 任务完成后设置为悬停状态
        
        # 计算速度向量 (velocity) 以供 GlobalMonitor 检测超速
        # 速度 = 方向向量 * 标量速度
        if self.progress < 1.0 and self.orientation != (0, 0, 0):
            velocity = (
                self.orientation[0] * self.speed,
                self.orientation[1] * self.speed,
                self.orientation[2] * self.speed
            )
        else:
            velocity = (0.0, 0.0, 0.0)  # 任务完成或未移动时速度为0
        
        # 基本状态更新
        return {
            'position': self.current_position,
            'orientation': self.orientation,
            'distance_traveled': self.distance_traveled,
            'altitude': self.current_position[2] if self.current_position else 0.0,
            'battery_level': self.battery_level,
            'status': 'active' if self.progress < 1.0 else 'idle',
            'moving_status': moving_status,
            'velocity': velocity
        }
    
    def _calculate_path(self, start_pos: Tuple, target_pos: Tuple) -> List[Tuple]:
        """计算移动路径"""
        if self.movement_type == 'direct':
            return [start_pos, target_pos]
        elif self.movement_type == 'waypoint' and self.waypoints:
            return [start_pos] + self.waypoints + [target_pos]
        elif self.movement_type == 'path_following':
            # 简化的路径规划，实际应该考虑障碍物避让
            return self._generate_smooth_path(start_pos, target_pos)
        else:
            return [start_pos, target_pos]
    
    def _generate_smooth_path(self, start_pos: Tuple, target_pos: Tuple) -> List[Tuple]:
        """生成平滑路径"""
        # 简化实现：在起点和终点之间插入中间点
        path = [start_pos]
        
        # 计算中间点数量（基于距离）
        distance = self._calculate_distance(start_pos, target_pos)
        num_segments = max(2, int(distance / 50.0))  # 每50米一个段
        
        for i in range(1, num_segments):
            t = i / num_segments
            intermediate_point = (
                start_pos[0] + t * (target_pos[0] - start_pos[0]),
                start_pos[1] + t * (target_pos[1] - start_pos[1]),
                start_pos[2] + t * (target_pos[2] - start_pos[2])
            )
            path.append(intermediate_point)
        
        path.append(target_pos)
        return path

    def _calculate_total_distance(self, path: List[Tuple]) -> float:
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += self._calculate_distance(path[i], path[i + 1])
        return total_distance

    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def complete(self):
        super().complete()
        if isinstance(self.result, dict):
            self.result.update({
                'target_position': self.target_position,
                'start_position': self.start_position,
                'distance_traveled': self.distance_traveled,
                'total_distance': self.total_distance,
                'agent_id': self.agent.id,
                'completion_time': self.end_time
            })
