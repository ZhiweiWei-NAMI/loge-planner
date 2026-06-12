"""
不明飞行物生成器

实现不明飞行物的动态生成、管理和状态控制，支持：
1. 动态生成和管理UFO对象
2. 基于配置文件的事件调度
3. 通过airspace_manager管理UFO位置
4. 检测与agent的碰撞和干扰
5. 与AirFogSim环境系统集成

@author: TCCN Auto Planner
"""

import random
import simpy
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from airfogsim.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UFOObject:
    """不明飞行物对象数据类"""
    id: str
    position: Tuple[float, float, float]  # (x, y, z)
    velocity: Tuple[float, float, float]  # (vx, vy, vz)
    size: float  # 半径（米）
    threat_level: str  # 威胁等级: 'low', 'medium', 'high'
    detection_range: float  # 检测范围（米）
    interference_strength: float  # 干扰强度 (0-1)
    created_time: float
    last_update_time: float
    status: str  # 'active', 'inactive', 'destroyed'


class UFOGenerator:
    """
    不明飞行物生成器
    
    负责动态生成和管理不明飞行物，通过AirFogSim的airspace_manager
    进行位置管理，并检测与agents的交互。
    """
    
    def __init__(self, env, config: Dict[str, Any] = None):
        """
        初始化不明飞行物生成器
        
        Args:
            env: AirFogSim仿真环境
            config: UFO生成配置字典
        """
        self.env = env
        self.config = config or {}
        
        # UFO管理
        self.active_ufos: Dict[str, UFOObject] = {}
        
        # 生成配置
        self.generation_events = self.config.get('ufo_events', [])
        self.max_ufos = self.config.get('max_ufos', 10)
        self.detection_interval = self.config.get('detection_interval', 5.0)
        
        # 空域边界（从环境配置获取）
        self.airspace_bounds = self.config.get('airspace_bounds', {
            'x_min': 0, 'x_max': 1000,
            'y_min': 0, 'y_max': 1000,
            'z_min': 50, 'z_max': 500
        })
        
        # UFO默认参数
        self.default_ufo_params = self.config.get('default_ufo_params', {
            'size': 5.0,
            'speed_range': [10, 30],
            'threat_level': 'medium',
            'detection_range': 100.0,
            'interference_strength': 0.3
        })
        
        # 启动UFO管理进程
        self.generator_process = self.env.process(self._event_scheduler())
        self.update_process = self.env.process(self._ufo_update_loop())
        self.detection_process = self.env.process(self._detection_loop())
        
        logger.info(f"时间 {self.env.now}: 不明飞行物生成器已初始化")
    
    def _event_scheduler(self):
        """事件调度器，处理配置文件中定义的UFO生成事件"""
        try:
            for event_config in self.generation_events:
                # 等待到事件触发时间
                trigger_time = event_config.get('trigger_time', 0)
                if trigger_time > self.env.now:
                    yield self.env.timeout(trigger_time - self.env.now)
                
                # 执行UFO生成事件
                self._execute_ufo_event(event_config)
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: UFO事件调度器被中断")
    
    def _ufo_update_loop(self):
        """UFO状态更新循环"""
        try:
            while True:
                # 等待更新间隔
                yield self.env.timeout(1.0)  # 每秒更新一次
                
                # 更新所有活跃UFO的位置
                self._update_ufo_positions()
                
                # 清理超出边界或过期的UFO
                self._cleanup_ufos()
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: UFO更新循环被中断")
    
    def _detection_loop(self):
        """检测循环，检测UFO与agents的交互"""
        try:
            while True:
                # 等待检测间隔
                yield self.env.timeout(self.detection_interval)
                
                # 检测UFO与agents的交互
                self._detect_ufo_agent_interactions()
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: UFO检测循环被中断")
    
    def _execute_ufo_event(self, event_config: Dict[str, Any]):
        """
        执行UFO生成事件
        
        Args:
            event_config: 事件配置字典
        """
        event_type = event_config.get('type', 'spawn_ufo')
        event_params = event_config.get('parameters', {})
        
        logger.info(f"时间 {self.env.now}: 执行UFO事件 - {event_type}")
        
        if event_type == 'spawn_ufo':
            self._spawn_ufo(event_params)
        elif event_type == 'spawn_multiple_ufos':
            count = event_params.get('count', 3)
            for _ in range(count):
                self._spawn_ufo(event_params)
        elif event_type == 'ufo_swarm':
            self._spawn_ufo_swarm(event_params)
        else:
            logger.warning(f"未知的UFO事件类型: {event_type}")
    
    def _spawn_ufo(self, params: Dict[str, Any]):
        """
        生成单个UFO
        
        Args:
            params: UFO参数字典
        """
        if len(self.active_ufos) >= self.max_ufos:
            logger.warning(f"时间 {self.env.now}: 已达到最大UFO数量限制 ({self.max_ufos})")
            return
        
        # 生成UFO ID
        ufo_id = f"ufo_{uuid.uuid4().hex[:8]}"
        
        # 确定生成位置
        position = self._generate_spawn_position(params)
        
        # 确定速度
        speed = random.uniform(*self.default_ufo_params['speed_range'])
        direction = random.uniform(0, 360)  # 度
        velocity = (
            speed * random.uniform(-1, 1),
            speed * random.uniform(-1, 1),
            speed * random.uniform(-0.5, 0.5)
        )
        
        # 创建UFO对象
        ufo = UFOObject(
            id=ufo_id,
            position=position,
            velocity=velocity,
            size=params.get('size', self.default_ufo_params['size']),
            threat_level=params.get('threat_level', self.default_ufo_params['threat_level']),
            detection_range=params.get('detection_range', self.default_ufo_params['detection_range']),
            interference_strength=params.get('interference_strength', 
                                           self.default_ufo_params['interference_strength']),
            created_time=self.env.now,
            last_update_time=self.env.now,
            status='active'
        )
        
        # 添加到活跃UFO列表
        self.active_ufos[ufo_id] = ufo
        
        # 在airspace_manager中注册UFO位置
        self.env.airspace_manager.register_object(position, obstacle_id=ufo_id)
        
        # 触发UFO生成事件
        self.env.event_registry.trigger_event(
            'ufo_generator',
            'ufo_spawned',
            {
                'ufo_id': ufo_id,
                'position': position,
                'threat_level': ufo.threat_level,
                'time': self.env.now
            }
        )
        
        logger.warning(f"时间 {self.env.now}: 生成UFO {ufo_id} 在位置 {position}, 威胁等级: {ufo.threat_level}")
    
    def _spawn_ufo_swarm(self, params: Dict[str, Any]):
        """
        生成UFO群
        
        Args:
            params: UFO群参数字典
        """
        swarm_size = params.get('swarm_size', 5)
        center_position = params.get('center_position', self._generate_random_position())
        spread_radius = params.get('spread_radius', 50.0)
        
        logger.warning(f"时间 {self.env.now}: 生成UFO群，数量: {swarm_size}")
        
        for i in range(swarm_size):
            # 在中心位置周围随机分布
            offset_x = random.uniform(-spread_radius, spread_radius)
            offset_y = random.uniform(-spread_radius, spread_radius)
            offset_z = random.uniform(-spread_radius/2, spread_radius/2)
            
            swarm_position = (
                center_position[0] + offset_x,
                center_position[1] + offset_y,
                center_position[2] + offset_z
            )
            
            # 确保位置在边界内
            swarm_position = self._clamp_position(swarm_position)
            
            swarm_params = params.copy()
            swarm_params['position'] = swarm_position
            self._spawn_ufo(swarm_params)
    
    def _generate_spawn_position(self, params: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        生成UFO出现位置
        
        Args:
            params: 参数字典
            
        Returns:
            位置坐标 (x, y, z)
        """
        if 'position' in params:
            return tuple(params['position'])
        
        return self._generate_random_position()
    
    def _generate_random_position(self) -> Tuple[float, float, float]:
        """生成随机位置"""
        x = random.uniform(self.airspace_bounds['x_min'], self.airspace_bounds['x_max'])
        y = random.uniform(self.airspace_bounds['y_min'], self.airspace_bounds['y_max'])
        z = random.uniform(self.airspace_bounds['z_min'], self.airspace_bounds['z_max'])
        return (x, y, z)
    
    def _clamp_position(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """将位置限制在边界内"""
        x, y, z = position
        x = max(self.airspace_bounds['x_min'], min(self.airspace_bounds['x_max'], x))
        y = max(self.airspace_bounds['y_min'], min(self.airspace_bounds['y_max'], y))
        z = max(self.airspace_bounds['z_min'], min(self.airspace_bounds['z_max'], z))
        return (x, y, z)
    
    def _update_ufo_positions(self):
        """更新所有UFO的位置"""
        for ufo_id, ufo in list(self.active_ufos.items()):
            if ufo.status != 'active':
                continue
            
            # 计算新位置
            dt = 1.0  # 时间步长（秒）
            new_position = (
                ufo.position[0] + ufo.velocity[0] * dt,
                ufo.position[1] + ufo.velocity[1] * dt,
                ufo.position[2] + ufo.velocity[2] * dt
            )
            
            # 更新UFO位置
            ufo.position = new_position
            ufo.last_update_time = self.env.now
            
            # 更新airspace_manager中的位置
            self.env.airspace_manager.update_object_position(new_position, obstacle_id=ufo_id)
    
    def _cleanup_ufos(self):
        """清理超出边界或过期的UFO"""
        to_remove = []
        
        for ufo_id, ufo in self.active_ufos.items():
            # 检查是否超出边界
            x, y, z = ufo.position
            if (x < self.airspace_bounds['x_min'] - 100 or x > self.airspace_bounds['x_max'] + 100 or
                y < self.airspace_bounds['y_min'] - 100 or y > self.airspace_bounds['y_max'] + 100 or
                z < self.airspace_bounds['z_min'] - 100 or z > self.airspace_bounds['z_max'] + 100):
                to_remove.append(ufo_id)
                continue
            
            # 检查是否过期（可选）
            max_lifetime = self.config.get('max_ufo_lifetime', 3600)  # 默认1小时
            if self.env.now - ufo.created_time > max_lifetime:
                to_remove.append(ufo_id)
        
        # 移除UFO
        for ufo_id in to_remove:
            self._remove_ufo(ufo_id)
    
    def _remove_ufo(self, ufo_id: str):
        """
        移除UFO
        
        Args:
            ufo_id: UFO ID
        """
        if ufo_id in self.active_ufos:
            ufo = self.active_ufos[ufo_id]
            
            # 从airspace_manager中移除
            self.env.airspace_manager.octree.remove(ufo_id)
            
            # 从活跃列表中移除
            del self.active_ufos[ufo_id]
            
            # 触发UFO移除事件
            self.env.event_registry.trigger_event(
                'ufo_generator',
                'ufo_removed',
                {
                    'ufo_id': ufo_id,
                    'position': ufo.position,
                    'time': self.env.now
                }
            )
            
            logger.info(f"时间 {self.env.now}: 移除UFO {ufo_id}")
    
    def _detect_ufo_agent_interactions(self):
        """检测UFO与agents的交互"""
        agents = self.env.get_all_agents()
        
        for agent in agents:
            agent_position = agent.get_state('position')
            if not agent_position:
                continue
            
            # 确保位置是三维的
            if len(agent_position) == 2:
                agent_position = (agent_position[0], agent_position[1], 0)
            
            for ufo_id, ufo in self.active_ufos.items():
                if ufo.status != 'active':
                    continue
                
                # 计算距离
                distance = self._calculate_distance(agent_position, ufo.position)
                
                # 检查是否在检测范围内
                if distance <= ufo.detection_range:
                    self._handle_ufo_agent_interaction(agent, ufo, distance)
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """计算两点间距离"""
        return ((pos1[0] - pos2[0])**2 + 
                (pos1[1] - pos2[1])**2 + 
                (pos1[2] - pos2[2])**2)**0.5
    
    def _handle_ufo_agent_interaction(self, agent, ufo: UFOObject, distance: float):
        """
        处理UFO与agent的交互
        
        Args:
            agent: 目标agent
            ufo: UFO对象
            distance: 距离
        """
        # 根据威胁等级和距离确定影响
        interference_factor = ufo.interference_strength * (1 - distance / ufo.detection_range)
        
        # 应用干扰效果
        effects = {}
        
        if ufo.threat_level == 'high':
            effects['communication_interference'] = interference_factor
            effects['sensor_accuracy'] = max(0.1, 1.0 - interference_factor)
            if distance < 20:  # 非常接近时影响导航
                effects['navigation_error'] = interference_factor * 10
        elif ufo.threat_level == 'medium':
            effects['communication_interference'] = interference_factor * 0.5
            effects['sensor_accuracy'] = max(0.5, 1.0 - interference_factor * 0.5)
        else:  # low
            effects['sensor_accuracy'] = max(0.8, 1.0 - interference_factor * 0.2)
        
        # 应用效果到agent
        for key, value in effects.items():
            agent.update_state(key, value)
        
        # 触发UFO检测事件
        agent.trigger_event('ufo_detected', {
            'ufo_id': ufo.id,
            'distance': distance,
            'threat_level': ufo.threat_level,
            'interference_factor': interference_factor,
            'time': self.env.now
        })
        
        logger.warning(f"时间 {self.env.now}: Agent {agent.id} 检测到UFO {ufo.id}, "
                      f"距离: {distance:.1f}m, 威胁等级: {ufo.threat_level}")
    
    def get_active_ufos(self) -> Dict[str, UFOObject]:
        """获取所有活跃的UFO"""
        return self.active_ufos.copy()
    
    def get_ufo_count(self) -> int:
        """获取活跃UFO数量"""
        return len(self.active_ufos)
