"""
气象环境控制器

实现气象条件的动态控制和事件触发机制，支持：
1. 动态天气条件控制（风速、能见度、温度等）
2. 基于配置文件的事件调度
3. 通过事件系统影响agent状态
4. 与AirFogSim环境系统集成

@author: TCCN Auto Planner
"""

import random
import simpy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from airfogsim.core.dataprovider import DataProvider
from airfogsim.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WeatherCondition:
    """天气条件数据类"""
    wind_speed: float  # m/s
    wind_direction: float  # degrees (0-360)
    visibility: float  # meters
    temperature: float  # celsius
    humidity: float  # percentage (0-100)
    precipitation: float  # mm/h
    pressure: float  # hPa
    timestamp: float  # simulation time


@dataclass
class WeatherZone:
    """天气区域数据类"""
    zone_id: str  # 区域ID
    center: Tuple[float, float, float]  # 区域中心坐标 (x, y, z)
    radius: float  # 影响半径（米）
    condition: WeatherCondition  # 天气条件
    start_time: float  # 开始时间
    duration: float  # 持续时间（秒），-1表示永久
    active: bool = True  # 是否激活


class WeatherController(DataProvider):
    """
    气象环境控制器
    
    继承自AirFogSim的DataProvider，提供实时天气数据和事件触发功能。
    支持通过配置文件定义天气事件的发生时间和参数。
    """
    
    def __init__(self, env, config: Dict[str, Any] = None):
        """
        初始化气象环境控制器

        Args:
            env: AirFogSim仿真环境
            config: 天气配置字典
        """
        super().__init__(env)
        self.env = env
        self.config = config or {}

        # 全局默认天气条件
        self.default_weather = WeatherCondition(
            wind_speed=self.config.get('initial_wind_speed', 5.0),
            wind_direction=self.config.get('initial_wind_direction', 0.0),
            visibility=self.config.get('initial_visibility', 10000.0),
            temperature=self.config.get('initial_temperature', 20.0),
            humidity=self.config.get('initial_humidity', 60.0),
            precipitation=self.config.get('initial_precipitation', 0.0),
            pressure=self.config.get('initial_pressure', 1013.25),
            timestamp=0.0
        )

        # 当前活跃的天气区域列表
        self.weather_zones: List[WeatherZone] = []

        # 天气事件配置
        self.weather_events = self.config.get('weather_events', [])

        # 天气更新间隔（秒）
        self.update_interval = self.config.get('update_interval', 60.0)

        # 天气变化范围配置
        self.weather_ranges = self.config.get('weather_ranges', {
            'wind_speed': [0, 30],
            'visibility': [100, 15000],
            'temperature': [-20, 45],
            'humidity': [20, 100],
            'precipitation': [0, 50],
            'pressure': [980, 1050]
        })

        # 启动天气控制进程
        self.weather_process = self.env.process(self._weather_control_loop())
        self.event_process = self.env.process(self._event_scheduler())

        logger.info(f"时间 {self.env.now}: 气象环境控制器已初始化")
    
    def get_data(self, key: str = None, position: Tuple[float, float, float] = None) -> Any:
        """
        获取天气数据（DataProvider接口实现）

        Args:
            key: 数据键，如果为None则返回完整天气条件
            position: 位置坐标，用于获取特定位置的天气

        Returns:
            天气数据
        """
        # 获取指定位置的天气条件
        weather_condition = self.get_weather_at_position(position) if position else self.default_weather

        if key is None:
            return weather_condition

        return getattr(weather_condition, key, None)

    def get_weather_at_position(self, position: Tuple[float, float, float]) -> WeatherCondition:
        """
        获取指定位置的天气条件

        Args:
            position: 位置坐标 (x, y, z)

        Returns:
            该位置的天气条件
        """
        if not position:
            return self.default_weather

        # 查找影响该位置的天气区域
        affecting_zones = []
        for zone in self.weather_zones:
            if not zone.active:
                continue

            # 检查是否在区域影响范围内
            distance = self._calculate_distance(position, zone.center)
            if distance <= zone.radius:
                affecting_zones.append((zone, distance))

        # 如果没有天气区域影响，返回默认天气
        if not affecting_zones:
            return self.default_weather

        # 如果有多个区域影响，选择距离最近的
        affecting_zones.sort(key=lambda x: x[1])
        closest_zone = affecting_zones[0][0]

        return closest_zone.condition

    def load_data(self):
        """加载数据（DataProvider接口实现）"""
        # 天气数据是动态生成的，不需要从外部加载
        pass

    def start_event_triggering(self):
        """启动事件触发（DataProvider接口实现）"""
        # 事件触发已在__init__中启动
        pass
    
    def _weather_control_loop(self):
        """天气控制主循环"""
        try:
            while True:
                # 等待更新间隔
                yield self.env.timeout(self.update_interval)
                
                # 更新天气条件
                self._update_weather_conditions()
                
                # 检查天气对agents的影响
                self._apply_weather_effects()
                
                # 触发天气更新事件
                self.env.event_registry.trigger_event(
                    'weather_controller',
                    'weather_updated',
                    {
                        'default_weather': self.default_weather,
                        'weather_zones': len(self.weather_zones),
                        'time': self.env.now
                    }
                )
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: 天气控制循环被中断")
    
    def _event_scheduler(self):
        """事件调度器，处理配置文件中定义的天气事件"""
        try:
            for event_config in self.weather_events:
                # 等待到事件触发时间
                trigger_time = event_config.get('trigger_time', 0)
                if trigger_time > self.env.now:
                    yield self.env.timeout(trigger_time - self.env.now)
                
                # 执行天气事件
                self._execute_weather_event(event_config)
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: 天气事件调度器被中断")
    
    def _update_weather_conditions(self):
        """更新天气条件（自然变化和区域管理）"""
        # 更新默认天气条件的自然变化
        wind_change = random.uniform(-2, 2)
        temp_change = random.uniform(-1, 1)
        visibility_change = random.uniform(-500, 500)

        # 更新默认天气条件，确保在合理范围内
        self.default_weather.wind_speed = max(0, min(
            self.weather_ranges['wind_speed'][1],
            self.default_weather.wind_speed + wind_change
        ))

        self.default_weather.temperature = max(
            self.weather_ranges['temperature'][0],
            min(self.weather_ranges['temperature'][1],
                self.default_weather.temperature + temp_change)
        )

        self.default_weather.visibility = max(
            self.weather_ranges['visibility'][0],
            min(self.weather_ranges['visibility'][1],
                self.default_weather.visibility + visibility_change)
        )

        # 更新时间戳
        self.default_weather.timestamp = self.env.now

        # 更新天气区域状态
        self._update_weather_zones()

        logger.debug(f"时间 {self.env.now}: 天气条件已更新 - "
                    f"默认风速: {self.default_weather.wind_speed:.1f}m/s, "
                    f"默认温度: {self.default_weather.temperature:.1f}°C, "
                    f"默认能见度: {self.default_weather.visibility:.0f}m, "
                    f"活跃天气区域: {len([z for z in self.weather_zones if z.active])}")

    def _update_weather_zones(self):
        """更新天气区域状态"""
        current_time = self.env.now

        # 检查并移除过期的天气区域
        for zone in self.weather_zones[:]:  # 使用切片复制列表以安全删除
            if zone.duration > 0 and (current_time - zone.start_time) >= zone.duration:
                zone.active = False
                logger.info(f"时间 {current_time}: 天气区域 {zone.zone_id} 已过期")

        # 移除非活跃的区域
        self.weather_zones = [zone for zone in self.weather_zones if zone.active]
    
    def _execute_weather_event(self, event_config: Dict[str, Any]):
        """
        执行天气事件

        Args:
            event_config: 事件配置字典
        """
        event_type = event_config.get('type', 'weather_change')
        event_params = event_config.get('parameters', {})

        logger.info(f"时间 {self.env.now}: 执行天气事件 - {event_type}")

        if event_type == 'storm':
            self._trigger_storm(event_params)
        elif event_type == 'fog':
            self._trigger_fog(event_params)
        elif event_type == 'high_wind':
            self._trigger_high_wind(event_params)
        elif event_type == 'temperature_extreme':
            self._trigger_temperature_extreme(event_params)
        elif event_type == 'weather_zone':
            self._create_weather_zone(event_params)
        else:
            logger.warning(f"未知的天气事件类型: {event_type}")

    def _create_weather_zone(self, params: Dict[str, Any]):
        """创建天气区域"""
        zone_id = params.get('zone_id', f"zone_{len(self.weather_zones)}")
        center = params.get('center', [0, 0, 100])
        radius = params.get('radius', 500.0)
        duration = params.get('duration', 600.0)  # 默认10分钟

        # 创建天气条件
        weather_condition = WeatherCondition(
            wind_speed=params.get('wind_speed', self.default_weather.wind_speed),
            wind_direction=params.get('wind_direction', self.default_weather.wind_direction),
            visibility=params.get('visibility', self.default_weather.visibility),
            temperature=params.get('temperature', self.default_weather.temperature),
            humidity=params.get('humidity', self.default_weather.humidity),
            precipitation=params.get('precipitation', self.default_weather.precipitation),
            pressure=params.get('pressure', self.default_weather.pressure),
            timestamp=self.env.now
        )

        # 创建天气区域
        weather_zone = WeatherZone(
            zone_id=zone_id,
            center=tuple(center),
            radius=radius,
            condition=weather_condition,
            start_time=self.env.now,
            duration=duration,
            active=True
        )

        self.weather_zones.append(weather_zone)

        logger.info(f"时间 {self.env.now}: 创建天气区域 {zone_id} - "
                   f"中心: {center}, 半径: {radius}m, 持续: {duration}s")
    
    def _trigger_storm(self, params: Dict[str, Any]):
        """触发暴风雨事件"""
        # 如果指定了区域参数，创建区域天气；否则影响全局天气
        if 'center' in params and 'radius' in params:
            zone_params = {
                'zone_id': f"storm_{self.env.now}",
                'center': params.get('center', [500, 500, 200]),
                'radius': params.get('radius', 1000.0),
                'duration': params.get('duration', 1800.0),  # 30分钟
                'wind_speed': params.get('wind_speed', 25.0),
                'precipitation': params.get('precipitation', 20.0),
                'visibility': params.get('visibility', 1000.0)
            }
            self._create_weather_zone(zone_params)
        else:
            # 全局天气变化
            self.default_weather.wind_speed = params.get('wind_speed', 25.0)
            self.default_weather.precipitation = params.get('precipitation', 20.0)
            self.default_weather.visibility = params.get('visibility', 1000.0)
            self.default_weather.timestamp = self.env.now

        logger.warning(f"时间 {self.env.now}: 暴风雨来袭！风速: {params.get('wind_speed', 25.0)}m/s")

    def _trigger_fog(self, params: Dict[str, Any]):
        """触发大雾事件"""
        if 'center' in params and 'radius' in params:
            zone_params = {
                'zone_id': f"fog_{self.env.now}",
                'center': params.get('center', [500, 500, 100]),
                'radius': params.get('radius', 800.0),
                'duration': params.get('duration', 1200.0),  # 20分钟
                'visibility': params.get('visibility', 500.0),
                'humidity': params.get('humidity', 95.0)
            }
            self._create_weather_zone(zone_params)
        else:
            self.default_weather.visibility = params.get('visibility', 500.0)
            self.default_weather.humidity = params.get('humidity', 95.0)
            self.default_weather.timestamp = self.env.now

        logger.warning(f"时间 {self.env.now}: 大雾天气！能见度: {params.get('visibility', 500.0)}m")

    def _trigger_high_wind(self, params: Dict[str, Any]):
        """触发大风事件"""
        if 'center' in params and 'radius' in params:
            zone_params = {
                'zone_id': f"high_wind_{self.env.now}",
                'center': params.get('center', [500, 500, 200]),
                'radius': params.get('radius', 600.0),
                'duration': params.get('duration', 900.0),  # 15分钟
                'wind_speed': params.get('wind_speed', 20.0),
                'wind_direction': params.get('wind_direction', random.uniform(0, 360))
            }
            self._create_weather_zone(zone_params)
        else:
            self.default_weather.wind_speed = params.get('wind_speed', 20.0)
            self.default_weather.wind_direction = params.get('wind_direction', random.uniform(0, 360))
            self.default_weather.timestamp = self.env.now

        logger.warning(f"时间 {self.env.now}: 大风天气！风速: {params.get('wind_speed', 20.0)}m/s")

    def _trigger_temperature_extreme(self, params: Dict[str, Any]):
        """触发极端温度事件"""
        if 'center' in params and 'radius' in params:
            zone_params = {
                'zone_id': f"temp_extreme_{self.env.now}",
                'center': params.get('center', [500, 500, 150]),
                'radius': params.get('radius', 700.0),
                'duration': params.get('duration', 2400.0),  # 40分钟
                'temperature': params.get('temperature', -10.0)
            }
            self._create_weather_zone(zone_params)
        else:
            self.default_weather.temperature = params.get('temperature', -10.0)
            self.default_weather.timestamp = self.env.now

        logger.warning(f"时间 {self.env.now}: 极端温度！温度: {params.get('temperature', -10.0)}°C")
    
    def _apply_weather_effects(self):
        """将天气条件应用到agents上"""
        # 通过airspace_manager获取所有agent位置信息
        agent_positions = self.env.airspace_manager.agent_positions

        for agent_id, position in agent_positions.items():
            agent = self.env.get_agent(agent_id)
            if agent:
                self._apply_weather_to_agent(agent, position)
    
    def _apply_weather_to_agent(self, agent, position=None):
        """
        将天气效果应用到特定agent

        Args:
            agent: 目标agent
            position: agent位置（可选，如果不提供则从agent状态获取）
        """
        try:
            # 获取agent位置信息
            if position is None:
                position = agent.get_state('position', [0, 0, 0])

            # 获取该位置的天气条件
            local_weather = self.get_weather_at_position(tuple(position))

            # 根据天气条件和位置影响agent状态
            weather_effects = {}

            # 风速影响（高度越高风速影响越大）
            if local_weather.wind_speed > 15:
                altitude_factor = 1.0 + (position[2] / 1000.0)  # 高度因子
                wind_force = local_weather.wind_speed * 0.1 * altitude_factor

                weather_effects['external_force'] = [
                    wind_force * 0.8,  # X方向
                    wind_force * 0.6,  # Y方向
                    wind_force * 0.2   # Z方向
                ]
                weather_effects['max_allowed_speed'] = max(5.0,
                    agent.get_state('max_speed', 20.0) * (0.9 - altitude_factor * 0.2))

            # 能见度影响（低空能见度更差）
            visibility_factor = max(0.5, 1.0 - (500 - position[2]) / 1000.0) if position[2] < 500 else 1.0
            effective_visibility = local_weather.visibility * visibility_factor

            if effective_visibility < 1000:
                weather_effects['sensor_accuracy'] = max(0.1, effective_visibility / 1000.0)
                weather_effects['communication_range_factor'] = max(0.5, effective_visibility / 2000.0)

            # 温度影响（高度越高温度越低）
            altitude_temp_drop = position[2] * 0.0065  # 每100米降温0.65度
            effective_temperature = local_weather.temperature - altitude_temp_drop

            if abs(effective_temperature) > 30:
                # 极端温度影响电池效率
                if effective_temperature < -10:
                    battery_efficiency = max(0.6, 1.0 + effective_temperature / 50.0)
                elif effective_temperature > 40:
                    battery_efficiency = max(0.7, 1.0 - (effective_temperature - 40) / 60.0)
                else:
                    battery_efficiency = 1.0

                weather_effects['battery_efficiency'] = battery_efficiency

            # 应用天气效果
            for key, value in weather_effects.items():
                agent.update_state(key, value)

            # 如果有严重天气条件，触发警告事件
            if (local_weather.wind_speed > 20 or
                local_weather.visibility < 500):

                agent.trigger_event('weather_warning', {
                    'weather_condition': local_weather,
                    'position': position,
                    'time': self.env.now,
                    'severity': 'high' if local_weather.wind_speed > 25 else 'medium'
                })

        except Exception as e:
            logger.error(f"应用天气效果到agent {agent.id}时出错: {e}")

    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """
        计算两点之间的3D距离

        Args:
            pos1: 位置1 (x, y, z)
            pos2: 位置2 (x, y, z)

        Returns:
            距离（米）
        """
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
