"""
移动能力组件

严格按照设计文档实现的 MobilityComponent。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

import math
from typing import Dict, Any
from ..core.component import Component
import logging

logger = logging.getLogger(__name__)

class MobilityComponent(Component):
    """移动能力组件"""
    
    # 严格按照设计文档的 PRODUCED_METRICS
    PRODUCED_METRICS = [
        'max_speed',           # km/h - 基于电池、载重、环境计算的最大飞行速度
        'mobility_base_power', # W - 基础飞行功耗
        'flight_efficiency',   # ratio - 综合飞行效率，受载重、风力等影响 0-1
        'navigation_accuracy', # m - 导航精度
        'flight_stability'     # % - 飞行稳定性
    ]
    MONITORED_STATES = [
        'battery_level',          # 电池影响最大速度和功耗
        'current_payload_weight', # 载重影响速度和功耗
        'wind_vector',           # 风向风速影响飞行性能
        'gps_signal_strength',   # GPS信号影响导航精度
        'visibility',            # 能见度影响安全飞行速度
        'precipitation_level'    # 降水影响功耗和稳定性
    ]
    
    def __init__(self, agent, **kwargs):
        """初始化移动组件"""
        super().__init__(agent, 'Mobility', **kwargs)
        
        # 硬件规格参数
        self.max_theoretical_speed = kwargs.get('max_speed', 15.0)  # km/h
        self.max_altitude = kwargs.get('max_altitude', 120.0)  # m
        self.base_power_consumption = kwargs.get('base_power', 100.0)  # W
        self.hovering_capability = kwargs.get('hovering_capability', True)
        self.precision_hovering = kwargs.get('precision_hovering', False)
        self.stable_flight = kwargs.get('stable_flight', False)
    
    def _calculate_performance_metrics(self):
        """计算移动性能指标"""
        # 获取监听的状态
        battery_level = self.agent.get_state('battery_level') or 100.0
        payload_weight = self.agent.get_state('current_payload_weight') or 0.0
        wind_vector = self.agent.get_state('wind_vector') or [0.0, 0.0, 0.0]
        gps_signal = self.agent.get_state('gps_signal_strength') or 1.0
        visibility = self.agent.get_state('visibility') or 1.0  # km
        precipitation = self.agent.get_state('precipitation_level') or 0.0
        
        # 计算最大飞行速度
        max_speed = self._calculate_max_speed(battery_level, payload_weight, wind_vector, visibility)
        
        # 计算基础飞行功耗
        mobility_base_power = self._calculate_base_power(payload_weight, wind_vector, precipitation)
        
        # 计算飞行效率
        flight_efficiency = self._calculate_flight_efficiency(payload_weight, wind_vector, precipitation)
        
        # 计算导航精度
        navigation_accuracy = self._calculate_navigation_accuracy(gps_signal, wind_vector)
        
        # 计算飞行稳定性
        flight_stability = self._calculate_flight_stability(wind_vector, precipitation, payload_weight)
        
        return {
            'max_speed': max_speed,
            'mobility_base_power': mobility_base_power,
            'flight_efficiency': flight_efficiency,
            'navigation_accuracy': navigation_accuracy,
            'flight_stability': flight_stability
        }
    
    def calculate_metrics(self, dt: float):
        """计算移动性能指标"""
        # 使用_calculate_performance_metrics获取指标
        metrics = self._calculate_performance_metrics()
        
        # 更新指标
        self.metrics.update(metrics)
        
        # 更新自身功耗
        self.update_power_consumption(activity_power=metrics['mobility_base_power'])
    
    def _calculate_max_speed(self, battery_level: float, payload_weight: float, 
                           wind_vector: list, visibility: float) -> float:
        """计算最大飞行速度"""
        base_speed = self.max_theoretical_speed
        
        # 电池电量影响（低电量时降低速度以节能）
        battery_factor = 1.0
        if battery_level < 30.0:
            battery_factor = 0.7 + (battery_level / 30.0) * 0.3
        elif battery_level < 50.0:
            battery_factor = 0.85 + (battery_level - 30.0) / 20.0 * 0.15
        
        # 载重影响（载重越大速度越慢）
        max_payload = self.agent.get_state('max_payload_weight') or 5.0
        payload_factor = 1.0 - (payload_weight / max_payload) * 0.3
        
        # 风力影响（逆风降低速度，顺风提高速度）
        wind_speed = math.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        wind_factor = 1.0 - wind_speed * 0.02  # 每m/s风速影响2%
        wind_factor = max(0.5, min(1.2, wind_factor))
        
        # 能见度影响（低能见度时安全限速）
        visibility_factor = 1.0
        if visibility < 0.1:  # 能见度小于100m时严重限速
            visibility_factor = 0.3
        elif visibility < 0.5:  # 能见度小于500m时适度限速
            visibility_factor = 0.6 + (visibility - 0.1) / 400 * 0.4
        
        return base_speed * battery_factor * payload_factor * wind_factor * visibility_factor
    
    def _calculate_base_power(self, payload_weight: float, wind_vector: list, 
                            precipitation: float) -> float:
        """计算基础飞行功耗"""
        base_power = self.base_power_consumption
        
        # 载重影响功耗
        max_payload = self.agent.get_state('max_payload_weight') or 5.0
        payload_factor = 1.0 + (payload_weight / max_payload) * 0.5
        
        # 风力影响功耗（需要更多功率对抗风阻）
        wind_speed = math.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        wind_factor = 1.0 + wind_speed * 0.05  # 每m/s风速增加5%功耗
        
        # 降水影响功耗（额外阻力）
        precipitation_factor = 1.0 + precipitation * 0.3
        
        return base_power * payload_factor * wind_factor * precipitation_factor
    
    def _calculate_flight_efficiency(self, payload_weight: float, wind_vector: list, 
                                   precipitation: float) -> float:
        """计算飞行效率"""
        base_efficiency = 0.85  # 基础效率
        
        # 载重影响效率
        max_payload = self.agent.get_state('max_payload_weight') or 5.0
        payload_factor = 1.0 - (payload_weight / max_payload) * 0.2
        
        # 风力影响效率
        wind_speed = math.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        wind_factor = 1.0 - wind_speed * 0.03
        
        # 降水影响效率
        precipitation_factor = 1.0 - precipitation * 0.25
        
        efficiency = base_efficiency * payload_factor * wind_factor * precipitation_factor
        return max(0.1, min(1.0, efficiency))
    
    def _calculate_navigation_accuracy(self, gps_signal: float, wind_vector: list) -> float:
        """计算导航精度（米）"""
        base_accuracy = 2.0  # 基础精度2米
        
        # GPS信号质量影响
        gps_factor = 2.0 - gps_signal  # 信号越好精度越高
        
        # 风力影响导航精度
        wind_speed = math.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        wind_factor = 1.0 + wind_speed * 0.1
        
        accuracy = base_accuracy * gps_factor * wind_factor
        return max(0.5, accuracy)
    
    def _calculate_flight_stability(self, wind_vector: list, precipitation: float, 
                                  payload_weight: float) -> float:
        """计算飞行稳定性（百分比）"""
        base_stability = 95.0  # 基础稳定性95%
        
        # 风力影响稳定性
        wind_speed = math.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        wind_penalty = wind_speed * 2.0  # 每m/s风速降低2%稳定性
        
        # 降水影响稳定性
        precipitation_penalty = precipitation * 15.0
        
        # 载重影响稳定性
        max_payload = self.agent.get_state('max_payload_weight') or 5.0
        payload_penalty = (payload_weight / max_payload) * 10.0
        
        stability = base_stability - wind_penalty - precipitation_penalty - payload_penalty
        return max(50.0, min(100.0, stability))
    
    def on_state_changed(self, state_key: str, old_value: Any, new_value: Any):
        """响应状态变化"""
        if state_key in self.monitored_states:
            # 状态变化时重新计算指标
            self.calculate_metrics(0.1)
            
            # 特殊处理
            if state_key == 'battery_level' and new_value < 15.0:
                self.trigger_event('low_battery_flight_warning', {
                    'battery_level': new_value,
                    'max_speed': self.metrics.get('max_speed', 0),
                    'recommendation': 'reduce_speed_and_land'
                })
