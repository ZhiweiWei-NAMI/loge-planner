"""
巡检感知组件

严格按照设计文档实现的 InspectionComponent。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

import math
from typing import Dict, Any, List
from ..core.component import Component
import logging

logger = logging.getLogger(__name__)

class InspectionComponent(Component):
    """巡检感知组件"""
    
    # 严格按照设计文档的 PRODUCED_METRICS
    PRODUCED_METRICS = [
        'sensing_capability',    # ratio - 感知能力 0-1
        'detection_accuracy',    # ratio - 检测精度 0-1
        'data_collection_rate',  # MB/s - 数据采集速率
        'inspection_base_power', # W - 传感器功耗
        'image_quality'          # ratio - 图像质量 0-1
    ]
    MONITORED_STATES = [
        'sensor_status',        # 传感器状态影响感知能力
        'altitude',            # 高度影响检测范围和精度
        'visibility',          # 能见度影响图像质量和检测能力
        'precipitation_level', # 降水影响传感器性能
        'ambient_temperature'  # 温度影响传感器工作状态
    ]
    
    def __init__(self, agent, **kwargs):
        """初始化巡检组件"""
        super().__init__(agent, 'Inspection', **kwargs)
        
        # 硬件规格参数
        self.sensors = kwargs.get('sensors', ['camera', 'gps'])  # 传感器类型
        self.base_power = kwargs.get('base_power', 30.0)  # W
        self.max_data_rate = kwargs.get('max_data_rate', 10.0)  # MB/s
        self.optimal_altitude = kwargs.get('optimal_altitude', 100.0)  # m
        self.max_detection_range = kwargs.get('max_detection_range', 100.0)  # m
        
        # 传感器特性
        self.sensor_specs = {
            'camera': {'power': 15.0, 'data_rate': 5.0, 'weather_sensitivity': 0.8},
            'lidar': {'power': 25.0, 'data_rate': 8.0, 'weather_sensitivity': 0.6},
            'thermal': {'power': 20.0, 'data_rate': 3.0, 'weather_sensitivity': 0.3},
            'gps': {'power': 2.0, 'data_rate': 0.1, 'weather_sensitivity': 0.1}
        }
    
    def _calculate_performance_metrics(self):
        """计算巡检感知指标"""
        # 获取监听的状态
        sensor_status = self.agent.get_state('sensor_status') or {}
        altitude = self.agent.get_state('altitude') or 50.0
        visibility = self.agent.get_state('visibility') or 1000.0  # m
        precipitation = self.agent.get_state('precipitation_level') or 0.0
        temperature = self.agent.get_state('ambient_temperature') or 25.0  # °C
        
        # 计算感知能力
        sensing_capability = self._calculate_sensing_capability(sensor_status, visibility, precipitation, temperature)
        
        # 计算检测精度
        detection_accuracy = self._calculate_detection_accuracy(sensor_status, altitude, visibility, precipitation)
        
        # 计算数据采集速率
        data_collection_rate = self._calculate_data_collection_rate(sensor_status, temperature)
        
        # 计算传感器功耗
        inspection_base_power = self._calculate_inspection_power(sensor_status, temperature)
        
        return {
            'sensing_capability': sensing_capability,
            'detection_accuracy': detection_accuracy,
            'data_collection_rate': data_collection_rate,
            'inspection_base_power': inspection_base_power
        }
    
    def _calculate_sensing_capability(self, sensor_status: Dict, visibility: float, 
                                    precipitation: float, temperature: float) -> float:
        """计算感知能力"""
        base_capability = 0.9  # 基础感知能力
        
        # 传感器状态影响
        sensor_factor = 1.0
        for sensor in self.sensors:
            status = sensor_status.get(sensor, 'normal')
            if status == 'failed':
                sensor_factor *= 0.0  # 关键传感器失效
            elif status == 'degraded':
                sensor_factor *= 0.6
            elif status == 'normal':
                sensor_factor *= 1.0
        
        # 能见度影响
        visibility_factor = 1.0
        if visibility < 50:  # 极低能见度
            visibility_factor = 0.2
        elif visibility < 200:  # 低能见度
            visibility_factor = 0.5 + (visibility - 50) / 150 * 0.3
        elif visibility < 1000:  # 中等能见度
            visibility_factor = 0.8 + (visibility - 200) / 800 * 0.2
        
        # 降水影响
        precipitation_factor = 1.0 - precipitation * 0.4
        
        # 温度影响（极端温度影响传感器性能）
        temperature_factor = 1.0
        if temperature < -10 or temperature > 50:
            temperature_factor = 0.7
        elif temperature < 0 or temperature > 40:
            temperature_factor = 0.85
        
        capability = base_capability * sensor_factor * visibility_factor * precipitation_factor * temperature_factor
        return max(0.0, min(1.0, capability))
    
    def _calculate_detection_accuracy(self, sensor_status: Dict, altitude: float, 
                                    visibility: float, precipitation: float) -> float:
        """计算检测精度"""
        base_accuracy = 1.0  # 基础检测精度
        
        # 传感器状态影响
        sensor_factor = 1.0
        active_sensors = 0
        for sensor in self.sensors:
            status = sensor_status.get(sensor, 'normal')
            if status == 'normal':
                active_sensors += 1
            elif status == 'degraded':
                active_sensors += 0.5
        
        sensor_factor = active_sensors / len(self.sensors) if self.sensors else 0
        
        # 高度影响（最优高度精度最高）
        altitude_factor = 1.0
        altitude_diff = abs(altitude - self.optimal_altitude)
        if altitude_diff > 50:
            altitude_factor = 0.8
        elif altitude_diff > 20:
            altitude_factor = 0.8 + (50 - altitude_diff) / 30 * 0.2
        
        # 能见度影响
        visibility_factor = min(1.0, visibility / 0.5)
        
        # 降水影响
        precipitation_factor = 1.0 - precipitation * 0.3
        
        accuracy = base_accuracy * sensor_factor * altitude_factor * visibility_factor * precipitation_factor
        return max(0.0, min(1.0, accuracy))
    
    def _calculate_data_collection_rate(self, sensor_status: Dict, temperature: float) -> float:
        """计算数据采集速率"""
        total_rate = 0.0
        
        # 计算每个传感器的数据速率
        for sensor in self.sensors:
            if sensor in self.sensor_specs:
                status = sensor_status.get(sensor, 'normal')
                base_rate = self.sensor_specs[sensor]['data_rate']
                
                if status == 'normal':
                    rate_factor = 1.0
                elif status == 'degraded':
                    rate_factor = 0.5
                else:  # failed
                    rate_factor = 0.0
                
                total_rate += base_rate * rate_factor
        
        # 温度影响数据处理速度
        temperature_factor = 1.0
        if temperature < -5 or temperature > 45:
            temperature_factor = 0.8
        elif temperature < 5 or temperature > 35:
            temperature_factor = 0.9
        
        return min(self.max_data_rate, total_rate * temperature_factor)
    
    def _calculate_inspection_power(self, sensor_status: Dict, temperature: float) -> float:
        """计算传感器功耗"""
        total_power = 0.0
        
        # 计算每个传感器的功耗
        for sensor in self.sensors:
            if sensor in self.sensor_specs:
                status = sensor_status.get(sensor, 'normal')
                base_power = self.sensor_specs[sensor]['power']
                
                if status == 'normal':
                    power_factor = 1.0
                elif status == 'degraded':
                    power_factor = 0.8  # 降级模式功耗略低
                else:  # failed
                    power_factor = 0.1  # 失效但仍有待机功耗
                
                total_power += base_power * power_factor
        
        # 温度影响功耗（极端温度需要更多功耗维持工作）
        temperature_factor = 1.0
        if temperature < -10 or temperature > 50:
            temperature_factor = 1.3
        elif temperature < 0 or temperature > 40:
            temperature_factor = 1.1
        
        return total_power * temperature_factor
    
