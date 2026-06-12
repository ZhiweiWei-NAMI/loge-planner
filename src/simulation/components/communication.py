"""
通信组件

严格按照设计文档实现的 CommunicationComponent。
支持数据传输、网络连接、信号处理等通信任务。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

import math
from typing import Dict, Any, Tuple
from ..core.component import Component
import logging

logger = logging.getLogger(__name__)

class CommunicationComponent(Component):
    """通信组件"""

    # 严格按照精简设计的 PRODUCED_METRICS
    PRODUCED_METRICS = [
        'uplink_throughput',    # MB/s - 上行有效吞吐量 (最重要的指标)
        'link_quality'          # 0-1 - 链路质量 (综合指标)
    ]

    # 严格按照精简设计的 MONITORED_STATES
    MONITORED_STATES = [
        'position',                      # 无人机当前位置，影响信号衰减
        'transmission_power',            # 发射功率，影响信号强度
        'interference_signal_strength',  # 干扰信号强度，影响链路质量
        'visibility',                    # 能见度，影响高频信号衰减
        'precipitation_level'            # 降水强度，影响信号衰减
    ]
    
    def __init__(self, agent, **kwargs):
        """初始化通信组件"""
        super().__init__(agent, 'Communication', **kwargs)

        # 硬件规格参数 - 精简版
        self.max_throughput = kwargs.get('max_throughput', 10.0)  # MB/s 最大吞吐量
        self.base_station_position = kwargs.get('base_station_position', (0, 0, 0))  # 地面站位置
        self.optimal_distance = kwargs.get('optimal_distance', 500.0)  # m 最优通信距离

        # 环境影响参数
        self.weather_sensitivity = kwargs.get('weather_sensitivity', 0.5)  # 天气敏感度
        self.interference_threshold = kwargs.get('interference_threshold', -80.0)  # dBm 干扰阈值
    
    def _calculate_performance_metrics(self):
        """计算通信性能指标 - 精简版"""
        # 获取监听的状态
        position = self.agent.get_state('position') or (0, 0, 0)
        transmission_power = self.agent.get_state('transmission_power') or 1.0  # W
        interference_strength = self.agent.get_state('interference_signal_strength') or -90.0  # dBm
        visibility = self.agent.get_state('visibility') or 10.0  # m
        precipitation = self.agent.get_state('precipitation_level') or 0.0  # mm/h

        # 计算链路质量 (综合指标)
        link_quality = self._calculate_link_quality(position, transmission_power,
                                                   interference_strength, visibility, precipitation)

        # 计算上行有效吞吐量 (最重要的指标)
        uplink_throughput = self._calculate_uplink_throughput(link_quality, transmission_power)

        return {
            'uplink_throughput': uplink_throughput,  # MB/s
            'link_quality': link_quality             # 0-1
        }
    
    def _calculate_link_quality(self, position: Tuple[float, float, float],
                               transmission_power: float, interference_strength: float,
                               visibility: float, precipitation: float) -> float:
        """计算链路质量 (0-1)"""
        # 1. 距离因子 - 基于与地面站的距离
        distance = math.sqrt(
            (position[0] - self.base_station_position[0])**2 +
            (position[1] - self.base_station_position[1])**2 +
            (position[2] - self.base_station_position[2])**2
        )
        distance_factor = max(0.1, min(1.0, self.optimal_distance / max(distance, 1.0)))

        # 2. 功率因子 - 发射功率越高，质量越好
        power_factor = min(1.0, transmission_power / 2.0)  # 假设2W为最优功率

        # 3. 干扰因子 - 干扰越强，质量越差
        interference_factor = max(0.1, 1.0 - max(0, (interference_strength + 80) / 40))

        # 4. 天气因子 - 能见度和降水影响
        visibility_factor = min(1.0, visibility / 1.0)  # 1000m为最优能见度
        precipitation_factor = max(0.3, 1.0 - precipitation * self.weather_sensitivity * 0.1)

        # 综合链路质量
        link_quality = distance_factor * power_factor * interference_factor * visibility_factor * precipitation_factor

        return max(0.0, min(1.0, link_quality))
    
    def _calculate_uplink_throughput(self, link_quality: float, transmission_power: float) -> float:
        """计算上行有效吞吐量 (MB/s)"""
        # 基础吞吐量基于链路质量
        base_throughput = self.max_throughput * link_quality

        # 功率影响 - 功率越高，吞吐量可能越高（但有上限）
        power_boost = min(1.2, 1.0 + (transmission_power - 1.0) * 0.1)

        # 最终吞吐量
        uplink_throughput = base_throughput * power_boost

        return max(0.1, min(self.max_throughput, uplink_throughput))

    def on_state_changed(self, state_key: str, old_value: Any, new_value: Any):
        """响应状态变化"""
        if state_key in self.MONITORED_STATES:
            # 状态变化时重新计算指标
            self.calculate_metrics(0.1)

            # 特殊处理
            if state_key == 'position':
                # 位置变化时检查通信范围
                distance = math.sqrt(
                    (new_value[0] - self.base_station_position[0])**2 +
                    (new_value[1] - self.base_station_position[1])**2 +
                    (new_value[2] - self.base_station_position[2])**2
                )
                if distance > self.optimal_distance * 2:
                    self.trigger_event('communication_range_warning', {
                        'current_distance': distance,
                        'optimal_distance': self.optimal_distance,
                        'signal_degradation': 'high'
                    })

            elif state_key == 'interference_signal_strength':
                # 干扰强度变化时的处理
                if new_value > self.interference_threshold:
                    self.trigger_event('high_interference_detected', {
                        'interference_level': new_value,
                        'threshold': self.interference_threshold,
                        'recommendation': 'consider_frequency_change'
                    })

    def get_communication_info(self) -> Dict[str, Any]:
        """获取通信能力信息"""
        position = self.agent.get_state('position') or (0, 0, 0)
        distance = math.sqrt(
            (position[0] - self.base_station_position[0])**2 +
            (position[1] - self.base_station_position[1])**2 +
            (position[2] - self.base_station_position[2])**2
        )

        return {
            'max_throughput': self.max_throughput,
            'current_distance_to_base': distance,
            'optimal_distance': self.optimal_distance,
            'link_quality': self.metrics.get('link_quality', 0.5),
            'uplink_throughput': self.metrics.get('uplink_throughput', 1.0)
        }
    