"""
计算组件

严格按照设计文档实现的 ComputationComponent。
支持边缘计算、数据处理、AI推理等计算任务。

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

class ComputeComponent(Component):
    """计算能力评估组件 - 实时监控无人机的核心计算状态，动态评估其性能"""

    # 严格按照设计理念的 PRODUCED_METRICS
    PRODUCED_METRICS = [
        'processing_throughput_gflops',  # float - 实时处理吞吐量 (GFLOPs/s)
        'memory_pressure'                # float - 内存压力 (0-1)
    ]

    # 严格按照设计理念的 MONITORED_STATES - 与DroneAgent状态模板严格匹配
    MONITORED_STATES = [
        'cpu_utilization',      # float - CPU使用率，直接影响可用的计算吞吐量
        'memory_usage',         # float - 内存使用量，用于计算内存压力
        'ambient_temperature',  # float - 环境温度，用于模拟因过热导致的性能下降（热节流）
        'battery_level'         # float - 电池电量，用于模拟低电量下的节能降频模式
    ]
    
    def __init__(self, agent, **kwargs):
        """初始化计算组件"""
        super().__init__(agent, 'Compute', **kwargs)

        # 从agent properties获取硬件规格参数
        self.cpu_cores = agent.get_state('cpu_cores', 4)  # CPU核心数
        self.cpu_frequency = agent.get_state('cpu_frequency', 3.0)  # GHz
        self.memory_size = agent.get_state('memory_size', 8.0)  # GB

        # 简化的指令/周期(IPC)模型，假设每个核心每个时钟周期执行2个操作
        self.max_gflops = self.cpu_cores * self.cpu_frequency * 2.0  # 最大GFLOPs
    
    def _calculate_performance_metrics(self):
        """计算性能指标 - 动态的 Metrics 计算逻辑 (State -> Metrics)"""
        # 1. 获取被监控的当前状态值
        current_cpu_util = self.agent.get_state('cpu_utilization') or 0.0
        current_temp = self.agent.get_state('ambient_temperature') or 25.0
        current_battery = self.agent.get_state('battery_level') or 100.0
        current_memory_usage = self.agent.get_state('memory_usage') or 0.0

        # 2. 计算各种影响因子 (0-1)
        # a. CPU可用率因子：CPU被占用的越多，留给新任务的就越少
        available_cpu_factor = 1.0

        # b. 温度节流因子：温度超过85°C后，性能线性下降
        if current_temp < 85.0:
            thermal_throttle_factor = 1.0
        else:
            thermal_throttle_factor = max(0.1, 1.0 - (current_temp - 85.0) / 15.0)  # 在85-100度之间线性降至10%

        # c. 低电量节能因子：电量低于20%时，性能减半
        power_saving_factor = 0.5 if current_battery < 20.0 else 1.0

        # 3. 计算最终的动态吞吐量
        dynamic_throughput_gflops = self.max_gflops * available_cpu_factor * thermal_throttle_factor * power_saving_factor

        # 4. 计算内存压力
        memory_pressure = current_memory_usage / self.memory_size if self.memory_size > 0 else 0.0
        memory_pressure = min(1.0, memory_pressure)  # 限制在0-1范围内

        # 5. 返回计算出的所有 metrics
        return {
            'processing_throughput_gflops': dynamic_throughput_gflops,
            'memory_pressure': memory_pressure
        }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            'cpu_cores': self.cpu_cores,
            'cpu_frequency': self.cpu_frequency,
            'memory_size': self.memory_size,
            'max_gflops': self.max_gflops
        }