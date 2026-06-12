"""通信任务实现

实现通信相关的任务：CommunicationTask

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from airfogsim.core.task import Task
from airfogsim.core.enums import TaskStatus
from typing import Dict, Any
from airfogsim.utils.logging_config import get_logger

logger = get_logger(__name__)

class CommunicationTask(Task):
    """
    Perform communication/data relay operations.

    This task handles communication operations such as data transmission, relay, or broadcasting.
    The agent should be positioned appropriately for communication before starting this task.

    Required parameters:
    - No required target_state (communication happens at current location)

    Required properties:
    - data_size_mb: Amount of data to transmit in MB
    - target_location: Target coordinates [x, y, z] for communication

    Optional properties:
    - data_id: Unique identifier for the data being transmitted (default: 'unknown')
    - communication_type: 'relay', 'broadcast', 'point_to_point' (default: 'relay')
    - max_end_to_end_latency_s: Maximum acceptable latency in seconds (default: 5.0)
    - transmission_power: Transmission power in dBm (default: 20.0)
    - frequency_band: Operating frequency in GHz (default: 2.4)

    Example usage:
    properties = {'data_size_mb': 100.0, 'target_location': [200, 300, 50], 'communication_type': 'relay'}
    """

    # AirFogSim 要求的类属性 - 严格按照精简设计
    # CommunicationTask的PRODUCED_STATES - 必须与DroneAgent状态模板匹配
    PRODUCED_STATES = [
        'communication_power_consumption'  # 通信功耗状态 (W)
    ]

    # 严格按照精简的 CommunicationComponent 的 PRODUCED_METRICS
    NECESSARY_METRICS = [
        'uplink_throughput'  # MB/s - 上行有效吞吐量 (最重要的指标)
    ]

    def __init__(self, env, agent, component_name, task_name,
                 workflow_id=None, target_state=None, properties=None):
        """
        初始化通信任务

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

        # 从properties获取任务参数 - 精简版
        props = properties or {}
        self.data_id = props.get('data_id', 'unknown')  # 数据ID
        self.data_size_mb = props.get('data_size_mb', 10.0)  # 数据大小 MB

        # 通信任务状态 - 精简版
        self.start_time = None
        self.bytes_sent = 0.0  # 已发送数据量 MB
        self.last_update_time = env.now

        # 功耗状态
        self.communication_power_consumption = 5.0  # 基础功耗 W

        logger.info(f"Communication task {task_name} initialized: {self.data_size_mb}MB data_id={self.data_id}")
    
    def estimate_remaining_time(self, performance_metrics: Dict) -> float:
        """估算剩余时间 - 基于数据传输进度"""
        # 获取上行吞吐量
        uplink_throughput = performance_metrics.get('uplink_throughput', 1.0)  # MB/s

        if uplink_throughput <= 0:
            return float('inf')

        # 剩余数据量
        remaining_data = self.data_size_mb - self.bytes_sent

        if remaining_data <= 0:
            return 0.0

        # 估算剩余时间
        remaining_time = remaining_data / uplink_throughput

        return max(0.0, remaining_time)
    
    def _update_task_state(self, performance_metrics: Dict):
        """更新任务状态 - 基于数据传输进度"""
        current_time = self.env.now
        time_delta = current_time - self.last_update_time

        if time_delta <= 0:
            return

        if self.start_time is None:
            self.start_time = current_time

        # 获取上行吞吐量
        uplink_throughput = performance_metrics.get('uplink_throughput', 1.0)  # MB/s

        # 计算本周期传输的数据量
        if uplink_throughput > 0:
            transmitted_this_cycle = uplink_throughput * time_delta  # MB
            self.bytes_sent = min(self.bytes_sent + transmitted_this_cycle, self.data_size_mb)

        # 计算进度
        progress = self.bytes_sent / self.data_size_mb if self.data_size_mb > 0 else 1.0

        # 更新通信功耗 - 基于传输活动
        if uplink_throughput > 0:
            # 传输时功耗较高
            activity_factor = min(2.0, 1.0 + uplink_throughput / 5.0)  # 基于吞吐量调整功耗
            self.communication_power_consumption = 5.0 * activity_factor
        else:
            # 空闲时基础功耗
            self.communication_power_consumption = 5.0

        # 更新agent状态
        self.agent.set_state('communication_power_consumption', self.communication_power_consumption)

        # 检查任务完成
        if progress >= 1.0:
            logger.info(f"Communication task {self.name} completed: {self.bytes_sent:.2f}MB sent")

        self.last_update_time = current_time

    def _get_task_specific_state_repr(self) -> Dict[str, Any]:
        """返回任务特定状态的表示"""
        return {
            'communication_power_consumption': self.communication_power_consumption
        }
    
    def complete(self) -> Dict[str, Any]:
        """
        完成任务并生成最终结果。
        这是工作流状态机触发的关键。
        """
        # 首先调用父类方法获取基础结果字典
        super().complete()

        # 计算任务总耗时
        runtime = self.env.now - self.start_time if self.start_time is not None else 0.0
        
        # 计算平均吞吐量
        avg_throughput = self.bytes_sent / runtime if runtime > 0 else 0.0

        # 添加此任务特定的结果
        # 注意：这里的key必须与Workflow中的检查逻辑对应
        task_results = {
            'data_id': self.data_id,
            'data_size_mb': self.bytes_sent, # 返回实际发送的数据量
            'runtime': runtime,
            'avg_throughput_mbps': avg_throughput,
            'final_power_consumption': self.communication_power_consumption
        }
        
        # 将特定结果合并到父类的结果字典中
        self.result.update(task_results)
    # ------------------------------------