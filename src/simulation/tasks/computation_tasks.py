"""计算任务实现

实现计算相关的任务：ComputationTask

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

class ComputeTask(Task):
    """
    Perform computational operations.

    This task executes computational workloads such as AI inference, data processing, or analysis.
    The agent should have sufficient computational resources before starting this task.

    Required parameters:
    - No required target_state (computation happens at current location)

    Required properties:
    - required_gflops: Total computational workload in GFLOPs (Giga Floating Point Operations)
    - job_id: Unique identifier for the computation job

    Optional properties:
    - compute_location: Specific coordinates [x, y, z] for computation (default: current position)
    - workload_profile: Dict with 'cpu_intensity' and 'memory_intensity' (0-1 scale)
    - priority: Computation priority 'low', 'normal', 'high' (default: 'normal')
    - max_execution_time: Maximum allowed execution time in seconds (default: 3600)

    Example usage:
    properties = {'required_gflops': 1500.0, 'job_id': 'inference_001', 'workload_profile': {'cpu_intensity': 0.8, 'memory_intensity': 0.6}}
    """

    # 任务执行会改变的无人机状态 - 与DroneAgent状态模板严格匹配
    PRODUCED_STATES = [
        'computing_power_consumption',  # float - 计算功耗 (W)，任务运行时必须更新此状态
        'cpu_utilization',             # float - CPU使用率，任务根据其workload_profile按比例增加此值
        'memory_usage'                 # float - 内存使用量，任务根据其workload_profile增加此值
    ]

    # 任务决策依赖的性能指标 - 严格按照 ComputeComponent 的 PRODUCED_METRICS
    NECESSARY_METRICS = [
        'processing_throughput_gflops',  # float - 从ComputeComponent获取，用于计算任务进度
        'memory_pressure'                # float - 从ComputeComponent获取，用于资源检查
    ]

    def __init__(self, env, agent, component_name, task_name,
                 workflow_id=None, target_state=None, properties=None):
        """
        初始化计算任务

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

        # 从properties获取任务参数
        props = properties or {}
        self.job_id = props.get('job_id', 'unknown_job')  # 作业ID
        self.required_gflops = props.get('required_gflops', 1000.0)  # 需要完成的总计算量 (GFLOPs)
        self.workload_profile = props.get('workload_profile', {})  # 资源消耗特征
        self.cpu_intensity = self.workload_profile.get('cpu_intensity', 0.5)  # CPU密集型 (0-1)
        self.memory_intensity = self.workload_profile.get('memory_intensity', 0.3)  # 内存占用 (0-1)
        
        # 计算任务状态
        self.computation_status = 'preparing'
        self.start_time = None
        self.progress = 0.0
        self.completed_gflops = 0.0  # 已完成的计算量
        self.last_update_time = env.now

        # 资源使用状态
        self.current_cpu_usage = 0.0  # 当前CPU使用增量
        self.current_memory_usage = 0.0  # 当前内存使用增量
        self.current_power_consumption = 15.0  # 当前功耗

        # 初始化计算任务
        self._initialize_computation()
    
    def _initialize_computation(self):
        """初始化计算任务状态"""
        # 设置开始时间
        self.start_time = self.env.now

        # 初始化状态
        self.computation_status = 'initialized'
        logger.info(f"Compute task {self.name} initialized for job {self.job_id} with {self.required_gflops} GFLOPs")
    
    def estimate_remaining_time(self, performance_metrics: Dict) -> float:
        """估算剩余时间 - 基于GFLOPs处理进度"""
        # 获取处理吞吐量 (GFLOPs/s)
        throughput_gflops = performance_metrics.get('processing_throughput_gflops', 1.0)

        if throughput_gflops <= 0:
            return float('inf')

        # 剩余计算量
        remaining_gflops = self.required_gflops - self.completed_gflops

        # 估算剩余时间
        remaining_time = remaining_gflops / throughput_gflops

        return max(0.0, remaining_time)
    
    def _update_task_state(self, performance_metrics: Dict):
        """更新任务状态 - 核心方法"""
        current_time = self.env.now
        time_delta = current_time - self.last_update_time

        if time_delta <= 0:
            return

        # 获取性能指标
        throughput_gflops = performance_metrics.get('processing_throughput_gflops', 1.0)
        memory_pressure = performance_metrics.get('memory_pressure', 0.0)

        # 检查内存压力是否过高
        if memory_pressure > 0.9:
            logger.warning(f"High memory pressure {memory_pressure:.2f} for task {self.name}")
            # 可以选择降低任务优先级或暂停任务

        # 更新计算进度
        gflops_completed = throughput_gflops * time_delta
        self.completed_gflops = min(
            self.completed_gflops + gflops_completed,
            self.required_gflops
        )

        # 更新进度百分比
        self.progress = self.completed_gflops / self.required_gflops if self.required_gflops > 0 else 1.0

        # 更新状态
        if self.computation_status == 'initialized':
            self.computation_status = 'processing'

        # 计算资源使用增量
        self.current_cpu_usage = self.cpu_intensity * 100.0  # 转换为百分比
        self.current_memory_usage = self.memory_intensity * 2.0  # GB
        self.current_power_consumption = 15.0 + self.cpu_intensity * 30.0  # W

        self.agent.set_state('cpu_utilization', self.current_cpu_usage)
        self.agent.set_state('memory_usage', self.current_memory_usage)
        self.agent.set_state('computing_power_consumption', self.current_power_consumption)

        # 检查任务完成
        if self.progress >= 1.0:
            # 任务完成时释放资源
            self._release_resources()

        self.last_update_time = current_time

    def _release_resources(self):
        """释放任务占用的资源"""
        current_cpu = self.agent.get_state('cpu_utilization') or 0.0
        current_memory = self.agent.get_state('memory_usage') or 0.0
        current_power = self.agent.get_state('computing_power_consumption') or 0.0

        # 释放资源
        self.agent.set_state('cpu_utilization', max(0.0, current_cpu - self.current_cpu_usage))
        self.agent.set_state('memory_usage', max(0.0, current_memory - self.current_memory_usage))
        self.agent.set_state('computing_power_consumption', max(15.0, current_power - self.current_power_consumption))

        logger.info(f"Released resources for task {self.name}: CPU {self.current_cpu_usage}%, Memory {self.current_memory_usage}GB")

        # ------------------ 代码补全部分 ------------------
    def _get_task_specific_state_repr(self) -> Dict[str, Any]:
        """
        返回任务特定状态的表示，用于日志记录和调试。
        
        这个字典提供了计算任务在任何时刻的关键内部状态快照。
        """
        return {
            'computing_power_consumption': self.current_power_consumption,
            'cpu_utilization': self.current_cpu_usage,
            'memory_usage': self.current_memory_usage
        }
    # ----------------------------------------------------

    def complete(self) -> Dict[str, Any]:
        """
        完成任务并生成最终结果。
        """
        super().complete()
        
        runtime = self.env.now - self.start_time if self.start_time is not None else 0.0
        
        # 计算平均处理吞吐量
        avg_throughput_gflops = self.completed_gflops / runtime if runtime > 0 else 0.0

        task_results = {
            'job_id': self.job_id,
            'completed_gflops': self.completed_gflops,
            'runtime': runtime,
            'avg_throughput_gflops': avg_throughput_gflops,
            'final_power_consumption': self.current_power_consumption
        }
        
        self.result.update(task_results)
        
        # 确保在任务结束时资源已被正确释放
        if self.computation_status != 'completed':
            self._release_resources()
            self.computation_status = 'completed'

        return self.result