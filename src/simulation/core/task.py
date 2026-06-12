"""
任务基类

基于 AirFogSim 架构设计文档的 Task 设计。
使用 AirFogSim 的 Task 基类，增强工作流ID支持。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../src'))

from airfogsim.core.task import Task as BaseTask
from airfogsim.core.enums import TaskStatus
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class Task(BaseTask):
    """
    任务基类
    
    基于 AirFogSim Task 基类实现，增强了工作流集成能力。
    支持：
    1. 工作流ID关联
    2. 增强的任务完成事件数据
    3. 任务执行监控
    4. 结果数据结构化
    """
    
    def __init__(self, env, task_name: str, agent, component_name: str = None,
                 workflow_id: str = None, **kwargs):
        """
        初始化任务
        
        Args:
            env: 仿真环境
            task_name: 任务名称
            agent: 执行任务的智能体
            component_name: 负责执行的组件名称
            workflow_id: 关联的工作流ID
            **kwargs: 其他任务参数
        """
        super().__init__(env, agent, component_name, task_name, **kwargs)
        
        # 工作流集成
        self.workflow_id = workflow_id
        
        # 任务元数据
        self.task_class = self.__class__.__name__
        self.target_state = kwargs.get('target_state', {})
        self.task_properties = kwargs.get('properties', {})
        
        # 执行监控
        self.execution_metrics = {
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'retries': 0,
            'power_consumed': 0.0
        }
    
    def start(self):
        """开始执行任务"""
        self.execution_metrics['start_time'] = self.env.now
        logger.info(f"任务开始: {self.name} (工作流: {self.workflow_id})")
        
        # 调用父类开始方法
        super().start()
    
    def complete(self):
        """完成任务"""
        self.execution_metrics['end_time'] = self.env.now
        self.execution_metrics['duration'] = (
            self.execution_metrics['end_time'] - self.execution_metrics['start_time']
            if self.execution_metrics['start_time'] else 0.0
        )
        
        logger.info(f"任务完成: {self.name} (耗时: {self.execution_metrics['duration']:.2f}s)")
        
        # 调用父类完成方法
        super().complete()
        
        # 触发增强的任务完成事件
        self._trigger_enhanced_completion_event()
    
    def fail(self, reason: str = None):
        """任务失败"""
        self.execution_metrics['end_time'] = self.env.now
        self.execution_metrics['duration'] = (
            self.execution_metrics['end_time'] - self.execution_metrics['start_time']
            if self.execution_metrics['start_time'] else 0.0
        )
        
        logger.warning(f"任务失败: {self.name} (原因: {reason})")
        
        # 调用父类失败方法
        super().fail(reason)
        
        # 触发增强的任务失败事件
        self._trigger_enhanced_failure_event(reason)
    
    def cancel(self, reason: str = None):
        """取消任务"""
        self.execution_metrics['end_time'] = self.env.now
        self.execution_metrics['duration'] = (
            self.execution_metrics['end_time'] - self.execution_metrics['start_time']
            if self.execution_metrics['start_time'] else 0.0
        )
        
        logger.info(f"任务取消: {self.name} (原因: {reason})")
        
        # 调用父类取消方法
        super().cancel(reason)
        
        # 触发增强的任务取消事件
        self._trigger_enhanced_cancellation_event(reason)
    
    def _trigger_enhanced_completion_event(self):
        """触发增强的任务完成事件"""
        event_data = {
            'task_id': self.id,
            'task_name': self.name,
            'task_class': self.task_class,
            'component': self.component_name,
            'workflow_id': self.workflow_id,
            'target_state': self.target_state,
            'status': self.status.name,
            'result': self.result,
            'execution_metrics': self.execution_metrics,
            'properties': self.task_properties,
            'agent_id': self.agent.id,
            'completion_time': self.env.now
        }
        
        # 使用环境的事件系统触发事件
        if hasattr(self.env, 'event_registry'):
            self.env.event_registry.trigger_task_completed_event(self.agent.id, event_data)
        else:
            # 回退到智能体事件系统
            if hasattr(self.agent, 'trigger_event'):
                self.agent.trigger_event('task_completed', event_data)
    
    def _trigger_enhanced_failure_event(self, reason: str):
        """触发增强的任务失败事件"""
        event_data = {
            'task_id': self.id,
            'task_name': self.name,
            'task_class': self.task_class,
            'component': self.component_name,
            'workflow_id': self.workflow_id,
            'target_state': self.target_state,
            'status': self.status.name,
            'result': self.result,
            'execution_metrics': self.execution_metrics,
            'properties': self.task_properties,
            'agent_id': self.agent.id,
            'failure_time': self.env.now
        }
        
        error_info = {
            'error_type': 'task_execution_error',
            'error_message': reason or 'Unknown error'
        }
        
        if hasattr(self.env, 'event_registry'):
            self.env.event_registry.trigger_task_failed_event(self.agent.id, event_data, error_info)
        else:
            event_data['error_info'] = error_info
            if hasattr(self.agent, 'trigger_event'):
                self.agent.trigger_event('task_failed', event_data)
    
    def _trigger_enhanced_cancellation_event(self, reason: str):
        """触发增强的任务取消事件"""
        event_data = {
            'task_id': self.id,
            'task_name': self.name,
            'task_class': self.task_class,
            'component': self.component_name,
            'workflow_id': self.workflow_id,
            'target_state': self.target_state,
            'status': self.status.name,
            'result': self.result,
            'execution_metrics': self.execution_metrics,
            'properties': self.task_properties,
            'agent_id': self.agent.id,
            'cancellation_time': self.env.now
        }
        
        if hasattr(self.env, 'event_registry'):
            self.env.event_registry.trigger_task_canceled_event(self.agent.id, event_data, reason or 'Unknown reason')
        else:
            event_data['reason'] = reason or 'Unknown reason'
            if hasattr(self.agent, 'trigger_event'):
                self.agent.trigger_event('task_canceled', event_data)
    
    def update_progress(self, progress: float, details: Dict[str, Any] = None):
        """
        更新任务进度
        
        Args:
            progress: 进度百分比 (0.0-1.0)
            details: 进度详情
        """
        self.result['progress'] = progress
        if details:
            self.result['progress_details'] = details
        
        # 触发进度更新事件
        if hasattr(self.agent, 'trigger_event'):
            self.agent.trigger_event('task_progress_updated', {
                'task_id': self.id,
                'task_name': self.name,
                'workflow_id': self.workflow_id,
                'progress': progress,
                'details': details
            })
    
    def add_execution_metric(self, key: str, value: Any):
        """添加执行指标"""
        self.execution_metrics[key] = value
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'task_id': self.id,
            'task_name': self.name,
            'task_class': self.task_class,
            'workflow_id': self.workflow_id,
            'status': self.status.name,
            'execution_metrics': self.execution_metrics,
            'result_summary': {
                'success': self.status == TaskStatus.COMPLETED,
                'progress': self.result.get('progress', 0.0),
                'error': self.result.get('error')
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        base_dict = super().to_dict() if hasattr(super(), 'to_dict') else {}
        
        enhanced_dict = {
            'id': self.id,
            'name': self.name,
            'task_class': self.task_class,
            'component': self.component_name,
            'workflow_id': self.workflow_id,
            'status': self.status.name,
            'target_state': self.target_state,
            'properties': self.task_properties,
            'execution_metrics': self.execution_metrics,
            'result': self.result,
            'agent_id': self.agent.id if self.agent else None
        }
        
        # 合并基类字典
        base_dict.update(enhanced_dict)
        return base_dict
