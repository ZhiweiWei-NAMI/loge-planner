"""
数据收集器模块

该模块提供了仿真数据收集功能，包括：
- 任务执行时间统计
- 智能体移动距离统计  
- 工作流时间统计
- 数据导出功能
"""

import logging
import json
import os
import time
import math
from typing import Dict, List, Any, Optional
from collections import defaultdict
from airfogsim.core.enums import WorkflowStatus

logger = logging.getLogger(__name__)


class DataCollector:
    """仿真数据收集器
    
    负责收集仿真过程中的各种数据，包括任务执行时间、移动距离、工作流时间等。
    """
    
    def __init__(self, env, event_registry, method_name: str = "default"):
        """初始化数据收集器

        Args:
            env: 仿真环境对象
            event_registry: 事件注册器对象
            method_name: 规划方法名称，用于区分不同实验
        """
        self.env = env
        self.event_registry = event_registry
        self.method_name = method_name

        # 任务执行时间数据收集
        self.task_execution_data = defaultdict(list)  # agent_id -> [{'task_id', 'start_time', 'end_time', 'duration', 'workflow_type'}]
        self.task_start_times = {}  # task_id -> {'agent_id', 'start_time', 'workflow_type'}

        # 移动距离数据收集
        self.movement_data = defaultdict(list)  # agent_id -> [{'task_id', 'distance', 'start_pos', 'end_pos'}]

        # 工作流时间数据收集
        self.workflow_data = defaultdict(dict)  # workflow_id -> {'start_time', 'end_time', 'duration', 'workflow_type'}

        # 工作流类型映射
        self.workflow_types = {}  # workflow_id -> workflow_type

        # 注册事件监听器
        self._register_event_listeners()

        logger.info(f"数据收集器初始化完成 (方法: {method_name})")
    
    def _register_event_listeners(self):
        """注册数据收集事件监听器"""
        # 监听所有source_id的所有事件，然后在回调中过滤task_started和task_completed事件
        self.event_registry.subscribe(
            source_id="*",
            event_name="*",
            listener_id="data_collector_all_events",
            callback=self._on_any_event
        )
        
        # 监听工作流开始和完成事件
        self.event_registry.subscribe(
            source_id="*",
            event_name="workflow_started",
            listener_id="data_collector_workflow_started",
            callback=self._on_workflow_started
        )

        self.event_registry.subscribe(
            source_id="*",
            event_name="workflow_completed",
            listener_id="data_collector_workflow_completed",
            callback=self._on_workflow_completed
        )

    def _on_any_event(self, event_data):
        """处理所有事件，过滤出我们需要的任务事件"""
        event_name = event_data.get('event_name', '')

        # 检查是否是任务相关事件 (AirFogSim 组件触发的是 task_started/task_completed)
        if event_name == 'task_started' or event_name.endswith('.task_started'):
            self._on_task_started(event_data)
        elif event_name == 'task_completed' or event_name.endswith('.task_completed'):
            self._on_task_completed(event_data)
        
        logger.debug("数据收集事件监听器注册完成")

    def _on_task_started(self, event_data):
        """处理任务开始事件"""
        try:
            # 从事件数据中获取信息
            task_id = event_data.get('task_id')
            agent_id = event_data.get('agent_id')
            start_time = event_data.get('time', self.env.now)
            workflow_id = event_data.get('workflow_id')

            # 打印调试信息
            logger.debug(f"任务开始事件数据: {event_data}")

            # 获取工作流类型
            workflow_type = "Unknown"
            if workflow_id and workflow_id in self.workflow_types:
                workflow_type = self.workflow_types[workflow_id]
            elif workflow_id:
                # 尝试从环境中获取工作流信息
                workflow_type = self._get_workflow_type_from_env(workflow_id)
                if workflow_type:
                    self.workflow_types[workflow_id] = workflow_type

            if task_id and agent_id:
                self.task_start_times[task_id] = {
                    'agent_id': agent_id,
                    'start_time': start_time,
                    'workflow_type': workflow_type,
                    'workflow_id': workflow_id
                }
                logger.debug(f"记录任务开始: Agent {agent_id}, Task {task_id}, Workflow {workflow_id} ({workflow_type}), Time {start_time}")
        except Exception as e:
            logger.error(f"处理任务开始事件时出错: {e}")

    def _on_task_completed(self, event_data):
        """处理任务完成事件"""
        try:
            # 从事件数据中获取信息
            task_id = event_data.get('task_id')
            agent_id = event_data.get('agent_id')
            task_class = event_data.get('task_class', '')
            end_time = event_data.get('completion_time', self.env.now)
            workflow_id = event_data.get('workflow_id')

            # 打印调试信息
            logger.debug(f"任务完成事件数据: {event_data}")

            if not task_id or not agent_id:
                return

            # 1. 记录任务执行时间
            if task_id in self.task_start_times:
                start_info = self.task_start_times[task_id]
                start_time = start_info['start_time']
                workflow_type = start_info.get('workflow_type', 'Unknown')
                stored_workflow_id = start_info.get('workflow_id')

                # 如果完成事件中有workflow_id，优先使用它
                if workflow_id and workflow_id != stored_workflow_id:
                    workflow_type = self._get_workflow_type_from_env(workflow_id)
                    if workflow_type:
                        self.workflow_types[workflow_id] = workflow_type

                duration = end_time - start_time

                self.task_execution_data[agent_id].append({
                    'task_id': task_id,
                    'task_class': task_class,
                    'workflow_type': workflow_type,
                    'workflow_id': workflow_id or stored_workflow_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })

                # 清理已完成的任务记录
                del self.task_start_times[task_id]
                logger.debug(f"记录任务执行时间: Agent {agent_id}, Task {task_id}, Workflow {workflow_id or stored_workflow_id} ({workflow_type}), Duration {duration:.2f}s")

            # 2. 如果是MoveToTask，记录移动距离
            if task_class == 'MoveToTask':
                self._record_movement_data(event_data, agent_id, task_id)

        except Exception as e:
            logger.error(f"处理任务完成事件时出错: {e}")

    def _record_movement_data(self, event_data, agent_id, task_id):
        """记录移动任务的距离数据"""
        try:
            # 从事件数据中获取移动相关信息
            result = event_data.get('result', {})
            properties = event_data.get('properties', {})
            
            # 尝试从不同位置获取距离信息
            distance = None
            start_pos = None
            end_pos = None
            
            # 方法1: 从result中获取
            if isinstance(result, dict):
                distance = result.get('distance_traveled') or result.get('total_distance')
                start_pos = result.get('start_position')
                end_pos = result.get('end_position') or result.get('target_position')
            
            # 方法2: 从properties中获取
            if distance is None and isinstance(properties, dict):
                distance = properties.get('distance_traveled') or properties.get('total_distance')
                start_pos = properties.get('start_position')
                end_pos = properties.get('target_position')
            
            # 方法3: 如果有起始和结束位置，计算距离
            if distance is None and start_pos and end_pos:
                distance = self._calculate_distance(start_pos, end_pos)
            
            # 记录移动数据
            if distance is not None:
                self.movement_data[agent_id].append({
                    'task_id': task_id,
                    'distance': distance,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'time': self.env.now
                })
                logger.debug(f"记录移动距离: Agent {agent_id}, Task {task_id}, Distance {distance:.2f}m")
                
        except Exception as e:
            logger.error(f"记录移动数据时出错: {e}")

    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的距离"""
        if not pos1 or not pos2:
            return 0.0
        try:
            x1, y1, z1 = pos1[:3] if len(pos1) >= 3 else (pos1[0], pos1[1], 0)
            x2, y2, z2 = pos2[:3] if len(pos2) >= 3 else (pos2[0], pos2[1], 0)
            return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        except:
            return 0.0

    def _get_workflow_type_from_env(self, workflow_id):
        """从环境中获取工作流类型"""
        try:
            # 尝试从工作流管理器获取工作流信息
            if hasattr(self.env, 'workflow_manager') and hasattr(self.env.workflow_manager, 'workflows'):
                workflow = self.env.workflow_manager.workflows.get(workflow_id)
                if workflow:
                    # 获取工作流类型
                    workflow_type = workflow.__class__.__name__
                    # 简化工作流类型名称
                    if 'Logistics' in workflow_type:
                        return 'Logistics'
                    elif 'Communication' in workflow_type:
                        return 'Communication'
                    elif 'Compute' in workflow_type:
                        return 'Computation'
                    elif 'Inspection' in workflow_type:
                        return 'Inspection'
                    else:
                        return workflow_type
            return "Unknown"
        except Exception as e:
            logger.debug(f"获取工作流类型失败: {e}")
            return "Unknown"

    def _on_workflow_started(self, event_data):
        """处理工作流开始事件"""
        try:
            workflow_id = event_data.get('workflow_id')
            start_time = event_data.get('time', self.env.now)

            if workflow_id:
                # 获取工作流类型
                workflow_type = self._get_workflow_type_from_env(workflow_id)
                self.workflow_types[workflow_id] = workflow_type

                self.workflow_data[workflow_id]['start_time'] = start_time
                self.workflow_data[workflow_id]['workflow_type'] = workflow_type
                logger.debug(f"记录工作流开始: {workflow_id}, Type {workflow_type}, Time {start_time}")
        except Exception as e:
            logger.error(f"处理工作流开始事件时出错: {e}")

    def _on_workflow_completed(self, event_data):
        """处理工作流完成事件"""
        try:
            workflow_id = event_data.get('workflow_id')
            end_time = event_data.get('time', self.env.now)

            if workflow_id:
                if workflow_id not in self.workflow_data:
                    self.workflow_data[workflow_id] = {}

                # 确保工作流类型信息存在
                if 'workflow_type' not in self.workflow_data[workflow_id]:
                    workflow_type = self._get_workflow_type_from_env(workflow_id)
                    self.workflow_data[workflow_id]['workflow_type'] = workflow_type
                    self.workflow_types[workflow_id] = workflow_type

                self.workflow_data[workflow_id]['end_time'] = end_time

                # 计算持续时间
                if 'start_time' in self.workflow_data[workflow_id]:
                    start_time = self.workflow_data[workflow_id]['start_time']
                    duration = end_time - start_time
                    self.workflow_data[workflow_id]['duration'] = duration
                    workflow_type = self.workflow_data[workflow_id].get('workflow_type', 'Unknown')
                    logger.debug(f"记录工作流完成: {workflow_id}, Type {workflow_type}, Duration {duration:.2f}s")
        except Exception as e:
            logger.error(f"处理工作流完成事件时出错: {e}")

    def collect_workflow_times_from_existing(self):
        """从现有工作流中收集时间数据"""
        try:
            # 优先使用环境中的真实 Workflow 实例来补全信息
            if not hasattr(self.env, 'workflow_manager') or not hasattr(self.env.workflow_manager, 'workflows'):
                return

            for workflow_id, workflow in self.env.workflow_manager.workflows.items():
                # 确保字典条目存在
                if workflow_id not in self.workflow_data:
                    self.workflow_data[workflow_id] = {}

                entry = self.workflow_data[workflow_id]

                # 仅在缺失时补全工作流类型，避免覆盖事件回调中已写入的类型
                if 'workflow_type' not in entry:
                    workflow_type = workflow.__class__.__name__
                    if 'Logistics' in workflow_type:
                        workflow_type = 'Logistics'
                    elif 'Communication' in workflow_type:
                        workflow_type = 'Communication'
                    elif 'Compute' in workflow_type:
                        workflow_type = 'Computation'
                    elif 'Inspection' in workflow_type:
                        workflow_type = 'Inspection'

                    entry['workflow_type'] = workflow_type
                    self.workflow_types[workflow_id] = workflow_type

                # 仅在缺失时补全开始时间
                if getattr(workflow, 'start_time', None) is not None and 'start_time' not in entry:
                    entry['start_time'] = workflow.start_time

                # 若 Workflow 已完成且有结束时间，则补全结束时间和持续时间
                if (
                    getattr(workflow, 'end_time', None) is not None
                    and getattr(workflow, 'status', None) == WorkflowStatus.COMPLETED
                ):
                    entry['end_time'] = workflow.end_time

                    if 'start_time' in entry:
                        entry['duration'] = workflow.end_time - entry['start_time']

        except Exception as e:
            logger.error(f"从现有工作流收集时间数据时出错: {e}")

    def export_data_to_json(self, output_dir: str):
        """将收集的数据导出为JSON文件

        Args:
            output_dir: 输出目录路径
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 生成时间戳作为文件名前缀
            timestamp = int(time.time())

            # 定义文件名 - 使用方法名作为前缀
            task_execution_file = os.path.join(output_dir, f'{self.method_name}_{timestamp}_task_execution.json')
            movement_file = os.path.join(output_dir, f'{self.method_name}_{timestamp}_movement.json')
            workflow_file = os.path.join(output_dir, f'{self.method_name}_{timestamp}_workflow_summary.json')

            # 导出任务执行数据
            self._export_task_execution_data(task_execution_file)

            # 导出移动数据
            self._export_movement_data(movement_file)

            # 导出工作流数据
            self._export_workflow_data(workflow_file)

            logger.info(f"数据已成功导出到目录: {output_dir}")

        except Exception as e:
            logger.error(f"导出数据时出错: {e}")

    def _export_task_execution_data(self, filename: str):
        """导出任务执行时间数据"""
        try:
            # 转换为可序列化的格式
            data = {
                'description': '每个agent的任务执行时间记录',
                'simulation_time': self.env.now,
                'total_agents': len(self.task_execution_data),
                'total_tasks': sum(len(tasks) for tasks in self.task_execution_data.values()),
                'data': dict(self.task_execution_data)
            }

            # 计算统计信息
            all_durations = []
            for agent_tasks in self.task_execution_data.values():
                for task in agent_tasks:
                    all_durations.append(task['duration'])

            if all_durations:
                data['statistics'] = {
                    'average_task_duration': sum(all_durations) / len(all_durations),
                    'min_task_duration': min(all_durations),
                    'max_task_duration': max(all_durations)
                }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"任务执行时间数据已导出: {filename}")

        except Exception as e:
            logger.error(f"导出任务执行时间数据时出错: {e}")

    def _export_movement_data(self, filename: str):
        """导出移动距离数据"""
        try:
            data = {
                'description': '每个agent的移动距离统计',
                'simulation_time': self.env.now,
                'total_agents': len(self.movement_data),
                'data': dict(self.movement_data)
            }

            # 计算统计信息
            all_distances = []
            for agent_movements in self.movement_data.values():
                for movement in agent_movements:
                    all_distances.append(movement['distance'])

            if all_distances:
                data['statistics'] = {
                    'total_distance': sum(all_distances),
                    'average_distance_per_move': sum(all_distances) / len(all_distances),
                    'min_distance': min(all_distances),
                    'max_distance': max(all_distances)
                }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"移动距离数据已导出: {filename}")

        except Exception as e:
            logger.error(f"导出移动距离数据时出错: {e}")

    def _export_workflow_data(self, filename: str):
        """导出工作流时间数据"""
        try:
            data = {
                'description': '工作流时间统计',
                'simulation_time': self.env.now,
                'total_workflows': len(self.workflow_data),
                'data': dict(self.workflow_data)
            }

            # 计算统计信息（对失败的workflow使用1800s惩罚）
            all_workflow_durations = []
            completed_count = 0
            failed_count = 0
            
            for workflow_info in self.workflow_data.values():
                if 'duration' in workflow_info:
                    # 成功完成的workflow
                    all_workflow_durations.append(workflow_info['duration'])
                    completed_count += 1
                else:
                    # 失败的workflow，使用1800s惩罚
                    all_workflow_durations.append(1800.0)
                    failed_count += 1

            if all_workflow_durations:
                data['statistics'] = {
                    'completed_workflows': completed_count,
                    'failed_workflows': failed_count,
                    'average_completion_time': sum(all_workflow_durations) / len(all_workflow_durations),
                    'min_completion_time': min(all_workflow_durations),
                    'max_completion_time': max(all_workflow_durations)
                }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"工作流时间数据已导出: {filename}")

        except Exception as e:
            logger.error(f"导出工作流时间数据时出错: {e}")

    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        summary = {
            'simulation_time': self.env.now,
            'task_execution': {
                'total_agents_with_tasks': len(self.task_execution_data),
                'total_tasks': sum(len(tasks) for tasks in self.task_execution_data.values())
            },
            'movement': {
                'total_agents_with_movement': len(self.movement_data),
                'total_movements': sum(len(movements) for movements in self.movement_data.values())
            },
            'workflow': {
                'total_workflows': len(self.workflow_data),
                'completed_workflows': len([w for w in self.workflow_data.values() if 'duration' in w])
            }
        }

        # 计算任务执行时间统计
        all_durations = []
        for agent_tasks in self.task_execution_data.values():
            for task in agent_tasks:
                all_durations.append(task['duration'])

        if all_durations:
            summary['task_execution'].update({
                'average_duration': sum(all_durations) / len(all_durations),
                'min_duration': min(all_durations),
                'max_duration': max(all_durations)
            })

        # 计算移动距离统计
        all_distances = []
        for agent_movements in self.movement_data.values():
            for movement in agent_movements:
                all_distances.append(movement['distance'])

        if all_distances:
            summary['movement'].update({
                'total_distance': sum(all_distances),
                'average_distance': sum(all_distances) / len(all_distances),
                'min_distance': min(all_distances),
                'max_distance': max(all_distances)
            })

        # 计算工作流时间统计（对失败的workflow使用1800s惩罚）
        all_durations = []
        completed_count = 0
        failed_count = 0
        
        for workflow_info in self.workflow_data.values():
            if 'duration' in workflow_info:
                all_durations.append(workflow_info['duration'])
                completed_count += 1
            else:
                all_durations.append(1800.0)
                failed_count += 1

        if all_durations:
            summary['workflow'].update({
                'completed_workflows': completed_count,
                'failed_workflows': failed_count,
                'average_completion_time': sum(all_durations) / len(all_durations),
                'min_completion_time': min(all_durations),
                'max_completion_time': max(all_durations)
            })

        return summary

    def print_summary(self):
        """打印数据收集统计摘要"""
        logger.info(f"\n--- 数据收集统计 ---")

        # 任务执行时间统计
        total_agents_with_tasks = len(self.task_execution_data)
        total_tasks = sum(len(tasks) for tasks in self.task_execution_data.values())
        logger.info(f"任务执行统计: {total_agents_with_tasks} 个智能体执行了 {total_tasks} 个任务")

        if total_tasks > 0:
            all_durations = []
            for agent_tasks in self.task_execution_data.values():
                for task in agent_tasks:
                    all_durations.append(task['duration'])

            avg_duration = sum(all_durations) / len(all_durations)
            logger.info(f"  平均任务执行时间: {avg_duration:.2f} 秒")
            logger.info(f"  最短任务时间: {min(all_durations):.2f} 秒")
            logger.info(f"  最长任务时间: {max(all_durations):.2f} 秒")

        # 移动距离统计
        total_agents_with_movement = len(self.movement_data)
        total_movements = sum(len(movements) for movements in self.movement_data.values())
        logger.info(f"移动统计: {total_agents_with_movement} 个智能体进行了 {total_movements} 次移动")

        if total_movements > 0:
            all_distances = []
            for agent_movements in self.movement_data.values():
                for movement in agent_movements:
                    all_distances.append(movement['distance'])

            total_distance = sum(all_distances)
            avg_distance = total_distance / len(all_distances)
            logger.info(f"  总移动距离: {total_distance:.2f} 米")
            logger.info(f"  平均单次移动距离: {avg_distance:.2f} 米")

        # 工作流统计（对失败的workflow使用1800s惩罚）
        total_workflows = len(self.workflow_data)
        completed_workflows = len([w for w in self.workflow_data.values() if 'duration' in w])
        failed_workflows = total_workflows - completed_workflows
        logger.info(f"工作流统计: {total_workflows} 个工作流，{completed_workflows} 个已完成，{failed_workflows} 个失败")

        if total_workflows > 0:
            all_durations = []
            for workflow_info in self.workflow_data.values():
                if 'duration' in workflow_info:
                    all_durations.append(workflow_info['duration'])
                else:
                    all_durations.append(1800.0)

            avg_completion_time = sum(all_durations) / len(all_durations)
            logger.info(f"  平均完成时间（含失败惩罚1800s）: {avg_completion_time:.2f} 秒")
            logger.info(f"  最短完成时间: {min(all_durations):.2f} 秒")
            logger.info(f"  最长完成时间: {max(all_durations):.2f} 秒")
