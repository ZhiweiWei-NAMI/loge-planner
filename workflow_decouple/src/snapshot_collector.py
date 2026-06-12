"""
快照数据收集器

每隔固定时间（如10秒）收集一次仿真状态的快照，包括：
- Agent位置
- Agent正在执行的任务类型
- Workflow状态和checkpoint信息
- 任务参数等

@author: TCCN Auto Planner
"""

import logging
import pandas as pd
from typing import Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class SnapshotCollector:
    """快照数据收集器
    
    基于事件驱动采集仿真状态，监听task_started和task_completed事件
    """
    
    def __init__(self, env, event_registry, scheduler=None):
        """初始化快照收集器
        
        Args:
            env: AirFogSim环境对象
            event_registry: 事件注册表
            scheduler: workflow调度器（用于获取workflow对象）
        """
        self.env = env
        self.event_registry = event_registry
        self.scheduler = scheduler
        
        # 注册的对象
        self.agents = {}  # agent_id -> agent对象
        self.workflows = {}  # workflow_id -> workflow对象
        self.completed_workflows = set()  # 已完成的workflow_id集合，不再采集数据
        self.workflow_init_info = []  # 存储workflow初始化信息
        self.snapshots = []  # agent快照列表
        
        # 注册任务事件监听器（事件驱动采集）
        self._register_task_event_listeners()
        
        # 注册workflow事件监听器（用于采集workflow初始化信息）
        self._register_workflow_listeners()
        
        logger.info(f"快照收集器初始化完成，使用事件驱动采集模式")
    
    def _register_task_event_listeners(self):
        """注册任务事件监听器（task_started和task_completed）
        仅监听 Agent 级别的裸事件，避免捕获组件前缀事件导致字段缺失
        """
        # 只订阅 Agent 触发的裸事件
        self.event_registry.subscribe(
            source_id="*",
            event_name="task_started",
            listener_id="snapshot_collector_task_started",
            callback=self._on_task_started
        )
        self.event_registry.subscribe(
            source_id="*",
            event_name="task_completed",
            listener_id="snapshot_collector_task_completed",
            callback=self._on_task_completed
        )
        
        logger.info("已注册Agent级任务事件监听器 (task_started / task_completed)")
    
    def _register_workflow_listeners(self):
        """注册workflow相关事件监听器"""
        # 监听workflow开始事件，捕获初始化信息
        self.event_registry.subscribe(
            source_id="*",
            event_name="workflow_started",
            listener_id="snapshot_collector_workflow_init",
            callback=self._on_workflow_started
        )
        
        logger.debug("Workflow事件监听器注册完成")
    
    def _on_task_started(self, event_data):
        """处理task_started事件，采集agent快照"""
        try:
            # 从事件数据中提取信息
            # event_data格式: {'task_id': '...', 'task_name': '...', 'time': ..., 'source_id': 'agent_xxx'}
            agent_id = event_data['source_id'] if 'source_id' in event_data else event_data['agent_id']
            task_id = event_data['task_id']
            task_name = event_data['task_name']
            event_time = event_data['time']
            
            logger.debug(f"收到task_started事件: agent_id={agent_id}, task_id={task_id}, task_name={task_name}")
            
            if not agent_id or agent_id not in self.agents:
                logger.debug(f"跳过：agent_id={agent_id}, 在agents中={'是' if agent_id and agent_id in self.agents else '否'}")
                return
            
            agent = self.agents[agent_id]
            
            # 采集agent状态快照，标记为task_started事件
            snapshot_record = self._extract_agent_state(agent_id, agent, event_time, event_type='task_started')
            
            if snapshot_record:
                self.snapshots.append(snapshot_record)
                logger.debug(f"时间 {event_time}: 采集task_started快照 - Agent: {agent_id}, Task: {task_name}")
        
        except Exception as e:
            logger.error(f"处理task_started事件时出错: {e}")
    
    def _on_task_completed(self, event_data):
        """处理task_completed事件，采集agent快照"""
        try:
            # 从事件数据中提取信息
            # event_data格式: {'task_id': '...', 'task_name': '...', 'status': '...', 'result': ..., 'time': ..., 'source_id': 'agent_xxx'}
            agent_id = event_data['source_id'] if 'source_id' in event_data else event_data['agent_id']
            task_id = event_data['task_id']
            task_name = event_data['task_name']
            task_status = event_data['status']
            event_time = event_data['time']
            
            logger.debug(f"收到task_completed事件: agent_id={agent_id}, task_id={task_id}, task_name={task_name}, status={task_status}")
            
            if not agent_id or agent_id not in self.agents:
                logger.debug(f"跳过：agent_id={agent_id}, 在agents中={'是' if agent_id and agent_id in self.agents else '否'}")
                return
            
            agent = self.agents[agent_id]
            
            # 采集agent状态快照，标记为task_completed事件
            snapshot_record = self._extract_agent_state(agent_id, agent, event_time, event_type='task_completed')
            
            if snapshot_record:
                self.snapshots.append(snapshot_record)
                logger.debug(f"时间 {event_time}: 采集task_completed快照 - Agent: {agent_id}, Task: {task_name}, Status: {task_status}")
        
        except Exception as e:
            logger.error(f"处理task_completed事件时出错: {e}")
    
    def _on_workflow_started(self, event_data):
        """处理workflow开始事件，记录初始化信息"""
        try:
            workflow_id = event_data.get('workflow_id')
            if not workflow_id:
                return
            
            # 获取workflow对象
            workflow = None
            if hasattr(self.env, 'workflow_manager'):
                workflow = self.env.workflow_manager.workflows.get(workflow_id)
            
            if not workflow:
                logger.warning(f"无法获取workflow对象: {workflow_id}")
                return
            
            # 记录workflow初始化信息（要求存在start_delay）
            start_delay = workflow.properties['start_delay']
            
            init_info = {
                'workflow_id': workflow_id,
                'workflow_type': workflow.__class__.__name__,
                'start_time': start_delay,  # 使用start_delay而不是self.env.now
                'agent_id': getattr(workflow, 'agent_id', None),
                'status': workflow.status.name if hasattr(workflow, 'status') else 'UNKNOWN'
            }
            
            # 根据workflow类型提取特定信息
            if hasattr(workflow, 'pickup_location'):
                init_info['pickup_location'] = workflow.pickup_location
            if hasattr(workflow, 'delivery_location'):
                init_info['delivery_location'] = workflow.delivery_location
            if hasattr(workflow, 'payload_id'):
                init_info['payload_id'] = workflow.payload_id
            if hasattr(workflow, 'payload_weight'):
                init_info['payload_weight'] = workflow.payload_weight
            
            if hasattr(workflow, 'inspection_points'):
                init_info['inspection_points'] = workflow.inspection_points
                init_info['num_inspection_points'] = len(workflow.inspection_points)
            
            # Checkpoint信息（如果存在）
            if hasattr(workflow, 'checkpoints'):
                init_info['checkpoints'] = workflow.checkpoints
            if hasattr(workflow, 'current_checkpoint'):
                init_info['current_checkpoint'] = workflow.current_checkpoint
            
            self.workflow_init_info.append(init_info)
            self.workflows[workflow_id] = workflow
            
            logger.debug(f"记录workflow初始化信息: {workflow_id} ({init_info['workflow_type']})")
            
        except Exception as e:
            logger.error(f"处理workflow开始事件时出错: {e}")
    
    def register_agent(self, agent_id: str, agent):
        """注册agent以便收集其状态
        
        Args:
            agent_id: Agent ID
            agent: Agent对象
        """
        self.agents[agent_id] = agent
        logger.debug(f"注册agent: {agent_id}")
    
    def register_agents(self, agents: Dict):
        """批量注册agents
        
        Args:
            agents: agent_id -> agent_object的字典（配置文件中的ID -> agent对象）
        """
        # 重要：使用agent对象的ID（agent.id）作为key，因为事件中使用的是agent.id
        for config_id, agent in agents.items():
            self.agents[agent.id] = agent  # 使用agent.id，而不是config_id
            logger.debug(f"注册agent: config_id={config_id}, agent.id={agent.id}")
        logger.info(f"批量注册{len(agents)}个agents")
    
    def _extract_agent_state(self, agent_id: str, agent, current_time: float, event_type: str = None):
        """提取agent状态信息
        
        Args:
            agent_id: Agent ID
            agent: Agent对象
            current_time: 当前时间
            event_type: 事件类型（'task_started' 或 'task_completed'）
            
        Returns:
            状态记录字典
        """
        # 基本信息
        record = {
            'time': current_time,
            'agent_id': agent_id,
            'agent_type': getattr(agent, 'agent_type', 'unknown'),
            'event_type': event_type  # 记录是task_started还是task_completed
        }
        
        # 位置信息 - 使用get_state方法获取
        try:
            pos = agent.get_state('position')
            if pos and isinstance(pos, (tuple, list)) and len(pos) >= 3:
                record['x'] = pos[0]
                record['y'] = pos[1]
                record['z'] = pos[2]
            else:
                record['x'] = record['y'] = record['z'] = None
        except:
            record['x'] = record['y'] = record['z'] = None
        
        # 当前工作流信息
        current_workflow_id = None
        workflow_status = None
        checkpoint_info = None
        
        # 方法1：从scheduler获取此agent被分配的workflows
        if self.scheduler:
            for wf_id, wf_assign_info in self.scheduler.assigned_workflows.items():
                if wf_assign_info['agent_id'] == agent_id:
                    current_workflow_id = wf_id
                    break
        
        # 方法2：如果没有scheduler，直接从workflows字典中查找owner是此agent的workflow
        # 注意：agent对象本身的ID（agent.id）可能与我们记录的agent_id不同
        # 所以我们需要通过比较agent对象来查找
        # 关键：跳过已完成的workflows
        if not current_workflow_id and self.workflows and agent_id in self.agents:
            agent_obj = self.agents[agent_id]
            for wf_id, wf in self.workflows.items():
                # 跳过已完成的workflow
                if wf_id in self.completed_workflows:
                    continue
                if hasattr(wf, 'owner') and wf.owner and wf.owner == agent_obj:
                    current_workflow_id = wf_id
                    break
        
        # 如果找到workflow，获取详细信息
        if current_workflow_id and current_workflow_id in self.workflows:
            workflow = self.workflows[current_workflow_id]
            workflow_status = workflow.status.name if hasattr(workflow, 'status') else 'UNKNOWN'
            
            # 如果workflow刚刚完成，记录到completed_workflows并采集最后一次快照
            if workflow_status == 'COMPLETED' and current_workflow_id not in self.completed_workflows:
                self.completed_workflows.add(current_workflow_id)
                logger.info(f"时间 {current_time}: Workflow {current_workflow_id} 完成，标记为已完成，这是最后一次采集")
            # 如果workflow已经在completed集合中，返回None表示不采集
            elif current_workflow_id in self.completed_workflows:
                return None
            
            # 获取checkpoint信息
            checkpoint_info = self._extract_checkpoint_info(workflow)
        
        record['workflow_id'] = current_workflow_id
        record['workflow_status'] = workflow_status
        record['checkpoint_info'] = str(checkpoint_info) if checkpoint_info else None
        
        # 当前任务信息
        current_task = None
        task_type = None
        task_params = None
        
        # 方法1: 从managed_tasks获取正在执行的任务（最准确）
        # managed_tasks是字典: {task_id: {'task': Task对象, 'status': 'running', ...}}
        if hasattr(agent, 'managed_tasks') and agent.managed_tasks:
            # 遍历managed_tasks字典，获取正在运行的任务
            for task_id, task_info in agent.managed_tasks.items():
                if task_info.get('status') == 'running' and 'task' in task_info:
                    current_task = task_info['task']
                    # Task对象有task_class属性，直接使用
                    task_type = current_task.task_class if hasattr(current_task, 'task_class') else current_task.__class__.__name__
                    task_params = self._extract_task_params(current_task)
                    break  # 只取第一个正在运行的任务
        
        # 方法2: 从task_queue获取下一个待执行的任务
        # task_queue存储的是任务信息字典: {'task_class': class, 'task_name': str, 'properties': dict, ...}
        if not task_type and hasattr(agent, 'task_queue') and len(agent.task_queue) > 0:
            task_info = agent.task_queue[0]  # 这是一个字典
            if isinstance(task_info, dict):
                # 从task_class获取类名
                task_class = task_info.get('task_class')
                if task_class:
                    task_type = task_class.__name__ if hasattr(task_class, '__name__') else str(task_class)
                else:
                    task_type = task_info.get('task_name', 'Unknown')
                
                # 从字典中提取参数
                task_params = {
                    'task_name': task_info.get('task_name'),
                    'component': task_info.get('component_name'),
                    'target_state': task_info.get('target_state'),
                    'properties': task_info.get('properties')
                }
        
        record['task_type'] = task_type
        record['task_params'] = str(task_params) if task_params else None
        
        # Agent状态信息
        if hasattr(agent, 'state'):
            record['agent_state'] = agent.state.name if hasattr(agent.state, 'name') else str(agent.state)
        else:
            record['agent_state'] = None
        
        return record
    
    def _extract_checkpoint_info(self, workflow) -> Dict[str, Any]:
        """提取workflow的checkpoint信息
        
        Args:
            workflow: workflow对象
            
        Returns:
            checkpoint信息字典
        """
        checkpoint_info = None
        
        try:
            workflow_type = workflow.__class__.__name__
            
            # 1. 优先使用workflow自身的current_checkpoint属性
            if hasattr(workflow, 'current_checkpoint'):
                checkpoint_info = workflow.current_checkpoint
            
            # 2. 巡检workflow：使用completed_waypoints
            elif hasattr(workflow, 'completed_waypoints'):
                checkpoint_info = {
                    'completed': len(workflow.completed_waypoints),
                    'total': len(getattr(workflow, 'inspection_points', []))
                }
            
            # 3. 物流workflow：使用状态机状态和取货/交付标志
            elif workflow_type == 'LogisticsWorkflow' or hasattr(workflow, 'is_picked_up'):
                # 获取当前状态机状态
                current_state = workflow.status_machine.current_status if hasattr(workflow, 'status_machine') else 'unknown'
                
                # 定义物流workflow的checkpoint阶段
                logistics_checkpoints = ['picking_up', 'transporting', 'delivering', 'completed']
                
                checkpoint_info = {
                    'current_state': current_state,
                    'is_picked_up': getattr(workflow, 'is_picked_up', False),
                    'is_delivered': getattr(workflow, 'is_delivered', False),
                    'payload_id': getattr(workflow, 'payload_id', None)
                }
                
                # 计算进度（基于状态）
                if current_state in logistics_checkpoints:
                    checkpoint_info['progress'] = f"{logistics_checkpoints.index(current_state) + 1}/{len(logistics_checkpoints)}"
            
            # 4. 通信workflow：使用状态机状态
            elif workflow_type == 'CommunicationWorkflow' or hasattr(workflow, 'data_id'):
                current_state = workflow.status_machine.current_status if hasattr(workflow, 'status_machine') else 'unknown'
                
                # 定义通信workflow的checkpoint阶段
                comm_checkpoints = ['moving_to_relay_position', 'relaying_data', 'completed']
                
                checkpoint_info = {
                    'current_state': current_state,
                    'data_id': getattr(workflow, 'data_id', None),
                    'data_size_mb': getattr(workflow, 'data_size_mb', None),
                    'transmission_result': getattr(workflow, 'transmission_result', None)
                }
                
                # 计算进度
                if current_state in comm_checkpoints:
                    checkpoint_info['progress'] = f"{comm_checkpoints.index(current_state) + 1}/{len(comm_checkpoints)}"
            
            # 5. 计算workflow：使用状态机状态和计算状态
            elif workflow_type == 'ComputeWorkflow' or hasattr(workflow, 'job_id'):
                current_state = workflow.status_machine.current_status if hasattr(workflow, 'status_machine') else 'unknown'
                
                # 定义计算workflow的checkpoint阶段
                compute_checkpoints = ['moving_to_compute_location', 'computing', 'completed']
                
                checkpoint_info = {
                    'current_state': current_state,
                    'job_id': getattr(workflow, 'job_id', None),
                    'required_gflops': getattr(workflow, 'required_gflops', None),
                    'is_completed': getattr(workflow, 'is_completed', False),
                    'compute_result': getattr(workflow, 'compute_result', None)
                }
                
                # 计算进度
                if current_state in compute_checkpoints:
                    checkpoint_info['progress'] = f"{compute_checkpoints.index(current_state) + 1}/{len(compute_checkpoints)}"
                
        except Exception as e:
            logger.error(f"提取checkpoint信息时出错: {e}")
            checkpoint_info = None
        
        return checkpoint_info
    
    def _extract_task_params(self, task) -> Dict[str, Any]:
        """提取任务的参数信息
        
        Args:
            task: 任务对象（airfogsim.core.task.Task）
            
        Returns:
            任务参数字典
        """
        params = {}
        
        try:
            # 基础属性（Task类的标准属性）
            if hasattr(task, 'id'):
                params['task_id'] = task.id
            if hasattr(task, 'name'):
                params['task_name'] = task.name
            if hasattr(task, 'component_name'):
                params['component'] = task.component_name
            if hasattr(task, 'workflow_id'):
                params['workflow_id'] = task.workflow_id
            
            # 目标状态（Task.target_state）
            if hasattr(task, 'target_state') and task.target_state:
                # target_state是字典，包含任务的目标
                if isinstance(task.target_state, dict):
                    for key, value in task.target_state.items():
                        # 避免过长的值
                        if isinstance(value, (list, tuple)) and len(value) <= 3:
                            params[f'target_{key}'] = value
                        elif isinstance(value, (int, float, str, bool)):
                            params[f'target_{key}'] = value
            
            # 任务属性（Task.properties）
            if hasattr(task, 'properties') and task.properties:
                # properties是字典，包含任务的各种参数
                if isinstance(task.properties, dict):
                    # 只提取关键参数，避免过长
                    key_params = ['priority', 'preemptive', 'duration', 'distance', 
                                  'payload_id', 'inspection_type', 'data_type']
                    for key in key_params:
                        if key in task.properties:
                            params[key] = task.properties[key]
            
            # 特定任务类型的特殊属性
            # MoveToTask
            if hasattr(task, 'target_position'):
                params['target_position'] = task.target_position
            if hasattr(task, 'target_location'):
                params['target_location'] = task.target_location
            
            # PickupTask / DeliveryTask
            if hasattr(task, 'payload_id'):
                params['payload_id'] = task.payload_id
            if hasattr(task, 'pickup_location'):
                params['pickup_location'] = task.pickup_location
            if hasattr(task, 'delivery_location'):
                params['delivery_location'] = task.delivery_location
            
            # InspectionTask
            if hasattr(task, 'inspection_point'):
                params['inspection_point'] = task.inspection_point
            if hasattr(task, 'inspection_type'):
                params['inspection_type'] = task.inspection_type
            
        except Exception as e:
            logger.error(f"提取任务参数时出错: {e}")
        
        return params
    
    def export_snapshot_data(self, output_path: str):
        """导出快照数据为CSV文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            if not self.snapshots:
                logger.warning("没有快照数据可导出")
                return
            
            df = pd.DataFrame(self.snapshots)
            df.to_csv(output_path, index=False)
            
            logger.info(f"快照数据已导出到: {output_path}")
            logger.info(f"总计导出 {len(self.snapshots)} 条快照记录")
            
        except Exception as e:
            logger.error(f"导出快照数据时出错: {e}")
    
    def export_workflow_init_data(self, output_path: str):
        """导出workflow初始化数据为CSV文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            if not self.workflow_init_info:
                logger.warning("没有workflow初始化数据可导出")
                return
            
            # 转换为DataFrame
            records = []
            for init_info in self.workflow_init_info:
                # 将嵌套的dict/list转换为字符串
                record = {}
                for key, value in init_info.items():
                    if isinstance(value, (dict, list)):
                        record[key] = str(value)
                    else:
                        record[key] = value
                records.append(record)
            
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Workflow初始化数据已导出到: {output_path}")
            logger.info(f"总计导出 {len(self.workflow_init_info)} 个workflow的初始化信息")
            
        except Exception as e:
            logger.error(f"导出workflow初始化数据时出错: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取收集统计摘要
        
        Returns:
            统计摘要字典
        """
        return {
            'total_snapshots': len(self.snapshots),
            'num_agents': len(self.agents),
            'num_workflows': len(self.workflow_init_info),
            'simulation_time': self.env.now,
            'snapshot_interval': getattr(self, 'snapshot_interval', 'event-driven')
        }
    
    def print_summary(self):
        """打印收集统计摘要"""
        summary = self.get_summary()
        
        logger.info("\n" + "="*60)
        logger.info("快照数据收集统计")
        logger.info("="*60)
        logger.info(f"总快照数: {summary['total_snapshots']}")
        logger.info(f"Agent数量: {summary['num_agents']}")
        logger.info(f"Workflow数量: {summary['num_workflows']}")
        logger.info(f"仿真时长: {summary['simulation_time']:.2f} 秒")
        logger.info(f"快照间隔: {summary['snapshot_interval']}")
        logger.info("="*60)
