from airfogsim.agent.drone import DroneAgent as BaseDroneAgent
from airfogsim.utils.logging_config import get_logger

logger = get_logger(__name__)


class DroneAgent(BaseDroneAgent):
    """TCCN 扩展的 DroneAgent
    
    支持两种模式：
    1. offline 模式：任务由 OfflineTaskLoader 预先加载到队列，不依赖 workflow 生成任务
    2. dynamic 模式：任务由 workflow 的 get_current_suggested_task() 动态生成
    """
    
    def _process_custom_logic(self):
        """执行无人机特定的逻辑
        
        根据 use_offline_plan 属性区分模式：
        - offline 模式：只执行预加载的离线任务，不生成新任务
        - dynamic 模式：使用 workflow 任务生成路径
        """
        # 检查是否是 offline 模式
        use_offline_plan = self.properties.get('use_offline_plan', False)
        
        if use_offline_plan:
            # Offline 模式：任务已由 OfflineTaskLoader 预加载
            # 只需要更新状态，不生成新任务
            active_workflows = self.get_active_workflows()
            if not active_workflows and not self.task_queue and not self.managed_tasks:
                self.update_state('moving_status', 'idle')
                self.update_state('status', 'idle')
            return
        
        # Dynamic 模式：调用基类的 _process_custom_logic
        # 基类会处理 task_planner_callback 和充电工作流
        super()._process_custom_logic()
        
        # 如果没有 task_planner_callback，使用 workflow 任务生成路径
        if not self.task_planner_callback:
            self._process_workflow_suggested_tasks()
    
    def _process_workflow_suggested_tasks(self):
        """从活跃的 workflow 获取建议任务并加入队列
        
        这是 dynamic 模式下的默认机制，确保 workflow 的任务能被执行
        """
        active_workflows = self.get_active_workflows()
        if not active_workflows:
            return
        
        for workflow in active_workflows:
            # 检查 workflow 是否有 get_current_suggested_task 方法
            if not hasattr(workflow, 'get_current_suggested_task'):
                continue
            
            try:
                suggested_task = workflow.get_current_suggested_task()
                if not suggested_task:
                    continue
                
                # suggested_task 是一个字典，包含任务信息
                # 检查是否已经在队列中（避免重复添加）
                task_name = suggested_task.get('task_name', '')
                workflow_id = suggested_task.get('workflow_id', '')
                
                # 检查是否已有相同的任务在队列或执行中
                is_duplicate = False
                for queued in self.task_queue:
                    if (queued.get('task_name') == task_name and 
                        queued.get('workflow_id') == workflow_id):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    for task_id, task_info in self.managed_tasks.items():
                        task = task_info.get('task')
                        if task and task.name == task_name and task.workflow_id == workflow_id:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    # 添加到任务队列
                    # 注意：workflow 返回的是 'component' 键，需要兼容
                    component_name = suggested_task.get('component_name') or suggested_task.get('component')
                    
                    # 处理 priority：确保是字符串或 None
                    priority = suggested_task.get('priority')
                    if isinstance(priority, int):
                        priority = None  # 使用默认值
                    
                    self.add_task_to_queue(
                        component_name=component_name,
                        task_name=suggested_task.get('task_name'),
                        task_class=suggested_task.get('task_class'),
                        priority=priority,
                        preemptive=suggested_task.get('preemptive', False),
                        target_state=suggested_task.get('target_state'),
                        properties=suggested_task.get('properties'),
                        workflow_id=suggested_task.get('workflow_id'),
                    )
                    logger.debug(f"时间 {self.env.now}: Agent {self.id} 从 workflow {workflow_id} 添加任务 {task_name}")
                    
            except Exception as e:
                logger.error(f"时间 {self.env.now}: Agent {self.id} 处理 workflow 建议任务失败: {e}")

DroneAgent.register_state_template('current_payload_weight', value_type=(int, float), required=False)
DroneAgent.register_state_template('max_payload_weight', value_type=(int, float), required=False)
DroneAgent.register_state_template('attached_items', value_type=list, required=False)
DroneAgent.register_state_template('wind_vector', value_type=(tuple, list), required=False)
DroneAgent.register_state_template('gps_signal_strength', value_type=(int, float), required=False)
DroneAgent.register_state_template('visibility', value_type=(int, float), required=False)
DroneAgent.register_state_template('precipitation_level', value_type=(int, float), required=False)
DroneAgent.register_state_template('ambient_temperature', value_type=(int, float), required=False)
DroneAgent.register_state_template('transmission_power', value_type=(int, float), required=False)
DroneAgent.register_state_template('interference_signal_strength', value_type=(int, float), required=False)
DroneAgent.register_state_template('cpu_utilization', value_type=(int, float), required=False)
DroneAgent.register_state_template('sensor_status', value_type=dict, required=False)
DroneAgent.register_state_template(
    'orientation',
    value_type=tuple,
    required=False,
    validator=lambda v: len(v) == 3 and all(isinstance(x, (int, float)) for x in v),
)
DroneAgent.register_state_template('payload_ids', value_type=list, required=False)
DroneAgent.register_state_template('delivery_status', value_type=str, required=False)
DroneAgent.register_state_template('data_collection_rate', value_type=(int, float), required=False)
DroneAgent.register_state_template('communication_power_consumption', value_type=(int, float), required=False)
DroneAgent.register_state_template(
    'velocity',
    value_type=(tuple, list),
    required=False,
    validator=lambda v: len(v) == 3 and all(isinstance(x, (int, float)) for x in v),
)

__all__ = ["DroneAgent"]
