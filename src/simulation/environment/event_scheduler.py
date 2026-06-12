"""
事件调度管理器

统一管理所有突发事件的调度，包括：
1. 基于配置文件的事件调度
2. 通过airspace_manager获取agent位置
3. 动态修改agent状态
4. 事件链和复合事件支持

@author: TCCN Auto Planner
"""

import simpy
from typing import Dict, List, Any, Optional, Callable
from airfogsim.utils.logging_config import get_logger

logger = get_logger(__name__)


class EventScheduler:
    """
    事件调度管理器
    
    负责统一调度和管理所有突发事件，包括天气事件、UFO事件和自定义事件。
    支持基于时间、条件和位置的事件触发机制。
    """
    
    def __init__(self, env, config: Dict[str, Any] = None):
        """
        初始化事件调度管理器
        
        Args:
            env: AirFogSim仿真环境
            config: 事件调度配置字典
        """
        self.env = env
        self.config = config or {}
        
        # 事件队列和处理器
        self.scheduled_events = []
        self.event_handlers: Dict[str, Callable] = {}
        self.active_events: Dict[str, Dict] = {}
        
        # 事件监控配置
        self.monitoring_config = self.config.get('monitoring', {})
        self.log_interval = self.monitoring_config.get('log_interval', 30.0)
        
        # 注册默认事件处理器
        self._register_default_handlers()
        
        # 启动事件调度进程
        self.scheduler_process = self.env.process(self._event_scheduler_loop())
        if self.monitoring_config.get('enable_agent_state_logging', False):
            self.monitoring_process = self.env.process(self._monitoring_loop())
        
        logger.info(f"时间 {self.env.now}: 事件调度管理器已初始化")
    
    def _register_default_handlers(self):
        """注册默认的事件处理器"""
        self.event_handlers.update({
            'agent_state_change': self._handle_agent_state_change,
            'position_based_effect': self._handle_position_based_effect,
            'area_effect': self._handle_area_effect,
            'conditional_event': self._handle_conditional_event,
            'chain_event': self._handle_chain_event
        })
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """
        注册自定义事件处理器
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        self.event_handlers[event_type] = handler
        logger.info(f"注册事件处理器: {event_type}")
    
    def schedule_event(self, trigger_time: float, event_type: str, 
                      event_params: Dict[str, Any], event_id: str = None):
        """
        调度事件
        
        Args:
            trigger_time: 触发时间
            event_type: 事件类型
            event_params: 事件参数
            event_id: 事件ID（可选）
        """
        if event_id is None:
            event_id = f"event_{len(self.scheduled_events)}"
        
        event = {
            'id': event_id,
            'trigger_time': trigger_time,
            'type': event_type,
            'parameters': event_params,
            'status': 'scheduled'
        }
        
        self.scheduled_events.append(event)
        self.scheduled_events.sort(key=lambda x: x['trigger_time'])
        
        logger.info(f"调度事件 {event_id} (类型: {event_type}) 在时间 {trigger_time}")
    
    def _event_scheduler_loop(self):
        """事件调度主循环"""
        try:
            while self.scheduled_events:
                # 获取下一个要执行的事件
                next_event = self.scheduled_events[0]
                
                # 等待到事件触发时间
                if next_event['trigger_time'] > self.env.now:
                    yield self.env.timeout(next_event['trigger_time'] - self.env.now)
                
                # 移除已处理的事件
                event = self.scheduled_events.pop(0)
                
                # 执行事件
                self._execute_event(event)
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: 事件调度循环被中断")
    
    def _monitoring_loop(self):
        """监控循环，定期记录agent状态"""
        try:
            while True:
                yield self.env.timeout(self.log_interval)
                self._log_agent_states()
                
        except simpy.Interrupt:
            logger.info(f"时间 {self.env.now}: 监控循环被中断")
    
    def _execute_event(self, event: Dict[str, Any]):
        """
        执行事件
        
        Args:
            event: 事件字典
        """
        event_type = event['type']
        event_params = event['parameters']
        event_id = event['id']
        
        logger.info(f"时间 {self.env.now}: 执行事件 {event_id} (类型: {event_type})")
        
        try:
            # 查找事件处理器
            if event_type in self.event_handlers:
                handler = self.event_handlers[event_type]
                handler(event_params)
                event['status'] = 'completed'
            else:
                logger.warning(f"未找到事件类型 {event_type} 的处理器")
                event['status'] = 'failed'
            
            # 记录活跃事件
            self.active_events[event_id] = event
            
        except Exception as e:
            logger.error(f"执行事件 {event_id} 时出错: {e}")
            event['status'] = 'error'
    
    def _handle_agent_state_change(self, params: Dict[str, Any]):
        """
        处理agent状态变化事件
        
        Args:
            params: 事件参数
        """
        target_agents = params.get('target_agents', 'all')
        state_changes = params.get('state_changes', {})
        
        if target_agents == 'all':
            agents = self.env.get_all_agents()
        elif isinstance(target_agents, list):
            agents = [self.env.get_agent(agent_id) for agent_id in target_agents]
            agents = [agent for agent in agents if agent is not None]
        else:
            logger.warning(f"无效的目标agent配置: {target_agents}")
            return
        
        for agent in agents:
            for state_key, state_value in state_changes.items():
                old_value = agent.get_state(state_key)
                agent.update_state(state_key, state_value)
                logger.info(f"Agent {agent.id} 状态变化: {state_key} {old_value} -> {state_value}")
    
    def _handle_position_based_effect(self, params: Dict[str, Any]):
        """
        处理基于位置的效果事件
        
        Args:
            params: 事件参数
        """
        center_position = params.get('center_position', [0, 0, 0])
        effect_radius = params.get('effect_radius', 100.0)
        effects = params.get('effects', {})
        
        # 通过airspace_manager获取所有agent位置
        agent_positions = self.env.airspace_manager.agent_positions
        
        for agent_id, position in agent_positions.items():
            # 计算距离
            distance = self._calculate_distance(position, center_position)
            
            if distance <= effect_radius:
                agent = self.env.get_agent(agent_id)
                if agent:
                    # 根据距离计算效果强度
                    intensity = 1.0 - (distance / effect_radius)
                    
                    for effect_key, effect_value in effects.items():
                        if isinstance(effect_value, (int, float)):
                            final_value = effect_value * intensity
                        else:
                            final_value = effect_value
                        
                        agent.update_state(effect_key, final_value)
                        logger.info(f"Agent {agent_id} 受到位置效果影响: {effect_key} = {final_value}")
    
    def _handle_area_effect(self, params: Dict[str, Any]):
        """
        处理区域效果事件
        
        Args:
            params: 事件参数
        """
        area_bounds = params.get('area_bounds', {})
        effects = params.get('effects', {})
        
        x_min = area_bounds.get('x_min', 0)
        x_max = area_bounds.get('x_max', 1000)
        y_min = area_bounds.get('y_min', 0)
        y_max = area_bounds.get('y_max', 1000)
        z_min = area_bounds.get('z_min', 0)
        z_max = area_bounds.get('z_max', 500)
        
        # 通过airspace_manager获取所有agent位置
        agent_positions = self.env.airspace_manager.agent_positions
        
        for agent_id, position in agent_positions.items():
            x, y, z = position
            
            # 检查是否在区域内
            if (x_min <= x <= x_max and 
                y_min <= y <= y_max and 
                z_min <= z <= z_max):
                
                agent = self.env.get_agent(agent_id)
                if agent:
                    for effect_key, effect_value in effects.items():
                        agent.update_state(effect_key, effect_value)
                        logger.info(f"Agent {agent_id} 受到区域效果影响: {effect_key} = {effect_value}")
    
    def _handle_conditional_event(self, params: Dict[str, Any]):
        """
        处理条件事件
        
        Args:
            params: 事件参数
        """
        conditions = params.get('conditions', [])
        actions = params.get('actions', [])
        
        # 检查条件
        conditions_met = True
        for condition in conditions:
            if not self._evaluate_condition(condition):
                conditions_met = False
                break
        
        # 如果条件满足，执行动作
        if conditions_met:
            for action in actions:
                self._execute_action(action)
    
    def _handle_chain_event(self, params: Dict[str, Any]):
        """
        处理链式事件
        
        Args:
            params: 事件参数
        """
        chain_events = params.get('chain_events', [])
        
        for i, chain_event in enumerate(chain_events):
            delay = chain_event.get('delay', 0)
            event_type = chain_event.get('type')
            event_params = chain_event.get('parameters', {})
            
            # 调度链式事件
            trigger_time = self.env.now + delay
            event_id = f"chain_{i}_{self.env.now}"
            self.schedule_event(trigger_time, event_type, event_params, event_id)
    
    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """计算两点间距离"""
        return ((pos1[0] - pos2[0])**2 + 
                (pos1[1] - pos2[1])**2 + 
                (pos1[2] - pos2[2])**2)**0.5
    
    def _evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """
        评估条件
        
        Args:
            condition: 条件字典
            
        Returns:
            条件是否满足
        """
        # 简单的条件评估实现
        condition_type = condition.get('type', 'agent_state')
        
        if condition_type == 'agent_state':
            agent_id = condition.get('agent_id')
            state_key = condition.get('state_key')
            operator = condition.get('operator', '==')
            target_value = condition.get('target_value')
            
            agent = self.env.get_agent(agent_id)
            if not agent:
                return False
            
            current_value = agent.get_state(state_key)
            
            if operator == '==':
                return current_value == target_value
            elif operator == '>':
                return current_value > target_value
            elif operator == '<':
                return current_value < target_value
            elif operator == '>=':
                return current_value >= target_value
            elif operator == '<=':
                return current_value <= target_value
        
        return False
    
    def _execute_action(self, action: Dict[str, Any]):
        """
        执行动作
        
        Args:
            action: 动作字典
        """
        action_type = action.get('type', 'state_change')
        
        if action_type == 'state_change':
            self._handle_agent_state_change(action)
        elif action_type == 'position_effect':
            self._handle_position_based_effect(action)
        elif action_type == 'area_effect':
            self._handle_area_effect(action)
    
    def _log_agent_states(self):
        """记录所有agent的状态"""
        agents = self.env.get_all_agents()
        
        for agent in agents:
            position = agent.get_state('position', [0, 0, 0])
            battery = agent.get_state('battery_level', 100.0)
            status = agent.get_state('status', 'unknown')
            
            logger.info(f"时间 {self.env.now}: Agent {agent.id} - "
                       f"位置: {position}, 电量: {battery:.1f}%, 状态: {status}")
    
    def get_active_events(self) -> Dict[str, Dict]:
        """获取所有活跃事件"""
        return self.active_events.copy()
    
    def get_scheduled_events_count(self) -> int:
        """获取待调度事件数量"""
        return len(self.scheduled_events)
