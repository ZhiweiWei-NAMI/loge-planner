"""
Workflow调度策略

轻量级调度器，负责agent匹配和选择逻辑。
Workflow管理由env.workflow_manager负责，避免重复实现。

- Greedy: 贪心分配给最近的可用agent
- Random: 随机分配给可用agent
- Nearest: 分配给最近的agent

@author: TCCN Auto Planner
"""

import random
import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WorkflowScheduler:
    """轻量级Workflow调度器
    
    只负责agent匹配和选择逻辑，workflow管理由env.workflow_manager处理
    """
    
    def __init__(self, env, strategy: str = "greedy"):
        """初始化调度器
        
        Args:
            env: AirFogSim环境对象
            strategy: 调度策略 ("greedy", "random", "nearest")
        """
        self.env = env
        self.strategy = strategy
        
        # Agent能力映射
        self.agents = {}  # agent_id -> agent_object
        self.agent_capabilities = {}  # agent_id -> [capability_list]
        
        # 统计信息
        self.success_count = 0
        self.failed_count = 0
        
        logger.info(f"Workflow调度器初始化，策略: {strategy}")
    
    def register_agent(self, agent_id: str, agent, capabilities: List[str]):
        """注册agent
        
        Args:
            agent_id: Agent ID
            agent: Agent对象
            capabilities: Agent能力列表
        """
        self.agents[agent_id] = agent
        self.agent_capabilities[agent_id] = capabilities
        logger.debug(f"注册agent: {agent_id}, 能力: {capabilities}")
    
    def register_agents(self, agents: Dict, capabilities_map: Dict[str, List[str]]):
        """批量注册agents
        
        Args:
            agents: agent_id -> agent_object的字典
            capabilities_map: agent_id -> capabilities的字典
        """
        for agent_id, agent in agents.items():
            caps = capabilities_map[agent_id]
            self.register_agent(agent_id, agent, caps)
        
        logger.info(f"批量注册{len(agents)}个agents")
    
    def select_agent_for_workflow(self, workflow_type: str, workflow_info: Dict) -> Optional[str]:
        """为workflow选择合适的agent（只负责选择逻辑，不创建workflow）
        
        Args:
            workflow_type: workflow类型
            workflow_info: workflow信息字典
            
        Returns:
            被选中的agent_id，失败返回None
        """
        # 筛选有能力的agents
        capable_agents = self._filter_capable_agents(workflow_type)
        
        if not capable_agents:
            logger.warning(f"没有agent能够处理 {workflow_type} 类型的workflow")
            self.failed_count += 1
            return None
        
        # 根据策略选择agent
        selected_agent_id = self._select_agent(capable_agents, workflow_info)
        
        if selected_agent_id:
            self.success_count += 1
            logger.debug(f"为workflow类型 {workflow_type} 选择了agent {selected_agent_id}")
            return selected_agent_id
        else:
            self.failed_count += 1
            return None
    
    @property
    def assigned_workflows(self) -> Dict:
        """获取已分配的workflows（从workflow_manager获取）
        
        Returns:
            workflow_id -> workflow_info的字典
        """
        # 利用env.workflow_manager获取所有workflows
        workflows_dict = {}
        if hasattr(self.env, 'workflow_manager'):
            for wf_id, wf in self.env.workflow_manager.workflows.items():
                workflows_dict[wf_id] = {
                    'workflow_object': wf,
                    'agent_id': wf.owner.id if hasattr(wf, 'owner') and wf.owner else None
                }
        return workflows_dict
    
    def _filter_capable_agents(self, workflow_type: str) -> List[str]:
        """筛选有能力处理特定workflow类型的agents
        
        Args:
            workflow_type: workflow类型
            
        Returns:
            有能力的agent_id列表
        """
        capable = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            # 检查agent是否具有相应能力
            if workflow_type.lower() in [cap.lower() for cap in capabilities]:
                capable.append(agent_id)
        
        return capable
    
    def _select_agent(self, capable_agents: List[str], workflow_info: Dict) -> Optional[str]:
        """根据策略选择agent
        
        Args:
            capable_agents: 有能力的agent_id列表
            workflow_info: workflow信息
            
        Returns:
            选中的agent_id
        """
        if not capable_agents:
            return None
        
        if self.strategy == "random":
            return self._random_selection(capable_agents)
        elif self.strategy == "greedy":
            return self._greedy_selection(capable_agents, workflow_info)
        elif self.strategy == "nearest":
            return self._nearest_selection(capable_agents, workflow_info)
        else:
            logger.warning(f"未知的调度策略: {self.strategy}，使用random")
            return self._random_selection(capable_agents)
    
    def _random_selection(self, capable_agents: List[str]) -> str:
        """随机选择agent"""
        return random.choice(capable_agents)
    
    def _greedy_selection(self, capable_agents: List[str], workflow_info: Dict) -> Optional[str]:
        """贪心选择：选择最近的空闲agent"""
        # 简化实现：直接选择最近的
        return self._nearest_selection(capable_agents, workflow_info)
    
    def _nearest_selection(self, capable_agents: List[str], workflow_info: Dict) -> Optional[str]:
        """选择最近的agent"""
        workflow_type = workflow_info['workflow_type']
        
        # 确定workflow的初始位置
        target_position = None
        wf_type_lower = workflow_type.lower()
        if 'logistics' in wf_type_lower and 'pickup_location' in workflow_info:
            target_position = workflow_info['pickup_location']
        elif 'inspection' in wf_type_lower and 'inspection_points' in workflow_info:
            points = workflow_info['inspection_points']
            if points:
                target_position = points[0]
        
        if not target_position:
            return random.choice(capable_agents)
        
        # 计算距离，选择最近的
        min_distance = float('inf')
        nearest_agent = None
        
        for agent_id in capable_agents:
            agent = self.agents[agent_id]
            agent_pos = agent.get_state('position')
            if agent_pos:
                distance = self._calculate_distance(agent_pos, target_position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_agent = agent_id
        
        return nearest_agent if nearest_agent else random.choice(capable_agents)
    
    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """计算两点之间的欧氏距离"""
        x1, y1, z1 = pos1[:3] if len(pos1) >= 3 else (pos1[0], pos1[1], 0)
        x2, y2, z2 = pos2[:3] if len(pos2) >= 3 else (pos2[0], pos2[1], 0)
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    def get_workflow_objects(self) -> Dict:
        """获取所有workflow对象（从workflow_manager获取）
        
        Returns:
            workflow_id -> workflow_object的字典
        """
        if hasattr(self.env, 'workflow_manager'):
            return self.env.workflow_manager.workflows
        return {}
    
    def get_statistics(self) -> Dict:
        """获取调度统计"""
        total_workflows = len(self.env.workflow_manager.workflows) if hasattr(self.env, 'workflow_manager') else 0
        return {
            'strategy': self.strategy,
            'total_agents': len(self.agents),
            'success_assignments': self.success_count,
            'failed_assignments': self.failed_count,
            'total_workflows': total_workflows,
            'simulation_time': self.env.now
        }
    
    def print_statistics(self):
        """打印调度统计"""
        stats = self.get_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("Workflow调度统计")
        logger.info("="*60)
        logger.info(f"调度策略: {stats['strategy']}")
        logger.info(f"Agent总数: {stats['total_agents']}")
        logger.info(f"成功分配: {stats['success_assignments']}")
        logger.info(f"失败分配: {stats['failed_assignments']}")
        logger.info(f"总Workflows: {stats['total_workflows']}")
        logger.info(f"仿真时长: {stats['simulation_time']:.2f} 秒")
        logger.info("="*60)
