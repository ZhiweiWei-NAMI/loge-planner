"""对象工厂模块

提供从配置文件创建智能体和工作流的统一API。

@author: TCCN Auto Planner
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import random

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../..')
sys.path.append(project_root)

from airfogsim.core.environment import Environment
from src.simulation.core.agent import DroneAgent
from src.simulation.workflows.logistics import LogisticsWorkflow
from src.simulation.workflows.inspection import InspectionWorkflow
from src.simulation.workflows.communication import CommunicationWorkflow
from src.simulation.workflows.computation import ComputeWorkflow
from .config_loader import ConfigLoader

import logging

logger = logging.getLogger(__name__)

def create_agents_from_config(env: Environment, config: Dict[str, Any], online_model_config: Dict[str, Any]=None, planning_mode: str='hierarchical') -> Dict[str, DroneAgent]:
    """从配置创建智能体

    Args:
        env: 仿真环境
        config: 完整配置字典
        online_model_config: 在线模型配置；公开版本默认禁用
        planning_mode: 规划模式

    Returns:
        智能体字典，键为智能体ID，值为智能体对象
    """
    config_loader = ConfigLoader()
    agents_config = config_loader.get_agents_config(config)
    agents = {}

    logger.info(f"开始创建 {len(agents_config)} 个智能体 (规划模式: {planning_mode})")

    for agent_config in agents_config:
        # 验证配置
        if not config_loader.validate_agent_config(agent_config):
            logger.error(f"智能体配置验证失败: {agent_config.get('id', 'unknown')}")
            continue

        try:
            agent = _create_single_agent(env, agent_config, online_model_config, planning_mode)
            agents[agent_config['id']] = agent
            logger.info(f"成功创建智能体: {agent_config['id']}")

        except Exception as e:
            logger.error(f"创建智能体失败 {agent_config.get('id', 'unknown')}: {e}")
            continue

    logger.info(f"成功创建了 {len(agents)} 个智能体")
    return agents

def _create_single_agent(env: Environment, agent_config: Dict[str, Any], online_model_config: Dict[str, Any]=None, planning_mode: str='hierarchical') -> DroneAgent:
    """创建单个智能体

    Args:
        env: 仿真环境
        agent_config: 智能体配置
        online_model_config: 在线模型配置；公开版本默认禁用
        planning_mode: 规划模式

    Returns:
        智能体对象
    """
    agent_id = agent_config['id']
    agent_type = agent_config['type']
    properties = agent_config['properties']
    position = agent_config['initial_position']
    enable_online_model = online_model_config.get('enable_online_model', False) if online_model_config else False
    position_tuple = (position['x'], position['y'], position['z'])

    # 离线模式：不需要在线模型客户端
    # 所有任务通过OfflineTaskLoader在仿真开始前加载


    # 创建智能体 - 统一使用TCCN的DroneAgent
    if agent_type == 'logistics_drone':
        # 为物流无人机添加物流组件配置
        if 'components' not in properties:
            properties['components'] = ['logistics']
        elif 'logistics' not in properties['components']:
            properties['components'].append('logistics')

    # 从properties中提取components配置
    components_config = properties.get('components', []) or []

    # 如果规划阶段没有针对UAV能力做差异化处理，
    # 在离线仿真模式下默认假设所有UAV同质且具备全能力组件
    if planning_mode == 'offline':
        default_components = ['mobility', 'compute', 'communication', 'inspection', 'logistics']
        existing_lower = {c.lower() for c in components_config}
        for cname in default_components:
            if cname not in existing_lower:
                components_config.append(cname)
    component_list = []
    
    # 将字符串形式的组件名转换为组件类（不传name，组件内部已硬编码为 'Mobility' 等）
    component_mapping = {
        'mobility': {'class': env.component_manager.get_component_class('MobilityComponent')},
        'logistics': {'class': env.component_manager.get_component_class('LogisticsComponent')},
        'inspection': {'class': env.component_manager.get_component_class('InspectionComponent')},
        'communication': {'class': env.component_manager.get_component_class('CommunicationComponent')},
        'compute': {'class': env.component_manager.get_component_class('ComputeComponent')}
    }
    
    # 所有drone都需要Mobility组件
    if 'mobility' not in [c.lower() for c in components_config]:
        components_config = ['mobility'] + components_config
    
    for comp_name in components_config:
        if comp_name.lower() in component_mapping:
            component_list.append(component_mapping[comp_name.lower()])
    
    logger.info(f"Agent {agent_id} 配置组件: {[c.lower() for c in components_config]}")
    
    # 统一使用TCCN的DroneAgent，它已包含完整的状态模板
    # 注意：这里不再通过 Environment.create_agent 的 components 参数自动附加组件，
    # 而是手动实例化并调用 agent.add_component，确保组件名称与任务中的 component 字符串一致
    agent = env.create_agent(
        DroneAgent,
        agent_name=agent_id,
        properties=properties
    )
    
    # Override agent.id to use config agent_id for consistent ID across PDDL plans and simulation
    # This ensures OfflineTaskLoader can map tasks correctly using config agent IDs
    agent.id = agent_id

    # 手动为agent附加组件
    for comp_cfg in component_list:
        comp_cls = comp_cfg.get('class')
        if not comp_cls:
            continue
        # TCCN 组件内部使用 super().__init__(agent, 'Mobility'/...) 设置组件名称
        component = comp_cls(agent)
        agent.add_component(component)

    if enable_online_model:
        raise RuntimeError(
            "enable_online_model=True requires an explicit online terminal monitor component. "
            "The public simulator path is offline-only."
        )

    # 设置初始位置
    agent.set_state('position', position_tuple)

    return agent

def create_workflows_from_config(env: Environment, agents: Dict[str, DroneAgent],
                               config: Dict[str, Any]) -> Dict[str, Union[LogisticsWorkflow, InspectionWorkflow, CommunicationWorkflow, ComputeWorkflow]]:
    """从配置创建工作流
    
    Args:
        env: 仿真环境
        agents: 智能体字典
        config: 完整配置字典
        
    Returns:
        工作流字典，键为工作流ID，值为工作流对象
    """
    config_loader = ConfigLoader()
    workflows_config = config_loader.get_workflows_config(config)
    workflows = {}
    
    logger.info(f"开始创建 {len(workflows_config)} 个工作流")
    logger.debug(f"工作流配置: {workflows_config}")
    
    for workflow_config in workflows_config:
        logger.debug(f"处理工作流配置: {workflow_config.get('id', 'unknown')}")
        # 验证配置
        if not config_loader.validate_workflow_config(workflow_config):
            logger.error(f"工作流配置验证失败: {workflow_config.get('id', 'unknown')}")
            continue
            
        try:
            workflow = _create_single_workflow(env, agents, workflow_config)
            workflows[workflow_config['id']] = workflow
            logger.info(f"成功创建工作流: {workflow_config['id']}")
            
        except Exception as e:
            logger.error(f"创建工作流失败 {workflow_config.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"成功创建了 {len(workflows)} 个工作流")
    return workflows

def _create_single_workflow(env: Environment, agents: Dict[str, DroneAgent],
                          workflow_config: Dict[str, Any]) -> Union[LogisticsWorkflow, InspectionWorkflow, CommunicationWorkflow, ComputeWorkflow]:
    """创建单个工作流
    
    Args:
        env: 仿真环境
        agents: 智能体字典
        workflow_config: 工作流配置
        
    Returns:
        工作流对象
    """
    workflow_id = workflow_config['id']
    workflow_name = workflow_config['name']
    workflow_type = workflow_config['type']
    properties = workflow_config['properties']
    start_delay = workflow_config.get('start_delay', 0)  # 获取start_delay配置，默认为0
    
    # 优先使用配置中指定的agent_id，如果没有则随机选择
    if 'agent_id' in workflow_config and workflow_config['agent_id'] in agents:
        agent_id = workflow_config['agent_id']
        logger.debug(f"使用配置指定的agent: {agent_id}")
    else:
        # 随机从agents中选择一个作为owner
        agent_id = random.choice(list(agents.keys()))
        logger.debug(f"随机选择agent: {agent_id}")
    
    agent = agents[agent_id]
    
    # 根据start_delay创建触发器
    from airfogsim.core.trigger import TimeTrigger
    start_trigger = TimeTrigger(env, trigger_time=start_delay, name=f"start_trigger_{workflow_id}")
    logger.debug(f"创建TimeTrigger: {start_trigger.name}, trigger_time={start_delay}")
    
    # 根据工作流类型创建不同的工作流
    if workflow_type == 'logistics':
        # 转换位置格式
        pickup_location = properties['pickup_location']
        delivery_location = properties['delivery_location']
        
        pickup_tuple = (pickup_location['x'], pickup_location['y'], pickup_location['z'])
        delivery_tuple = (delivery_location['x'], delivery_location['y'], delivery_location['z'])
        
        # 构建工作流属性
        workflow_properties = {
            'pickup_location': pickup_tuple,
            'delivery_location': delivery_tuple,
            'payload_id': properties['payload_id'],
            'payload_weight': properties['payload_weight'],
            'source_agent_id': properties.get('source_agent_id', agent_id),
            'target_agent_id': properties.get('target_agent_id', 'unknown'),
            'payloads': properties.get('payloads', []),
            'start_delay': start_delay  # 保存start_delay到workflow属性中
        }
        
        workflow = LogisticsWorkflow(
            env, workflow_id, agent,
            properties=workflow_properties
        )
        workflow.id = workflow_id  # Override auto-generated UUID
        
    elif workflow_type == 'inspection':
        # 转换巡检点格式
        inspection_points = properties.get('inspection_points', [])
        converted_points = []
        for point in inspection_points:
            if isinstance(point, dict):
                converted_points.append((point['x'], point['y'], point['z']))
            else:
                converted_points.append(point)
        
        inspection_properties = {
            'inspection_points': converted_points,
            'inspection_mode': properties.get('inspection_mode', 'sequential'),
            'data_collection_type': properties.get('data_collection_type', 'image'),
            'inspection_interval': properties.get('inspection_interval', 60),
            'quality_threshold': properties.get('quality_threshold', 0.8),
            'start_delay': start_delay  # 保存start_delay到workflow属性中
        }
        
        workflow = InspectionWorkflow(
            env, workflow_id, agent,
            properties=inspection_properties
        )
        workflow.id = workflow_id  # Override auto-generated UUID
        # --- 新增: 创建 communication 工作流 ---
    elif workflow_type == 'communication':
        # 通信工作流的属性通常不需要转换格式
        communication_properties = {
            'data_id': properties['data_id'],
            'data_size_mb': properties['data_size_mb'],
            'target_location': tuple(properties['target_location']), # 转换为元组更佳
            'max_end_to_end_latency_s': properties.get('max_end_to_end_latency_s'),
            'start_delay': start_delay  # 保存start_delay到workflow属性中
        }
        
        workflow = CommunicationWorkflow(
            env, workflow_id, agent,
            properties=communication_properties
        )
        workflow.id = workflow_id  # Override auto-generated UUID
    elif workflow_type == 'computation':
        # 计算工作流的属性处理
        compute_location = properties.get('compute_location', [0, 0, 0])

        # 确保compute_location是元组格式
        if isinstance(compute_location, list):
            compute_location = tuple(compute_location)

        computation_properties = {
            'compute_location': compute_location,
            'job_id': properties['job_id'],
            'required_gflops': properties['required_gflops'],
            'workload_profile': properties['workload_profile'],
            'start_delay': start_delay  # 保存start_delay到workflow属性中
        }

        workflow = ComputeWorkflow(
            env, workflow_id, agent,
            properties=computation_properties
        )
        workflow.id = workflow_id  # Override auto-generated UUID
    else:
        # 未知类型，抛出错误
        raise ValueError(f"不支持的工作流类型: {workflow_type}")
    
    env.workflow_manager.register_workflow(workflow, start_trigger, 1)
    logger.info(f"已注册workflow到WorkflowManager: {workflow_id}, trigger_time={start_delay}")
    return workflow

def load_and_create_from_config(env: Environment, config_file: str = "logistics_config.yaml") -> Tuple[Dict[str, DroneAgent], Dict[str, Union[LogisticsWorkflow, InspectionWorkflow, CommunicationWorkflow, ComputeWorkflow]], Dict[str, Any]]:
    """从配置文件加载并创建智能体和工作流
    
    Args:
        env: 仿真环境
        config_file: 配置文件名
        
    Returns:
        (智能体字典, 工作流字典, 配置字典)
    """
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_logistics_config(config_file)
    
    # 创建智能体
    agents = create_agents_from_config(env, config)
    
    # 创建工作流
    workflows = create_workflows_from_config(env, agents, config)
    
    return agents, workflows, config

def create_workflow_from_config(env: Environment, agent: DroneAgent, workflow_config: Dict[str, Any]):
    """从配置创建单个工作流

    Args:
        env: 仿真环境
        agent: 智能体对象
        workflow_config: 工作流配置

    Returns:
        创建的工作流对象
    """
    try:
        workflow_type = workflow_config.get('type')
        workflow_id = workflow_config.get('id')
        workflow_name = workflow_config.get('name', workflow_id)
        properties = workflow_config.get('properties', {})

        # 根据工作流类型创建相应的工作流
        if workflow_type == 'LogisticsWorkflow':
            workflow = LogisticsWorkflow(
                env=env,
                workflow_id=workflow_id,
                owner_agent=agent,
                properties=properties
            )
        elif workflow_type == 'InspectionWorkflow':
            workflow = InspectionWorkflow(
                env=env,
                workflow_id=workflow_id,
                owner_agent=agent,
                properties=properties
            )
        elif workflow_type == 'CommunicationWorkflow':
            workflow = CommunicationWorkflow(
                env=env,
                workflow_id=workflow_id,
                owner_agent=agent,
                properties=properties
            )
        elif workflow_type == 'ComputeWorkflow':
            workflow = ComputeWorkflow(
                env=env,
                workflow_id=workflow_id,
                owner_agent=agent,
                properties=properties
            )
        else:
            logger.error(f"不支持的工作流类型: {workflow_type}")
            return None

        logger.info(f"成功创建工作流: {workflow_id} ({workflow_type})")
        return workflow

    except Exception as e:
        logger.error(f"创建工作流失败 {workflow_config.get('id', 'unknown')}: {str(e)}")
        return None
