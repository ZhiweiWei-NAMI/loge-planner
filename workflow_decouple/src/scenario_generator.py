"""
场景生成器 - 动态生成agents和workflows配置
"""

import yaml
import random
import math
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from .safe_yaml import safe_yaml_read, safe_yaml_write
except ImportError:
    from safe_yaml import safe_yaml_read, safe_yaml_write

try:
    # 仅在当前工程内可用，用于snap-to-graph
    from src.simulation.environment.topology_manager import TopologyManager
except Exception:  # pragma: no cover - 允许独立使用
    TopologyManager = None

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """场景生成器 - 根据配置动态生成agents和workflows"""
    
    def __init__(self, uav_types_config: str, generation_config: str, topology_file: Optional[str] = None, snap_to_graph: bool = False):
        """
        初始化场景生成器
        
        Args:
            uav_types_config: UAV类型配置文件路径
            generation_config: 场景生成配置文件路径
        """
        # 加载配置
        with open(uav_types_config, 'r', encoding='utf-8') as f:
            self.uav_types = yaml.safe_load(f)['uav_types']
        
        with open(generation_config, 'r', encoding='utf-8') as f:
            self.gen_config = yaml.safe_load(f)
        
        self.sim_space = self.gen_config['simulation_space']
        
        # 加载热点区域配置
        self.hotspot_config = self.gen_config['hotspot_regions']
        self.hotspots_enabled = self.hotspot_config['enable']
        self.hotspot_probability = self.hotspot_config['hotspot_probability']
        self.hotspot_regions = self.hotspot_config['regions']
        
        # 可选拓扑与snap-to-graph
        self.snap_to_graph = bool(snap_to_graph)
        self.topology_manager = None
        if topology_file and TopologyManager is not None:
            topo_path = Path(topology_file)
            if topo_path.is_file():
                bounds = {
                    'x_min': self.sim_space.get('x_min', 0.0),
                    'x_max': self.sim_space.get('x_max', 10000.0),
                    'y_min': self.sim_space.get('y_min', 0.0),
                    'y_max': self.sim_space.get('y_max', 10000.0),
                    'z_min': self.sim_space.get('z_min', 0.0),
                    'z_max': self.sim_space.get('z_max', 500.0),
                }
                try:
                    self.topology_manager = TopologyManager(bounds=bounds)
                    self.topology_manager.load_from_file(str(topo_path))
                    self.snap_to_graph = True
                    logger.info("TopologyManager loaded from %s, snap_to_graph enabled", topo_path)
                except Exception as e:
                    self.topology_manager = None
                    logger.error("Failed to load topology from %s: %s", topo_path, e)
        
        logger.info(f"场景生成器初始化完成")
        logger.info(f"UAV类型: {list(self.uav_types.keys())}")
        if self.hotspots_enabled:
            logger.info(f"热点区域已启用: {len(self.hotspot_regions)}个热点, 热点概率={self.hotspot_probability}")
    
    def generate_scenario(self, scenario_name: str = None) -> Dict[str, Any]:
        """
        生成完整场景配置
        
        Args:
            scenario_name: 场景名称 ('small', 'medium', 'large' 或 None使用默认配置)
        
        Returns:
            场景配置字典
        """
        if scenario_name and scenario_name in self.gen_config['test_scenarios']:
            logger.info(f"使用预定义测试场景: {scenario_name}")
            test_config = self.gen_config['test_scenarios'][scenario_name]
            agents_config = test_config['agents']
            workflows_config = test_config['workflows']
        else:
            logger.info("使用默认配置生成场景")
            agents_config = self.gen_config['agents']
            workflows_config = self.gen_config['workflows']
        
        # 生成agents
        agents = self._generate_agents(agents_config)
        
        # 生成workflows（需要agents信息来分配）
        workflows = self._generate_workflows(workflows_config, agents)
        
        # 构建场景配置
        scenario = {
            'scenario_name': self.gen_config['scenario_name'],
            'scenario_description': self.gen_config['scenario_description'],
            'agents': agents,
            'workflows': workflows
        }
        
        logger.info(f"场景生成完成: {len(agents)}个agents, {len(workflows)}个workflows")
        
        return scenario
    
    def _generate_agents(self, agents_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成agents配置"""
        agents = []
        agent_counter = 0
        
        for uav_type, type_config in agents_config.items():
            if uav_type not in self.uav_types:
                logger.warning(f"未知的UAV类型: {uav_type}, 跳过")
                continue
            
            # 支持两种格式：整数或字典
            if isinstance(type_config, int):
                count = type_config
                battery_level = 100.0
            else:
                count = type_config['count']
                battery_level = type_config['battery_level']
            
            # 获取UAV类型的基础配置
            base_config = self.uav_types[uav_type]
            
            for i in range(count):
                agent_id = f"agent_{uav_type}_{i+1}"
                
                # 随机生成初始位置
                position = self._random_position()
                
                # 构建agent配置
                agent = {
                    'id': agent_id,
                    'type': 'drone',
                    'initial_position': {
                        'x': position[0],
                        'y': position[1],
                        'z': position[2]
                    },
                    'properties': {
                        'battery_capacity': base_config['battery_capacity'],
                        'battery_level': battery_level,
                        'components': base_config['components'].copy(),
                        'payload_capacity': base_config['payload_capacity'],
                        'weight': base_config['weight']
                    },
                    'uav_type': uav_type  # 记录UAV类型，用于workflow分配
                }
                
                # 添加特殊属性（如果有）
                for key in ['cpu_cores', 'memory_gb', 'transmission_power']:
                    if key in base_config:
                        agent['properties'][key] = base_config[key]
                
                agents.append(agent)
                agent_counter += 1
        
        logger.info(f"生成了 {agent_counter} 个agents")
        return agents
    
    def _generate_workflows(self, workflows_config: Dict[str, Any], 
                           agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        生成workflows配置
        
        Args:
            workflows_config: workflow生成配置
            agents: 已生成的agents列表
        
        Returns:
            workflows配置列表
        """
        workflows = []
        workflow_counter = 1
        
        # 按类型分组agents
        agents_by_type = self._group_agents_by_capability(agents)
        
        # 跨类型追踪已分配的agents（agent_id -> workflow_id）
        self.assigned_agents = {}
        
        # 生成物流workflows
        if 'logistics' in workflows_config:
            logistics_wfs = self._generate_logistics_workflows(
                workflows_config['logistics'],
                agents_by_type['logistics'],
                workflow_counter
            )
            workflows.extend(logistics_wfs)
            workflow_counter += len(logistics_wfs)
        
        # 生成巡检workflows
        if 'inspection' in workflows_config:
            inspection_wfs = self._generate_inspection_workflows(
                workflows_config['inspection'],
                agents_by_type['inspection'],
                workflow_counter
            )
            workflows.extend(inspection_wfs)
            workflow_counter += len(inspection_wfs)
        
        # 生成通信workflows
        if 'communication' in workflows_config:
            communication_wfs = self._generate_communication_workflows(
                workflows_config['communication'],
                agents_by_type['communication'],
                workflow_counter
            )
            workflows.extend(communication_wfs)
            workflow_counter += len(communication_wfs)
        
        # 生成计算workflows
        if 'computation' in workflows_config:
            computation_wfs = self._generate_computation_workflows(
                workflows_config['computation'],
                agents_by_type['computation'],
                workflow_counter
            )
            workflows.extend(computation_wfs)
            workflow_counter += len(computation_wfs)
        
        logger.info(f"生成了 {len(workflows)} 个workflows")
        logger.info(f"Agent分配统计: {len(self.assigned_agents)} 个agents被分配workflows")
        return workflows
    
    def _group_agents_by_capability(self, agents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """按能力分组agents"""
        grouped = {
            'logistics': [],
            'inspection': [],
            'communication': [],
            'computation': [],
            'relay': []
        }
        
        for agent in agents:
            components = agent['properties']['components']
            agent_id = agent['id']
            
            if 'logistics' in components:
                grouped['logistics'].append(agent_id)
            if 'inspection' in components:
                grouped['inspection'].append(agent_id)
            if 'communication' in components:
                grouped['communication'].append(agent_id)
            if 'compute' in components:  # 注意：配置文件中是'compute'而不是'computation'
                grouped['computation'].append(agent_id)
            if 'relay' in components:
                grouped['relay'].append(agent_id)
        
        return grouped
    
    def _generate_logistics_workflows(self, config: Any, 
                                     capable_agents: List[str],
                                     start_counter: int) -> List[Dict[str, Any]]:
        """生成物流workflows"""
        workflows = []
        
        # 支持整数或字典格式
        if isinstance(config, int):
            count = config
            start_delay_min = 0
            start_delay_max = 150
            start_delay_step = 2
            weight_min = 0.5
            weight_max = 3.0
        else:
            count = config['count']
            start_delay_min = config['start_delay_min']
            start_delay_max = config['start_delay_max']
            start_delay_step = config['start_delay_step']
            weight_min = config['payload_weight_min']
            weight_max = config['payload_weight_max']
        start_delay_min = max(start_delay_min, 1)
        if not capable_agents:
            logger.warning("没有可用的物流agents，跳过物流workflow生成")
            return workflows
        
        # 生成均匀分布的启动延迟
        delays = self._generate_delays(count, start_delay_min, start_delay_max, start_delay_step)
        
        # 获取未分配的agents（跨workflow类型）
        unassigned_agents = [a for a in capable_agents if a not in self.assigned_agents]
        
        for i in range(count):
            workflow_id = f"workflow_{start_counter + i}"
            
            # 优先从未分配的agents中选择
            if unassigned_agents:
                agent_id = random.choice(unassigned_agents)
                unassigned_agents.remove(agent_id)
            else:
                # 所有agents都已分配，允许重复分配
                agent_id = random.choice(capable_agents)
            
            # 记录分配
            self.assigned_agents[agent_id] = workflow_id
            
            # 生成取货和送货位置（使用热点区域）
            pickup_loc = self._random_position(z_fixed=50.0, task_type='logistics')
            delivery_loc = self._random_position(z_fixed=50.0, task_type='logistics')
            
            # 随机货物重量
            payload_weight = round(random.uniform(weight_min, weight_max), 1)
            payload_id = f"payload_{start_counter + i}"
            
            workflow = {
                'id': workflow_id,
                'name': f"物流{workflow_id}",
                'type': 'logistics',
                'agent_id': agent_id,
                'start_delay': delays[i],
                'properties': {
                    'payload_id': payload_id,
                    'payload_weight': payload_weight,
                    'pickup_location': {
                        'x': pickup_loc[0],
                        'y': pickup_loc[1],
                        'z': pickup_loc[2]
                    },
                    'delivery_location': {
                        'x': delivery_loc[0],
                        'y': delivery_loc[1],
                        'z': delivery_loc[2]
                    },
                    'payloads': []
                }
            }
            
            workflows.append(workflow)
        
        return workflows
    
    def _generate_inspection_workflows(self, config: Any,
                                      capable_agents: List[str],
                                      start_counter: int) -> List[Dict[str, Any]]:
        """生成巡检workflows"""
        workflows = []
        
        # 支持整数或字典格式
        if isinstance(config, int):
            count = config
            start_delay_min = 0
            start_delay_max = 150
            start_delay_step = 2
            points_min = 3
            points_max = 8
            data_type = 'image'
            inspection_mode = 'sequential'
        else:
            count = config['count']
            start_delay_min = config['start_delay_min']
            start_delay_max = config['start_delay_max']
            start_delay_step = config['start_delay_step']
            points_min = config['inspection_points_min']
            points_max = config['inspection_points_max']
            data_type = config['data_collection_type']
            inspection_mode = config['inspection_mode']
        start_delay_min = max(start_delay_min, 1)
        if not capable_agents:
            logger.warning("没有可用的巡检agents，跳过巡检workflow生成")
            return workflows
        
        # 生成均匀分布的启动延迟
        delays = self._generate_delays(count, start_delay_min, start_delay_max, start_delay_step)
        
        # 获取未分配的agents（跨workflow类型）
        unassigned_agents = [a for a in capable_agents if a not in self.assigned_agents]
        
        for i in range(count):
            workflow_id = f"workflow_{start_counter + i}"
            
            # 优先从未分配的agents中选择
            if unassigned_agents:
                agent_id = random.choice(unassigned_agents)
                unassigned_agents.remove(agent_id)
            else:
                # 所有agents都已分配，允许重复分配
                agent_id = random.choice(capable_agents)
            
            # 记录分配
            self.assigned_agents[agent_id] = workflow_id
            
            # 随机生成巡检点数量
            num_points = random.randint(points_min, points_max)
            
            # 生成巡检点 - 第一个点使用热点，后续点在前一个点500m半径内
            inspection_points = []
            max_radius = 500.0  # 最大半径500米
            
            for j in range(num_points):
                if j == 0:
                    # 第一个点使用热点区域
                    first_pos = self._random_position(task_type='inspection')
                    point = {
                        'x': first_pos[0],
                        'y': first_pos[1],
                        'z': first_pos[2]
                    }
                else:
                    # 后续点在前一个点的500m半径内随机生成
                    prev_point = inspection_points[j - 1]
                    
                    # 随机生成角度和距离
                    angle = random.uniform(0, 2 * 3.14159)  # 0-2π
                    distance = random.uniform(100, max_radius)  # 100-500米之间
                    
                    # 计算新坐标
                    new_x = prev_point['x'] + distance * math.cos(angle)
                    new_y = prev_point['y'] + distance * math.sin(angle)
                    
                    # 确保坐标在合法范围内
                    new_x = max(self.sim_space['x_min'], min(self.sim_space['x_max'], new_x))
                    new_y = max(self.sim_space['y_min'], min(self.sim_space['y_max'], new_y))
                    
                    # z坐标在一定范围内变化
                    z_variation = random.uniform(-50, 50)
                    new_z = prev_point['z'] + z_variation
                    new_z = max(self.sim_space['z_min'], min(self.sim_space['z_max'], new_z))
                    
                    point = {
                        'x': round(new_x, 2),
                        'y': round(new_y, 2),
                        'z': round(new_z, 2)
                    }
                
                inspection_points.append(point)
            
            workflow = {
                'id': workflow_id,
                'name': f"巡检{workflow_id}",
                'type': 'inspection',
                'agent_id': agent_id,
                'start_delay': delays[i],
                'properties': {
                    'inspection_points': inspection_points,
                    'data_collection_type': data_type,
                    'inspection_mode': inspection_mode
                }
            }
            
            workflows.append(workflow)
        
        return workflows
    
    def _generate_communication_workflows(self, config: Any,
                                         capable_agents: List[str],
                                         start_counter: int) -> List[Dict[str, Any]]:
        """生成通信workflows"""
        workflows = []
        
        # 解析配置
        if isinstance(config, dict):
            count = config['count']
            start_delay_min = config['start_delay_min']
            start_delay_max = config['start_delay_max']
            start_delay_step = config['start_delay_step']
            data_size_min = config['data_size_min_mb']
            data_size_max = config['data_size_max_mb']
        else:
            count = int(config) if config else 0
            start_delay_min = 0
            start_delay_max = 150
            start_delay_step = 2
            data_size_min = 10
            data_size_max = 100
        
        start_delay_min = max(start_delay_min, 1)
        if not capable_agents:
            logger.warning("没有可用的通信agents，跳过通信workflow生成")
            return workflows
        
        # 生成均匀分布的启动延迟
        delays = self._generate_delays(count, start_delay_min, start_delay_max, start_delay_step)
        
        # 获取未分配的agents
        unassigned_agents = [a for a in capable_agents if a not in self.assigned_agents]
        
        for i in range(count):
            workflow_id = f"workflow_{start_counter + i}"
            
            # 优先从未分配的agents中选择
            if unassigned_agents:
                agent_id = random.choice(unassigned_agents)
                unassigned_agents.remove(agent_id)
            else:
                agent_id = random.choice(capable_agents)
            
            # 记录分配
            self.assigned_agents[agent_id] = workflow_id
            
            # 生成通信任务参数（目标位置使用热点）
            data_id = f"data_{workflow_id}"
            data_size_mb = round(random.uniform(data_size_min, data_size_max), 2)
            target_location = self._random_position(task_type='communication')
            
            workflow = {
                'id': workflow_id,
                'name': f"通信{workflow_id}",
                'type': 'communication',
                'agent_id': agent_id,
                'start_delay': delays[i],
                'properties': {
                    'data_id': data_id,
                    'data_size_mb': data_size_mb,
                    'target_location': [target_location[0], target_location[1], target_location[2]],
                    'max_end_to_end_latency_s': 300.0
                }
            }
            
            workflows.append(workflow)
        
        return workflows
    
    def _generate_computation_workflows(self, config: Any,
                                       capable_agents: List[str],
                                       start_counter: int) -> List[Dict[str, Any]]:
        """生成计算workflows"""
        workflows = []
        
        # 解析配置
        if isinstance(config, dict):
            count = config['count']
            start_delay_min = config['start_delay_min']
            start_delay_max = config['start_delay_max']
            start_delay_step = config['start_delay_step']
            gflops_min = config['gflops_min']
            gflops_max = config['gflops_max']
        else:
            count = int(config) if config else 0
            start_delay_min = 0
            start_delay_max = 150
            start_delay_step = 2
            gflops_min = 100
            gflops_max = 1000
        
        start_delay_min = max(start_delay_min, 1)
        if not capable_agents:
            logger.warning("没有可用的计算agents，跳过计算workflow生成")
            return workflows
        
        # 生成均匀分布的启动延迟
        delays = self._generate_delays(count, start_delay_min, start_delay_max, start_delay_step)
        
        # 获取未分配的agents
        unassigned_agents = [a for a in capable_agents if a not in self.assigned_agents]
        
        for i in range(count):
            workflow_id = f"workflow_{start_counter + i}"
            
            # 优先从未分配的agents中选择
            if unassigned_agents:
                agent_id = random.choice(unassigned_agents)
                unassigned_agents.remove(agent_id)
            else:
                agent_id = random.choice(capable_agents)
            
            # 记录分配
            self.assigned_agents[agent_id] = workflow_id
            
            # 生成计算任务参数（计算位置使用热点）
            job_id = f"job_{workflow_id}"
            required_gflops = round(random.uniform(gflops_min, gflops_max), 2)
            compute_location = self._random_position(task_type='computation')
            
            workflow = {
                'id': workflow_id,
                'name': f"计算{workflow_id}",
                'type': 'computation',
                'agent_id': agent_id,
                'start_delay': delays[i],
                'properties': {
                    'job_id': job_id,
                    'required_gflops': required_gflops,
                    'compute_location': [
                        compute_location[0],
                        compute_location[1],
                        compute_location[2]
                    ],
                    'workload_profile': {
                        'type': 'batch',
                        'priority': 'normal',
                        'cpu_intensity': round(random.uniform(0.5, 1.0), 2),
                        'memory_intensity': round(random.uniform(0.3, 0.8), 2),
                        'io_intensity': round(random.uniform(0.2, 0.6), 2)
                    }
                }
            }
            
            workflows.append(workflow)
        
        return workflows
    
    def _random_position(self, z_fixed: float = None, task_type: str = None) -> tuple:
        """生成随机位置（支持热点区域）
        
        Args:
            z_fixed: 固定Z坐标
            task_type: 任务类型 ('logistics', 'inspection', 'communication', 'computation')
        
        Returns:
            (x, y, z) 坐标元组
        """
        # 如果启用热点且有适用的热点区域，则优先在热点内生成
        if self.hotspots_enabled and task_type and random.random() < self.hotspot_probability:
            pos = self._random_position_hotspot(task_type, z_fixed)
            if pos is not None:
                return pos
        
        # 否则在整个空间随机生成
        x = round(random.uniform(self.sim_space['x_min'], self.sim_space['x_max']), 2)
        y = round(random.uniform(self.sim_space['y_min'], self.sim_space['y_max']), 2)
        
        if z_fixed is not None:
            z = z_fixed
        else:
            z = round(random.uniform(self.sim_space['z_min'], self.sim_space['z_max']), 2)

        pos = (x, y, z)

        # 可选：snap 到图节点
        if self.snap_to_graph and self.topology_manager is not None:
            try:
                type_filter = None
                if task_type == 'logistics':
                    type_filter = ['vertiport', 'charging']
                elif task_type in ('inspection', 'communication', 'computation'):
                    type_filter = ['waypoint']
                nid = self.topology_manager.get_nearest_node(x, y, z, type_filter=type_filter)
                if nid is not None:
                    node = self.topology_manager.nodes.get(nid)
                    if node is not None:
                        pos = (round(node.x, 2), round(node.y, 2), round(node.z, 2))
            except Exception as e:  # pragma: no cover - 防御性
                logger.error("Snap-to-graph failed for task_type=%s: %s", task_type, e)
        
        return pos
    
    def _sample_from_hotspot(self, task_type: str):
        """根据任务类型和权重采样一个热点区域
        
        Args:
            task_type: 任务类型
        
        Returns:
            热点区域配置dict，如果没有适用热点则返回None
        """
        # 筛选适用于该任务类型的热点
        applicable_hotspots = [
            h for h in self.hotspot_regions 
            if task_type in h['types']
        ]
        
        if not applicable_hotspots:
            return None
        
        # 按权重采样（权重越大越容易被选中）
        weights = [h['weight'] for h in applicable_hotspots]
        return random.choices(applicable_hotspots, weights=weights, k=1)[0]
    
    def _random_position_hotspot(self, task_type: str, z_fixed: float = None) -> tuple:
        """在热点区域内生成随机位置
        
        Args:
            task_type: 任务类型
            z_fixed: 固定Z坐标
        
        Returns:
            (x, y, z) 坐标元组，如果没有适用热点返回None
        """
        hotspot = self._sample_from_hotspot(task_type)
        if hotspot is None:
            return None
        
        center = hotspot['center']  # [x, y, z]
        radius = hotspot['radius']
        
        # 在圆形区域内均匀采样（极坐标）
        r = radius * math.sqrt(random.random())  # sqrt确保在圆内均匀分布
        theta = random.uniform(0, 2 * math.pi)
        
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)
        
        # 确保坐标在合法范围内
        x = max(self.sim_space['x_min'], min(self.sim_space['x_max'], x))
        y = max(self.sim_space['y_min'], min(self.sim_space['y_max'], y))
        
        # Z坐标
        if z_fixed is not None:
            z = z_fixed
        else:
            # 在热点中心高度附近随机波动
            z_center = center[2]
            z_variation = random.uniform(-20, 20)  # ±20米
            z = z_center + z_variation
            z = max(self.sim_space['z_min'], min(self.sim_space['z_max'], z))
        
        return (round(x, 2), round(y, 2), round(z, 2))
    
    def _generate_delays(self, count: int, min_delay: int, max_delay: int, step: int) -> List[int]:
        """生成均匀分布的启动延迟"""
        if count == 0:
            return []
        
        # 尽量均匀分布
        if count * step <= (max_delay - min_delay):
            # 可以均匀分布
            delays = [min_delay + i * step for i in range(count)]
        else:
            # 随机分布
            delays = sorted([random.randint(min_delay, max_delay) for _ in range(count)])
        
        return delays
    
    def save_scenario(self, scenario: Dict[str, Any], output_path: str):
        """保存场景配置到yaml文件（使用原子写入防止并发损坏）"""
        safe_yaml_write(
            output_path, scenario,
            allow_unicode=True, default_flow_style=False, sort_keys=False
        )
        
        logger.info(f"场景配置已保存到: {output_path}")


def generate_and_save_scenario(
    scenario_size: str = None, 
    output_path: str = None,
    uav_types_config: str = None,
    generation_config: str = None,
    scenario_name: str = None,
    seed: int = None,
    topology_file: str = None,
    snap_to_graph: bool = False,
):
    """
    生成并保存场景配置
    
    Args:
        scenario_size: 场景规模 ('small', 'medium', 'large' 或 None使用默认) [已弃用，使用scenario_name]
        output_path: 输出文件路径
        uav_types_config: UAV类型配置文件路径
        generation_config: 场景生成配置文件路径
        scenario_name: 场景名称（如'large_mixed', 'small_logistics_inspection'等）
        seed: 随机种子
    """
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        logger.info(f"设置随机种子: {seed}")
    
    # 配置文件路径
    config_dir = Path(__file__).parent.parent / 'config'
    if uav_types_config is None:
        uav_types_config = config_dir / 'uav_types.yaml'
    if generation_config is None:
        generation_config = config_dir / 'scenario_generation.yaml'
    
    if output_path is None:
        name = scenario_name or scenario_size or 'default'
        output_path = config_dir / f'generated_scenario_{name}.yaml'
    
    # 创建生成器
    generator = ScenarioGenerator(
        str(uav_types_config),
        str(generation_config),
        topology_file=topology_file,
        snap_to_graph=snap_to_graph,
    )
    
    # 生成场景（优先使用scenario_name，然后scenario_size）
    final_scenario_name = scenario_name or scenario_size
    scenario = generator.generate_scenario(final_scenario_name)

    # 确保仿真时长配置为1800秒，以便离线仿真和规划使用统一的simulation_time
    # 这里只在场景未显式指定duration时设置默认值，避免覆盖用户自定义配置
    try:
        sim_cfg = scenario.get("simulation", {})
        if not isinstance(sim_cfg, dict):
            sim_cfg = {}
        sim_cfg.setdefault("duration", 1800)
        scenario["simulation"] = sim_cfg
    except Exception:
        # 若场景结构异常，则不强行修改，避免影响其他流程
        pass

    # 保存场景
    generator.save_scenario(scenario, str(output_path))
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"场景生成完成!")
    print(f"{'='*60}")
    if seed is not None:
        print(f"随机种子: {seed}")
    print(f"场景名称: {final_scenario_name or '默认'}")
    print(f"Agents: {len(scenario['agents'])}个")
    print(f"Workflows: {len(scenario['workflows'])}个")
    print(f"输出文件: {output_path}")
    print(f"{'='*60}\n")
    
    return scenario, output_path  # 返回2个值以保持向后兼容


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='生成仿真场景配置')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], 
                       help='场景规模 (small/medium/large)')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    generate_and_save_scenario(args.size, args.output)
