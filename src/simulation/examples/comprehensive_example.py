"""
综合场景仿真示例

展示如何同时运行多个场景（物流、通信、计算、巡检）的综合仿真。
自动发现并加载config目录下的所有场景配置文件。

@author: LoGe Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import random
random.seed(2025)
from pathlib import Path
# 导入 AirFogSim 核心组件
from airfogsim.core.environment import Environment
from src.config import ConfigLoader
from src.config.factory import create_agents_from_config, create_workflows_from_config
from src.offline_pipeline.runtime_loader import OfflineTaskLoader

# 导入所有任务类
from src.simulation.tasks import (
    MoveToTask, PickupTask, DeliveryTask,
    CommunicationTask, ComputeTask, InspectionTask
)

# 导入所有组件类
from src.simulation.components import (
    CommunicationComponent, MobilityComponent, LogisticsComponent,
    ComputeComponent, InspectionComponent
)

# 导入环境控制器
from src.simulation.environment import WeatherController, UFOGenerator, EventScheduler, TopologyManager, GlobalMonitor

# 导入数据收集器
from src.simulation.core import DataCollector

import logging
from typing import Dict, Any
import time
import os


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSimulation:
    """综合场景仿真类"""

    def __init__(self, config_dir: str = None, method_name: str = "default", planning_mode: str = "hierarchical",
                 offline_plan_path: str = None):
        """初始化综合仿真

        Args:
            config_dir: 配置文件目录，默认使用项目config目录
            method_name: 规划方法名称
            planning_mode: 规划模式 ('hierarchical', 'monolithic', 'greedy', 'none', 'offline')
            offline_plan_path: 离线PDDL计划JSON路径（仅当planning_mode='offline'时使用）
        """
        self.offline_plan_path = offline_plan_path
        self.offline_task_loader = None
        # 配置加载器
        self.config_loader = ConfigLoader(config_dir)
        self.method_name = method_name
        self.planning_mode = planning_mode

        # 加载所有场景配置
        self.scenario_configs = self.config_loader.load_all_scenario_configs()

        if not self.scenario_configs:
            raise ValueError("未找到任何场景配置文件")

        # 获取公共配置（从第一个场景配置中提取）
        first_config = next(iter(self.scenario_configs.values()))
        sim_config = self.config_loader.get_simulation_config(first_config)
        visual_interval = sim_config.get('visual_interval', 1.0)

        # 创建AirFogSim环境
        self.env = Environment(visual_interval=visual_interval)
        self.env.task_manager.clear_registered_task_classes()
        self.env.component_manager.clear_registered_component_classes()
        # 注册所有任务类
        self.env.task_manager.register_task_classes([
            MoveToTask, PickupTask, DeliveryTask,
            CommunicationTask, ComputeTask, InspectionTask
        ])

        # 注册所有组件类
        self.env.component_manager.register_component_classes([
            CommunicationComponent, MobilityComponent, LogisticsComponent,
            ComputeComponent, InspectionComponent
        ])

        # 存储所有智能体和工作流
        self.all_agents = {}
        self.all_workflows = {}
        self.scenario_agents = {}  # 按场景分组的智能体
        self.scenario_workflows = {}  # 按场景分组的工作流

        # 仿真参数
        self.simulation_time = sim_config.get('duration', 3600)

        # 公开版本只运行离线仿真，不加载在线模型配置。
        self.online_model_config = {"enable_online_model": False}

        # 初始化环境控制器
        self.weather_controller = None
        self.ufo_generator = None
        self.event_scheduler = None

        # 初始化拓扑管理器
        self.topology_manager = None
        self.global_monitor = None

        # 初始化数据收集器 - 传入方法名
        self.data_collector = DataCollector(self.env, self.env.event_registry, method_name=method_name)

        self._setup_environment_controllers()
        self._setup_topology_manager()

        # 如果是离线模式，加载离线任务loader
        if planning_mode == 'offline' and offline_plan_path:
            try:
                self.offline_task_loader = OfflineTaskLoader(plan_path=Path(offline_plan_path))
                logger.info(f"已加载离线PDDL计划: {offline_plan_path}")
            except Exception as e:
                logger.error(f"加载离线PDDL计划失败: {e}")
                self.offline_task_loader = None

        logger.info(f"初始化综合仿真，发现 {len(self.scenario_configs)} 个场景")

    def _setup_environment_controllers(self):
        """设置环境控制器"""
        try:
            # 检查是否有突发事件配置
            emergency_config_file = "emergency_events_config.yaml"
            emergency_config = self.config_loader.load_config(emergency_config_file)
            logger.info(f"加载突发事件配置: {emergency_config_file}")

            # 初始化气象环境控制器
            weather_config = emergency_config.get('weather_controller', {})
            if weather_config:
                self.weather_controller = WeatherController(self.env, weather_config)
                self.env.add_data_provider('weather', self.weather_controller)
                logger.info("气象环境控制器已初始化")

            # 初始化不明飞行物生成器
            ufo_config = emergency_config.get('ufo_generator', {})
            if ufo_config:
                self.ufo_generator = UFOGenerator(self.env, ufo_config)
                logger.info("不明飞行物生成器已初始化")

            # 初始化事件调度器
            self.event_scheduler = EventScheduler(self.env, emergency_config)
            logger.info("事件调度器已初始化")

        except Exception as e:
            logger.error(f"设置环境控制器时出错: {e}")
            # 继续运行，但不使用环境控制器

    def _setup_topology_manager(self):
        """设置拓扑管理器"""
        try:
            # 从第一个场景配置获取边界信息
            first_config = next(iter(self.scenario_configs.values()))
            sim_config = self.config_loader.get_simulation_config(first_config)

            # 获取仿真空间边界
            sim_space = sim_config.get('simulation_space', {})
            bounds = {
                'x_min': sim_space.get('x_min', 0),
                'x_max': sim_space.get('x_max', 10000),
                'y_min': sim_space.get('y_min', 0),
                'y_max': sim_space.get('y_max', 10000),
                'z_min': sim_space.get('z_min', 0),
                'z_max': sim_space.get('z_max', 500)
            }

            # 创建拓扑管理器
            self.topology_manager = TopologyManager(bounds=bounds, scale_factor=1.0)

            # Try to load topology from offline plan directory if available
            topo_loaded = False
            if self.planning_mode == 'offline' and self.offline_plan_path:
                try:
                    plan_path = Path(self.offline_plan_path).resolve()  # Resolve to absolute path
                    # Topology file: .../offline_init/scenario/run_seed_X/graph_topology.json
                    scenario_name = plan_path.parents[1].name
                    run_dir_name = plan_path.parent.name
                    data_root = plan_path.parents[3]
                    topo_file = data_root / 'offline_init' / scenario_name / run_dir_name / 'graph_topology.json'
                    traffic_file = topo_file.with_name('graph_topology_traffic.csv')
                    
                    if topo_file.exists():
                        self.topology_manager.load_from_file(str(topo_file))
                        logger.info(f"Loaded topology from {topo_file}")
                        topo_loaded = True
                        
                        # Load traffic profiles for runtime speed limits (same for both topo-aware and non-topo-aware)
                        if traffic_file.exists():
                            self.topology_manager.load_traffic_from_csv(str(traffic_file))
                            logger.info(f"Runtime traffic profiles loaded: {len(self.topology_manager.traffic_profiles)} edges")
                        else:
                            logger.warning(f"Traffic profiles not found: {traffic_file}")
                except Exception as e:
                    logger.warning(f"Failed to load offline topology: {e}")
            
            # Fall back to generating topology if not loaded
            if not topo_loaded:
                self.topology_manager.generate_grid_topology(
                    n_vertiports=30,
                    n_waypoints=80,
                    n_charging=10,
                    k_neighbors=5,
                    seed=42
                )

            logger.info(f"拓扑管理器已初始化: {len(self.topology_manager.nodes)} 节点, "
                       f"{len(self.topology_manager.edges)} 边, "
                       f"traffic_profiles: {len(self.topology_manager.traffic_profiles)} 边")

            # 将拓扑管理器挂接到环境
            self.env.topology_manager = self.topology_manager

            # 初始化全局监控器
            output_dir = Path("data/results")
            self.global_monitor = GlobalMonitor(
                env=self.env,
                agents=[],  # agents在create_all_agents_and_workflows后补充
                topology_manager=self.topology_manager,
                output_path=str(output_dir / "replan_triggers.csv")
            )
            logger.info("全局监控器已初始化")

        except Exception as e:
            logger.error(f"设置拓扑管理器时出错: {e}")
            # 继续运行，但不使用拓扑管理器

    def create_all_agents_and_workflows(self):
        """创建所有场景的智能体和工作流"""
        logger.info("=== 开始创建所有场景的智能体和工作流 ===")
        cnt = 0
        for scenario_name, config in self.scenario_configs.items():
            logger.info(f"\n--- 处理场景: {scenario_name} ---")
            # if cnt in [1,2,3]: 
            #     cnt += 1
            #     continue
            # if cnt > 4: break
            try:
                # 创建该场景的智能体 - 传入planning_mode
                scenario_agents = create_agents_from_config(
                    self.env, config, online_model_config=self.online_model_config, planning_mode=self.planning_mode
                )

                # 创建该场景的工作流
                scenario_workflows = create_workflows_from_config(
                    self.env, scenario_agents, config
                )
                
                # 存储到总集合中
                self.all_agents.update(scenario_agents)
                self.all_workflows.update(scenario_workflows)
                
                # 按场景分组存储
                self.scenario_agents[scenario_name] = scenario_agents
                self.scenario_workflows[scenario_name] = scenario_workflows
                
                logger.info(f"场景 {scenario_name}: 创建了 {len(scenario_agents)} 个智能体, {len(scenario_workflows)} 个工作流")
                
            except Exception as e:
                logger.error(f"创建场景 {scenario_name} 失败: {e}")
                continue
            cnt += 1
            # if cnt > 1:
            #     break
        
        logger.info(f"\n=== 总计创建了 {len(self.all_agents)} 个智能体, {len(self.all_workflows)} 个工作流 ===")

        # 更新全局监控器的agents列表
        if self.global_monitor:
            self.global_monitor.agents = list(self.all_agents.values())

        # 如果是离线模式，为所有agent加载离线任务
        if self.planning_mode == 'offline' and self.offline_task_loader:
            self._load_offline_tasks_to_agents()

    def _load_offline_tasks_to_agents(self):
        """将离线PDDL计划任务加载到所有agent的任务队列中"""
        if not self.offline_task_loader:
            logger.warning("离线任务加载器未初始化，跳过离线任务加载")
            return

        # Load agent ID mapping (runtime_agent_id -> config_agent_id)
        # PDDL plans use config_agent_id, simulation uses runtime_agent_id
        import pandas as pd
        runtime_to_config = {}
        config_to_runtime = {}
        try:
            # Find agent_id_mapping.csv in the same directory structure
            plan_path = Path(self.offline_plan_path)
            # Plan path: .../offline_plans/scenario/run_seed_X/pddl_plan_*.json
            # Mapping: .../offline_init/scenario/run_seed_X/agent_id_mapping.csv
            scenario_name = plan_path.parents[1].name  # e.g., 'large_mixed'
            run_dir_name = plan_path.parent.name  # e.g., 'run_seed_42'
            data_root = plan_path.parents[3]  # .../data
            mapping_path = data_root / 'offline_init' / scenario_name / run_dir_name / 'agent_id_mapping.csv'
            
            if mapping_path.exists():
                mapping_df = pd.read_csv(mapping_path)
                for _, row in mapping_df.iterrows():
                    runtime_id = row['runtime_agent_id']
                    config_id = row['config_agent_id']
                    runtime_to_config[runtime_id] = config_id
                    config_to_runtime[config_id] = runtime_id
                logger.info(f"Loaded agent ID mapping: {len(runtime_to_config)} agents from {mapping_path}")
            else:
                logger.warning(f"Agent ID mapping not found at {mapping_path}")
        except Exception as e:
            logger.error(f"Failed to load agent ID mapping: {e}")

        total_tasks_loaded = 0
        for agent_id, agent in self.all_agents.items():
            try:
                # Try to get tasks using runtime agent_id first
                offline_tasks = self.offline_task_loader.get_tasks_for_agent(agent_id)
                
                # If no tasks found, try using config_agent_id (PDDL uses config IDs)
                if not offline_tasks and agent_id in runtime_to_config:
                    config_agent_id = runtime_to_config[agent_id]
                    offline_tasks = self.offline_task_loader.get_tasks_for_agent(config_agent_id)
                    if offline_tasks:
                        logger.debug(f"Found tasks for {agent_id} using config ID {config_agent_id}")
                
                if not offline_tasks:
                    logger.debug(f"Agent {agent_id} 没有离线任务")
                    continue

                # 标记agent使用离线计划（禁用workflow自动生成任务）
                agent.properties['use_offline_plan'] = True

                for t in offline_tasks:
                    # Inject topology_manager for runtime speed limit queries
                    props = dict(t.get("properties") or {})
                    if self.topology_manager is not None:
                        props["topology_manager"] = self.topology_manager
                    
                    agent.add_task_to_queue(
                        component_name=t["component"],
                        task_name=t["task_name"],
                        task_class=t["task_class"],
                        priority=t.get("priority"),
                        preemptive=t.get("preemptive", False),
                        target_state=t.get("target_state"),
                        properties=props,
                        workflow_id=t.get("workflow_id"),
                    )
                    total_tasks_loaded += 1

                logger.info(f"Agent {agent_id}: 加载了 {len(offline_tasks)} 个离线任务")

            except Exception as e:
                logger.error(f"为Agent {agent_id} 加载离线任务失败: {e}")
                continue

        logger.info(f"=== 离线任务加载完成: 共 {total_tasks_loaded} 个任务 ===")

    def run_simulation(self):
        """运行综合仿真"""
        logger.info("=== 开始综合场景仿真 ===")
        logger.info(f"仿真时长: {self.simulation_time} 秒")
        logger.info(f"参与场景: {list(self.scenario_configs.keys())}")
        
        start_time = time.time()

        # 启动全局监控器进程
        if self.global_monitor is not None:
            try:
                self.env.process(self.global_monitor.run())
                logger.info("GlobalMonitor process started")
            except Exception as e:
                logger.error(f"Failed to start GlobalMonitor process: {e}")

        # 运行仿真
        self.env.run(until=self.simulation_time)
        
        end_time = time.time()
        real_duration = end_time - start_time
        
        logger.info("=== 仿真完成 ===")
        logger.info(f"实际运行时间: {real_duration:.2f} 秒")

        # 保存收集的数据
        results_dir = os.path.join(os.path.dirname(__file__), '../../../data/results')
        self.data_collector.collect_workflow_times_from_existing()
        self.data_collector.export_data_to_json(results_dir)

        # 输出仿真结果
        self.print_comprehensive_results()

    def print_comprehensive_results(self):
        """打印综合仿真结果"""
        logger.info("\n" + "="*60)
        logger.info("综合仿真结果报告")
        logger.info("="*60)
        
        # 按场景输出结果
        for scenario_name in self.scenario_configs.keys():
            logger.info(f"\n--- {scenario_name} 场景结果 ---")
            
            # 该场景的工作流状态
            if scenario_name in self.scenario_workflows:
                workflows = self.scenario_workflows[scenario_name]
                logger.info(f"\n工作流数量: {len(workflows)}")
                
                for workflow_id, workflow in workflows.items():
                    logger.info(f"\n工作流 {workflow_id}:")
                    logger.info(f"  状态机状态: {workflow.status_machine.current_status}")
                    logger.info(f"  工作流状态: {workflow.status.name}")
                    logger.info(f"  开始时间 {workflow.start_time}")
                    logger.info(f"  完成时间 {workflow.end_time}")
                    
                    # 根据工作流类型输出特定信息
                    if hasattr(workflow, 'payload_id'):
                        logger.info(f"  物品ID: {workflow.payload_id}")
                        logger.info(f"  已取件: {workflow.is_picked_up}")
                        logger.info(f"  已配送: {workflow.is_delivered}")
                    elif hasattr(workflow, 'data_id'):
                        logger.info(f"  数据ID: {workflow.data_id}")
                        logger.info(f"  数据大小: {workflow.data_size_mb} MB")
                    elif hasattr(workflow, 'job_id'):
                        logger.info(f"  作业ID: {workflow.job_id}")
                        logger.info(f"  是否完成: {getattr(workflow, 'is_completed', False)}")
                    elif hasattr(workflow, 'inspection_points'):
                        logger.info(f"  巡检点总数: {len(workflow.inspection_points)}")
                        logger.info(f"  已完成点数: {len(workflow.completed_waypoints)}")
        
        # 环境控制器状态
        logger.info(f"\n--- 环境控制器状态 ---")
        if self.weather_controller:
            default_weather = self.weather_controller.default_weather
            active_zones = len([z for z in self.weather_controller.weather_zones if z.active])
            logger.info(f"默认天气条件:")
            logger.info(f"  风速: {default_weather.wind_speed:.1f} m/s")
            logger.info(f"  能见度: {default_weather.visibility:.0f} m")
            logger.info(f"  温度: {default_weather.temperature:.1f} °C")
            logger.info(f"  湿度: {default_weather.humidity:.1f} %")
            logger.info(f"活跃天气区域数量: {active_zones}")

            # 显示活跃的天气区域信息
            for zone in self.weather_controller.weather_zones:
                if zone.active:
                    logger.info(f"  区域 {zone.zone_id}: 中心({zone.center[0]:.0f}, {zone.center[1]:.0f}, {zone.center[2]:.0f}), "
                               f"半径{zone.radius:.0f}m, 风速{zone.condition.wind_speed:.1f}m/s")
        else:
            logger.info("气象环境控制器未启用")

        if self.ufo_generator:
            ufo_count = self.ufo_generator.get_ufo_count()
            logger.info(f"活跃UFO数量: {ufo_count}")
            if ufo_count > 0:
                active_ufos = self.ufo_generator.get_active_ufos()
                for ufo_id, ufo in active_ufos.items():
                    logger.info(f"  UFO {ufo_id}: 位置 {ufo.position}, 威胁等级 {ufo.threat_level}")
        else:
            logger.info("不明飞行物生成器未启用")

        if self.event_scheduler:
            active_events = self.event_scheduler.get_active_events()
            scheduled_count = self.event_scheduler.get_scheduled_events_count()
            logger.info(f"事件调度器状态:")
            logger.info(f"  已执行事件数: {len(active_events)}")
            logger.info(f"  待调度事件数: {scheduled_count}")
            if active_events:
                for event_id, event in active_events.items():
                    logger.info(f"  事件 {event_id}: {event['type']} - {event['status']}")
        else:
            logger.info("事件调度器未启用")

        # 拓扑管理器状态
        logger.info(f"\n--- 拓扑管理器状态 ---")
        if self.topology_manager:
            logger.info(f"节点总数: {len(self.topology_manager.nodes)}")
            logger.info(f"边总数: {len(self.topology_manager.edges)}")
            logger.info(f"仿真空间边界: {self.topology_manager.bounds}")

            # 统计节点类型
            node_types = {}
            for node in self.topology_manager.nodes.values():
                node_types[node.type] = node_types.get(node.type, 0) + 1

            for ntype, count in node_types.items():
                logger.info(f"  {ntype}: {count}")

            # 拥堵边统计
            congested_edges = sum(1 for e in self.topology_manager.edges.values()
                                 if e.capacity > 0)
            logger.info(f"可用边数: {congested_edges}/{len(self.topology_manager.edges)}")
        else:
            logger.info("拓扑管理器未启用")

        # 全局监控器状态
        logger.info(f"\n--- 全局监控器状态 ---")
        if self.global_monitor:
            logger.info(f"监控的agents数: {len(self.global_monitor.agents)}")
            logger.info(f"收集到的replan触发记录数: {len(self.global_monitor.triggers)}")
        else:
            logger.info("全局监控器未启用")

        # 数据收集统计
        self.data_collector.print_summary()

        # 总体统计
        logger.info(f"\n--- 总体统计 ---")
        logger.info(f"参与场景数: {len(self.scenario_configs)}")
        logger.info(f"总智能体数: {len(self.all_agents)}")
        logger.info(f"总工作流数: {len(self.all_workflows)}")
        logger.info(f"仿真时长: {self.env.now} 秒")



    def get_scenario_summary(self) -> Dict[str, Any]:
        """获取场景摘要信息"""
        summary = {
            'total_scenarios': len(self.scenario_configs),
            'total_agents': len(self.all_agents),
            'total_workflows': len(self.all_workflows),
            'simulation_time': self.env.now,
            'scenarios': {}
        }

        for scenario_name in self.scenario_configs.keys():
            scenario_summary = {
                'agents_count': len(self.scenario_agents.get(scenario_name, {})),
                'workflows_count': len(self.scenario_workflows.get(scenario_name, {})),
                'agent_ids': list(self.scenario_agents.get(scenario_name, {}).keys()),
                'workflow_ids': list(self.scenario_workflows.get(scenario_name, {}).keys())
            }
            summary['scenarios'][scenario_name] = scenario_summary

        # 添加拓扑相关信息
        if self.topology_manager:
            summary['topology'] = {
                'nodes_count': len(self.topology_manager.nodes),
                'edges_count': len(self.topology_manager.edges),
                'bounds': self.topology_manager.bounds
            }

        # 添加监控信息
        if self.global_monitor:
            summary['monitoring'] = {
                'replan_triggers': len(self.global_monitor.triggers)
            }

        return summary

    def save_topology_config(self, output_path: str = None):
        """保存拓扑配置到文件"""
        if not self.topology_manager:
            logger.warning("拓扑管理器未启用，跳过拓扑配置保存")
            return

        if output_path is None:
            output_dir = Path("data/results")
            output_path = str(output_dir / "topology_config.json")

        try:
            self.topology_manager.save_to_file(output_path)
            logger.info(f"拓扑配置已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存拓扑配置失败: {e}")

    def save_monitoring_results(self, output_path: str = None):
        """保存监控结果到文件"""
        if not self.global_monitor:
            logger.warning("全局监控器未启用，跳过监控结果保存")
            return

        if output_path is None:
            output_dir = Path("data/results")
            output_path = str(output_dir / "replan_triggers.csv")

        try:
            self.global_monitor.output_path = output_path
            self.global_monitor.save_triggers()
            logger.info(f"监控结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存监控结果失败: {e}")


def run_single_experiment(method_name: str, enable_online_model: bool, planning_mode: str,
                          offline_plan_path: str = None, config_dir: str = None):
    """运行单个实验

    Args:
        method_name: 方法名称
        enable_online_model: 是否启用在线模型；公开版本应保持为False
        planning_mode: 规划模式
        offline_plan_path: 离线PDDL计划路径（仅offline模式使用）
        config_dir: 配置文件目录（离线模式时使用offline_init目录）
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"开始实验: {method_name}")
        logger.info(f"启用在线模型: {enable_online_model}, 规划模式: {planning_mode}")
        if offline_plan_path:
            logger.info(f"离线计划路径: {offline_plan_path}")
        logger.info(f"{'='*80}\n")

        # 创建综合仿真
        sim = ComprehensiveSimulation(
            config_dir=config_dir,
            method_name=method_name, 
            planning_mode=planning_mode,
            offline_plan_path=offline_plan_path
        )

        if enable_online_model:
            raise RuntimeError("公开版本只支持离线仿真，不能启用在线模型。")

        # 创建所有智能体和工作流
        sim.create_all_agents_and_workflows()

        # 运行仿真
        sim.run_simulation()

        # 保存拓扑配置和监控结果
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        sim.save_topology_config(str(output_dir / f"topology_config_{method_name}.json"))
        sim.save_monitoring_results(str(output_dir / f"replan_triggers_{method_name}.csv"))

        # 输出摘要
        summary = sim.get_scenario_summary()
        logger.info(f"\n实验 {method_name} 完成")
        logger.info(f"仿真摘要: {summary}\n")

        return summary

    except Exception as e:
        logger.error(f"实验 {method_name} 运行失败: {e}")
        raise


def main():
    """主函数 - 运行离线仿真实验
    
    所有方法都是完全离线的，不依赖在线模型调用。
    请使用 run_offline_simulation.py 作为主入口。
    """
    import argparse

    # 所有可用的离线方法
    CLUSTERING_METHODS = ['kmeans', 'triplet', 'hungarian', 'htt_recursive']
    NON_CLUSTERING_METHODS = ['monolithic']
    ALL_METHODS = CLUSTERING_METHODS + NON_CLUSTERING_METHODS

    parser = argparse.ArgumentParser(description='运行离线仿真实验')
    parser.add_argument('--method', type=str, default='htt_recursive',
                       choices=ALL_METHODS + ['all'],
                       help='选择运行的方法 (默认: htt_recursive)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--offline-plan', type=str, default=None,
                       help='显式指定离线计划JSON文件路径')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    
    # 确定要运行的方法
    if args.method == 'all':
        methods_to_run = ALL_METHODS
    else:
        methods_to_run = [args.method]

    # 配置目录
    config_dir = str(project_root / "data" / "offline_init" / f"run_seed_{args.seed}")

    logger.info(f"\n{'='*80}")
    logger.info(f"离线仿真实验")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"方法: {methods_to_run}")
    logger.info(f"{'='*80}\n")

    # 运行实验
    all_results = {}
    for method in methods_to_run:
        # 构建计划文件路径
        if args.offline_plan:
            offline_plan_path = args.offline_plan
        else:
            offline_plan_path = str(
                project_root / "data" / "offline_plans" / f"run_seed_{args.seed}" / f"pddl_plan_{method}.json"
            )

        if not Path(offline_plan_path).exists():
            logger.warning(f"跳过 {method}: 计划文件不存在 {offline_plan_path}")
            continue

        logger.info(f"\n运行方法: {method}")
        logger.info(f"计划文件: {offline_plan_path}")

        try:
            summary = run_single_experiment(
                method_name=f"Offline-{method}",
                enable_online_model=False,
                planning_mode='offline',
                offline_plan_path=offline_plan_path,
                config_dir=config_dir
            )
            all_results[method] = summary
            logger.info(f"✓ {method}: {summary.get('total_agents', 'N/A')} agents")
        except Exception as e:
            logger.error(f"✗ {method} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 输出总结
    logger.info(f"\n{'='*80}")
    logger.info("实验完成!")
    logger.info(f"{'='*80}")
    for method, summary in all_results.items():
        logger.info(f"  {method:15s}: {summary.get('total_agents', 'N/A'):3d} agents, "
                   f"{summary.get('total_workflows', 'N/A'):3d} workflows")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
