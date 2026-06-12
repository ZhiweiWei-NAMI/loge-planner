"""
动态数据收集脚本 - 支持不同规模的场景测试
"""

import sys
import os

# 添加必要的路径
current_dir = os.path.dirname(__file__)
workflow_decouple_root = os.path.abspath(os.path.join(current_dir, '..'))
tccn_auto_planner_root = os.path.abspath(os.path.join(current_dir, '../..'))

# 确保能找到 src.config 模块
if tccn_auto_planner_root not in sys.path:
    sys.path.insert(0, tccn_auto_planner_root)

from airfogsim.core.environment import Environment
from src.config.factory import create_agents_from_config, create_workflows_from_config
from snapshot_collector import SnapshotCollector
from scenario_generator import generate_and_save_scenario
try:
    from .safe_yaml import safe_yaml_read
except ImportError:
    from safe_yaml import safe_yaml_read
import yaml
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('airfogsim').setLevel(logging.WARNING)
logging.getLogger('snapshot_collector').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def run_data_collection(scenario_config_path: str, simulation_time: int = 1800, 
                       output_dir: str = '../data/dynamic'):
    """
    运行数据收集仿真
    
    Args:
        scenario_config_path: 场景配置文件路径
        simulation_time: 仿真时长（秒）
        output_dir: 数据输出目录
    """
    
    # 加载配置文件（使用安全读取防止并发损坏）
    config = safe_yaml_read(scenario_config_path)
    
    # 验证配置坐标范围
    _validate_coordinates(config)
    
    logger.info(f"=" * 70)
    logger.info(f"场景: {config['scenario_name']}")
    logger.info(f"Agents数量: {len(config['agents'])}")
    logger.info(f"Workflows数量: {len(config['workflows'])}")
    logger.info(f"仿真时长: {simulation_time}秒 ({simulation_time/60:.1f}分钟)")
    logger.info(f"=" * 70)
    
    # 创建环境
    env = Environment(visual_interval=5.0)
    
    # 清空并重新注册任务类和组件类
    env.task_manager.clear_registered_task_classes()
    env.component_manager.clear_registered_component_classes()
    
    # 导入并注册任务类
    from src.simulation.tasks import (
        MoveToTask, PickupTask, DeliveryTask,
        CommunicationTask, ComputeTask, InspectionTask
    )
    env.task_manager.register_task_classes([
        MoveToTask, PickupTask, DeliveryTask,
        CommunicationTask, ComputeTask, InspectionTask
    ])
    
    # 导入并注册组件类
    from src.simulation.components import (
        MobilityComponent, LogisticsComponent, InspectionComponent,
        CommunicationComponent, ComputeComponent
    )
    env.component_manager.register_component_classes([
        MobilityComponent, LogisticsComponent, InspectionComponent,
        CommunicationComponent, ComputeComponent
    ])
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建agents和workflows
    logger.info("创建agents...")
    # 使用hierarchical planning模式，让agents能够自动规划和执行workflow任务
    online_model_config = {'enable_online_model': False}
    agents = create_agents_from_config(env, config, online_model_config=online_model_config)
    
    # 导出agent ID映射（运行时ID → 配置文件名称）
    logger.info("导出agent ID映射...")
    agent_mapping = []
    for config_agent_id, agent in agents.items():
        agent_mapping.append({
            'runtime_agent_id': agent.id,  # 运行时生成的随机ID
            'config_agent_id': config_agent_id  # 配置文件中的agent名称
        })
    
    import pandas as pd
    agent_mapping_df = pd.DataFrame(agent_mapping)
    mapping_file = output_path / 'agent_id_mapping.csv'
    agent_mapping_df.to_csv(mapping_file, index=False)
    logger.info(f"Agent映射已保存: {mapping_file} ({len(agent_mapping)} 个映射)")
    
    logger.info("创建workflows...")
    workflows = create_workflows_from_config(env, agents, config)
    
    # 创建快照收集器
    logger.info("初始化快照收集器...")
    
    snapshot_collector = SnapshotCollector(
        env=env,
        event_registry=env.event_registry
    )
    
    # 更改输出目录
    snapshot_collector.output_dir = str(output_path)
    
    # 注册agents（workflows通过事件自动监听）
    snapshot_collector.register_agents(agents)
    
    # 注册workflows到snapshot_collector，以便追踪workflow状态
    snapshot_collector.workflows = workflows
    
    # 记录workflow初始化信息
    agent_workflow_mapping = []
    for workflow_id, wf in workflows.items():
        agent_workflow_mapping.append({
            'workflow_id': workflow_id,
            'workflow_type': wf.__class__.__name__,
            'owner_agent': wf.owner.id if wf.owner else None,
            'start_time': wf.properties['start_delay'],  # 使用start_delay，缺失则报错
            'properties': wf.properties
        })
    
    logger.info(f"Agent-Workflow映射: {len(agent_workflow_mapping)} 个映射")
    
    # 运行仿真
    logger.info(f"开始运行仿真 {simulation_time} 秒（{simulation_time/60:.0f}分钟）...")
    env.run(until=simulation_time)
    
    # 打印结果
    logger.info("=" * 70)
    logger.info("=== 仿真完成 ===")
    logger.info(f"总agents: {len(agents)}")
    logger.info(f"总workflows: {len(workflows)}")
    
    # 详细检查workflow状态
    completed_wfs = []
    running_wfs = []
    failed_wfs = []
    pending_wfs = []
    
    for workflow_id, wf in workflows.items():
        status = wf.status.name
        if status == 'COMPLETED':
            completed_wfs.append((workflow_id, wf))
        elif status == 'RUNNING':
            running_wfs.append((workflow_id, wf))
        elif status == 'FAILED':
            failed_wfs.append((workflow_id, wf))
        elif status == 'PENDING':
            pending_wfs.append((workflow_id, wf))
    
    logger.info("")
    logger.info("Workflow状态统计:")
    logger.info(f"  COMPLETED: {len(completed_wfs)}")
    logger.info(f"  RUNNING: {len(running_wfs)}")
    logger.info(f"  FAILED: {len(failed_wfs)}")
    logger.info(f"  PENDING: {len(pending_wfs)}")
    
    # 显示前几个COMPLETED workflows
    logger.info("")
    logger.info("前5个COMPLETED workflows:")
    for workflow_id, wf in completed_wfs[:5]:
        logger.info(f"  {workflow_id}: status={wf.status.name}, current_state={wf.status_machine.current_status}")
    
    # 显示前几个RUNNING workflows（诊断用）
    if running_wfs:
        logger.info("")
        logger.info("前5个RUNNING workflows:")
        for workflow_id, wf in running_wfs[:5]:
            logger.info(f"  {workflow_id}: status={wf.status.name}, current_state={wf.status_machine.current_status}")
            if wf.owner:
                agent = wf.owner
                logger.info(f"  Task queue length: {len(agent.task_queue)}")
    
    # 导出数据
    snapshots_file = output_path / 'agent_snapshots.csv'
    workflow_init_file = output_path / 'workflow_init_info.csv'
    
    snapshot_collector.export_snapshot_data(str(snapshots_file))
    
    # 将agent_workflow_mapping存到snapshot_collector
    snapshot_collector.workflow_init_info = agent_workflow_mapping
    snapshot_collector.export_workflow_init_data(str(workflow_init_file))
    
    logger.info("")
    logger.info(f"数据已导出到: {output_path}")
    logger.info(f"  - agent_snapshots.csv: {len(snapshot_collector.snapshots)} 条记录")
    logger.info(f"  - workflow_init_info.csv: {len(agent_workflow_mapping)} 条记录")
    logger.info("=" * 70)
    logger.info("=== 数据收集完成 ===")
    
    return {
        'total_agents': len(agents),
        'total_workflows': len(workflows),
        'completed': len(completed_wfs),
        'running': len(running_wfs),
        'failed': len(failed_wfs),
        'pending': len(pending_wfs),
        'snapshots': len(snapshot_collector.snapshots)
    }


def _validate_coordinates(config: dict, max_coord: float = 5000.0):
    """
    验证配置中的坐标是否在合理范围内
    
    Args:
        config: 场景配置字典
        max_coord: 最大允许坐标值
    
    Raises:
        ValueError: 如果发现坐标超出范围
    """
    logger.info("验证场景坐标范围...")
    
    all_coords = []
    errors = []
    
    # 检查agent位置
    for agent in config['agents']:
        pos = agent['initial_position']
        x, y = pos['x'], pos['y']
        all_coords.extend([x, y])
        if x < 0 or x > max_coord or y < 0 or y > max_coord:
            errors.append(f"Agent {agent['id']}: 位置({x:.2f}, {y:.2f})超出范围")
    
    # 检查workflow位置
    for wf in config['workflows']:
        props = wf['properties']
        wf_id = wf['id']
        
        # 物流workflow
        if 'pickup_location' in props:
            for loc_name, loc in [('pickup', props['pickup_location']), 
                                   ('delivery', props['delivery_location'])]:
                x, y = loc['x'], loc['y']
                all_coords.extend([x, y])
                if x < 0 or x > max_coord or y < 0 or y > max_coord:
                    errors.append(f"Workflow {wf_id} {loc_name}: 位置({x:.2f}, {y:.2f})超出范围")
        
        # 巡检workflow
        elif 'inspection_points' in props:
            for i, pt in enumerate(props['inspection_points']):
                x, y = pt['x'], pt['y']
                all_coords.extend([x, y])
                if x < 0 or x > max_coord or y < 0 or y > max_coord:
                    errors.append(f"Workflow {wf_id} 巡检点{i}: 位置({x:.2f}, {y:.2f})超出范围")
    
    if errors:
        logger.error(f"发现 {len(errors)} 个坐标超出范围的错误:")
        for err in errors[:10]:
            logger.error(f"  {err}")
        if len(errors) > 10:
            logger.error(f"  ... 还有 {len(errors)-10} 个错误")
        raise ValueError(f"场景配置中有坐标超出范围[0, {max_coord}]，请检查配置文件")
    
    if all_coords:
        min_coord, max_coord_actual = min(all_coords), max(all_coords)
        logger.info(f"✓ 坐标范围验证通过: {min_coord:.2f} ~ {max_coord_actual:.2f}")

def run_test_scenarios():
    """运行多个测试场景"""
    scenarios = [
        ('small_logistics_inspection', 1800, '小规模-物流巡检'),
        ('small_comm_compute', 1800, '小规模-通信计算'),
        ('medium_logistics_heavy', 2400, '中规模-物流主导'),
        ('medium_inspection_heavy', 2400, '中规模-巡检主导'),
        ('large_mixed', 3600, '大规模-全业务混合')
    ]
    results = {}
    
    config_dir = Path(__file__).parent.parent / 'config'
    
    for scenario_name, sim_time, desc in scenarios:
        logger.info("\n" + "="*70)
        logger.info(f"测试场景: {desc}")
        logger.info("="*70)
        
        # 生成场景配置
        scenario, config_path = generate_and_save_scenario(
            scenario_name,
            output_path=str(config_dir / f'test_{scenario_name}.yaml')
        )
        
        # 运行仿真
        output_dir = f'../data/test_{scenario_name}'
        result = run_data_collection(
            str(config_path),
            simulation_time=sim_time,
            output_dir=output_dir
        )
        
        results[scenario_name] = result
    
    # 汇总报告
    logger.info("\n" + "="*70)
    logger.info("多场景测试汇总报告")
    logger.info("="*70)
    
    for scenario_name, sim_time, desc in scenarios:
        result = results[scenario_name]
        logger.info(f"\n【{desc}】")
        logger.info(f"  Agents: {result['total_agents']}")
        logger.info(f"  Workflows: {result['total_workflows']}")
        logger.info(f"  仿真时长: {sim_time}秒")
        logger.info(f"  完成率: {result['completed']}/{result['total_workflows']} ({result['completed']/result['total_workflows']*100:.1f}%)")
        logger.info(f"  运行中: {result['running']}")
        logger.info(f"  失败: {result['failed']}")
        logger.info(f"  快照数: {result['snapshots']}")
    
    logger.info("\n" + "="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='动态数据收集')
    parser.add_argument('--scenario', 
                       choices=['test_all',
                               'small_logistics_inspection', 'small_comm_compute',
                               'medium_logistics_heavy', 'medium_inspection_heavy',
                               'large_mixed'],
                       default='small_logistics_inspection', help='场景类型')
    parser.add_argument('--time', type=int, default=1800, help='仿真时长（秒）')
    parser.add_argument('--config', help='自定义配置文件路径')
    parser.add_argument('--output', default='../data/dynamic', help='输出目录')
    
    args = parser.parse_args()
    
    if args.scenario == 'test_all':
        # 测试所有场景
        run_test_scenarios()
    else:
        # 单个场景
        if args.config:
            config_path = args.config
        else:
            # 先生成场景配置
            config_dir = Path(__file__).parent.parent / 'config'
            _, config_path = generate_and_save_scenario(
                args.scenario,
                output_path=str(config_dir / f'test_{args.scenario}.yaml')
            )
        
        run_data_collection(config_path, args.time, args.output)
