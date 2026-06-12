"""
多Seed数据生成和Triplet Learning Pipeline

工作流程：
1. 运行10次large_mixed仿真（不同random seed）
2. 对每次运行进行预处理和聚类，找到最佳threshold
3. 合并所有运行的数据
4. 使用合并数据训练Triplet Learning模型
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
import json
import subprocess
import time
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simulation(seed: int, output_base_dir: Path, config_dir: Path, scenario_name: str = 'large_mixed', simulation_time: int = 1800):
    """
    运行一次仿真
    
    Args:
        seed: 随机种子
        output_base_dir: 输出基目录
        config_dir: 配置文件目录
        scenario_name: 场景名称（如'large_mixed', 'small_logistics_inspection'等）
        simulation_time: 仿真时长（秒）
    
    Returns:
        output_dir: 该次运行的输出目录
    """
    logger.info("="*70)
    logger.info(f"运行仿真 - Seed {seed}")
    logger.info("="*70)
    
    # 创建输出目录
    output_dir = output_base_dir / f"run_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置random seed
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # 生成场景配置
    from scenario_generator import generate_and_save_scenario
    
    generation_config = config_dir / 'scenario_generation.yaml'
    uav_types_config = config_dir / 'uav_types.yaml'
    
    _, scenario_path = generate_and_save_scenario(
        uav_types_config=str(uav_types_config),
        generation_config=str(generation_config),
        output_path=str(output_dir / 'scenario.yaml'),
        scenario_name=scenario_name,  # 使用指定的场景配置
        seed=seed
    )
    
    logger.info(f"场景配置已生成: {scenario_path}")
    
    # 运行仿真
    from run_dynamic_collection import run_data_collection
    
    run_data_collection(
        scenario_config_path=scenario_path,
        simulation_time=simulation_time,
        output_dir=str(output_dir)
    )
    
    logger.info(f"仿真完成 - Seed {seed}, 输出目录: {output_dir}")
    
    return output_dir


def preprocess_and_cluster(run_dir: Path, thresholds: list = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                          preprocess_config: Path = None):
    """
    对单次运行进行预处理和聚类，找到最佳threshold
    
    Args:
        run_dir: 运行目录
        thresholds: 要测试的阈值列表
        preprocess_config: 预处理配置文件路径
    
    Returns:
        best_result: 最佳聚类结果
    """
    logger.info("="*70)
    logger.info(f"预处理和聚类: {run_dir.name}")
    logger.info("="*70)
    
    # 执行预处理
    logger.info("Step 1: 预处理workflow和checkpoint数据")
    
    # 确保模块路径正确
    import sys
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from process_task_spatial_temporal import TaskSpatialTemporalProcessor
    
    # 直接处理当前目录
    input_file = run_dir / 'agent_snapshots.csv'
    output_file = run_dir / 'processed_task_spatial_temporal_coupling.csv'
    
    if not input_file.exists():
        raise FileNotFoundError(f"未找到agent_snapshots.csv: {input_file}")
    
    # 加载预处理配置
    if preprocess_config and preprocess_config.exists():
        import yaml
        with open(preprocess_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        time_granularity = config.get('time_granularity', 1.0)
        clustering_cfg = config.get('clustering', {})
        enable_clustering = clustering_cfg.get('enable_clustering', True)
        clustering_method = clustering_cfg.get('method', 'kmeans')
        distance_threshold = clustering_cfg.get('distance_threshold', 100.0)
        time_window = clustering_cfg.get('time_window', None)
        grid_size = clustering_cfg.get('grid_size', 100.0)
        
        logger.info(f"  使用配置文件: {preprocess_config}")
        logger.info(f"    - time_granularity: {time_granularity}s")
        logger.info(f"    - clustering_method: {clustering_method}")
        logger.info(f"    - distance_threshold: {distance_threshold}m")
    else:
        # 使用默认参数
        time_granularity = 1.0
        enable_clustering = True
        clustering_method = 'kmeans'
        distance_threshold = 100.0
        time_window = None
        grid_size = 100.0
        logger.info("  使用默认预处理参数")
    
    processor = TaskSpatialTemporalProcessor(
        time_granularity=time_granularity,
        enable_clustering=enable_clustering,
        clustering_method=clustering_method,
        distance_threshold=distance_threshold,
        time_window=time_window,
        grid_size=grid_size
    )
    processor.process_file(input_file, output_file)
    logger.info(f"  预处理完成: {output_file}")
    
    # 测试不同threshold找最佳
    logger.info(f"\nStep 2: 测试不同threshold找最佳聚类 ({len(thresholds)}个阈值)")
    
    # 确保模块路径正确
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from workflow_checkpoint_clustering import process_test_directory
    
    best_threshold = None
    best_synergy_ratio = -float('inf')
    best_clustering_file = None
    
    results = []
    
    for th in thresholds:
        logger.info(f"  测试 threshold = {th}")
        
        # 进行聚类
        process_test_directory(
            test_dir=run_dir,
            cohesion_threshold=th,
            save_with_suffix=True
        )
        
        clustering_file = run_dir / f'workflow_checkpoint_clustering_th{th:.2f}.csv'
        
        # 计算synergy
        # 确保模块路径正确
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        from compute_synergy_metric import SynergyMetricCalculator
        
        calculator = SynergyMetricCalculator(data_dir=run_dir, threshold=th)
        synergy_result = calculator.evaluate_clustering()
        
        # 计算Sum Synergy Ratio
        sum_synergy_ratio = sum(
            cluster['synergy_ratio'] 
            for cluster in synergy_result.get('cluster_scores', [])
        )
        
        results.append({
            'threshold': th,
            'sum_synergy_ratio': sum_synergy_ratio,
            'n_clusters': synergy_result.get('num_clusters', 0),
            'clustering_file': str(clustering_file)
        })
        
        logger.info(f"    Sum Synergy Ratio: {sum_synergy_ratio:.4f}, Clusters: {synergy_result.get('num_clusters', 0)}")
        
        if sum_synergy_ratio > best_synergy_ratio:
            best_synergy_ratio = sum_synergy_ratio
            best_threshold = th
            best_clustering_file = clustering_file
    
    # 保存所有threshold结果
    results_file = run_dir / 'threshold_comparison.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best_threshold': best_threshold,
            'best_synergy_ratio': best_synergy_ratio,
            'best_clustering_file': str(best_clustering_file)
        }, f, indent=2)
    
    logger.info(f"\n最佳配置:")
    logger.info(f"  Threshold: {best_threshold}")
    logger.info(f"  Sum Synergy Ratio: {best_synergy_ratio:.4f}")
    logger.info(f"  聚类文件: {best_clustering_file}")
    
    # 复制最佳聚类结果为默认文件
    import shutil
    default_clustering = run_dir / 'workflow_checkpoint_clustering.csv'
    shutil.copy(best_clustering_file, default_clustering)
    logger.info(f"  已复制到: {default_clustering}")
    
    return {
        'run_dir': str(run_dir),
        'best_threshold': best_threshold,
        'best_synergy_ratio': best_synergy_ratio,
        'best_clustering_file': str(default_clustering),
        'all_results': results
    }


def merge_datasets(run_results: list, output_dir: Path):
    """
    合并多次运行的数据集
    
    Args:
        run_results: 每次运行的结果列表
        output_dir: 合并数据输出目录
    
    Returns:
        merged_data_dir: 合并后的数据目录
    """
    logger.info("="*70)
    logger.info("合并数据集")
    logger.info("="*70)
    
    merged_dir = output_dir / 'merged_dataset'
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    import shutil
    
    # 合并所有运行的数据
    all_checkpoints = []
    all_workflows = []
    all_snapshots = []  # 合并所有agent_snapshots
    all_agent_mappings = []  # 合并所有agent映射
    is_first_run = True  # 标记是否是第一个run
    
    for result in run_results:
        run_dir = Path(result['run_dir'])
        # 提取真实的seed作为run_id（从run_seed_42 → 42）
        seed_str = run_dir.name.split('_')[-1]
        run_id = int(seed_str)
        
        logger.info(f"合并 Run {run_id} (seed {seed_str}): {run_dir.name}")
        
        # 读取checkpoint clustering结果
        clustering_file = Path(result['best_clustering_file'])
        if clustering_file.exists():
            df = pd.read_csv(clustering_file)
            # 添加run_id标识
            df['source_run_id'] = run_id
            df['source_seed'] = run_dir.name.split('_')[-1]
            # 调整workflow_id以匹配合并后的workflow (需要与workflow合并保持一致)
            df['original_workflow_id'] = df['workflow_id']
            df['workflow_id'] = df['workflow_id'].astype(str) + f'_run{run_id}'
            # 调整cluster_group以避免冲突（每个run的cluster独立编号）
            df['original_cluster'] = df['cluster_group']
            df['cluster_group'] = df['cluster_group'].astype(str) + f'_run{run_id}'
            
            all_checkpoints.append(df)
            
            logger.info(f"  Checkpoints: {len(df)}, Clusters: {df['original_cluster'].nunique()}, Best Threshold: {result['best_threshold']}")
        
        # 合并每次运行的workflow信息
        workflow_file = run_dir / 'workflow_init_info.csv'
        if workflow_file.exists():
            wf_df = pd.read_csv(workflow_file)
            # 添加run_id标识，并调整workflow_id避免冲突
            wf_df['source_run_id'] = run_id
            wf_df['source_seed'] = run_dir.name.split('_')[-1]
            wf_df['original_workflow_id'] = wf_df['workflow_id']
            wf_df['workflow_id'] = wf_df['workflow_id'].astype(str) + f'_run{run_id}'
            
            all_workflows.append(wf_df)
            logger.info(f"  Workflows: {len(wf_df)}")
        
        # 合并agent_snapshots.csv
        snapshot_file = run_dir / 'agent_snapshots.csv'
        if snapshot_file.exists():
            snap_df = pd.read_csv(snapshot_file)
            snap_df['source_run_id'] = run_id
            snap_df['source_seed'] = run_dir.name.split('_')[-1]
            all_snapshots.append(snap_df)
            logger.info(f"  Snapshots: {len(snap_df)}")
        
        # 合并agent_id_mapping.csv
        mapping_file = run_dir / 'agent_id_mapping.csv'
        if mapping_file.exists():
            mapping_df = pd.read_csv(mapping_file)
            mapping_df['source_run_id'] = run_id
            mapping_df['source_seed'] = run_dir.name.split('_')[-1]
            all_agent_mappings.append(mapping_df)
            logger.info(f"  Agent映射: {len(mapping_df)}")
        
        # 对于第一次运行，复制workflows目录和processed文件
        if is_first_run:
            # 复制workflows目录（包含workflow日志等）
            workflows_dir = run_dir / 'workflows'
            if workflows_dir.exists():
                dest_workflows = merged_dir / 'workflows'
                if dest_workflows.exists():
                    shutil.rmtree(dest_workflows)
                shutil.copytree(workflows_dir, dest_workflows)
                logger.info(f"  复制目录: workflows/ (仅第一次)")
            
            # 复制processed文件（仅第一次，用于参考）
            processed_file = run_dir / 'processed_task_spatial_temporal_coupling.csv'
            if processed_file.exists():
                dest_file = merged_dir / 'processed_task_spatial_temporal_coupling.csv'
                shutil.copy(processed_file, dest_file)
                logger.info(f"  复制文件: processed_task_spatial_temporal_coupling.csv (仅第一次)")
        
        is_first_run = False  # 后续循环不再是第一次
    
    # 合并所有checkpoints
    if all_checkpoints:
        merged_checkpoints = pd.concat(all_checkpoints, ignore_index=True)
        
        # 保存合并后的checkpoint数据
        merged_file = merged_dir / 'workflow_checkpoint_clustering.csv'
        merged_checkpoints.to_csv(merged_file, index=False)
        
        logger.info(f"\n合并完成:")
        logger.info(f"  总Checkpoints: {len(merged_checkpoints)}")
        logger.info(f"  总Runs: {len(run_results)}")
        logger.info(f"  总Clusters: {merged_checkpoints['cluster_group'].nunique()}")
        logger.info(f"  保存到: {merged_file}")
    
    # 合并所有workflows
    if all_workflows:
        merged_workflows = pd.concat(all_workflows, ignore_index=True)
        workflow_file = merged_dir / 'workflow_init_info.csv'
        merged_workflows.to_csv(workflow_file, index=False)
        logger.info(f"  总Workflows: {len(merged_workflows)}")
        logger.info(f"  保存到: {workflow_file}")
    
    # 合并所有agent_snapshots
    if all_snapshots:
        merged_snapshots = pd.concat(all_snapshots, ignore_index=True)
        snapshot_file = merged_dir / 'agent_snapshots.csv'
        merged_snapshots.to_csv(snapshot_file, index=False)
        logger.info(f"  总Snapshots: {len(merged_snapshots)}")
        logger.info(f"  保存到: {snapshot_file}")
    
    # 合并所有agent映射
    if all_agent_mappings:
        merged_mappings = pd.concat(all_agent_mappings, ignore_index=True)
        mapping_file = merged_dir / 'agent_id_mapping.csv'
        merged_mappings.to_csv(mapping_file, index=False)
        logger.info(f"  总Agent映射: {len(merged_mappings)}")
        logger.info(f"  保存到: {mapping_file}")
    
    # 保存合并元数据
    if all_checkpoints:
        metadata = {
            'n_runs': len(run_results),
            'total_checkpoints': len(merged_checkpoints),
            'total_workflows': len(merged_workflows) if all_workflows else 0,
            'total_snapshots': len(merged_snapshots) if all_snapshots else 0,
            'total_clusters': int(merged_checkpoints['cluster_group'].nunique()),
            'run_summaries': [
                {
                    'run_id': int(Path(result['run_dir']).name.split('_')[-1]),  # 使用真实seed
                    'seed': result['run_dir'].split('_')[-1],
                    'best_threshold': result['best_threshold'],
                    'best_synergy_ratio': result['best_synergy_ratio'],
                    'n_checkpoints': len(all_checkpoints[i])
                }
                for i, result in enumerate(run_results)
            ],
            'merged_file': str(merged_file),
            'creation_time': datetime.now().isoformat()
        }
        
        metadata_file = merged_dir / 'merge_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  元数据保存到: {metadata_file}")
    
    return merged_dir


def train_triplet_model(merged_data_dir: Path, output_dir: Path, config: dict):
    """
    在合并的数据集上训练Triplet Learning模型
    
    Args:
        merged_data_dir: 合并数据目录
        output_dir: 训练输出目录
        config: 训练配置
    """
    logger.info("="*70)
    logger.info("训练Triplet Learning模型")
    logger.info("="*70)
    
    # 读取传入的train/val runs（透传给训练脚本）
    train_runs_arg = config.get('train_runs', None)
    val_runs_arg = config.get('val_runs', None)
    if train_runs_arg is not None:
        logger.info(f"训练Runs(传入): {train_runs_arg}")
    if val_runs_arg is not None:
        logger.info(f"验证Runs(传入): {val_runs_arg}")
    
    # 准备训练脚本参数
    cmd = [
        'python', 'ml_training/train_triplet_learning.py',
        '--data-dir', str(merged_data_dir),
        '--save-dir', str(output_dir / 'triplet_learning_results'),
        # 透传train/val runs到训练脚本（若提供）
        *(['--train-runs', train_runs_arg] if train_runs_arg else []),
        *(['--val-runs', val_runs_arg] if val_runs_arg else []),
        '--epochs', str(config.get('epochs', 100)),
        '--n-triplets', str(config.get('n_triplets', 20000)),
        '--batch-size', str(config.get('batch_size', 128)),
        '--lr', str(config.get('lr', 0.001)),
        '--margin', str(config.get('margin', 0.5)),
        '--threshold-high', str(config.get('threshold_high', 0.2)),
        '--threshold-low', str(config.get('threshold_low', 0.05)),
        '--triplet-strategy', config.get('triplet_strategy', 'hard'),
        '--eval-every', str(config.get('eval_every', 5)),
        '--early-stopping', str(config.get('early_stopping', 20)),
        '--use-cache'
    ]
    
    # 添加模型架构参数
    hidden_dims = config.get('hidden_dims', [64, 128, 64])
    cmd.extend(['--hidden-dims'] + [str(d) for d in hidden_dims])
    cmd.extend(['--embed-dim', str(config.get('embed_dim', 32))])
    
    logger.info(f"运行命令: {' '.join(cmd)}")
    
    # 运行训练
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode == 0:
        logger.info("Triplet Learning训练完成！")
    else:
        logger.error(f"训练失败，返回码: {result.returncode}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='多Seed数据生成和Triplet Learning Pipeline')
    
    # Pipeline控制
    parser.add_argument('--n-seeds', type=int, default=10,
                       help='运行次数（不同random seed）')
    parser.add_argument('--start-seed', type=int, default=42,
                       help='起始seed')
    parser.add_argument('--skip-simulation', action='store_true',
                       help='跳过仿真步骤（使用已有数据）')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='跳过聚类步骤（使用已有聚类结果）')
    parser.add_argument('--only-clustering', action='store_true',
                       help='只执行预处理和聚类步骤（不合并和训练）')
    parser.add_argument('--only-merge', action='store_true',
                       help='只执行合并步骤')
    parser.add_argument('--only-train', action='store_true',
                       help='只执行训练步骤')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练步骤（只收集和处理数据）')
    
    # 路径配置
    parser.add_argument('--output-dir', type=str,
                       default='data/multi_seed_runs',
                       help='输出基目录')
    parser.add_argument('--config-dir', type=str,
                       default='config',
                       help='配置文件目录')
    parser.add_argument('--scenario-name', type=str,
                       default='large_mixed',
                       choices=['small_logistics_inspection', 'small_comm_compute',
                               'medium_logistics_heavy', 'medium_inspection_heavy',
                               'large_mixed'],
                       help='场景名称')
    
    # 仿真参数
    parser.add_argument('--simulation-time', type=int, default=1800,
                       help='每次仿真时长（秒）')
    
    # 聚类参数
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                       help='测试的threshold列表')
    parser.add_argument('--preprocess-config', type=str,
                       default='config/preprocessing.yaml',
                       help='预处理配置文件路径')
    
    # Triplet训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--n-triplets', type=int, default=20000,
                       help='Triplet样本数量')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='Triplet Loss margin')
    parser.add_argument('--triplet-strategy', type=str, default='hard',
                       choices=['random', 'hard', 'semi-hard'],
                       help='Triplet采样策略')
    parser.add_argument('--eval-every', type=int, default=5,
                       help='每N个epoch评估一次')
    parser.add_argument('--early-stopping', type=int, default=20,
                       help='早停patience')
    parser.add_argument('--train-runs', type=str, default='42,43,44,45,46,47,48',
                       help='训练run IDs（逗号分隔）')
    parser.add_argument('--val-runs', type=str, default='49,50',
                       help='验证run IDs（逗号分隔）')
    
    # 模型架构参数
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                       default=[64, 128, 64],
                       help='MLP隐藏层维度')
    parser.add_argument('--embed-dim', type=int, default=32,
                       help='Embedding维度')
    
    args = parser.parse_args()
    
    # 转换路径
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir
    
    # 在输出目录下创建scenario子目录
    output_dir = output_dir / args.scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = Path(__file__).parent.parent / config_dir
    
    preprocess_config = Path(args.preprocess_config)
    if not preprocess_config.is_absolute():
        preprocess_config = Path(__file__).parent.parent / preprocess_config
    
    logger.info("="*70)
    logger.info("多Seed数据生成和Triplet Learning Pipeline")
    logger.info("="*70)
    logger.info(f"场景名称: {args.scenario_name}")
    logger.info(f"运行次数: {args.n_seeds}")
    logger.info(f"起始Seed: {args.start_seed}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*70)
    
    # 记录开始时间
    start_time = time.time()
    
    # ========================================
    # 阶段1: 运行多次仿真
    # ========================================
    if not args.only_merge and not args.only_train and not args.only_clustering and not args.skip_simulation:
        logger.info("\n【阶段1】运行多次仿真")
        logger.info("-"*70)
        
        for i in range(args.n_seeds):
            seed = args.start_seed + i
            try:
                run_dir = run_simulation(
                    seed=seed,
                    output_base_dir=output_dir,
                    config_dir=config_dir,
                    scenario_name=args.scenario_name,
                    simulation_time=args.simulation_time
                )
                logger.info(f"✅ Seed {seed} 完成\n")
            except Exception as e:
                logger.error(f"❌ Seed {seed} 失败: {e}\n")
                continue
    
    # ========================================
    # 阶段2: 预处理和聚类
    # ========================================
    if not args.only_merge and not args.only_train and not args.skip_clustering:
        logger.info("\n【阶段2】预处理和聚类")
        logger.info("-"*70)
        
        run_results = []
        
        for i in range(args.n_seeds):
            seed = args.start_seed + i
            run_dir = output_dir / f"run_seed_{seed}"
            
            if not run_dir.exists():
                logger.warning(f"跳过 Seed {seed}: 目录不存在")
                continue
            
            try:
                result = preprocess_and_cluster(
                    run_dir=run_dir,
                    thresholds=args.thresholds,
                    preprocess_config=preprocess_config
                )
                run_results.append(result)
                logger.info(f"✅ Seed {seed} 预处理完成\n")
            except Exception as e:
                logger.error(f"❌ Seed {seed} 预处理失败: {e}\n")
                continue
        
        # 保存所有运行结果汇总
        summary_file = output_dir / 'all_runs_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'n_runs': len(run_results),
                'runs': run_results,
                'creation_time': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"所有运行汇总保存到: {summary_file}")
    
    # 如果只执行聚类，到此结束
    if args.only_clustering:
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("预处理和聚类完成")
        logger.info("="*70)
        logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"  - 各次运行: {output_dir}/run_seed_*")
        logger.info(f"  - 运行汇总: {output_dir}/all_runs_summary.json")
        logger.info("="*70)
        return
    
    # ========================================
    # 阶段3: 合并数据集
    # ========================================
    if not args.only_train:
        logger.info("\n【阶段3】合并数据集")
        logger.info("-"*70)
        
        # 如果没有运行前面的步骤，从文件加载结果
        if args.only_merge or args.skip_clustering:
            summary_file = output_dir / 'all_runs_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    run_results = data['runs']
                logger.info(f"从汇总文件加载了 {len(run_results)} 次运行结果")
            else:
                # 如果汇总文件不存在，尝试从各个运行目录自动构建
                logger.warning("找不到运行汇总文件，尝试从运行目录自动构建...")
                run_results = []
                
                for i in range(args.n_seeds):
                    seed = args.start_seed + i
                    run_dir = output_dir / f"run_seed_{seed}"
                    
                    if not run_dir.exists():
                        logger.warning(f"  跳过 Seed {seed}: 目录不存在")
                        continue
                    
                    # 查找最佳聚类文件（默认的或threshold_comparison.json）
                    default_clustering = run_dir / 'workflow_checkpoint_clustering.csv'
                    threshold_comparison = run_dir / 'threshold_comparison.json'
                    
                    if default_clustering.exists():
                        # 如果有threshold_comparison.json，读取最佳配置
                        with open(threshold_comparison, 'r') as f:
                            comp_data = json.load(f)
                            best_threshold = comp_data.get('best_threshold')
                            best_synergy_ratio = comp_data.get('best_synergy_ratio')
                        
                        run_results.append({
                            'run_dir': str(run_dir),
                            'best_threshold': best_threshold,
                            'best_synergy_ratio': best_synergy_ratio,
                            'best_clustering_file': str(default_clustering)
                        })
                        logger.info(f"  ✓ Seed {seed}: threshold={best_threshold}, synergy={best_synergy_ratio:.4f}")
                    else:
                        logger.warning(f"  跳过 Seed {seed}: 找不到聚类文件")
                
                if not run_results:
                    logger.error("没有找到任何可用的运行结果，无法合并")
                    return
                
                logger.info(f"成功构建了 {len(run_results)} 次运行结果")
        
        try:
            merged_dir = merge_datasets(
                run_results=run_results,
                output_dir=output_dir
            )
            logger.info(f"✅ 数据集合并完成: {merged_dir}\n")
        except Exception as e:
            logger.error(f"❌ 数据集合并失败: {e}\n")
            return
    
    # 如果只执行合并，到此结束
    if args.only_merge or args.skip_train:
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("数据收集和处理完成")
        logger.info("="*70)
        logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"  - 合并数据集: {output_dir}/merged_dataset")
        logger.info("="*70)
        return
    
    # ========================================
    # 阶段4: Triplet Learning训练
    # ========================================
    logger.info("\n【阶段4】Triplet Learning训练")
    logger.info("-"*70)
    
    merged_dir = output_dir / 'merged_dataset'
    
    if not merged_dir.exists():
        logger.error("合并数据集不存在，无法训练")
        return
    
    training_config = {
        'epochs': args.epochs,
        'n_triplets': args.n_triplets,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'margin': args.margin,
        'threshold_high': 0.2,  # 降低以覆盖更多中高协同的样本对
        'threshold_low': 0.05,
        'triplet_strategy': args.triplet_strategy,
        'eval_every': args.eval_every,
        'early_stopping': args.early_stopping,
        'hidden_dims': args.hidden_dims,
        'embed_dim': args.embed_dim,
        'start_seed': args.start_seed,
        'n_seeds': args.n_seeds,
        'train_runs': args.train_runs,
        'val_runs': args.val_runs,
    }
    
    try:
        success = train_triplet_model(
            merged_data_dir=merged_dir,
            output_dir=output_dir,
            config=training_config
        )
        
        if success:
            logger.info("✅ Triplet Learning训练完成\n")
        else:
            logger.error("❌ Triplet Learning训练失败\n")
    except Exception as e:
        logger.error(f"❌ 训练过程出错: {e}\n")
    
    # ========================================
    # 总结
    # ========================================
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("Pipeline完成")
    logger.info("="*70)
    logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"  - 各次运行: {output_dir}/run_seed_*")
    logger.info(f"  - 合并数据集: {output_dir}/merged_dataset")
    logger.info(f"  - 训练结果: {output_dir}/triplet_learning_results")
    logger.info("="*70)


if __name__ == '__main__':
    main()
