"""
Synergy评估模块 - 计算聚类结果的协同增效指标

将聚类结果转换为compute_synergy_metric.py可用的格式，
然后计算synergy ratio并与oracle对比
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import tempfile
import shutil

# 添加src目录到路径
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from compute_synergy_metric import SynergyMetricCalculator

logger = logging.getLogger(__name__)


class CheckpointSynergyEvaluator:
    """Checkpoint聚类结果的Synergy评估器"""
    
    def __init__(self, data_dir: Path):
        """
        初始化评估器
        
        Args:
            data_dir: 数据目录（包含运行时日志等）
        """
        self.data_dir = Path(data_dir)
        
        # 加载oracle聚类结果（用于对比）
        self.oracle_clustering_df = None
        self.oracle_synergy_result = None
        self._load_oracle_data()
    
    def _load_oracle_data(self):
        """加载oracle聚类结果"""
        oracle_file = self.data_dir / 'workflow_checkpoint_clustering.csv'
        if oracle_file.exists():
            self.oracle_clustering_df = pd.read_csv(oracle_file)
            logger.info(f"  加载oracle聚类: {len(self.oracle_clustering_df)} checkpoints")
        else:
            logger.warning(f"  Oracle聚类文件不存在: {oracle_file}")
    
    def create_clustering_result_file(
        self,
        checkpoint_df: pd.DataFrame,
        labels: np.ndarray,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        创建聚类结果文件（格式与workflow_checkpoint_clustering.csv兼容）
        
        Args:
            checkpoint_df: checkpoint DataFrame（从extractor提取的）
            labels: 聚类标签数组
            output_file: 输出文件路径（如果为None，使用临时文件）
            
        Returns:
            聚类结果文件路径
        """
        # 合并oracle的运行时信息
        if self.oracle_clustering_df is None:
            raise ValueError("Oracle聚类数据未加载，无法创建完整的聚类结果文件")
        
        # 创建结果DataFrame
        # ⚠️ 关键：必须包含original_workflow_id列（如果存在）
        columns_to_include = ['workflow_id', 'checkpoint_idx']
        if 'original_workflow_id' in checkpoint_df.columns:
            columns_to_include.append('original_workflow_id')
        
        result_df = checkpoint_df[columns_to_include].copy()
        result_df['cluster_group'] = labels
        
        # 从oracle中获取运行时信息（start_time, end_time等）
        # 通过workflow_id和checkpoint_idx匹配
        oracle_info = self.oracle_clustering_df[[
            'workflow_id', 'checkpoint_idx',
            'start_time', 'end_time', 'duration',
            'num_tasks', 'spatial_footprint_size', 'task_types',
            'avg_position_x', 'avg_position_y', 'avg_position_z',
            'unique_cells'
        ]].copy()
        
        # 合并
        result_df = result_df.merge(
            oracle_info,
            on=['workflow_id', 'checkpoint_idx'],
            how='left'
        )
        
        # 检查未匹配的checkpoints
        unmatched = result_df[result_df['start_time'].isna()]
        if len(unmatched) > 0:
            logger.warning(f"  ⚠️ {len(unmatched)}/{len(result_df)} checkpoints无法匹配到Oracle运行时信息")
            logger.warning(f"  这些checkpoints将被排除在synergy计算之外")
            
            # 只保留成功匹配的checkpoints
            result_df = result_df[result_df['start_time'].notna()].copy()
            logger.info(f"  保留 {len(result_df)} 个有效checkpoints用于synergy计算")
        
        # 保存到文件
        if output_file is None:
            # 使用临时文件
            temp_dir = Path(tempfile.mkdtemp())
            output_file = temp_dir / 'workflow_checkpoint_clustering.csv'
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        
        logger.info(f"  创建聚类结果文件: {output_file}")
        logger.info(f"    {len(result_df)} checkpoints, {len(set(labels))} 个簇")
        
        return output_file
    
    def evaluate_clustering_by_run(
        self,
        checkpoint_df: pd.DataFrame,
        labels: np.ndarray,
        save_result: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        按Run分组评估聚类结果
        
        Args:
            checkpoint_df: checkpoint DataFrame（必须包含source_run_id列）
            labels: 聚类标签
            save_result: 是否保存结果
            verbose: 是否输出详细日志
            
        Returns:
            汇总的synergy指标
        """
        if verbose:
            logger.info("\n" + "="*70)
            logger.info("按Run分组评估聚类结果")
            logger.info("="*70)
        
        # 检查必须有source_run_id列
        if 'source_run_id' not in checkpoint_df.columns:
            raise ValueError("checkpoint_df必须包含source_run_id列")
        
        # 按run分组
        run_ids = checkpoint_df['source_run_id'].unique()
        if verbose:
            logger.info(f"  发现 {len(run_ids)} 个runs")
        
        all_results = []
        all_cluster_scores = []
        total_sum_synergy = 0.0
        total_clusters = 0
        
        for run_id in sorted(run_ids):
            if verbose:
                logger.info(f"\n  评估 Run {run_id}...")
            
            # 提取该run的数据
            run_mask = checkpoint_df['source_run_id'] == run_id
            run_checkpoint_df = checkpoint_df[run_mask].copy()
            run_labels = labels[run_mask]
            
            # ⚠️ 关键修复：为每个run找到对应的数据文件夹
            # 跨场景模式：同一个source_run_id包含来自多个scenario的checkpoints
            # 所以必须使用all_scenarios_merged目录（包含所有scenario的合并agent数据）
            run_data_dir = None
            if self.data_dir.name == 'all_scenarios_merged':
                # 跨场景模式：使用合并数据目录，不使用单个scenario的run目录
                run_data_dir = self.data_dir  # 使用all_scenarios_merged
            else:
                # 单场景模式：使用该scenario的run目录
                run_data_dir = self.data_dir.parent / f'run_seed_{run_id}'
                if not run_data_dir.exists():
                    run_data_dir = None
                    logger.warning(f"  Run {run_id} 的数据目录不存在，使用merged数据")
            
            # 评估该run（静默模式），传入run的数据目录
            run_result = self.evaluate_clustering(
                run_checkpoint_df,
                run_labels,
                save_result=False,
                verbose=verbose,
                run_data_dir=run_data_dir  # 传入run的数据目录
            )
            
            all_results.append({
                'run_id': int(run_id),
                'sum_synergy_ratio': run_result['sum_synergy_ratio'],
                'n_clusters': run_result['n_clusters'],
                'n_checkpoints': len(run_checkpoint_df)
            })
            
            total_sum_synergy += run_result['sum_synergy_ratio']
            total_clusters += run_result['n_clusters']
            
            # 收集cluster scores（添加run_id标识）
            for score in run_result.get('cluster_scores', []):
                score['run_id'] = int(run_id)
                all_cluster_scores.append(score)
            
            if verbose:
                logger.info(f"    Sum Synergy Ratio: {run_result['sum_synergy_ratio']:.4f}")
                logger.info(f"    Clusters: {run_result['n_clusters']}")
        
        # 汇总结果
        avg_sum_synergy = total_sum_synergy / len(run_ids)
        
        summary = {
            'n_runs': len(run_ids),
            'avg_sum_synergy_ratio': avg_sum_synergy,
            'total_sum_synergy_ratio': total_sum_synergy,
            'total_clusters': total_clusters,
            'run_results': all_results,
            'cluster_scores': all_cluster_scores
        }
        
        if verbose:
            logger.info("\n" + "="*70)
            logger.info("汇总结果")
            logger.info("="*70)
            logger.info(f"  平均Sum Synergy Ratio: {avg_sum_synergy:.4f}")
            logger.info(f"  总Sum Synergy Ratio: {total_sum_synergy:.4f}")
            logger.info(f"  总Clusters: {total_clusters}")
        else:
            # 简洁输出
            logger.info(f"  Sum Synergy Ratio: {total_sum_synergy:.4f} ({len(run_ids)} runs, {total_clusters} clusters)")
        
        if save_result:
            output_dir = self.data_dir / 'ml_training_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = output_dir / 'clustering_synergy_by_run.json'
            with open(result_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"\n结果已保存到: {result_file}")
        
        return summary
    
    def evaluate_clustering(
        self,
        checkpoint_df: pd.DataFrame,
        labels: np.ndarray,
        save_result: bool = True,
        verbose: bool = True,
        run_data_dir: Path = None
    ) -> Dict:
        """
        评估聚类结果的synergy指标
        
        Args:
            checkpoint_df: checkpoint DataFrame
            labels: 聚类标签
            save_result: 是否保存结果文件
            verbose: 是否输出详细日志
            run_data_dir: 该run的数据目录（如果为None，使用self.data_dir）
            
        Returns:
            包含synergy指标的字典
        """
        if verbose:
            logger.info("\n" + "="*70)
            logger.info("评估聚类结果的Synergy指标")
            logger.info("="*70)
        
        # 创建临时数据目录
        temp_dir = Path(tempfile.mkdtemp())
        temp_data_dir = temp_dir / 'eval_data'
        temp_data_dir.mkdir(parents=True)
        
        # ⚠️ 使用run的数据目录（如果提供），否则使用merged数据
        source_data_dir = run_data_dir if run_data_dir is not None else self.data_dir
        
        try:
            # 复制必要的数据文件到临时目录
            required_files = [
                'processed_task_spatial_temporal_coupling.csv',
                'workflow_init_info.csv',
                'agent_snapshots.csv',
                'agent_id_mapping.csv',  # ⚠️ 关键：必须包含agent映射文件
                'scenario.yaml',
                'merge_metadata.json'  # 跨场景模式需要此文件
            ]
            
            for filename in required_files:
                src_file = source_data_dir / filename
                if src_file.exists():
                    shutil.copy(src_file, temp_data_dir / filename)
                elif filename == 'agent_id_mapping.csv' and source_data_dir != self.data_dir:
                    # 备用路径：如果是从run_data_dir复制但缺少agent_id_mapping.csv，
                    # 尝试从merged_dataset复制（合并数据的映射文件在merged目录下）
                    alternate_file = self.data_dir / filename
                    if alternate_file.exists():
                        shutil.copy(alternate_file, temp_data_dir / filename)
                        if verbose:
                            logger.info(f"  从merged_dataset备用复制: {filename}")
                    else:
                        raise FileNotFoundError(f"找不到{filename}，既不在{source_data_dir}也不在{self.data_dir}")
                elif verbose:
                    logger.warning(f"  缺少文件: {filename} (from {source_data_dir})")
            
            # 创建聚类结果（删除verbose的鲁棒性分支，统一处理）
            columns_to_include = ['workflow_id', 'checkpoint_idx']
            if 'original_workflow_id' in checkpoint_df.columns:
                columns_to_include.append('original_workflow_id')
            if 'source_run_id' in checkpoint_df.columns:
                columns_to_include.append('source_run_id')  # 跨场景检测需要此列
            
            result_df = checkpoint_df[columns_to_include].copy()
            result_df['cluster_group'] = labels
            
            # 从oracle中获取运行时信息（只选择存在的列，删除鲁棒性）
            oracle_cols = [
                'workflow_id', 'checkpoint_idx',
                'start_time', 'end_time', 'duration',
                'num_tasks', 'spatial_footprint_size', 'task_types',
                'avg_position_x', 'avg_position_y', 'avg_position_z',
                'unique_cells'
            ]
            # original_workflow_id可能不存在
            if 'original_workflow_id' in self.oracle_clustering_df.columns:
                oracle_cols.append('original_workflow_id')
            
            oracle_info = self.oracle_clustering_df[oracle_cols].copy()
            result_df = result_df.merge(oracle_info, on=['workflow_id', 'checkpoint_idx'], how='left')
            # 只保留成功匹配的
            result_df = result_df[result_df['start_time'].notna()].copy()
            
            result_df.to_csv(temp_data_dir / 'workflow_checkpoint_clustering.csv', index=False)
            
            if verbose:
                logger.info(f"  创建聚类结果文件: {temp_data_dir / 'workflow_checkpoint_clustering.csv'}")
                logger.info(f"    {len(result_df)} checkpoints, {len(set(labels))} 个簇")
            
            # 使用SynergyMetricCalculator计算
            if verbose:
                logger.info("\n计算Synergy指标...")
            
            # 临时关闭compute_synergy_metric的日志（如果不是verbose模式）
            if not verbose:
                import logging
                synergy_logger = logging.getLogger('compute_synergy_metric')
                original_level = synergy_logger.level
                synergy_logger.setLevel(logging.CRITICAL)  # 完全禁用所有日志
            
            # 直接传入result_df，避免文件I/O和数据不一致
            calculator = SynergyMetricCalculator(temp_data_dir, clustering_df=result_df)
            
            result = calculator.evaluate_clustering(output_file=None)
            
            # 恢复日志级别
            if not verbose:
                synergy_logger.setLevel(original_level)
            
            # 计算Sum Synergy Ratio（所有簇的synergy_ratio之和）
            cluster_scores = result.get('cluster_scores', [])
            sum_synergy_ratio = sum(score['synergy_ratio'] for score in cluster_scores)
            result['sum_synergy_ratio'] = sum_synergy_ratio
            result['n_clusters'] = len(cluster_scores)
            
            # 如果需要保存结果
            if save_result:
                output_dir = self.data_dir / 'ml_training_results'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                result_file = output_dir / 'clustering_synergy_result.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"\n结果已保存到: {result_file}")
            
            return result
            
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def compare_with_oracle(self, predicted_result: Dict, oracle_result: Dict = None) -> Dict:
        """
        对比预测聚类结果与Oracle（ground truth）
        
        Args:
            predicted_result: 预测的聚类结果（来自evaluate_clustering_by_run）
            oracle_result: Oracle的聚类结果（可选，如果不提供则从metadata读取）
            
        Returns:
            对比结果字典
        """
        logger.info("\n" + "="*70)
        logger.info("对比预测结果与Oracle")
        logger.info("="*70)
        
        # 如果没有提供oracle结果，从metadata读取
        if oracle_result is None:
            logger.info("\n从metadata读取Oracle的Synergy指标...")
            oracle_result = self._load_oracle_from_metadata()
        
        # 计算Sum Synergy Ratio（正确的评价指标）
        # Sum Synergy Ratio = sum of all cluster synergy ratios
        pred_cluster_scores = predicted_result.get('cluster_scores', [])
        oracle_cluster_scores = oracle_result.get('cluster_scores', [])
        
        pred_sum_synergy_ratio = sum(score['synergy_ratio'] for score in pred_cluster_scores)
        
        # Oracle的Sum Synergy Ratio：优先使用预计算的值（从metadata读取）
        oracle_sum_synergy_ratio = oracle_result.get('oracle_sum_synergy_ratio')
        if oracle_sum_synergy_ratio is None:
            # 如果没有预计算值，则从cluster_scores计算
            oracle_sum_synergy_ratio = sum(score['synergy_ratio'] for score in oracle_cluster_scores)
        
        # 提取其他指标
        pred_avg_synergy = predicted_result.get('avg_synergy_ratio', 0.0)
        oracle_avg_synergy = oracle_result.get('avg_synergy_ratio', 0.0)
        
        pred_gain = predicted_result.get('total_synergy_gain', 0.0)
        oracle_gain = oracle_result.get('total_synergy_gain', 0.0)
        
        pred_clusters = predicted_result.get('num_clusters', 0)
        oracle_clusters = oracle_result.get('num_clusters', 0)
        
        # 计算相对性能（基于Sum Synergy Ratio）
        relative_performance = pred_sum_synergy_ratio / oracle_sum_synergy_ratio if oracle_sum_synergy_ratio > 0 else 0.0
        
        comparison = {
            'predicted_sum_synergy_ratio': pred_sum_synergy_ratio,
            'oracle_sum_synergy_ratio': oracle_sum_synergy_ratio,
            'predicted_avg_synergy_ratio': pred_avg_synergy,
            'oracle_avg_synergy_ratio': oracle_avg_synergy,
            'relative_performance': relative_performance,
            'performance_gap': oracle_sum_synergy_ratio - pred_sum_synergy_ratio,
            'predicted_gain': pred_gain,
            'oracle_gain': oracle_gain,
            'predicted_clusters': pred_clusters,
            'oracle_clusters': oracle_clusters
        }
        
        logger.info(f"\n【正确的评价指标：Sum Synergy Ratio】")
        logger.info(f"预测 Sum Synergy Ratio: {pred_sum_synergy_ratio:.4f} ({pred_sum_synergy_ratio*100:.2f}%) - {pred_clusters} 簇")
        logger.info(f"Oracle Sum Synergy Ratio: {oracle_sum_synergy_ratio:.4f} ({oracle_sum_synergy_ratio*100:.2f}%) - {oracle_clusters} 簇")
        logger.info(f"相对性能: {relative_performance:.2%}")
        logger.info(f"性能差距: {oracle_sum_synergy_ratio - pred_sum_synergy_ratio:.4f}")
        logger.info(f"\n【参考：Avg Synergy Ratio】")
        logger.info(f"预测 Avg: {pred_avg_synergy:.4f}, Oracle Avg: {oracle_avg_synergy:.4f}")
        
        return comparison
    
    def _load_oracle_from_metadata(self) -> Dict:
        """
        从merge_metadata.json读取Oracle的synergy结果
        
        Returns:
            Oracle评估结果
        """
        metadata_file = self.data_dir / 'merge_metadata.json'
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"找不到metadata文件: {metadata_file}")
        
        logger.info("  从merge_metadata.json读取Oracle结果")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        run_summaries = metadata.get('run_summaries', [])
        
        # 计算Oracle的Sum Synergy Ratio（所有runs的best_synergy_ratio之和）
        oracle_sum_synergy_ratio = sum(run['best_synergy_ratio'] for run in run_summaries)
        n_runs = len(run_summaries)
        
        # 创建cluster scores列表（每个run一个）
        all_cluster_scores = []
        for run_summary in run_summaries:
            run_id = run_summary['run_id']
            run_synergy = run_summary['best_synergy_ratio']
            
            logger.info(f"  Run {run_id}: Sum Synergy Ratio: {run_synergy:.4f}")
            
            all_cluster_scores.append({
                'cluster_id': f'run{run_id}_oracle',
                'synergy_ratio': run_synergy,
                'run_id': run_id
            })
        
        summary = {
            'cluster_scores': all_cluster_scores,
            'num_clusters': n_runs,
            'avg_synergy_ratio': oracle_sum_synergy_ratio / n_runs if n_runs > 0 else 0.0,
            'num_runs': n_runs,
            'oracle_sum_synergy_ratio': oracle_sum_synergy_ratio
        }
        
        logger.info(f"\n  Oracle汇总（从metadata读取）:")
        logger.info(f"    Runs数: {n_runs}")
        logger.info(f"    Sum Synergy Ratio: {oracle_sum_synergy_ratio:.4f}")
        
        return summary


def evaluate_and_compare(
    data_dir: Path,
    checkpoint_df: pd.DataFrame,
    labels: np.ndarray,
    save_result: bool = True
) -> Dict:
    """
    便捷函数：按run评估聚类结果并与oracle对比
    
    Args:
        data_dir: 数据目录
        checkpoint_df: checkpoint DataFrame（必须包含source_run_id列）
        labels: 聚类标签
        save_result: 是否保存结果
        
    Returns:
        包含评估和对比结果的字典
    """
    evaluator = CheckpointSynergyEvaluator(data_dir)
    
    # 按run分组评估聚类结果
    predicted_result = evaluator.evaluate_clustering_by_run(
        checkpoint_df,
        labels,
        save_result=save_result
    )
    
    # 与oracle对比（从metadata读取）
    comparison = evaluator.compare_with_oracle(predicted_result)
    
    return {
        'predicted_result': predicted_result,
        'comparison': comparison
    }
