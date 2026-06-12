"""
工作流Checkpoint聚类算法（改进版）

对于每个工作流ID，按照agent_snapshots.csv中每次任务完成的时间点作为一个checkpoint，
基于checkpoint的空间足迹（global_cell_id集合）计算相似度，并使用层次聚类进行分组。

相似度定义：两个checkpoint的空间足迹的Jaccard相似度（集合交集/集合并集）。
空间足迹：checkpoint执行过程中涉及的所有唯一的全局网格单元（global_cell_id）。

改进点：
1. 使用时间不敏感的global_cell_id替代时间窗口相关的cluster_id
2. 对空间足迹去重，使用集合Jaccard相似度
3. 避免了时间窗口标签不一致导致的相似度低估问题

@author: TCCN Auto Planner
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowCheckpoint:
    """工作流Checkpoint类"""
    
    def __init__(self, workflow_id: str, checkpoint_idx: int, 
                 start_time: float, end_time: float,
                 tasks: List[str], task_types: List[str],
                 positions: List[tuple], cluster_ids: List[str],
                 global_cell_ids: List[str]):
        self.workflow_id = workflow_id
        self.checkpoint_idx = checkpoint_idx  # 该工作流的第几个checkpoint
        self.start_time = start_time
        self.end_time = end_time
        self.tasks = tasks  # 任务实例列表
        self.task_types = task_types
        self.positions = positions
        self.cluster_ids = cluster_ids  # 每个任务所属的空间聚类ID（旧版，保留作辅助）
        self.global_cell_ids = global_cell_ids  # 每个任务所属的全局网格单元ID（稳定空间编码）
        
        # 预计算空间足迹（去重后的global_cell_id集合）
        self.spatial_footprint = set(global_cell_ids) if global_cell_ids else set()
        
    def __repr__(self):
        return f"Checkpoint({self.workflow_id}_ckp{self.checkpoint_idx}, t={self.start_time:.1f}-{self.end_time:.1f}, tasks={len(self.tasks)})"


class WorkflowCheckpointClustering:
    """工作流Checkpoint聚类"""
    
    def __init__(self, cohesion_threshold: float = 0.3):
        """
        Args:
            cohesion_threshold: 内聚度阈值，用于判断两个checkpoint是否应该聚在一起
        """
        self.cohesion_threshold = cohesion_threshold
        self.checkpoints: List[WorkflowCheckpoint] = []
        self.cohesion_matrix = None
        self.cluster_labels = None
        
    def extract_checkpoints_from_snapshots(self, snapshots_csv: Path, 
                                          processed_csv: Path) -> List[WorkflowCheckpoint]:
        """
        从agent_snapshots.csv中提取checkpoint
        
        每个工作流每次任务完成时就是一个checkpoint。
        提取该checkpoint时间段内的空间足迹（global_cell_id集合）。
        """
        logger.info(f"从 {snapshots_csv} 提取checkpoints...")
        
        # 读取snapshots
        df_snapshots = pd.read_csv(snapshots_csv)
        
        # 只保留task_completed事件
        df_completed = df_snapshots[df_snapshots['event_type'] == 'task_completed'].copy()
        
        # 读取processed数据以获取cluster_id信息
        df_processed = pd.read_csv(processed_csv)
        
        # 按workflow_id和时间排序
        df_completed = df_completed.sort_values(['workflow_id', 'time'])
        
        checkpoints = []
        
        # 按workflow分组
        for workflow_id, wf_group in df_completed.groupby('workflow_id'):
            # 每个任务完成时间点是一个checkpoint
            for idx, (_, row) in enumerate(wf_group.iterrows()):
                end_time = row['time']
                agent_id = row['agent_id']
                task_type = row['task_type']
                position = (row['x'], row['y'], row['z'])
                
                # 获取该checkpoint前一段时间内的所有任务（作为该checkpoint涉及的任务）
                # 我们定义checkpoint的开始时间为上一个checkpoint结束时间（或workflow开始时间）
                if idx == 0:
                    # 第一个checkpoint，查找该workflow的起始时间
                    start_time = df_snapshots[df_snapshots['workflow_id'] == workflow_id]['time'].min()
                else:
                    # 取上一个checkpoint的结束时间
                    start_time = wf_group.iloc[idx-1]['time']
                
                # 获取这段时间内processed_data中该workflow的所有任务实例
                mask = ((df_processed['workflow_id'] == workflow_id) &
                       (df_processed['time'] >= start_time) &
                       (df_processed['time'] <= end_time))
                
                checkpoint_tasks = df_processed[mask]
                
                if len(checkpoint_tasks) == 0:
                    continue
                
                # 提取任务信息
                tasks = checkpoint_tasks.index.tolist()  # 使用索引作为任务ID
                task_types_list = checkpoint_tasks['task_type'].tolist()
                positions_list = [
                    tuple(map(float, pos.strip('()').split(','))) 
                    for pos in checkpoint_tasks['position']
                ]
                cluster_ids = checkpoint_tasks['cluster_id'].tolist() if 'cluster_id' in checkpoint_tasks.columns else []
                
                # 提取global_cell_id（如果存在）
                if 'global_cell_id' in checkpoint_tasks.columns:
                    global_cell_ids = checkpoint_tasks['global_cell_id'].tolist()
                else:
                    # 如果processed_csv中没有global_cell_id，使用cluster_id作为后备
                    logger.warning(f"processed_csv中缺少global_cell_id列，使用cluster_id作为后备方案")
                    global_cell_ids = cluster_ids
                
                checkpoint = WorkflowCheckpoint(
                    workflow_id=workflow_id,
                    checkpoint_idx=idx,
                    start_time=start_time,
                    end_time=end_time,
                    tasks=tasks,
                    task_types=task_types_list,
                    positions=positions_list,
                    cluster_ids=cluster_ids,
                    global_cell_ids=global_cell_ids
                )
                
                checkpoints.append(checkpoint)
        
        logger.info(f"提取了 {len(checkpoints)} 个checkpoints")
        return checkpoints
    
    def compute_cohesion(self, ckp1: WorkflowCheckpoint, ckp2: WorkflowCheckpoint) -> float:
        """
        计算两个checkpoint之间的相似度（改用集合Jaccard相似度）
        
        相似度定义：两个checkpoint的空间足迹（global_cell_id集合）的Jaccard相似度。
        Jaccard相似度 = |S1 ∩ S2| / |S1 ∪ S2|
        
        改进：使用去重后的global_cell_id集合，避免了时间窗口标签不一致问题。
        
        Returns:
            相似度值 (0-1)
        """
        s1 = ckp1.spatial_footprint
        s2 = ckp2.spatial_footprint
        
        # 如果两个集合都为空，定义相似度为0
        if len(s1) == 0 and len(s2) == 0:
            return 0.0
        
        # 计算交集和并集
        intersection = s1 & s2
        union = s1 | s2
        
        # Jaccard相似度
        if len(union) == 0:
            return 0.0
        
        similarity = len(intersection) / len(union)
        
        return similarity
    
    def build_cohesion_matrix(self, checkpoints: List[WorkflowCheckpoint]) -> np.ndarray:
        """构建checkpoint间的相似度矩阵（基于Jaccard相似度）"""
        n = len(checkpoints)
        cohesion_matrix = np.zeros((n, n))
        
        logger.info(f"计算 {n} x {n} 内聚度矩阵...")
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    cohesion_matrix[i, j] = 1.0
                else:
                    # 计算Jaccard相似度（对称的，无需双向计算）
                    similarity = self.compute_cohesion(checkpoints[i], checkpoints[j])
                    cohesion_matrix[i, j] = similarity
        
        return cohesion_matrix
    
    def cluster_checkpoints(self, checkpoints: List[WorkflowCheckpoint]) -> np.ndarray:
        """
        使用相似度矩阵对checkpoints进行聚类
        
        Returns:
            cluster_labels: 每个checkpoint的聚类标签
        """
        self.checkpoints = checkpoints
        
        # 构建相似度矩阵（Jaccard相似度）
        self.cohesion_matrix = self.build_cohesion_matrix(checkpoints)
        
        # 将相似度转换为距离：distance = 1 - similarity
        distance_matrix = 1.0 - self.cohesion_matrix
        
        # 使用层次聚类，基于距离阈值
        # distance_threshold = 1 - similarity_threshold
        distance_threshold = 1.0 - self.cohesion_threshold
        
        logger.info(f"使用层次聚类（Jaccard相似度），距离阈值={distance_threshold:.3f} (相似度阈值={self.cohesion_threshold:.3f})")
        
        # 使用AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        self.cluster_labels = cluster_labels
        
        n_clusters = len(set(cluster_labels))
        logger.info(f"聚类完成，生成 {n_clusters} 个聚类")
        
        return cluster_labels
    
    def save_results(self, output_path: Path):
        """保存聚类结果"""
        if self.checkpoints is None or self.cluster_labels is None:
            logger.error("尚未进行聚类，请先调用cluster_checkpoints()")
            return
        
        # 准备输出数据
        results = []
        for checkpoint, label in zip(self.checkpoints, self.cluster_labels):
            results.append({
                'workflow_id': checkpoint.workflow_id,
                'checkpoint_idx': checkpoint.checkpoint_idx,
                'cluster_group': int(label),
                'start_time': checkpoint.start_time,
                'end_time': checkpoint.end_time,
                'duration': checkpoint.end_time - checkpoint.start_time,
                'num_tasks': len(checkpoint.tasks),
                'spatial_footprint_size': len(checkpoint.spatial_footprint),
                'task_types': ','.join(set(checkpoint.task_types)),
                'avg_position_x': np.mean([p[0] for p in checkpoint.positions]),
                'avg_position_y': np.mean([p[1] for p in checkpoint.positions]),
                'avg_position_z': np.mean([p[2] for p in checkpoint.positions]),
                'unique_cells': len(checkpoint.spatial_footprint)
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, index=False)
        logger.info(f"聚类结果已保存到 {output_path}")
        
        # 打印统计信息
        logger.info("\n聚类统计:")
        for cluster_id, group in df_results.groupby('cluster_group'):
            num_workflows = group['workflow_id'].nunique()
            avg_footprint_size = group['num_tasks'].mean()
            logger.info(f"  簇 {cluster_id}: {len(group)} 个checkpoints, "
                       f"来自 {num_workflows} 个不同工作流, "
                       f"平均空间足迹大小: {avg_footprint_size:.1f}")


def process_test_directory(test_dir: Path, cohesion_threshold: float = 0.3, save_with_suffix: bool = True):
    """处理单个测试目录
    
    Args:
        test_dir: 测试数据目录
        cohesion_threshold: 聚类阈值
        save_with_suffix: 是否在输出文件名中包含阈值后缀（默认True）
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"处理目录: {test_dir.name}")
    logger.info(f"{'='*80}")
    
    snapshots_csv = test_dir / 'agent_snapshots.csv'
    processed_csv = test_dir / 'processed_task_spatial_temporal_coupling.csv'
    
    # 根据配置决定输出文件名
    if save_with_suffix:
        output_csv = test_dir / f'workflow_checkpoint_clustering_th{cohesion_threshold:.2f}.csv'
    else:
        output_csv = test_dir / 'workflow_checkpoint_clustering.csv'
    
    if not snapshots_csv.exists():
        logger.warning(f"找不到 {snapshots_csv}，跳过")
        return
    
    if not processed_csv.exists():
        logger.warning(f"找不到 {processed_csv}，跳过")
        return
    
    # 创建聚类器
    clusterer = WorkflowCheckpointClustering(cohesion_threshold=cohesion_threshold)
    
    # 提取checkpoints
    checkpoints = clusterer.extract_checkpoints_from_snapshots(
        snapshots_csv, processed_csv
    )
    
    if len(checkpoints) == 0:
        logger.warning("未找到任何checkpoints，跳过")
        return
    
    # 聚类
    clusterer.cluster_checkpoints(checkpoints)
    
    # 保存结果
    clusterer.save_results(output_csv)


def main():
    parser = argparse.ArgumentParser(description='工作流Checkpoint聚类')
    parser.add_argument('--data-dir', type=str, 
                       default='../data',
                       help='数据目录路径')
    parser.add_argument('--cohesion-threshold', type=float, 
                       default=0.3,
                       help='相似度阈值 (0-1)，基于Jaccard相似度。较高值产生更多簇，较低值产生更少簇。推荐0.1-0.4')
    parser.add_argument('--test-dir', type=str,
                       help='指定单个测试目录名称，不指定则处理所有目录')
    parser.add_argument('--save-with-suffix', action='store_true', default=True,
                       help='输出文件名包含阈值后缀（默认True）')
    parser.add_argument('--no-suffix', dest='save_with_suffix', action='store_false',
                       help='输出文件名不包含阈值后缀（向后兼容）')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.test_dir:
        # 处理指定目录
        test_dir = data_dir / args.test_dir
        if test_dir.exists() and test_dir.is_dir():
            process_test_directory(test_dir, args.cohesion_threshold, args.save_with_suffix)
        else:
            logger.error(f"目录不存在: {test_dir}")
    else:
        # 处理所有test_开头的目录
        test_dirs = sorted([d for d in data_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('test_')])
        
        logger.info(f"找到 {len(test_dirs)} 个测试目录")
        
        for test_dir in test_dirs:
            process_test_directory(test_dir, args.cohesion_threshold, args.save_with_suffix)
        
        logger.info(f"\n{'='*80}")
        logger.info("所有目录处理完成!")


if __name__ == '__main__':
    main()
