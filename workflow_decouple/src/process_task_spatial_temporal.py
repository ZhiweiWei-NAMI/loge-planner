"""
任务时空耦合数据预处理脚本
Process workflow logs to generate spatial-temporal task instances for clustering analysis.

功能:
1. 读取工作流日志 (agent_snapshots.csv)
2. 提取 task_started 和 task_completed 事件对
3. 对于 MoveToTask: 在起点和终点之间进行线性插值，生成每个时间粒度的位置
4. 对于其他任务 (Compute, Communicate, Deliver等): 在每个时间粒度重复相同位置
5. 输出预处理数据到 processed_task_spatial_temporal_coupling.csv

输出格式: time, agent_id, position, workflow_id, task_type
"""

import pandas as pd
import numpy as np
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import yaml
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskSpatialTemporalProcessor:
    """任务时空数据处理器"""
    
    def __init__(self, time_granularity: float = 1.0, 
                 enable_clustering: bool = True,
                 clustering_method: str = 'kmeans',
                 distance_threshold: float = 100.0,
                 time_window: float = None,
                 grid_size: float = 100.0):
        """
        Args:
            time_granularity: 时间粒度 (秒), 默认1.0秒
            enable_clustering: 是否启用空间聚类
            clustering_method: 聚类方法 ('kmeans' 或 'hierarchical')
            distance_threshold: 聚类距离阈值 (米)
            time_window: 时间窗口大小 (秒), None则使用time_granularity
            grid_size: 全局网格大小 (米), 用于生成稳定的global_cell_id, 默认100.0米
        """
        self.time_granularity = time_granularity
        self.enable_clustering = enable_clustering
        self.clustering_method = clustering_method
        self.distance_threshold = distance_threshold
        self.time_window = time_window if time_window is not None else time_granularity
        self.grid_size = grid_size
        
    def parse_dict_string(self, dict_str: str) -> dict:
        """安全地解析字符串形式的字典"""
        return ast.literal_eval(dict_str)
    
    def extract_position(self, agent_state: dict) -> Tuple[float, float, float]:
        """从agent_state中提取位置信息"""
        if isinstance(agent_state, str):
            agent_state = self.parse_dict_string(agent_state)
        
        position = agent_state['position']
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            return tuple(position[:3])
        return (0, 0, 0)
    
    def extract_target_position(self, task_params: dict) -> Optional[Tuple[float, float, float]]:
        """从task_params中提取目标位置 (仅用于MoveToTask)"""
        if isinstance(task_params, str):
            task_params = self.parse_dict_string(task_params)
        
        target_pos = task_params['target_position']
        if target_pos and isinstance(target_pos, (list, tuple)) and len(target_pos) >= 3:
            return tuple(target_pos[:3])
        return None
    
    def linear_interpolate_positions(
        self, 
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        start_time: float,
        end_time: float
    ) -> List[Tuple[float, Tuple[float, float, float]]]:
        """
        在起点和终点之间进行线性插值
        
        Returns:
            List of (time, position) tuples
        """
        positions = []
        
        # 计算需要生成的时间点
        num_steps = int((end_time - start_time) / self.time_granularity) + 1
        time_points = np.linspace(start_time, end_time, num_steps)
        
        # 线性插值计算每个时间点的位置
        for t in time_points:
            # 插值因子
            alpha = (t - start_time) / (end_time - start_time) if end_time > start_time else 0
            
            # 对x, y, z分别插值
            x = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
            y = start_pos[1] + alpha * (end_pos[1] - start_pos[1])
            z = start_pos[2] + alpha * (end_pos[2] - start_pos[2])
            
            positions.append((float(t), (x, y, z)))
        
        return positions
    
    def generate_static_positions(
        self,
        position: Tuple[float, float, float],
        start_time: float,
        end_time: float
    ) -> List[Tuple[float, Tuple[float, float, float]]]:
        """
        为静态任务 (Compute, Communicate等) 生成固定位置的时间序列
        
        Returns:
            List of (time, position) tuples
        """
        positions = []
        
        # 计算需要生成的时间点
        num_steps = int((end_time - start_time) / self.time_granularity) + 1
        time_points = np.linspace(start_time, end_time, num_steps)
        
        # 每个时间点使用相同位置
        for t in time_points:
            positions.append((float(t), position))
        
        return positions
    
    def match_task_events(self, df: pd.DataFrame) -> List[Dict]:
        """
        匹配 task_started 和 task_completed 事件对
        
        Returns:
            List of matched task events with start and end information
        """
        matched_tasks = []
        
        # 按agent_id和workflow_id分组
        grouped = df.groupby(['agent_id', 'workflow_id'])
        
        for (agent_id, workflow_id), group in grouped:
            # 按时间排序
            group = group.sort_values('time')
            
            # 用字典存储已开始的任务 (key: task_type)
            started_tasks = {}
            
            for _, row in group.iterrows():
                event_type = row['event_type']
                task_type = row['task_type']
                
                if event_type == 'task_started':
                    # 记录任务开始
                    started_tasks[task_type] = {
                        'start_time': row['time'],
                        'start_position': self.extract_position(row['agent_state']),
                        'task_params': row['task_params'],
                        'agent_id': agent_id,
                        'workflow_id': workflow_id,
                        'task_type': task_type
                    }
                
                elif event_type == 'task_completed' and task_type in started_tasks:
                    # 找到匹配的任务完成事件
                    start_info = started_tasks.pop(task_type)
                    
                    matched_tasks.append({
                        'agent_id': agent_id,
                        'workflow_id': workflow_id,
                        'task_type': task_type,
                        'start_time': start_info['start_time'],
                        'end_time': row['time'],
                        'start_position': start_info['start_position'],
                        'end_position': self.extract_position(row['agent_state']),
                        'task_params': start_info['task_params']
                    })
        
        return matched_tasks
    
    def process_matched_tasks(self, matched_tasks: List[Dict]) -> pd.DataFrame:
        """
        处理匹配的任务，生成时空耦合数据
        
        Returns:
            DataFrame with columns: time, agent_id, position, workflow_id, task_type
        """
        processed_data = []
        
        for task in matched_tasks:
            task_type = task['task_type']
            start_time = task['start_time']
            end_time = task['end_time']
            start_pos = task['start_position']
            end_pos = task['end_position']
            agent_id = task['agent_id']
            workflow_id = task['workflow_id']
            
            # 根据任务类型选择处理方式
            if task_type == 'MoveToTask':
                # MoveToTask: 使用线性插值
                # 尝试从task_params获取目标位置，否则使用end_position
                target_pos = self.extract_target_position(task['task_params'])
                if target_pos is None:
                    target_pos = end_pos
                
                positions = self.linear_interpolate_positions(
                    start_pos, target_pos, start_time, end_time
                )
            else:
                # 其他任务 (Compute, Communicate, Deliver, Inspection, Pickup等): 固定位置
                # 使用开始位置作为任务执行位置
                positions = self.generate_static_positions(
                    start_pos, start_time, end_time
                )
            
            # 添加到结果数据
            for time, position in positions:
                global_cell_id = self.compute_global_cell_id(position)
                processed_data.append({
                    'time': time,
                    'agent_id': agent_id,
                    'position': f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})",
                    'workflow_id': workflow_id,
                    'task_type': task_type,
                    'global_cell_id': global_cell_id
                })
        
        # 转换为DataFrame并按时间排序
        df = pd.DataFrame(processed_data)
        if not df.empty:
            df = df.sort_values('time').reset_index(drop=True)
            
            # 如果启用聚类，添加cluster_id列
            if self.enable_clustering:
                df = self.add_spatial_clustering(df)
        
        return df
    
    def parse_position_string(self, pos_str: str) -> Tuple[float, float, float]:
        """解析位置字符串为坐标元组"""
        # 移除括号和空格，分割
        pos_str = pos_str.strip('()')
        coords = [float(x.strip()) for x in pos_str.split(',')]
        return tuple(coords[:3])
    
    def compute_global_cell_id(self, position: Tuple[float, float, float]) -> str:
        """计算全局网格单元ID（时间不敏感的稳定空间编码）
        
        Args:
            position: (x, y, z) 坐标元组
            
        Returns:
            全局网格单元ID，格式: "cell_X_Y_Z"
        """
        x, y, z = position
        cell_x = int(x // self.grid_size)
        cell_y = int(y // self.grid_size)
        cell_z = int(z // self.grid_size)
        return f"cell_{cell_x}_{cell_y}_{cell_z}"
    
    def add_spatial_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为时空数据添加空间聚类标签
        
        在每个时间窗口下，对空间位置进行聚类（不考虑任务类型）
        
        Args:
            df: 包含time, position的DataFrame
            
        Returns:
            添加了cluster_id列的DataFrame
        """
        logger.info(f"  执行空间聚类 (方法: {self.clustering_method}, 距离阈值: {self.distance_threshold}m)")
        
        # 解析位置字符串为坐标
        df['x'] = df['position'].apply(lambda p: self.parse_position_string(p)[0])
        df['y'] = df['position'].apply(lambda p: self.parse_position_string(p)[1])
        df['z'] = df['position'].apply(lambda p: self.parse_position_string(p)[2])
        
        # 创建时间窗口
        df['time_window'] = (df['time'] // self.time_window).astype(int)
        
        # 初始化cluster_id列
        df['cluster_id'] = ''
        
        # 仅按时间窗口分组（不再按任务类型分组）
        grouped = df.groupby('time_window')
        
        cluster_count = 0
        for time_window, group in grouped:
            
            if len(group) < 2:
                # 只有一个点，直接分配cluster_id
                t_start = time_window * self.time_window
                cluster_name = f"ts_{t_start:.1f}_cluster_0"
                df.loc[group.index, 'cluster_id'] = cluster_name
                continue
            
            # 提取空间坐标
            positions = group[['x', 'y', 'z']].values
            
            # 执行聚类
            if self.clustering_method == 'kmeans':
                # 使用KMeans，估计簇数量
                # 基于距离阈值估算簇数量
                max_distance = np.max(np.ptp(positions, axis=0))
                n_clusters = max(1, int(max_distance / self.distance_threshold))
                n_clusters = min(n_clusters, len(group))  # 不超过样本数
                
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(positions)
                
            elif self.clustering_method == 'hierarchical':
                # 使用层次聚类
                n_clusters = None
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    distance_threshold=self.distance_threshold,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(positions)
                
            else:
                logger.warning(f"  未知的聚类方法: {self.clustering_method}，使用默认kmeans")
                n_clusters = max(1, len(group) // 5)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(positions)
            
            # 分配cluster_id（不再包含task_type）
            t_start = time_window * self.time_window
            for cluster_label in np.unique(labels):
                cluster_mask = labels == cluster_label
                cluster_indices = group.index[cluster_mask]
                cluster_name = f"ts_{t_start:.1f}_cluster_{cluster_label}"
                df.loc[cluster_indices, 'cluster_id'] = cluster_name
                cluster_count += 1
        
        # 删除临时列
        df = df.drop(columns=['x', 'y', 'z', 'time_window'])
        
        logger.info(f"  生成 {cluster_count} 个空间聚类")
        
        return df
    
    def process_file(self, input_path: Path, output_path: Path):
        """
        处理单个agent_snapshots.csv文件
        
        Args:
            input_path: 输入CSV文件路径
            output_path: 输出CSV文件路径
        """
        logger.info(f"处理文件: {input_path}")
        
        # 读取CSV
        df = pd.read_csv(input_path, engine='python', on_bad_lines='skip')
        logger.info(f"  读取 {len(df)} 行数据")
        
        # 筛选task_started和task_completed事件
        task_events = df[df['event_type'].isin(['task_started', 'task_completed'])]
        logger.info(f"  找到 {len(task_events)} 个任务事件")
        
        # 匹配任务事件对
        matched_tasks = self.match_task_events(task_events)
        logger.info(f"  匹配到 {len(matched_tasks)} 个完整任务")
        
        # 处理任务生成时空数据
        result_df = self.process_matched_tasks(matched_tasks)
        logger.info(f"  生成 {len(result_df)} 个时空实例")
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"  保存到: {output_path}")
        
        return result_df


def process_all_test_directories(
    data_dir: Path,
    time_granularity: float = 1.0,
    output_filename: str = 'processed_task_spatial_temporal_coupling.csv',
    enable_clustering: bool = True,
    clustering_method: str = 'kmeans',
    distance_threshold: float = 100.0,
    time_window: float = None,
    grid_size: float = 100.0
):
    """
    处理data目录下所有test开头的目录
    
    Args:
        data_dir: 数据根目录
        time_granularity: 时间粒度
        output_filename: 输出文件名
        enable_clustering: 是否启用聚类
        clustering_method: 聚类方法
        distance_threshold: 距离阈值
        time_window: 时间窗口
    """
    processor = TaskSpatialTemporalProcessor(
        time_granularity=time_granularity,
        enable_clustering=enable_clustering,
        clustering_method=clustering_method,
        distance_threshold=distance_threshold,
        time_window=time_window,
        grid_size=grid_size
    )
    
    # 查找所有test开头的目录
    test_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('test')])
    
    if not test_dirs:
        logger.warning(f"未在 {data_dir} 中找到test开头的目录")
        return
    
    logger.info(f"找到 {len(test_dirs)} 个测试目录")
    logger.info(f"时间粒度: {time_granularity} 秒")
    if enable_clustering:
        logger.info(f"聚类方法: {clustering_method}")
        logger.info(f"距离阈值: {distance_threshold} 米")
        logger.info(f"时间窗口: {time_window if time_window else time_granularity} 秒")
    else:
        logger.info("聚类功能: 已禁用")
    logger.info("=" * 70)
    
    # 处理每个目录
    for test_dir in test_dirs:
        logger.info(f"\n处理目录: {test_dir.name}")
        
        input_file = test_dir / 'agent_snapshots.csv'
        output_file = test_dir / output_filename
        
        if not input_file.exists():
            logger.warning(f"  未找到文件: {input_file}")
            continue
        
        processor.process_file(input_file, output_file)
    
    logger.info("\n" + "=" * 70)
    logger.info("所有文件处理完成!")


def main():
    parser = argparse.ArgumentParser(
        description='处理工作流日志生成任务时空耦合数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认时间粒度 (1.0秒)
  python process_task_spatial_temporal.py
  
  # 指定时间粒度为0.5秒
  python process_task_spatial_temporal.py --time-granularity 0.5
  
  # 使用配置文件
  python process_task_spatial_temporal.py --config ../config/preprocessing.yaml
  
  # 指定数据目录
  python process_task_spatial_temporal.py --data-dir ../data
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='数据目录路径 (默认: ../data)'
    )
    
    parser.add_argument(
        '--time-granularity',
        type=float,
        default=5.0,
        help='时间粒度，单位：秒 (默认: 5.0)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径 (可选)'
    )
    
    parser.add_argument(
        '--output-filename',
        type=str,
        default='processed_task_spatial_temporal_coupling.csv',
        help='输出文件名 (默认: processed_task_spatial_temporal_coupling.csv)'
    )
    
    args = parser.parse_args()
    
    # 聚类相关参数（添加默认值）
    enable_clustering = True
    clustering_method = 'kmeans'
    distance_threshold = 100.0
    time_window = None
    grid_size = 100.0
    
    # 如果提供了配置文件，从中读取参数
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 配置文件中的参数优先级低于命令行参数
            if 'time_granularity' in config:
                args.time_granularity = config['time_granularity']
            if 'output_filename' in config:
                args.output_filename = config['output_filename']
            
            # 读取聚类配置
            if 'clustering' in config:
                clustering_config = config['clustering']
                enable_clustering = clustering_config.get('enable_clustering', True)
                clustering_method = clustering_config.get('method', 'kmeans')
                distance_threshold = clustering_config.get('distance_threshold', 100.0)
                time_window = clustering_config.get('time_window', None)
                grid_size = clustering_config.get('grid_size', 100.0)
        else:
            logger.warning(f"配置文件不存在: {config_path}")
    
    # 解析数据目录路径
    data_dir = Path(__file__).parent / args.data_dir
    data_dir = data_dir.resolve()
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    # 处理所有测试目录
    process_all_test_directories(
        data_dir=data_dir,
        time_granularity=args.time_granularity,
        output_filename=args.output_filename,
        enable_clustering=enable_clustering,
        clustering_method=clustering_method,
        distance_threshold=distance_threshold,
        time_window=time_window,
        grid_size=grid_size
    )


if __name__ == '__main__':
    main()
