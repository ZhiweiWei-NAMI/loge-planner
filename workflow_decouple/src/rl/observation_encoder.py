"""
观测编码器 - 将workflow初始信息编码为固定维度的观测向量

采用统计聚合方式：
- checkpoint特征矩阵的逐维统计：mean/std/min/max → 104维
- workflow类型分布 → 4维
- checkpoint数量统计 → 2维
- agent初始状态统计 → 10维
总计约120维
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
import logging

# 添加父目录到路径以导入ml_training模块
parent_path = Path(__file__).parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

from ml_training.initial_data_loader import InitialDataLoader
from ml_training.checkpoint_extractor import CheckpointExtractor

logger = logging.getLogger(__name__)


class ObservationEncoder:
    """
    观测编码器
    
    将数据目录的初始信息编码为约120维的向量
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: 是否做归一化（基于训练集统计）
        """
        self.normalize = normalize
        self.stats = None  # 训练集统计（用于归一化）
        
        # 特征维度
        self.checkpoint_feature_dim = 26
        self.obs_dim = self._compute_obs_dim()
    
    def _compute_obs_dim(self) -> int:
        """计算观测向量总维度"""
        checkpoint_stats_dim = self.checkpoint_feature_dim * 4  # mean/std/min/max
        workflow_stats_dim = 4  # 4种类型计数
        global_stats_dim = 2  # 总checkpoint数、平均checkpoint数
        agent_stats_dim = 10  # 5个属性 × 2 (mean/std)
        
        total_dim = checkpoint_stats_dim + workflow_stats_dim + global_stats_dim + agent_stats_dim
        return total_dim
    
    def encode(self, data_dir: Path) -> torch.Tensor:
        """
        编码单个数据目录
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            obs: 观测向量 [obs_dim]
        """
        # 加载数据
        loader = InitialDataLoader(data_dir)
        workflow_df, agent_df = loader.load_all()
        
        # 提取workflow特征
        workflows_features = loader.get_all_workflows_with_agents()
        
        # 提取checkpoint特征
        extractor = CheckpointExtractor()
        checkpoint_df = extractor.extract_all_checkpoints(workflows_features)
        feature_matrix = extractor.prepare_features_for_clustering(checkpoint_df)
        
        # 构造观测向量
        obs_list = []
        
        # 1. Checkpoint特征统计 (104维)
        obs_list.extend(np.mean(feature_matrix, axis=0))  # 26维
        obs_list.extend(np.std(feature_matrix, axis=0))   # 26维
        obs_list.extend(np.min(feature_matrix, axis=0))   # 26维
        obs_list.extend(np.max(feature_matrix, axis=0))   # 26维
        
        # 2. Workflow类型分布 (4维)
        workflow_types = ['LogisticsWorkflow', 'InspectionWorkflow', 
                         'CommunicationWorkflow', 'ComputeWorkflow']
        type_counts = workflow_df['workflow_type'].value_counts()
        for wf_type in workflow_types:
            # 如果场景中没有该类型，填充0（兼容不同规模的场景）
            count = type_counts.get(wf_type, 0)
            obs_list.append(float(count))
        
        # 3. 全局统计 (2维)
        obs_list.append(float(len(checkpoint_df)))  # 总checkpoint数
        avg_checkpoints_per_workflow = len(checkpoint_df) / max(len(workflow_df), 1)
        obs_list.append(avg_checkpoints_per_workflow)
        
        # 4. Agent初始状态统计 (10维)
        # 重构为真正有信息量的agent特征：
        # - 空间分布 (6维)：初始位置的 x/y/z mean/std
        # - 数量与构成 (2维)：agent总数、类型多样性
        # - 任务负载分布 (2维)：每agent的workflow数 mean/std
        
        # 4.1 Agent空间分布 (6维)
        # 从agent_df的位置列提取（初始时刻的position）
        if 'x' in agent_df.columns and 'y' in agent_df.columns and 'z' in agent_df.columns:
            x_coords = agent_df['x'].values
            y_coords = agent_df['y'].values
            z_coords = agent_df['z'].values
            
            obs_list.append(float(np.mean(x_coords)))
            obs_list.append(float(np.std(x_coords)))
            obs_list.append(float(np.mean(y_coords)))
            obs_list.append(float(np.std(y_coords)))
            obs_list.append(float(np.mean(z_coords)))
            obs_list.append(float(np.std(z_coords)))
        else:
            # 如果没有x/y/z列，尝试从agent_state_parsed的position提取
            positions = []
            for idx, row in agent_df.iterrows():
                state = row.get('agent_state_parsed', {})
                pos = state.get('position', (0, 0, 0))
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    positions.append(pos)
            
            if positions:
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                z_coords = [p[2] for p in positions]
                
                obs_list.append(float(np.mean(x_coords)))
                obs_list.append(float(np.std(x_coords)))
                obs_list.append(float(np.mean(y_coords)))
                obs_list.append(float(np.std(y_coords)))
                obs_list.append(float(np.mean(z_coords)))
                obs_list.append(float(np.std(z_coords)))
            else:
                obs_list.extend([0.0] * 6)
        
        # 4.2 Agent数量与构成 (2维)
        num_agents = len(agent_df)
        obs_list.append(float(num_agents))
        
        # 统计agent类型多样性（基于agent_id_mapping或agent_type）
        num_config_types = 1  # 默认1种
        if loader.agent_id_mapping_df is not None and len(loader.agent_id_mapping_df) > 0:
            # 从config_agent_id提取类型前缀（如 agent_logistics_inspection_X -> logistics_inspection）
            config_ids = loader.agent_id_mapping_df['config_agent_id'].values
            type_prefixes = set()
            for cid in config_ids:
                # 提取前缀（去掉最后的数字部分）
                parts = str(cid).split('_')
                if len(parts) >= 2:
                    # agent_logistics_inspection_1 -> logistics_inspection
                    type_prefix = '_'.join(parts[1:-1]) if parts[-1].isdigit() else '_'.join(parts[1:])
                    type_prefixes.add(type_prefix)
            num_config_types = len(type_prefixes) if type_prefixes else 1
        elif 'agent_type' in agent_df.columns:
            num_config_types = agent_df['agent_type'].nunique()
        
        obs_list.append(float(num_config_types))
        
        # 4.3 Task-Load分布 (2维)
        # 统计每个agent被分配了多少个workflow（owner_agent）
        if 'owner_agent' in workflow_df.columns:
            workflows_per_agent = workflow_df['owner_agent'].value_counts()
            obs_list.append(float(workflows_per_agent.mean()))
            obs_list.append(float(workflows_per_agent.std()))
        else:
            obs_list.extend([0.0, 0.0])
        
        # 转为tensor
        obs = torch.tensor(obs_list, dtype=torch.float32)
        
        # 归一化（如果已有训练集统计）
        if self.normalize and self.stats is not None:
            obs = (obs - self.stats['mean']) / (self.stats['std'] + 1e-8)
        
        return obs
    
    def fit_normalization(self, data_dirs: list, verbose: bool = False):
        """
        在训练集上计算归一化统计
        
        Args:
            data_dirs: 训练数据目录列表
            verbose: 是否输出详细信息
        """
        # 临时禁用logger
        import logging
        loggers_to_silence = [
            'ml_training.initial_data_loader',
            'ml_training.checkpoint_extractor'
        ]
        original_levels = {}
        for logger_name in loggers_to_silence:
            logger_obj = logging.getLogger(logger_name)
            original_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(logging.WARNING)
        
        try:
            all_obs = []
            for data_dir in data_dirs:
                obs = self.encode(Path(data_dir))
                all_obs.append(obs.numpy())
            
            all_obs = np.array(all_obs)
            
            self.stats = {
                'mean': torch.tensor(np.mean(all_obs, axis=0), dtype=torch.float32),
                'std': torch.tensor(np.std(all_obs, axis=0), dtype=torch.float32)
            }
            
            if verbose:
                logger.info(f"  观测维度: {len(self.stats['mean'])}")
                logger.info(f"  均值范围: [{self.stats['mean'].min():.2f}, {self.stats['mean'].max():.2f}]")
                logger.info(f"  标准差范围: [{self.stats['std'].min():.2f}, {self.stats['std'].max():.2f}]")
        finally:
            # 恢复logger级别
            for logger_name, level in original_levels.items():
                logging.getLogger(logger_name).setLevel(level)
    
    def save_stats(self, path: Path):
        """保存归一化统计"""
        if self.stats is None:
            logger.warning("未计算归一化统计，跳过保存")
            return
        
        torch.save(self.stats, path)
        logger.info(f"归一化统计已保存: {path}")
    
    def load_stats(self, path: Path):
        """加载归一化统计"""
        self.stats = torch.load(path)
        logger.info(f"归一化统计已加载: {path}")


def encode_observation(data_dir: Path, encoder: Optional[ObservationEncoder] = None) -> torch.Tensor:
    """
    便捷函数：编码单个数据目录
    
    Args:
        data_dir: 数据目录
        encoder: 编码器实例（可选，用于复用归一化统计）
        
    Returns:
        obs: 观测向量
    """
    if encoder is None:
        encoder = ObservationEncoder(normalize=False)
    
    return encoder.encode(data_dir)
