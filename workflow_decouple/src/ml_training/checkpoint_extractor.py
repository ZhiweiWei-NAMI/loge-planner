"""
Checkpoint提取器 - 从workflow定义中提取checkpoint信息

从 src/simulation/workflows 中的workflow类定义提取checkpoints
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 添加simulation模块路径
# checkpoint_extractor.py在 workflow_decouple/src/ml_training/
# simulation在 src/simulation/
sim_path = Path(__file__).parent.parent.parent.parent / 'src' / 'simulation'
if str(sim_path) not in sys.path:
    sys.path.insert(0, str(sim_path))

from workflows.logistics import LogisticsWorkflow
from workflows.inspection import InspectionWorkflow
from workflows.communication import CommunicationWorkflow
from workflows.computation import ComputeWorkflow

logger = logging.getLogger(__name__)


class CheckpointExtractor:
    """从workflow定义中提取checkpoint"""
    
    def __init__(self):
        """初始化提取器"""
        self.workflow_classes = {
            'LogisticsWorkflow': LogisticsWorkflow,
            'InspectionWorkflow': InspectionWorkflow,
            'CommunicationWorkflow': CommunicationWorkflow,
            'ComputeWorkflow': ComputeWorkflow
        }
    
    def extract_checkpoints_for_workflow(self, workflow_features: Dict) -> List[Dict]:
        """
        从workflow特征提取其checkpoints
        
        Args:
            workflow_features: workflow特征字典（包含workflow_type和properties）
            
        Returns:
            checkpoint列表
        """
        workflow_type = workflow_features.get('workflow_type', '')
        workflow_id = workflow_features.get('workflow_id', '')
        
        # 直接根据workflow类型生成checkpoints，而不实例化workflow对象
        if workflow_type == 'LogisticsWorkflow':
            return self._extract_logistics_checkpoints(workflow_features)
        elif workflow_type == 'InspectionWorkflow':
            return self._extract_inspection_checkpoints(workflow_features)
        elif workflow_type == 'CommunicationWorkflow':
            return self._extract_communication_checkpoints(workflow_features)
        elif workflow_type == 'ComputeWorkflow':
            return self._extract_compute_checkpoints(workflow_features)
        else:
            logger.warning(f"未知的workflow类型: {workflow_type}")
            return []
    
    def _extract_logistics_checkpoints(self, wf_feat: Dict) -> List[Dict]:
        """提取LogisticsWorkflow的checkpoints（基于logistics.py中的定义）"""
        workflow_id = wf_feat.get('workflow_id', '')
        pickup_loc = [
            wf_feat.get('pickup_x', 0),
            wf_feat.get('pickup_y', 0),
            wf_feat.get('pickup_z', 50)
        ]
        delivery_loc = [
            wf_feat.get('delivery_x', 0),
            wf_feat.get('delivery_y', 0),
            wf_feat.get('delivery_z', 50)
        ]
        payload_id = wf_feat.get('payload_id', '')
        payload_weight = float(wf_feat.get('payload_weight', 1.0))
        
        checkpoints = [
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 0,
                'state_name': 'picking_up',
                'description': f'前往坐标({pickup_loc[0]}, {pickup_loc[1]}, {pickup_loc[2]})取货，物品ID: {payload_id}，重量: {payload_weight}kg',
                'location': pickup_loc,
                'operation_type': 'pickup',
                'requirements': ['mobility_component', 'logistics_component'],
                'next_states': ['transporting'],
                'payload_weight': payload_weight
            },
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 1,
                'state_name': 'transporting',
                'description': f'运输物品{payload_id}从({pickup_loc[0]}, {pickup_loc[1]}, {pickup_loc[2]})到({delivery_loc[0]}, {delivery_loc[1]}, {delivery_loc[2]})',
                'location': delivery_loc,
                'operation_type': 'transport',
                'requirements': ['logistics_component'],
                'next_states': ['delivering', 'completed'],
                'payload_weight': payload_weight
            },
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 2,
                'state_name': 'delivering',
                'description': f'在坐标({delivery_loc[0]}, {delivery_loc[1]}, {delivery_loc[2]})交付物品{payload_id}，重量: {payload_weight}kg',
                'location': delivery_loc,
                'operation_type': 'delivery',
                'requirements': ['logistics_component'],
                'next_states': ['completed'],
                'payload_weight': payload_weight
            },
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 3,
                'state_name': 'completed',
                'description': f'物流任务完成，物品{payload_id}已成功交付',
                'location': delivery_loc,
                'operation_type': 'completion',
                'requirements': [],
                'next_states': [],
                'payload_weight': payload_weight
            }
        ]
        return checkpoints
    
    def _extract_inspection_checkpoints(self, wf_feat: Dict) -> List[Dict]:
        """
        提取InspectionWorkflow的checkpoints
        
        正确逻辑（基于Oracle）：
        - 每个inspection point产生2个checkpoints：
          1. MoveToTask: 移动到巡检点
          2. InspectionTask: 执行巡检
        - 最后加一个completed checkpoint
        """
        workflow_id = wf_feat.get('workflow_id', '')
        n_points = wf_feat.get('n_inspection_points', 0)
        data_collection_type = wf_feat.get('data_collection_type', 'image')
        quality_threshold = float(wf_feat.get('quality_threshold', 0.8))
        inspection_interval = float(wf_feat.get('inspection_interval', 60))
        
        # 提取所有巡检点
        inspection_points = []
        for i in range(n_points):
            x = wf_feat.get(f'insp_point_{i}_x', 0)
            y = wf_feat.get(f'insp_point_{i}_y', 0)
            z = wf_feat.get(f'insp_point_{i}_z', 50)
            inspection_points.append([x, y, z])
        
        checkpoints = []
        ckpt_idx = 0
        
        # 为每个inspection point生成2个checkpoints
        for point_idx, location in enumerate(inspection_points):
            # 1. MoveToTask: 移动到巡检点
            checkpoints.append({
                'workflow_id': workflow_id,
                'checkpoint_idx': ckpt_idx,
                'state_name': f'moving_to_point_{point_idx}',
                'description': f'移动到巡检点{point_idx+1}/{n_points}，位置({location[0]:.1f}, {location[1]:.1f}, {location[2]:.1f})',
                'location': location,
                'operation_type': 'move_to_inspection',
                'requirements': ['mobility_component'],
                'next_states': [f'inspecting_point_{point_idx}'],
                'quality_threshold': quality_threshold,
                'inspection_interval': inspection_interval,
                'n_inspection_points': n_points
            })
            ckpt_idx += 1
            
            # 2. InspectionTask: 执行巡检
            checkpoints.append({
                'workflow_id': workflow_id,
                'checkpoint_idx': ckpt_idx,
                'state_name': f'inspecting_point_{point_idx}',
                'description': f'在巡检点{point_idx+1}/{n_points}执行{data_collection_type}数据采集',
                'location': location,
                'operation_type': 'inspection',
                'requirements': ['inspection_component'],
                'next_states': [f'moving_to_point_{point_idx+1}' if point_idx < n_points-1 else 'completed'],
                'quality_threshold': quality_threshold,
                'inspection_interval': inspection_interval,
                'n_inspection_points': n_points
            })
            ckpt_idx += 1
        
        # InspectionWorkflow不需要completed状态（与Oracle一致）
        return checkpoints
    
    def _extract_communication_checkpoints(self, wf_feat: Dict) -> List[Dict]:
        """
        提取CommunicationWorkflow的checkpoints
        
        Oracle模式：只有2个checkpoints
        1. MoveToTask: 移动到中继位置
        2. CommunicationTask: 传输数据
        （没有completed状态）
        """
        workflow_id = wf_feat.get('workflow_id', '')
        target_loc = [
            wf_feat.get('target_x', 0),
            wf_feat.get('target_y', 0),
            wf_feat.get('target_z', 50)
        ]
        data_size_mb = float(wf_feat.get('data_size_mb', 0))
        max_latency_s = float(wf_feat.get('max_end_to_end_latency_s', 300.0))
        
        checkpoints = [
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 0,
                'state_name': 'moving_to_relay_position',
                'description': f'移动到中继位置({target_loc[0]:.1f}, {target_loc[1]:.1f}, {target_loc[2]:.1f})',
                'location': target_loc,
                'operation_type': 'move_to_communication',
                'requirements': ['mobility_component'],
                'next_states': ['transmitting_data'],
                'comm_bandwidth_mb': data_size_mb,
                'max_latency_s': max_latency_s
            },
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 1,
                'state_name': 'transmitting_data',
                'description': f'传输{data_size_mb}MB数据到目标位置',
                'location': target_loc,
                'operation_type': 'communication',
                'requirements': ['communication_component'],
                'next_states': [],
                'comm_bandwidth_mb': data_size_mb,
                'max_latency_s': max_latency_s
            }
        ]
        return checkpoints
    
    def _extract_compute_checkpoints(self, wf_feat: Dict) -> List[Dict]:
        """
        提取ComputeWorkflow的checkpoints
        
        Oracle模式：只有2个checkpoints
        1. MoveToTask: 移动到计算位置
        2. ComputeTask: 执行计算
        （没有completed状态）
        """
        workflow_id = wf_feat.get('workflow_id', '')
        target_loc = [
            wf_feat.get('compute_x', 0),
            wf_feat.get('compute_y', 0),
            wf_feat.get('compute_z', 50)
        ]
        required_gflops = float(wf_feat.get('required_gflops', 0))
        cpu_intensity = float(wf_feat.get('cpu_intensity', 0.0))
        memory_intensity = float(wf_feat.get('memory_intensity', 0.0))
        io_intensity = float(wf_feat.get('io_intensity', 0.0))
        
        checkpoints = [
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 0,
                'state_name': 'moving_to_compute_location',
                'description': f'移动到计算位置({target_loc[0]:.1f}, {target_loc[1]:.1f}, {target_loc[2]:.1f})',
                'location': target_loc,
                'operation_type': 'move_to_compute',
                'requirements': ['mobility_component'],
                'next_states': ['computing'],
                'compute_gflops': required_gflops,
                'cpu_intensity': cpu_intensity,
                'memory_intensity': memory_intensity,
                'io_intensity': io_intensity
            },
            {
                'workflow_id': workflow_id,
                'checkpoint_idx': 1,
                'state_name': 'computing',
                'description': f'执行计算任务，需求{required_gflops}GFLOPS',
                'location': target_loc,
                'operation_type': 'compute',
                'requirements': ['compute_component'],
                'next_states': [],
                'compute_gflops': required_gflops,
                'cpu_intensity': cpu_intensity,
                'memory_intensity': memory_intensity,
                'io_intensity': io_intensity
            }
        ]
        return checkpoints
    
    def _build_properties_from_features(self, workflow_features: Dict, workflow_type: str) -> Dict:
        """
        从workflow_features构建properties字典
        
        Args:
            workflow_features: 从initial_data_loader提取的特征
            workflow_type: workflow类型
            
        Returns:
            properties字典
        """
        properties = {}
        
        if workflow_type == 'LogisticsWorkflow':
            # 提取pickup和delivery位置
            if 'pickup_x' in workflow_features:
                properties['pickup_location'] = (
                    workflow_features['pickup_x'],
                    workflow_features['pickup_y'],
                    workflow_features['pickup_z']
                )
            
            if 'delivery_x' in workflow_features:
                properties['delivery_location'] = (
                    workflow_features['delivery_x'],
                    workflow_features['delivery_y'],
                    workflow_features['delivery_z']
                )
            
            properties['payload_id'] = workflow_features.get('payload_id', '')
            properties['payload_weight'] = workflow_features.get('payload_weight', 1.0)
            
        elif workflow_type == 'InspectionWorkflow':
            # 提取巡检点
            inspection_points = []
            n_points = workflow_features.get('n_inspection_points', 0)
            for i in range(n_points):
                x_key = f'insp_point_{i}_x'
                y_key = f'insp_point_{i}_y'
                z_key = f'insp_point_{i}_z'
                if x_key in workflow_features:
                    point = (
                        workflow_features[x_key],
                        workflow_features[y_key],
                        workflow_features[z_key]
                    )
                    inspection_points.append(point)
            
            properties['inspection_points'] = inspection_points
            properties['inspection_mode'] = workflow_features.get('inspection_mode', 'sequential')
            properties['data_collection_type'] = workflow_features.get('data_collection_type', 'image')
            properties['inspection_interval'] = workflow_features.get('inspection_interval', 60)
            properties['quality_threshold'] = workflow_features.get('quality_threshold', 0.8)
            
        elif workflow_type == 'CommunicationWorkflow':
            # 提取目标位置
            if 'target_x' in workflow_features:
                properties['target_location'] = (
                    workflow_features['target_x'],
                    workflow_features['target_y'],
                    workflow_features['target_z']
                )
            
            properties['data_id'] = workflow_features.get('data_id', '')
            properties['data_size_mb'] = workflow_features.get('data_size_mb', 0)
            properties['max_end_to_end_latency_s'] = workflow_features.get('max_end_to_end_latency_s', 300.0)
            
        elif workflow_type == 'ComputeWorkflow':
            # 提取计算位置
            if 'compute_x' in workflow_features:
                properties['compute_location'] = (
                    workflow_features['compute_x'],
                    workflow_features['compute_y'],
                    workflow_features['compute_z']
                )
            
            properties['job_id'] = workflow_features.get('job_id', '')
            properties['required_gflops'] = workflow_features.get('required_gflops', 0)
            
            # workload_profile
            properties['workload_profile'] = {
                'type': workflow_features.get('workload_type', 'batch'),
                'priority': workflow_features.get('workload_priority', 'normal'),
                'cpu_intensity': workflow_features.get('cpu_intensity', 0),
                'memory_intensity': workflow_features.get('memory_intensity', 0),
                'io_intensity': workflow_features.get('io_intensity', 0)
            }
        
        # 通用属性
        properties['initial_status'] = workflow_features.get('initial_status', 'idle')
        
        return properties
    
    def extract_all_checkpoints(self, workflows_features: List[Dict]) -> pd.DataFrame:
        """
        为所有workflows提取checkpoints
        
        Args:
            workflows_features: workflow特征列表
            
        Returns:
            checkpoint DataFrame
        """
        all_checkpoints = []
        
        for wf_feat in workflows_features:
            ckpts = self.extract_checkpoints_for_workflow(wf_feat)
            
            # 添加 workflow_type 和 source_run_id 到每个 checkpoint（供后续指标使用）
            wf_type = wf_feat.get('workflow_type', '')
            for ckpt in ckpts:
                ckpt['workflow_type'] = wf_type
                if 'checkpoint_idx' not in ckpt:
                    ckpt['checkpoint_idx'] = 0
                if 'source_run_id' in wf_feat:
                    ckpt['source_run_id'] = wf_feat['source_run_id']
            
            all_checkpoints.extend(ckpts)
        
        df = pd.DataFrame(all_checkpoints)
        
        # 确保数值列存在并填充 NaN
        numeric_cols = [
            'payload_weight', 'compute_gflops', 'cpu_intensity', 'memory_intensity', 'io_intensity',
            'comm_bandwidth_mb', 'max_latency_s', 'quality_threshold', 'inspection_interval', 'n_inspection_points'
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        # 规范化 location 字段为长度为3的有限浮点数列表
        if 'location' in df.columns:
            def _sanitize_loc(loc):
                if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                    out = []
                    for k in range(3):
                        try:
                            v = float(loc[k])
                        except Exception:
                            v = 0.0
                        if pd.isna(v) or not np.isfinite(v):
                            v = 0.0
                        out.append(v)
                    return out
                return [0.0, 0.0, 0.0]
            df['location'] = df['location'].apply(_sanitize_loc)

        logger.info(f"  提取 {len(df)} 个checkpoints (来自 {len(workflows_features)} 个workflows)")
        if 'source_run_id' in df.columns:
            logger.info(f"  包含 {df['source_run_id'].nunique()} 个不同的runs")
        
        return df
    
    def extract_checkpoint_feature_vector(self, checkpoint: Dict) -> np.ndarray:
        """
        从checkpoint提取特征向量用于聚类
        
        特征包括：
        - 位置（location）- 3个
        - operation_type one-hot - 8个
        - requirements one-hot - 5个
        - 数值属性（所有workflow类型的数值字段）- 10个
        
        总共：3 + 8 + 5 + 10 = 26个特征
        
        Returns:
            特征向量 (26维)
        """
        features = []
        
        # 1. 位置特征（归一化到[0, 5000]范围）- 3个
        location = checkpoint.get('location', [0, 0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 3:
            features.append(float(location[0]) / 5000.0)
            features.append(float(location[1]) / 5000.0)
            features.append(float(location[2]) / 150.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 2. operation_type one-hot - 8个
        operation_type = checkpoint.get('operation_type', '')
        all_operation_types = ['pickup', 'transport', 'delivery', 'inspection', 
                               'compute', 'communication', 'completion', '']
        
        for op_type in all_operation_types:
            features.append(1.0 if operation_type == op_type else 0.0)
        
        # 3. requirements one-hot - 5个
        requirements = checkpoint.get('requirements', [])
        if not isinstance(requirements, list):
            requirements = []
        
        all_requirements = ['mobility_component', 'logistics_component', 
                           'inspection_component', 'communication_component', 
                           'compute_component']
        
        for req in all_requirements:
            features.append(1.0 if req in requirements else 0.0)
        
        # 4. 数值属性（归一化）- 10个
        # 辅助函数：安全获取数值，处理NaN
        def safe_get_float(d, key, default=0.0):
            val = d.get(key, default)
            if pd.isna(val):
                return default
            return float(val)
        
        # Logistics
        features.append(safe_get_float(checkpoint, 'payload_weight', 0.0) / 20.0)  # 最大20kg
        
        # Compute
        features.append(safe_get_float(checkpoint, 'compute_gflops', 0.0) / 2000.0)  # 最大2000 GFLOPS
        features.append(safe_get_float(checkpoint, 'cpu_intensity', 0.0))  # 已经是0-1
        features.append(safe_get_float(checkpoint, 'memory_intensity', 0.0))  # 已经是0-1
        features.append(safe_get_float(checkpoint, 'io_intensity', 0.0))  # 已经是0-1
        
        # Communication
        features.append(safe_get_float(checkpoint, 'comm_bandwidth_mb', 0.0) / 500.0)  # 最大500MB
        features.append(safe_get_float(checkpoint, 'max_latency_s', 0.0) / 300.0)  # 最大300s
        
        # Inspection
        features.append(safe_get_float(checkpoint, 'quality_threshold', 0.0))  # 已经是0-1
        features.append(safe_get_float(checkpoint, 'inspection_interval', 0.0) / 100.0)  # 最大100s
        features.append(safe_get_float(checkpoint, 'n_inspection_points', 0) / 10.0)  # 最大10个点
        
        return np.array(features, dtype=np.float32)
    
    def prepare_features_for_clustering(self, checkpoint_df: pd.DataFrame) -> np.ndarray:
        """
        准备用于聚类的特征矩阵
        
        Args:
            checkpoint_df: checkpoint DataFrame
            
        Returns:
            特征矩阵 (n_checkpoints, n_features)
        """
        features_list = []
        
        for idx, row in checkpoint_df.iterrows():
            feat = self.extract_checkpoint_feature_vector(row.to_dict())
            features_list.append(feat)
        
        X = np.array(features_list, dtype=np.float32)
        
        logger.info(f"  准备特征矩阵: {X.shape}")
        
        return X


# 便捷函数
def extract_checkpoints_from_initial_data(workflows_features: List[Dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    便捷函数：从初始数据提取checkpoints并准备特征矩阵
    
    Args:
        workflows_features: workflow特征列表
        
    Returns:
        (checkpoint_df, feature_matrix)
    """
    extractor = CheckpointExtractor()
    checkpoint_df = extractor.extract_all_checkpoints(workflows_features)
    feature_matrix = extractor.prepare_features_for_clustering(checkpoint_df)
    
    return checkpoint_df, feature_matrix
