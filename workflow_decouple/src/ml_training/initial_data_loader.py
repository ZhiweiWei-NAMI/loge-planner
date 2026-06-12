"""
初始信息数据加载器

仅使用初始信息：
1. workflow_init_info.csv - workflow初始参数
2. agent初始状态 - 从agent_snapshots.csv中每个agent第一次出现时提取
"""

import pandas as pd
import numpy as np
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InitialDataLoader:
    """初始信息数据加载器"""
    
    def __init__(self, data_dir: Path):
        """
        初始化加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.workflow_init_df = None
        self.agent_init_df = None
        self.agent_id_mapping_df = None  # Agent ID映射表
        
    def load_workflow_init(self) -> pd.DataFrame:
        """加载workflow初始信息"""
        workflow_file = self.data_dir / 'workflow_init_info.csv'
        if not workflow_file.exists():
            raise FileNotFoundError(f"未找到文件: {workflow_file}")
        
        self.workflow_init_df = pd.read_csv(workflow_file)
        logger.info(f"  加载 {len(self.workflow_init_df)} 个workflows")
        
        # 解析properties列
        self.workflow_init_df['properties_parsed'] = self.workflow_init_df['properties'].apply(
            self._parse_properties
        )
        
        return self.workflow_init_df
    
    def load_agent_initial_states(self) -> pd.DataFrame:
        """加载agent初始状态（每个agent第一次出现时的状态）
        
        首先尝试从 agent_snapshots.csv 读取，如果不存在或为空，
        则从 scenario.yaml 读取 agent 初始位置。
        """
        snapshots_file = self.data_dir / 'agent_snapshots.csv'
        scenario_file = self.data_dir / 'scenario.yaml'
        
        # Try loading from snapshots first
        if snapshots_file.exists():
            try:
                df = pd.read_csv(snapshots_file)
                if len(df) > 0:
                    # 提取每个agent第一次出现的记录
                    self.agent_init_df = df.groupby('agent_id').first().reset_index()
                    logger.info(f"  加载 {len(self.agent_init_df)} 个agents的初始状态 (from snapshots)")
                    
                    # 解析agent_state列
                    self.agent_init_df['agent_state_parsed'] = self.agent_init_df['agent_state'].apply(
                        self._parse_agent_state
                    )
                    return self.agent_init_df
            except Exception as e:
                logger.warning(f"  无法从agent_snapshots.csv加载: {e}")
        
        # Fallback: load from scenario.yaml
        if scenario_file.exists():
            logger.info("  从scenario.yaml加载agent初始状态...")
            return self._load_agent_states_from_scenario(scenario_file)
        
        raise FileNotFoundError(f"未找到agent_snapshots.csv或scenario.yaml: {self.data_dir}")
    
    def _load_agent_states_from_scenario(self, scenario_file: Path) -> pd.DataFrame:
        """从scenario.yaml加载agent初始状态"""
        import yaml
        
        with open(scenario_file, 'r') as f:
            scenario = yaml.safe_load(f)
        
        agents_data = []
        for agent in scenario.get('agents', []):
            agent_id = agent.get('id', '')
            pos = agent.get('initial_position', {})
            props = agent.get('properties', {})
            
            # Build agent state dict similar to snapshots format
            agent_state = {
                'position': [pos.get('x', 0), pos.get('y', 0), pos.get('z', 100)],
                'battery_level': props.get('battery_level', 100.0),
                'payload_capacity': props.get('payload_capacity', 5.0),
                'weight': props.get('weight', 2.5),
                'components': props.get('components', []),
            }
            
            agents_data.append({
                'agent_id': agent_id,
                'timestep': 0,
                'position_x': pos.get('x', 0),
                'position_y': pos.get('y', 0),
                'position_z': pos.get('z', 100),
                'agent_state': json.dumps(agent_state),
                'agent_state_parsed': agent_state,
            })
        
        self.agent_init_df = pd.DataFrame(agents_data)
        logger.info(f"  加载 {len(self.agent_init_df)} 个agents的初始状态 (from scenario.yaml)")
        
        return self.agent_init_df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载所有初始信息"""
        logger.info(f"\n从 {self.data_dir.name} 加载初始信息...")
        
        workflow_df = self.load_workflow_init()
        agent_df = self.load_agent_initial_states()
        
        # 加载agent ID映射
        mapping_file = self.data_dir / 'agent_id_mapping.csv'
        if mapping_file.exists():
            self.agent_id_mapping_df = pd.read_csv(mapping_file)
            logger.info(f"  加载 {len(self.agent_id_mapping_df)} 个agent映射")
        else:
            logger.warning(f"  未找到agent ID映射文件: {mapping_file}")
            self.agent_id_mapping_df = pd.DataFrame()
        
        return workflow_df, agent_df
    
    def _parse_properties(self, props_str: str) -> Dict:
        """解析properties字符串"""
        if pd.isna(props_str):
            return {}
        
        try:
            # 尝试用ast.literal_eval
            return ast.literal_eval(props_str)
        except:
            try:
                # 尝试JSON解析（替换单引号为双引号）
                props_str = props_str.replace("'", '"')
                return json.loads(props_str)
            except:
                logger.warning(f"无法解析properties: {props_str[:100]}")
                return {}
    
    def _parse_agent_state(self, state_str: str) -> Dict:
        """解析agent_state字符串"""
        if pd.isna(state_str):
            return {}
        
        try:
            return ast.literal_eval(state_str)
        except:
            try:
                state_str = state_str.replace("'", '"')
                return json.loads(state_str)
            except:
                logger.warning(f"无法解析agent_state")
                return {}
    
    def get_workflow_by_id(self, workflow_id: str) -> Optional[pd.Series]:
        """
        根据workflow_id获取workflow信息
        
        支持两种查询方式：
        1. 直接用workflow_id匹配（优先）
        2. 如果失败且存在original_workflow_id列，尝试用original匹配
        """
        if self.workflow_init_df is None:
            raise ValueError("workflow数据未加载")
        
        # 优先用workflow_id直接匹配
        rows = self.workflow_init_df[self.workflow_init_df['workflow_id'] == workflow_id]
        
        # 如果没找到，且存在original_workflow_id列，尝试用original匹配
        if len(rows) == 0 and 'original_workflow_id' in self.workflow_init_df.columns:
            rows = self.workflow_init_df[self.workflow_init_df['original_workflow_id'] == workflow_id]
        
        if len(rows) == 0:
            return None
        return rows.iloc[0]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[pd.Series]:
        """根据agent_id获取agent初始状态"""
        if self.agent_init_df is None:
            raise ValueError("agent数据未加载")
        
        rows = self.agent_init_df[self.agent_init_df['agent_id'] == agent_id]
        if len(rows) == 0:
            return None
        return rows.iloc[0]
    
    def extract_workflow_features(self, workflow_row: pd.Series) -> Dict:
        """
        从workflow提取特征（完整提取所有properties字段）
        
        Returns:
            特征字典
        """
        features = {
            'workflow_id': workflow_row['workflow_id'],
            'workflow_type': workflow_row['workflow_type'],
            'owner_agent': workflow_row['owner_agent'],
            'start_time': workflow_row['start_time']
        }
        
        # 如果有source_run_id列，提取它（用于合并数据集）
        if 'source_run_id' in workflow_row:
            features['source_run_id'] = workflow_row['source_run_id']
        
        # 添加properties中的信息
        props = workflow_row.get('properties_parsed', {})
        workflow_type = workflow_row['workflow_type']
        
        # === LogisticsWorkflow ===
        if workflow_type == 'LogisticsWorkflow':
            # pickup_location
            if 'pickup_location' in props:
                loc = props['pickup_location']
                if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                    features['pickup_x'] = loc[0]
                    features['pickup_y'] = loc[1]
                    features['pickup_z'] = loc[2]
            
            # delivery_location
            if 'delivery_location' in props:
                loc = props['delivery_location']
                if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                    features['delivery_x'] = loc[0]
                    features['delivery_y'] = loc[1]
                    features['delivery_z'] = loc[2]
            
            # 其他字段
            features['payload_id'] = props.get('payload_id', '')
            features['payload_weight'] = props.get('payload_weight', 0)
            features['source_agent_id'] = props.get('source_agent_id', '')
            features['target_agent_id'] = props.get('target_agent_id', '')
            
        # === InspectionWorkflow ===
        elif workflow_type == 'InspectionWorkflow':
            # inspection_points
            if 'inspection_points' in props:
                points = props['inspection_points']
                if isinstance(points, list):
                    features['n_inspection_points'] = len(points)
                    # 提取所有巡检点坐标
                    for i, point in enumerate(points):
                        if isinstance(point, (list, tuple)) and len(point) >= 3:
                            features[f'insp_point_{i}_x'] = point[0]
                            features[f'insp_point_{i}_y'] = point[1]
                            features[f'insp_point_{i}_z'] = point[2]
                    
                    # 计算巡检区域的边界框
                    if len(points) > 0:
                        xs = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) >= 3]
                        ys = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) >= 3]
                        if xs and ys:
                            features['insp_min_x'] = min(xs)
                            features['insp_max_x'] = max(xs)
                            features['insp_min_y'] = min(ys)
                            features['insp_max_y'] = max(ys)
                            features['insp_range_x'] = max(xs) - min(xs)
                            features['insp_range_y'] = max(ys) - min(ys)
                            features['insp_center_x'] = (max(xs) + min(xs)) / 2
                            features['insp_center_y'] = (max(ys) + min(ys)) / 2
            
            # 其他字段
            features['inspection_mode'] = props.get('inspection_mode', 'sequential')
            features['data_collection_type'] = props.get('data_collection_type', 'image')
            features['inspection_interval'] = props.get('inspection_interval', 60)
            features['quality_threshold'] = props.get('quality_threshold', 0.8)
            
        # === CommunicationWorkflow ===
        elif workflow_type == 'CommunicationWorkflow':
            # target_location
            if 'target_location' in props:
                loc = props['target_location']
                if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                    features['target_x'] = loc[0]
                    features['target_y'] = loc[1]
                    features['target_z'] = loc[2]
            
            # 其他字段
            features['data_id'] = props.get('data_id', '')
            features['data_size_mb'] = props.get('data_size_mb', 0)
            features['max_end_to_end_latency_s'] = props.get('max_end_to_end_latency_s', 300.0)
            
        # === ComputeWorkflow ===
        elif workflow_type == 'ComputeWorkflow':
            # compute_location
            if 'compute_location' in props:
                loc = props['compute_location']
                if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                    features['compute_x'] = loc[0]
                    features['compute_y'] = loc[1]
                    features['compute_z'] = loc[2]
            
            # 其他字段
            features['job_id'] = props.get('job_id', '')
            features['required_gflops'] = props.get('required_gflops', 0)
            
            # workload_profile
            if 'workload_profile' in props:
                profile = props['workload_profile']
                if isinstance(profile, dict):
                    features['workload_type'] = profile.get('type', 'batch')
                    features['workload_priority'] = profile.get('priority', 'normal')
                    features['cpu_intensity'] = profile.get('cpu_intensity', 0)
                    features['memory_intensity'] = profile.get('memory_intensity', 0)
                    features['io_intensity'] = profile.get('io_intensity', 0)
        
        # 通用：initial_status
        features['initial_status'] = props.get('initial_status', 'unknown')
        
        return features
    
    def extract_agent_features(self, agent_row: pd.Series) -> Dict:
        """
        从agent初始状态提取特征
        
        Returns:
            特征字典
        """
        features = {
            'agent_id': agent_row['agent_id'],
            'agent_type': agent_row['agent_type'],
            'init_x': agent_row['x'],
            'init_y': agent_row['y'],
            'init_z': agent_row['z'],
            'init_time': agent_row['time']
        }
        
        # 添加agent_state中的能力和资源信息
        state = agent_row.get('agent_state_parsed', {})
        
        # 能力参数
        features['payload_capacity'] = state.get('payload_capacity', 0)
        features['storage_available'] = state.get('storage_available', 0)
        features['communication_range'] = state.get('communication_range', 0)
        
        # 资源状态
        features['battery_level'] = state.get('battery_level', 100)
        features['cpu_utilization'] = state.get('cpu_utilization', 0)
        
        return features
    
    def get_workflow_agent_features(self, workflow_id: str) -> Optional[Dict]:
        """
        获取workflow及其对应agent的组合特征
        
        Args:
            workflow_id: workflow ID
            
        Returns:
            组合特征字典，如果不存在返回None
        """
        workflow = self.get_workflow_by_id(workflow_id)
        if workflow is None:
            return None
        
        wf_features = self.extract_workflow_features(workflow)
        
        # 获取对应的agent
        owner_agent_id = workflow['owner_agent']
        agent = self.get_agent_by_id(owner_agent_id)
        
        if agent is not None:
            agent_features = self.extract_agent_features(agent)
            # 合并特征
            combined = {**wf_features, **agent_features}
        else:
            logger.warning(f"未找到agent {owner_agent_id}")
            combined = wf_features
        
        return combined
    
    def get_all_workflows_with_agents(self) -> List[Dict]:
        """
        获取所有workflows及其agent的组合特征
        
        Returns:
            特征字典列表
        """
        if self.workflow_init_df is None or self.agent_init_df is None:
            raise ValueError("数据未加载，请先调用load_all()")
        
        all_features = []
        
        for idx, workflow_row in self.workflow_init_df.iterrows():
            workflow_id = workflow_row['workflow_id']
            features = self.get_workflow_agent_features(workflow_id)
            if features is not None:
                all_features.append(features)
        
        logger.info(f"  提取 {len(all_features)} 个workflow-agent组合特征")
        return all_features


# 便捷函数
def load_initial_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    便捷函数：加载初始数据并提取特征
    
    Args:
        data_dir: 数据目录
        
    Returns:
        (workflow_df, agent_df, combined_features)
    """
    loader = InitialDataLoader(data_dir)
    workflow_df, agent_df = loader.load_all()
    combined_features = loader.get_all_workflows_with_agents()
    
    return workflow_df, agent_df, combined_features
