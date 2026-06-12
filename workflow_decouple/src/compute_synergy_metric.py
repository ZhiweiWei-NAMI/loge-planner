"""
协同增效指标计算 - 基于任务重分配的调度灵活性增益

核心思想：
对于同一簇内的checkpoints，通过重新分配任务给能力兼容的UAV，
计算理论上可节省的飞行距离（即协同增效）。

增益 = 基线成本 - 理论最优成本
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
from scipy.optimize import linear_sum_assignment
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynergyMetricCalculator:
    """协同增效指标计算器"""
    
    def __init__(self, data_dir: Path, threshold: float = 0, is_merged: bool = None, 
                 clustering_df: pd.DataFrame = None):
        """
        初始化协同增效计算器
        
        Args:
            data_dir: 数据目录（包含聚类结果、processed数据、workflow初始化信息）
            threshold: 聚类阈值（可选）。如果指定，将优先查找带阈值后缀的聚类文件
            is_merged: 是否为合并数据（None则自动检测）
            clustering_df: 直接传入聚类DataFrame（优先级最高，避免文件选择不一致）
        """
        self.data_dir = data_dir
        self.threshold = threshold
        self._external_clustering_df = clustering_df  # 外部传入的DataFrame
        
        # 加载数据
        self.clustering_df = None
        self.processed_df = None
        self.workflow_init_df = None
        self.agent_snapshots_df = None
        
        # 性能优化：缓存和索引
        self._uav_capabilities_cache = {}  # 单run: {agent_id: caps}, 合并: {run_id: {agent_id: caps}}
        self._all_agents_dict = {}  # 跨场景模式：所有unique agents的能力字典
        self._workflow_info_index = {}  # workflow_id -> row 的索引
        self._agent_snapshots_index = {}  # agent_id -> sorted snapshots 的索引
        self._agent_mapping_index = {}  # (runtime_agent_id, source_run_id) -> config_agent_id
        self._clustering_index = {}  # (workflow_id, checkpoint_idx) -> row 的索引
        self._checkpoint_info_cache = {}  # 缓存checkpoint信息
        
        self._load_data()
        
        # 自动检测是否为合并数据
        if is_merged is None:
            self.is_merged = self._detect_merged_data()
        else:
            self.is_merged = is_merged
        
        logger.info(f"数据类型: {'合并数据' if self.is_merged else '单run数据'}")
        
        self._build_indices()  # 构建索引以加速查找
    
    def _find_clustering_file(self) -> Path:
        """
        智能查找聚类结果文件

        Returns:
            聚类文件路径，如果找不到返回None
        """
        # 优先级1：指定阈值的文件
        specific_file = self.data_dir / f'workflow_checkpoint_clustering_th{self.threshold:.2f}.csv'
        if specific_file.exists():
            logger.info(f"  找到指定阈值文件: {specific_file.name}")
            return specific_file
        return None
    
    def _load_data(self):
        """加载所有必要的数据"""
        logger.info("加载数据...")
        
        # 聚类结果 - 优先使用外部传入的DataFrame
        if self._external_clustering_df is not None:
            self.clustering_df = self._external_clustering_df
            logger.info(f"  聚类结果: {len(self.clustering_df)} checkpoints (使用外部传入DataFrame)")
        else:
            # 否则从文件读取
            clustering_file = self._find_clustering_file()
            if clustering_file and clustering_file.exists():
                self.clustering_df = pd.read_csv(clustering_file)
                logger.info(f"  聚类结果: {len(self.clustering_df)} checkpoints (来自 {clustering_file.name})")
            else:
                raise FileNotFoundError(f"聚类结果文件不存在: {self.data_dir / 'workflow_checkpoint_clustering*.csv'}")
        
        # 处理后的时空数据
        processed_file = self.data_dir / 'processed_task_spatial_temporal_coupling.csv'
        if processed_file.exists():
            self.processed_df = pd.read_csv(processed_file)
            logger.info(f"  时空数据: {len(self.processed_df)} 记录")
        else:
            raise FileNotFoundError(f"处理数据文件不存在: {processed_file}")
        
        # Workflow初始化信息（严格要求存在且可解析）
        workflow_init_file = self.data_dir / 'workflow_init_info.csv'
        if not workflow_init_file.exists():
            raise FileNotFoundError(f"Workflow初始化文件不存在: {workflow_init_file}")
        self.workflow_init_df = pd.read_csv(workflow_init_file)

        # 解析properties列为字典
        import ast
        def parse_properties(props_str):
            if pd.isna(props_str):
                raise ValueError("workflow_init_info.properties 存在 NaN 值")
            # 严格解析，失败直接抛错
            return ast.literal_eval(props_str)
        
        self.workflow_init_df['properties_parsed'] = self.workflow_init_df['properties'].apply(parse_properties)
        logger.info(f"  Workflow信息: {len(self.workflow_init_df)} workflows")
        
        # Agent快照（严格要求存在）
        snapshots_file = self.data_dir / 'agent_snapshots.csv'
        if not snapshots_file.exists():
            raise FileNotFoundError(f"Agent快照文件不存在: {snapshots_file}")
        self.agent_snapshots_df = pd.read_csv(snapshots_file)
        logger.info(f"  Agent快照: {len(self.agent_snapshots_df)} 记录")
        
        # Agent ID映射（严格要求存在）
        mapping_file = self.data_dir / 'agent_id_mapping.csv'
        if not mapping_file.exists():
            raise FileNotFoundError(f"Agent映射文件不存在: {mapping_file}")
        self.agent_id_mapping_df = pd.read_csv(mapping_file)
        logger.info(f"  Agent映射: {len(self.agent_id_mapping_df)} 个映射")
    
    def _build_indices(self):
        """构建索引以加速数据查找"""
        logger.info("构建数据索引...")
        # 检测clustering中的运行后缀（例如: _run42），用于对齐workflow_id
        run_suffixes: Set[str] = set()
        if self.clustering_df is not None and 'workflow_id' in self.clustering_df.columns:
            try:
                import re
                for wf in self.clustering_df['workflow_id'].astype(str).unique():
                    m = re.search(r'_run\d+$', wf)
                    if m:
                        run_suffixes.add(m.group(0))
            except Exception as e:
                logger.warning(f"  解析workflow_id后缀失败: {e}")
        
        # 1. 构建workflow_info索引
        if not self.workflow_init_df.empty:
            # 支持original_workflow_id和workflow_id两种方式
            for idx, row in self.workflow_init_df.iterrows():
                # 使用original_workflow_id作为主键（合并数据）
                if 'original_workflow_id' in row and pd.notna(row['original_workflow_id']):
                    key = row['original_workflow_id']
                    self._workflow_info_index[key] = row
                # 同时支持workflow_id（单run数据）
                if 'workflow_id' in row and pd.notna(row['workflow_id']):
                    key = row['workflow_id']
                    if key not in self._workflow_info_index:
                        self._workflow_info_index[key] = row
                    # 若clustering使用了带_runXX后缀的workflow_id，为该run添加带后缀的镜像key
                    for suffix in run_suffixes:
                        suffixed_key = f"{key}{suffix}"
                        if suffixed_key not in self._workflow_info_index:
                            self._workflow_info_index[suffixed_key] = row
            logger.info(f"  Workflow索引: {len(self._workflow_info_index)} 条")
        
        # 2. 构建agent_snapshots索引（按agent_id分组并排序）
        if not self.agent_snapshots_df.empty:
            for agent_id, group in self.agent_snapshots_df.groupby('agent_id'):
                self._agent_snapshots_index[agent_id] = group.sort_values('time')
            logger.info(f"  Agent快照索引: {len(self._agent_snapshots_index)} 个agent")
        
        # 3. 构建agent_mapping索引
        if not self.agent_id_mapping_df.empty:
            for idx, row in self.agent_id_mapping_df.iterrows():
                runtime_id = row['runtime_agent_id']
                source_run = row.get('source_run_id', None)
                config_id = row['config_agent_id']
                # 支持两种查找方式
                self._agent_mapping_index[(runtime_id, source_run)] = config_id
                if source_run is not None:
                    # 也建立不带source_run_id的索引（兼容性）
                    if (runtime_id, None) not in self._agent_mapping_index:
                        self._agent_mapping_index[(runtime_id, None)] = config_id
            logger.info(f"  Agent映射索引: {len(self._agent_mapping_index)} 条")
        
        # 4. 构建clustering索引
        if self.clustering_df is not None:
            for idx, row in self.clustering_df.iterrows():
                key = (row['workflow_id'], row['checkpoint_idx'])
                self._clustering_index[key] = row
            logger.info(f"  Clustering索引: {len(self._clustering_index)} 条")
        
        # 5. 预加载scenario.yaml（一次性读取）
        if self.is_merged:
            self._load_merged_scenario_capabilities()
        else:
            self._load_single_scenario_capabilities()
    
    def _detect_merged_data(self) -> bool:
        """自动检测是否为合并数据"""
        # 检查clustering_df和workflow_init_df都有source_run_id列
        has_in_clustering = self.clustering_df is not None and 'source_run_id' in self.clustering_df.columns
        has_in_workflow = self.workflow_init_df is not None and 'source_run_id' in self.workflow_init_df.columns
        # 两者都必须有source_run_id才算是合并数据
        return has_in_clustering and has_in_workflow
    
    def _load_single_scenario_capabilities(self):
        """加载单run的scenario.yaml"""
        scenario_file = self.data_dir / 'scenario.yaml'
        if not scenario_file.exists():
            logger.warning(f"scenario.yaml不存在: {scenario_file}")
            return
        
        try:
            import yaml
            with open(scenario_file, 'r', encoding='utf-8') as f:
                scenario = yaml.safe_load(f)
            
            agents = scenario['agents']
            for agent in agents:
                agent_id = agent['id']
                properties = agent['properties']
                components = properties['components']
                
                capabilities = {
                    'agent_id': agent_id,
                    'can_logistics': 'logistics' in components,
                    'can_compute': 'compute' in components or 'computation' in components,
                    'can_communication': 'communication' in components,
                    'can_inspection': 'inspection' in components,
                    'max_payload': properties['payload_capacity'],
                    'initial_position': agent['initial_position']
                }
                self._uav_capabilities_cache[agent_id] = capabilities
            
            logger.info(f"  预加载UAV能力: {len(self._uav_capabilities_cache)} 个agent")
        except Exception as e:
            logger.warning(f"加载scenario.yaml失败: {e}")
    
    def _load_merged_scenario_capabilities(self):
        """加载合并数据的所有run的scenario.yaml"""
        # 检查是否是跨场景合并数据（检查merge_metadata.json是否存在）
        merge_meta_file = self.data_dir / 'merge_metadata.json'
        if merge_meta_file.exists():
            # 跨场景：从merge_metadata.json读取scenario列表和data_root
            with open(merge_meta_file, 'r') as f:
                meta = json.load(f)
            scenarios = meta.get('scenarios', [])
            # 使用data_root（绝对路径）来解析scenario目录
            data_root = Path(meta.get('data_root', str(self.data_dir.parent)))
            run_dirs = []
            for scenario in scenarios:
                scenario_dir = data_root / scenario
                run_dirs.extend(sorted(scenario_dir.glob('run_seed_*')))
        elif self.data_dir.name == 'all_scenarios_merged':
            logger.warning(f"跨场景模式但未找到merge_metadata.json: {merge_meta_file}")
            return
        else:
            # 单场景：merged目录和run_seed_XX在同一级别，去父目录找
            run_dirs = sorted(self.data_dir.parent.glob('run_seed_*'))
        
        if not run_dirs:
            logger.warning(f"未找到run_seed_*目录: {self.data_dir.parent}")
            return
        
        logger.info(f"加载{len(run_dirs)}个run的scenario.yaml...")
        
        import yaml
        # 跨场景模式：收集所有unique的agent配置
        all_agents_dict = {}  # agent_id -> capabilities
        
        for run_dir in run_dirs:
            scenario_file = run_dir / 'scenario.yaml'
            if not scenario_file.exists():
                logger.warning(f"  {run_dir.name}: scenario.yaml不存在，跳过")
                continue
            
            try:
                # 提取run_id（从目录名）
                run_id = int(run_dir.name.split('_')[-1])
                
                with open(scenario_file, 'r', encoding='utf-8') as f:
                    scenario = yaml.safe_load(f)
                
                # 为该run创建agent能力字典
                run_capabilities = {}
                agents = scenario['agents']
                for agent in agents:
                    agent_id = agent['id']
                    properties = agent['properties']
                    components = properties['components']
                    
                    capabilities = {
                        'agent_id': agent_id,
                        'can_logistics': 'logistics' in components,
                        'can_compute': 'compute' in components or 'computation' in components,
                        'can_communication': 'communication' in components,
                        'can_inspection': 'inspection' in components,
                        'max_payload': properties['payload_capacity'],
                        'initial_position': agent['initial_position']
                    }
                    run_capabilities[agent_id] = capabilities
                    # 同时记录到全局agent字典（跨场景合并时用）
                    if agent_id not in all_agents_dict:
                        all_agents_dict[agent_id] = capabilities
                
                self._uav_capabilities_cache[run_id] = run_capabilities
                logger.info(f"  Run {run_id}: {len(run_capabilities)} 个agent")
                
            except Exception as e:
                logger.warning(f"  {run_dir.name}: 加载失败 - {e}")
        
        total_runs = len(self._uav_capabilities_cache)
        logger.info(f"  预加载完成: {total_runs} 个run的UAV能力")
        # 保存全局agent字典（跨场景用）
        if all_agents_dict:
            self._all_agents_dict = all_agents_dict
            logger.info(f"  全局agent池: {len(all_agents_dict)} 个unique agents")
    
    def extract_checkpoint_info(self, workflow_id: str, checkpoint_idx: int, 
                                debug: bool = False) -> Dict:
        """
        提取checkpoint的完整信息
        
        Returns:
            {
                'agent_id': str,
                'location': (x, y, z),
                'time_start': float,
                'time_end': float,
                'requirements': dict,  # 资源需求
                'prev_location': (x, y, z),  # UAV上一个位置
            }
        """
        # 使用缓存
        cache_key = (workflow_id, checkpoint_idx)
        if cache_key in self._checkpoint_info_cache:
            return self._checkpoint_info_cache[cache_key]
        
        # 使用索引快速查找
        ckpt_row = self._clustering_index.get(cache_key)
        if ckpt_row is None:
            # 备用路径：直接从DataFrame查询（性能较低，但能保证找到）
            query_result = self.clustering_df[
                (self.clustering_df['workflow_id'] == workflow_id) &
                (self.clustering_df['checkpoint_idx'] == checkpoint_idx)
            ]
            if query_result.empty:
                raise KeyError(f"找不到checkpoint: {workflow_id}, idx={checkpoint_idx}")
            ckpt_row = query_result.iloc[0]
            # 补充到索引中（避免下次再次走备用路径）
            self._clustering_index[cache_key] = ckpt_row
        
        # 使用original_workflow_id来查询原始数据（合并数据必须提供）
        if self.is_merged:
            original_wf_id = ckpt_row['original_workflow_id'] if 'original_workflow_id' in ckpt_row.index else ckpt_row['workflow_id']
        else:
            original_wf_id = workflow_id
        
        # 使用索引快速查找workflow信息
        wf_row = self._workflow_info_index.get(original_wf_id)
        
        if wf_row is None:
            raise KeyError(f"找不到workflow信息: workflow_id={workflow_id}, original_workflow_id={original_wf_id}")
        
        agent_id = wf_row.get('owner_agent')
        if agent_id is None:
            raise KeyError(f"Workflow {workflow_id} 缺少owner_agent字段")
        
        # 使用索引快速查找config_agent_id（严格模式）
        source_run_id = wf_row['source_run_id'] if self.is_merged else None
        config_agent_id = self._agent_mapping_index[(agent_id, source_run_id)]
        
        # 提取位置信息（支持两种格式）
        if 'avg_position_x' in ckpt_row:
            # 格式1：聚类结果（有avg_position_x/y/z字段）
            location = (
                ckpt_row['avg_position_x'],
                ckpt_row['avg_position_y'],
                ckpt_row['avg_position_z']
            )
        elif 'location' in ckpt_row:
            # 格式2：CheckpointExtractor生成（location是列表或字符串）
            loc = ckpt_row['location']
            if isinstance(loc, str):
                # 如果是字符串，尝试解析
                import ast
                try:
                    loc = ast.literal_eval(loc)
                except:
                    loc = [0, 0, 0]
            if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                location = (float(loc[0]), float(loc[1]), float(loc[2]))
            else:
                location = (0.0, 0.0, 0.0)
        else:
            logger.warning(f"Checkpoint {workflow_id}_idx{checkpoint_idx} 缺少位置信息")
            location = (0.0, 0.0, 0.0)
        
        # 提取时间信息（可选：部分数据集中不包含checkpoint级时间）
        time_start = ckpt_row['start_time'] if 'start_time' in ckpt_row.index else None
        time_end = ckpt_row['end_time'] if 'end_time' in ckpt_row.index else None
        
        info = {
            'workflow_id': workflow_id,
            'checkpoint_idx': checkpoint_idx,
            'agent_id': agent_id,
            'config_agent_id': config_agent_id,  # 配置文件中的agent名称
            'location': location,
            'time_start': time_start,
            'time_end': time_end,
        }
        
        # 提取资源需求（从workflow_init_info）
        if wf_row is not None:
            info['requirements'] = self._extract_requirements(wf_row, debug=debug)
        else:
            info['requirements'] = {}
            if debug:
                logger.debug(f"  无法提取requirements: 未找到workflow")
        
        # 提取UAV上一个位置
        info['prev_location'] = self._get_prev_location(
            workflow_id=workflow_id,
            checkpoint_idx=checkpoint_idx,
            agent_id=info['agent_id'], 
            time_start=info['time_start'],
            debug=False  # 默认关闭debug，在需要时可以启用
        )
        
        # 缓存结果
        self._checkpoint_info_cache[cache_key] = info
        
        return info
    
    def _extract_requirements(self, workflow_row, debug: bool = False) -> Dict:
        """
        从workflow行提取**所有**资源需求，确保机器可读（全部转换为数值）
        
        提取所有可能影响UAV能力匹配和成本计算的字段
        """
        wf_type = workflow_row.get('workflow_type')
        if wf_type is None:
            raise KeyError(f"Workflow缺少workflow_type字段")
        
        if debug:
            logger.debug(f"  提取requirements for workflow_type={wf_type}")
        
        # 初始化requirements（包含所有workflow类型的所有数值字段）
        requirements = {
            'type': wf_type,
            # Logistics字段
            'payload_weight': 0.0,
            # Compute字段
            'compute_gflops': 0.0,
            'cpu_intensity': 0.0,
            'memory_intensity': 0.0,
            'io_intensity': 0.0,
            # Communication字段
            'comm_bandwidth_mb': 0.0,
            'max_latency_s': 0.0,
            # Inspection字段
            'quality_threshold': 0.0,
            'inspection_interval': 0.0,
            'n_inspection_points': 0
        }
        
        # 从properties_parsed提取数值字段
        if 'properties_parsed' not in workflow_row:
            raise KeyError("workflow_row 缺少 properties_parsed 列")
        props = workflow_row['properties_parsed']
        if not isinstance(props, dict):
            raise TypeError(f"properties_parsed 不是字典: {type(props)}")

        wf_type_lower = wf_type.lower()

        if 'logistics' in wf_type_lower:
            requirements['payload_weight'] = float(props['payload_weight'])

        elif 'comput' in wf_type_lower:
            requirements['compute_gflops'] = float(props['required_gflops'])
            workload = props['workload_profile']
            requirements['cpu_intensity'] = float(workload['cpu_intensity'])
            requirements['memory_intensity'] = float(workload['memory_intensity'])
            requirements['io_intensity'] = float(workload['io_intensity'])

        elif 'communication' in wf_type_lower:
            requirements['comm_bandwidth_mb'] = float(props['data_size_mb'])
            requirements['max_latency_s'] = float(props['max_end_to_end_latency_s'])

        elif 'inspection' in wf_type_lower:
            requirements['quality_threshold'] = float(props['quality_threshold'])
            requirements['inspection_interval'] = float(props['inspection_interval'])
            inspection_points = props['inspection_points']
            if not isinstance(inspection_points, list):
                raise TypeError("inspection_points 必须为列表")
            requirements['n_inspection_points'] = int(len(inspection_points))
        
        return requirements
    
    def _get_prev_location(self, workflow_id: str, checkpoint_idx: int, 
                          agent_id: str, time_start: float, debug: bool = False) -> Tuple[float, float, float]:
        """
        获取agent在当前checkpoint之前的位置
        
        逻辑：
        1. 如果checkpoint_idx > 0：返回前一个checkpoint（checkpoint_idx-1）结束时的位置
        2. 如果checkpoint_idx == 0：返回agent最早出现的位置（第一个快照）
        3. 如果time_start小于agent所有快照的最小时间：返回agent最早位置
        
        Args:
            workflow_id: workflow ID
            checkpoint_idx: 当前checkpoint索引
            agent_id: agent ID
            time_start: 当前checkpoint开始时间
            debug: 是否输出debug信息
        
        Returns:
            (x, y, z) 位置坐标
        """
        # 使用预构建的索引
        agent_snapshots = self._agent_snapshots_index.get(agent_id)
        
        if agent_snapshots is None or agent_snapshots.empty:
            raise ValueError(f"agent_id={agent_id} 没有快照")
        
        # 情况1：如果是第一个checkpoint（idx=0），返回agent最早位置
        if checkpoint_idx == 0:
            first_snapshot = agent_snapshots.iloc[0]
            x = float(first_snapshot['x'])
            y = float(first_snapshot['y'])
            z = float(first_snapshot['z'])
            if debug:
                logger.debug(f"    [_get_prev_location] checkpoint_idx=0，使用最早快照: ({x:.2f}, {y:.2f}, {z:.2f}) at time={first_snapshot['time']:.2f}")
            return (x, y, z)
        
        # 情况2：checkpoint_idx > 0，查找前一个checkpoint的结束位置
        # 使用索引快速查找前一个checkpoint
        prev_key = (workflow_id, checkpoint_idx - 1)
        prev_ckpt_row = self._clustering_index.get(prev_key)
        
        if prev_ckpt_row is None:
            # 明确回退：使用agent快照在time_start之前的最后位置；若没有，则使用最早快照
            if agent_snapshots is None or agent_snapshots.empty:
                raise ValueError(f"agent_id={agent_id} 没有快照，且缺少前一个checkpoint: {workflow_id}, idx={checkpoint_idx-1}")
            if time_start is None:
                # 没有时间信息，使用最早快照（保持确定性）
                snap = agent_snapshots.iloc[0]
                return (float(snap['x']), float(snap['y']), float(snap['z']))
            # 选择time<=time_start的最后一条快照
            candidates = agent_snapshots[agent_snapshots['time'] <= time_start]
            if candidates.empty:
                snap = agent_snapshots.iloc[0]
            else:
                snap = candidates.iloc[-1]
            return (float(snap['x']), float(snap['y']), float(snap['z']))
        
        # 使用前一个checkpoint的位置（支持两种格式）
        if 'avg_position_x' in prev_ckpt_row:
            prev_x = prev_ckpt_row['avg_position_x']
            prev_y = prev_ckpt_row['avg_position_y']
            prev_z = prev_ckpt_row['avg_position_z']
        elif 'location' in prev_ckpt_row:
            loc = prev_ckpt_row['location']
            if isinstance(loc, str):
                import ast
                try:
                    loc = ast.literal_eval(loc)
                except:
                    loc = [0, 0, 0]
            if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                prev_x, prev_y, prev_z = float(loc[0]), float(loc[1]), float(loc[2])
            else:
                prev_x, prev_y, prev_z = 0.0, 0.0, 0.0
        else:
            prev_x, prev_y, prev_z = 0.0, 0.0, 0.0
        
        if debug:
            logger.debug(f"    [_get_prev_location] 使用前checkpoint的位置: ({prev_x:.2f}, {prev_y:.2f}, {prev_z:.2f})")
            start_t = prev_ckpt_row['start_time']
            end_t = prev_ckpt_row['end_time']
            logger.debug(f"    [_get_prev_location] 前checkpoint时间: {start_t:.2f} - {end_t:.2f}")
        
        return (prev_x, prev_y, prev_z)
    
    def extract_uav_capabilities(self, agent_id: str, run_id: int = None) -> Dict:
        """
        从缓存中获取UAV能力（已预加载）
        
        Args:
            agent_id: agent ID（config_agent_id）
            run_id: run ID（仅合并数据需要）
        
        Returns:
            {
                'agent_id': str,
                'can_logistics': bool,
                'can_compute': bool,
                'can_communication': bool,
                'can_inspection': bool,
                'max_payload': float,
                'max_compute_gflops': float,
                'max_comm_bandwidth': float,
            }
        """
        if self.is_merged:
            if run_id is None:
                raise ValueError(f"合并数据模式下需要提供run_id来查询agent {agent_id}")
            run_cache = self._uav_capabilities_cache.get(run_id, {})
            # 优先从run-specific cache查找
            if agent_id in run_cache:
                capabilities = run_cache[agent_id]
                return capabilities.copy()
            # 使用全局agent字典（跨场景模式）
            elif agent_id in self._all_agents_dict:
                capabilities = self._all_agents_dict[agent_id]
                return capabilities.copy()
            else:
                raise KeyError(f"agent_id={agent_id} 在run {run_id}和全局agent池中都不存在")
        else:
            capabilities = self._uav_capabilities_cache[agent_id]
            return capabilities.copy()
    
    def check_capability_compatibility(self, uav_cap: Dict, checkpoint_req: Dict) -> bool:
        """
        检查UAV能力是否满足checkpoint需求（使用完整的requirements字段）
        
        Returns:
            True if compatible, False otherwise
        """
        req_type = checkpoint_req['type']
        req_type_lower = req_type.lower()
        
        if 'logistics' in req_type_lower:
            # 检查logistics能力
            if not uav_cap['can_logistics']:
                return False
            # 检查payload容量
            if checkpoint_req['payload_weight'] > uav_cap['max_payload']:
                return False
                
        elif 'comput' in req_type_lower:
            # 检查compute能力
            if not uav_cap['can_compute']:
                return False
            
        elif 'communication' in req_type_lower:
            # 检查communication能力
            if not uav_cap['can_communication']:
                return False
            
        elif 'inspection' in req_type_lower:
            # 检查inspection能力
            if not uav_cap['can_inspection']:
                return False
            # quality_threshold和inspection_interval是任务参数，不影响能力兼容性
        
        return True
    
    @staticmethod
    def compute_distance(pos1: Tuple[float, float, float], 
                        pos2: Tuple[float, float, float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def compute_cluster_synergy(self, cluster_id) -> Dict:
        """
        计算单个簇的协同增效
        
        Args:
            cluster_id: 簇ID（可以是int或str）
        
        Returns:
            {
                'cluster_id': str,
                'cluster_size': int,
                'baseline_cost': float,  # 基线成本（原始分配）
                'optimized_cost': float,  # 理论最优成本（重新分配）
                'synergy_gain': float,  # 增益（节省的距离）
                'synergy_ratio': float,  # 增益比例
                'assignment': list,  # 最优分配方案
            }
        """
        # 获取该簇的所有checkpoints
        cluster_checkpoints = self.clustering_df[
            self.clustering_df['cluster_group'] == cluster_id
        ]
        
        k = len(cluster_checkpoints)
        if k == 0:
            return None
        
        logger.info(f"处理簇 {cluster_id}: {k} 个checkpoints")
        
        # 调试：对簇1输出详细信息（包含多种checkpoint_idx）
        debug_cluster = False
        debug_zero_cost = 'start_time' in cluster_checkpoints.columns and 'end_time' in cluster_checkpoints.columns  # 调试基线成本为0的问题
        
        # 提取checkpoints和UAVs信息（批量处理）
        checkpoints_info = []
        uavs_info = []
        
        for idx, row in cluster_checkpoints.iterrows():
            ckpt_info = self.extract_checkpoint_info(
                row['workflow_id'],
                row['checkpoint_idx']
            )
            checkpoints_info.append(ckpt_info)
            
            # 提取对应的UAV能力（使用config_agent_id在scenario.yaml中查找）
            # 如果是合并数据，需要传递run_id
            run_id = row.get('source_run_id', None) if self.is_merged else None
            # config_agent_id 在 scenario.yaml 中就是字符串形式（如 'agent_logistics_inspection_23'）
            config_agent_id = str(ckpt_info['config_agent_id'])
            uav_cap = self.extract_uav_capabilities(config_agent_id, run_id=run_id)
            
            if uav_cap is None:
                raise ValueError(f"无法提取UAV能力: config_agent_id={ckpt_info['config_agent_id']}, "
                               f"run_id={run_id}, workflow_id={row['workflow_id']}, "
                               f"checkpoint_idx={row['checkpoint_idx']}")
            
            # 更新UAV的prev_location（从checkpoint_info中获取）
            uav_cap['prev_location'] = ckpt_info['prev_location']
            uavs_info.append(uav_cap)
        
        # 构建能力兼容矩阵 (k x k)
        cap_matrix = np.zeros((k, k), dtype=bool)
        for j in range(k):  # UAV j
            for i in range(k):  # checkpoint i
                cap_matrix[j, i] = self.check_capability_compatibility(
                    uavs_info[j],
                    checkpoints_info[i]['requirements']
                )
        
        # 向量化构建成本矩阵 (k x k) - 性能优化
        # 提取所有位置到NumPy数组
        uav_positions = np.array([uav['prev_location'] for uav in uavs_info])  # (k, 3)
        checkpoint_positions = np.array([ckpt['location'] for ckpt in checkpoints_info])  # (k, 3)
        
        # 向量化计算所有距离：使用广播计算 (k, 1, 3) - (1, k, 3) = (k, k, 3)
        # 然后计算欧氏距离
        diff = uav_positions[:, np.newaxis, :] - checkpoint_positions[np.newaxis, :, :]  # (k, k, 3)
        cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # (k, k)
        
        # 应用能力约束：不兼容的设为inf
        cost_matrix[~cap_matrix] = np.inf
        
        # 调试输出
        if debug_cluster:
            logger.info(f"\n  === 簇{cluster_id}详细信息 ===")
            for i in range(k):
                logger.info(f"  Checkpoint {i}: {checkpoints_info[i]['workflow_id']}_ckp{checkpoints_info[i]['checkpoint_idx']}")
                logger.info(f"    Agent: {checkpoints_info[i]['agent_id']}")
                logger.info(f"    Prev location: {uavs_info[i]['prev_location']}")
                logger.info(f"    Current location: {checkpoints_info[i]['location']}")
                logger.info(f"    Type: {checkpoints_info[i]['requirements']['type']}")
            
            logger.info(f"\n  能力兼容矩阵 (UAV x Checkpoint):")
            logger.info(f"  {cap_matrix.astype(int)}")
            
            logger.info(f"\n  成本矩阵 (UAV x Checkpoint, 单位:m):")
            logger.info(f"  {np.round(cost_matrix, 1)}")
        
        # 计算基线成本（对角线：每个UAV执行自己的任务）
        baseline_cost = sum(cost_matrix[i, i] for i in range(k) if cost_matrix[i, i] < np.inf)
        
        # 调试：如果基线成本为0，输出详细信息
        if baseline_cost == 0 and k > 0:
            logger.warning(f"  ⚠️ 簇 {cluster_id} 基线成本为0！")
            logger.warning(f"    簇大小: {k}")
            inf_diag = sum(1 for i in range(k) if cost_matrix[i, i] == np.inf)
            zero_diag = sum(1 for i in range(k) if cost_matrix[i, i] == 0)
            if inf_diag > 0:
                logger.warning(f"    对角线上有 {inf_diag} 个inf成本（能力不兼容）")
            if zero_diag > 0:
                logger.warning(f"    对角线上有 {zero_diag} 个0成本（距离为0）")
            
            # 输出详细的诊断信息
            if debug_zero_cost:
                logger.warning(f"\n  === 簇{cluster_id}详细诊断信息 ===")
                
                # 统计checkpoint_idx分布
                checkpoint_idx_counts = cluster_checkpoints['checkpoint_idx'].value_counts().sort_index()
                logger.warning(f"  Checkpoint_idx分布:")
                for idx, count in checkpoint_idx_counts.head(10).items():
                    logger.warning(f"    checkpoint_idx={idx}: {count} 个")
                if len(checkpoint_idx_counts) > 10:
                    logger.warning(f"    ... (共 {len(checkpoint_idx_counts)} 种不同的idx)")
                
                # 统计有多少是第一个checkpoint (idx=0)
                num_first_ckpt = (cluster_checkpoints['checkpoint_idx'] == 0).sum()
                logger.warning(f"  第一个checkpoint (idx=0): {num_first_ckpt}/{k} ({num_first_ckpt/k*100:.1f}%)")
                
                # 重新提取前几个checkpoint的信息，并启用debug模式
                logger.warning(f"\n  重新提取前几个checkpoint的详细位置信息（启用debug）：")
                for i in range(min(k, 3)):  # 只重新提取前3个
                    row = cluster_checkpoints.iloc[i]
                    logger.warning(f"\n  --- 重新提取 Checkpoint {i} ---")
                    # 临时启用debug模式重新获取prev_location
                    ckpt_info_debug = {
                        'workflow_id': row['workflow_id'],
                        'checkpoint_idx': row['checkpoint_idx'],
                        'time_start': row['start_time']
                    }
                    
                    # 获取agent_id（使用original_workflow_id）
                    original_wf_id_debug = row.get('original_workflow_id', row['workflow_id'])
                    if 'original_workflow_id' in self.workflow_init_df.columns:
                        wf_info_debug = self.workflow_init_df[
                            self.workflow_init_df['original_workflow_id'] == original_wf_id_debug
                        ]
                    else:
                        wf_info_debug = self.workflow_init_df[
                            self.workflow_init_df['workflow_id'] == original_wf_id_debug
                        ]
                    
                    if wf_info_debug.empty:
                        raise KeyError(f"Debug: 找不到workflow信息: {original_wf_id_debug}")
                    
                    agent_id_debug = wf_info_debug.iloc[0]['owner_agent']
                    
                    logger.warning(f"  Workflow: {row['workflow_id']}, Checkpoint: {row['checkpoint_idx']}")
                    logger.warning(f"  Agent: {agent_id_debug}")
                    logger.warning(f"  Time: {row['start_time']:.2f} - {row['end_time']:.2f}")
                    logger.warning(f"  位置获取过程：")
                    
                    # 启用debug重新获取prev_location
                    prev_loc_debug = self._get_prev_location(
                        workflow_id=row['workflow_id'],
                        checkpoint_idx=row['checkpoint_idx'],
                        agent_id=agent_id_debug,
                        time_start=row['start_time'],
                        debug=True
                    )
                
                logger.warning(f"\n  --- Checkpoint摘要信息 ---")
                for i in range(min(k, 10)):  # 只输出前10个，避免太多
                    ckpt = checkpoints_info[i]
                    uav = uavs_info[i]
                    logger.warning(f"  [{i}] Checkpoint: {ckpt['workflow_id']}_ckp{ckpt['checkpoint_idx']}")
                    logger.warning(f"      Agent: {ckpt['agent_id']}")
                    logger.warning(f"      Prev location: {uav['prev_location']}")
                    logger.warning(f"      Current location: {ckpt['location']}")
                    logger.warning(f"      Distance: {cost_matrix[i, i]:.2f}m")
                    logger.warning(f"      Type: {ckpt['requirements']['type']}")
                    logger.warning(f"      Time: {ckpt['time_start']:.2f} - {ckpt['time_end']:.2f}")
                    if cost_matrix[i, i] == 0:
                        logger.warning(f"      ⚠️ 距离为0 - prev和current位置相同")
                    if cost_matrix[i, i] == np.inf:
                        logger.warning(f"      ⚠️ 距离为inf - 能力不兼容")
                if k > 10:
                    logger.warning(f"  ... (省略剩余 {k-10} 个checkpoint)")
        
        # 使用匈牙利算法求解最优分配
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            optimized_cost = cost_matrix[row_ind, col_ind].sum()
            
            # 如果有不可行的分配（cost=inf），说明没有完全兼容的方案
            if np.isinf(optimized_cost):
                logger.warning(f"  簇 {cluster_id} 没有完全兼容的分配方案")
                optimized_cost = baseline_cost  # 回退到基线
                synergy_gain = 0.0
            else:
                synergy_gain = baseline_cost - optimized_cost
        except:
            logger.error(f"  簇 {cluster_id} 最优分配求解失败")
            optimized_cost = baseline_cost
            synergy_gain = 0.0
            row_ind, col_ind = list(range(k)), list(range(k))
        
        # 计算增益比例
        synergy_ratio = synergy_gain / baseline_cost if baseline_cost > 0 else 0.0
        
        # 构建分配方案
        assignment = [
            {
                'uav': uavs_info[j]['agent_id'],
                'original_checkpoint': f"{checkpoints_info[j]['workflow_id']}_ckp{checkpoints_info[j]['checkpoint_idx']}",
                'assigned_checkpoint': f"{checkpoints_info[i]['workflow_id']}_ckp{checkpoints_info[i]['checkpoint_idx']}",
                'distance': float(cost_matrix[j, i]) if not np.isinf(cost_matrix[j, i]) else None
            }
            for j, i in zip(row_ind, col_ind)
        ]
        
        result = {
            'cluster_id': str(cluster_id),  # 支持字符串类型的cluster_id（合并数据时使用）
            'cluster_size': k,
            'baseline_cost': float(baseline_cost),
            'optimized_cost': float(optimized_cost),
            'synergy_gain': float(synergy_gain),
            'synergy_ratio': float(synergy_ratio),
            'assignment': assignment
        }
        
        logger.info(f"  基线成本: {baseline_cost:.2f}m")
        logger.info(f"  最优成本: {optimized_cost:.2f}m")
        logger.info(f"  增益: {synergy_gain:.2f}m ({synergy_ratio*100:.1f}%)")
        
        return result
    
    def evaluate_clustering(self, output_file: Path = None) -> Dict:
        """
        评估整个聚类方案
        
        Returns:
            {
                'total_synergy_gain': float,
                'avg_synergy_ratio': float,
                'total_baseline_cost': float,
                'total_optimized_cost': float,
                'cluster_scores': list,

            }
        """
        logger.info("\n" + "="*70)
        logger.info("开始评估聚类方案...")
        logger.info("="*70)
        
        all_cluster_ids = sorted(self.clustering_df['cluster_group'].unique())
        logger.info(f"总共 {len(all_cluster_ids)} 个簇")
        
        cluster_scores = []
        total_synergy_gain = 0.0
        total_baseline_cost = 0.0
        total_optimized_cost = 0.0
        
        for cluster_id in all_cluster_ids:
            score = self.compute_cluster_synergy(cluster_id)
            if score and score['synergy_ratio'] >= self.threshold:
                cluster_scores.append(score)
                total_synergy_gain += score['synergy_gain']
                total_baseline_cost += score['baseline_cost']
                total_optimized_cost += score['optimized_cost']
        
        avg_synergy_ratio = total_synergy_gain / total_baseline_cost if total_baseline_cost > 0 else 0.0
        
        result = {
            'total_synergy_gain': total_synergy_gain,
            'avg_synergy_ratio': avg_synergy_ratio,
            'total_baseline_cost': total_baseline_cost,
            'total_optimized_cost': total_optimized_cost,
            'num_clusters': len(cluster_scores),
            'cluster_scores': cluster_scores
        }
        
        logger.info("\n" + "="*70)
        logger.info("评估完成！")
        logger.info(f"总基线成本: {total_baseline_cost:.2f}m")
        logger.info(f"总最优成本: {total_optimized_cost:.2f}m")
        logger.info(f"总协同增益: {total_synergy_gain:.2f}m ({avg_synergy_ratio*100:.1f}%)")
        logger.info("="*70)
        
        # 保存结果
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\n结果已保存到: {output_file}")
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='计算聚类方案的协同增效指标'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='数据目录路径（包含聚类结果等文件）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='synergy_evaluation.json',
        help='输出文件路径（默认: synergy_evaluation.json）'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='聚类阈值（可选）。如果指定，将查找对应阈值的聚类文件'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_file = data_dir / args.output
    
    # 创建计算器并评估
    calculator = SynergyMetricCalculator(data_dir, threshold=args.threshold)
    result = calculator.evaluate_clustering(output_file)
    
    # 打印摘要
    print(f"\n✅ 评估完成！")
    print(f"总协同增益: {result['total_synergy_gain']:.2f}m")
    print(f"平均增益比例: {result['avg_synergy_ratio']*100:.1f}%")
    print(f"详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
