"""
基于Oracle协同增益率生成训练样本对/三元组

使用SynergyMetricCalculator计算每对checkpoint的协同增益率，
作为"真值标签"来构造Positive/Negative样本。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from itertools import combinations
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SynergyPairGenerator:
    """
    基于Oracle协同增益率生成训练样本对
    
    核心思路：
    1. 使用Oracle聚类结果(th=0.3)计算成对协同增益率
    2. R(i,j) = G({i,j}) / B({i,j})，G是协同增益，B是基线成本
    3. R高 → Positive pair，R低 → Negative pair
    """
    
    def __init__(
        self,
        data_dir: Path,
        checkpoint_df: pd.DataFrame,
        threshold_high: float = 0.2,  # R > 0.2为正样本
        threshold_low: float = 0.05,  # R < 0.05为负样本
        cache_file: str = None
    ):
        """
        Args:
            data_dir: 数据目录（包含Oracle聚类结果）
            checkpoint_df: checkpoint DataFrame
            threshold_high: 正样本阈值
            threshold_low: 负样本阈值
            cache_file: 缓存文件路径
        """
        self.data_dir = Path(data_dir)
        self.checkpoint_df = checkpoint_df
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.cache_file = cache_file
        
        # 加载Oracle数据
        self.oracle_df = self._load_oracle_data()
        
        # 协同增益率矩阵（延迟计算）
        self.synergy_ratio_matrix = None
    
    def _load_oracle_data(self):
        """加载Oracle聚类数据"""
        oracle_file = self.data_dir / 'workflow_checkpoint_clustering.csv'
        if not oracle_file.exists():
            raise FileNotFoundError(f"Oracle文件不存在: {oracle_file}")
        
        oracle_df = pd.read_csv(oracle_file)
        logger.info(f"加载Oracle数据: {len(oracle_df)} checkpoints")
        return oracle_df
    
    def compute_pairwise_synergy_ratios(self, max_pairs: int = None):
        """
        真实计算每对checkpoint的协同增益率（使用实际的synergy计算公式）
        
        注意：这是一个O(n²)操作，对于大规模数据会很慢
        
        Args:
            max_pairs: 最多计算多少对（None表示全部，但会限制同run内）
        
        Returns:
            synergy_ratio_matrix: [n_samples, n_samples] 矩阵
        """
        # 尝试从缓存加载
        if self.cache_file and Path(self.cache_file).exists():
            logger.info(f"从缓存加载: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.synergy_ratio_matrix = cache_data['synergy_ratio_matrix']
                return self.synergy_ratio_matrix
        
        n_samples = len(self.checkpoint_df)
        synergy_ratio_matrix = np.zeros((n_samples, n_samples))
        
        # 导入计算器
        import sys
        src_path = Path(__file__).parent.parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from compute_synergy_metric import SynergyMetricCalculator
        
        # 初始化计算器，传入相同的DataFrame确保数据一致
        calculator = SynergyMetricCalculator(self.data_dir, clustering_df=self.checkpoint_df)
        
        logger.info("真实计算成对协同增益率（使用实际synergy公式）...")
        logger.info(f"  总checkpoints: {n_samples}")
        logger.info(f"  只计算同run内的pairs")
        
        # 临时禁用compute_synergy_metric的logger
        import logging
        synergy_logger = logging.getLogger('compute_synergy_metric')
        original_level = synergy_logger.level
        synergy_logger.setLevel(logging.CRITICAL)
        
        # 预提取所有checkpoint的信息和UAV能力，避免重复提取
        logger.info("预提取所有checkpoint信息和UAV能力...")
        checkpoint_infos = []  # 存储每个checkpoint的info
        checkpoint_uavs = []   # 存储每个checkpoint的UAV能力
        
        for idx in tqdm(range(n_samples), desc="预提取数据"):
            ckpt = self.checkpoint_df.iloc[idx]
            # 提取checkpoint信息
            info = calculator.extract_checkpoint_info(
                ckpt['workflow_id'], 
                ckpt['checkpoint_idx']
            )
            
            # 提取UAV能力
            run_id = ckpt.get('source_run_id', None) if calculator.is_merged else None
            uav = calculator.extract_uav_capabilities(info['config_agent_id'], run_id=run_id)
            
            if uav is None:
                raise ValueError(f"预提取checkpoint {idx} 失败: 无法提取UAV能力 - "
                               f"config_agent_id={info['config_agent_id']}, run_id={run_id}, "
                               f"workflow_id={ckpt['workflow_id']}, checkpoint_idx={ckpt['checkpoint_idx']}")
            
            # 更新UAV的prev_location
            uav['prev_location'] = info['prev_location']
            
            checkpoint_infos.append(info)
            checkpoint_uavs.append(uav)
        
        logger.info(f"预提取完成：{n_samples} 个checkpoint")
        
        # 预计算能力兼容性矩阵 (n x n) - 每个UAV对每个checkpoint的兼容性
        logger.info("预计算能力兼容性矩阵...")
        capability_matrix = np.zeros((n_samples, n_samples), dtype=bool)
        for i in tqdm(range(n_samples), desc="计算能力兼容性"):
            for j in range(n_samples):
                capability_matrix[i, j] = calculator.check_capability_compatibility(
                    checkpoint_uavs[i], 
                    checkpoint_infos[j]['requirements']
                )
        
        # 预计算距离矩阵 (n x n) - 使用numpy向量化操作加速
        logger.info("预计算距离矩阵（向量化）...")
        
        # 提取所有UAV位置和checkpoint位置
        uav_positions = np.array([checkpoint_uavs[i]['prev_location'] for i in range(n_samples)])  # (n, 3)
        checkpoint_positions = np.array([checkpoint_infos[i]['location'] for i in range(n_samples)])  # (n, 3)
        
        # 向量化计算所有距离：使用广播 (n, 1, 3) - (1, n, 3) = (n, n, 3)
        diff = uav_positions[:, np.newaxis, :] - checkpoint_positions[np.newaxis, :, :]  # (n, n, 3)
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
        
        logger.info(f"  距离矩阵计算完成：{n_samples} x {n_samples}")
        
        logger.info("预计算完成，开始计算协同增益率...")
        
        # 提取 run_id 信息（用于跳过不同 run 的配对）
        run_ids = self.checkpoint_df['source_run_id'].values
        
        # 统计
        computed_pairs = 0
        skipped_pairs = 0
        skipped_cross_run = 0
        
        for i in tqdm(range(n_samples), desc="计算协同增益"):
            # 直接使用预提取的数据
            info_i = checkpoint_infos[i]
            uav_i = checkpoint_uavs[i]
            
            for j in range(i+1, n_samples):
                # 跳过不同 run 的配对
                if run_ids[i] != run_ids[j]:
                    skipped_cross_run += 1
                    continue
                
                # 直接使用预提取的数据
                info_j = checkpoint_infos[j]
                uav_j = checkpoint_uavs[j]
                
                # 真实计算协同增益率
                try:
                    
                    # 从预计算的矩阵中提取能力兼容性 (2x2)
                    cap_matrix = np.array([
                        [capability_matrix[i, i], capability_matrix[i, j]],
                        [capability_matrix[j, i], capability_matrix[j, j]]
                    ])
                    
                    # 从预计算的矩阵中提取距离成本 (2x2)
                    # 只有在兼容的情况下才使用距离，否则为inf
                    cost_matrix = np.array([
                        [
                            distance_matrix[i, i] if cap_matrix[0, 0] else np.inf,
                            distance_matrix[i, j] if cap_matrix[0, 1] else np.inf
                        ],
                        [
                            distance_matrix[j, i] if cap_matrix[1, 0] else np.inf,
                            distance_matrix[j, j] if cap_matrix[1, 1] else np.inf
                        ]
                    ])
                    
                    # 计算baseline cost（对角线分配: UAV_i->ckpt_i, UAV_j->ckpt_j）
                    baseline_cost = cost_matrix[0, 0] + cost_matrix[1, 1]
                    
                    if baseline_cost > 0 and baseline_cost < np.inf:
                        # 对于2x2矩阵，直接比较两种分配方式（比匈牙利算法更快）
                        # 分配1: (0,0)+(1,1) - 对角线（baseline）
                        # 分配2: (0,1)+(1,0) - 反对角线
                        alternative_cost = cost_matrix[0, 1] + cost_matrix[1, 0]
                        optimized_cost = min(baseline_cost, alternative_cost)
                        
                        # 计算协同增益率
                        synergy_gain = baseline_cost - optimized_cost
                        synergy_ratio = synergy_gain / baseline_cost
                        
                        # 理论范围 [0, 1.0]
                        if synergy_ratio < -0.01 or synergy_ratio > 1.01:
                            logger.warning(f"异常synergy_ratio={synergy_ratio:.3f} at pair ({i},{j})")
                    synergy_ratio = max(0.1, min(1., synergy_ratio))
                    
                    synergy_ratio_matrix[i, j] = synergy_ratio
                    synergy_ratio_matrix[j, i] = synergy_ratio
                    computed_pairs += 1
                    
                except Exception as e:
                    logger.warning(f"计算pair ({i},{j})失败: {e}")
                    synergy_ratio_matrix[i, j] = 0.0
                    synergy_ratio_matrix[j, i] = 0.0
                    skipped_pairs += 1
                
                # 限制最大计算对数
                if max_pairs and computed_pairs >= max_pairs:
                    logger.info(f"达到最大计算对数 {max_pairs}，停止计算")
                    break
            
            if max_pairs and computed_pairs >= max_pairs:
                break
        
        # 恢复logger级别
        synergy_logger.setLevel(original_level)
        
        logger.info(f"计算完成: {computed_pairs} 对真实计算, {skipped_pairs} 对失败, {skipped_cross_run} 对跨run跳过")
        self.synergy_ratio_matrix = synergy_ratio_matrix
        
        # 保存缓存
        if self.cache_file:
            logger.info(f"保存缓存到: {self.cache_file}")
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'synergy_ratio_matrix': synergy_ratio_matrix,
                    'threshold_high': self.threshold_high,
                    'threshold_low': self.threshold_low,
                    'computed_pairs': computed_pairs
                }, f)
        
        logger.info("协同增益率矩阵计算完成")
        return synergy_ratio_matrix
    
    def generate_pairs(self, n_pairs: int = 10000):
        """
        生成训练样本对
        
        Args:
            n_pairs: 生成样本对数量
        
        Returns:
            pairs: [(idx1, idx2, label), ...], label=1为正样本，0为负样本
        """
        if self.synergy_ratio_matrix is None:
            self.compute_pairwise_synergy_ratios()
        
        n_samples = len(self.checkpoint_df)
        pairs = []
        
        logger.info(f"生成{n_pairs}个训练样本对...")
        
        # 采样策略：保证正负样本平衡
        n_positive = n_pairs // 2
        n_negative = n_pairs - n_positive
        
        # 生成正样本
        positive_count = 0
        attempts = 0
        max_attempts = n_positive * 10
        
        while positive_count < n_positive and attempts < max_attempts:
            i, j = np.random.randint(0, n_samples, size=2)
            if i != j and self.synergy_ratio_matrix[i, j] >= self.threshold_high:
                pairs.append((i, j, 1))
                positive_count += 1
            attempts += 1
        
        # 生成负样本
        negative_count = 0
        attempts = 0
        
        while negative_count < n_negative and attempts < max_attempts:
            i, j = np.random.randint(0, n_samples, size=2)
            if i != j and self.synergy_ratio_matrix[i, j] < self.threshold_low:
                pairs.append((i, j, 0))
                negative_count += 1
            attempts += 1
        
        logger.info(f"生成完成: {positive_count}个正样本, {negative_count}个负样本")
        
        return pairs
    
    def generate_triplets(self, n_triplets: int = 10000, strategy: str = 'hard'):
        """
        生成训练三元组 (anchor, positive, negative)
        
        **重要**: 确保三元组中的所有checkpoints都来自同一个run
        
        Args:
            n_triplets: 生成三元组数量
            strategy: 'random' | 'hard' | 'semi-hard'
        
        Returns:
            triplets: [(anchor_idx, positive_idx, negative_idx), ...]
        """
        if self.synergy_ratio_matrix is None:
            self.compute_pairwise_synergy_ratios()
        
        n_samples = len(self.checkpoint_df)
        triplets = []
        
        # 获取每个checkpoint的run_id（如果有）
        if 'source_run_id' in self.checkpoint_df.columns:
            run_ids = self.checkpoint_df['source_run_id'].values
            logger.info(f"生成{n_triplets}个训练三元组 (strategy={strategy}, 限制同run)...")
        else:
            run_ids = None
            logger.info(f"生成{n_triplets}个训练三元组 (strategy={strategy})...")
        
        attempts = 0
        max_attempts = n_triplets * 10  # 防止无限循环
        
        for _ in tqdm(range(n_triplets)):
            if attempts >= max_attempts:
                logger.warning(f"达到最大尝试次数，仅生成了{len(triplets)}个三元组")
                break
            
            # 1. 随机选择anchor
            anchor = np.random.randint(0, n_samples)
            anchor_run = run_ids[anchor] if run_ids is not None else None
            
            # 2. 选择positive：与anchor协同高的（且同run）
            positive_candidates = np.where(
                self.synergy_ratio_matrix[anchor] >= self.threshold_high
            )[0]
            
            # 限制同run
            if run_ids is not None:
                positive_candidates = positive_candidates[
                    run_ids[positive_candidates] == anchor_run
                ]
            
            if len(positive_candidates) == 0:
                attempts += 1
                continue
            
            positive = np.random.choice(positive_candidates)
            
            # 3. 选择negative：与anchor协同低的（且同run）
            if strategy == 'random':
                # 随机负样本
                negative_candidates = np.where(
                    self.synergy_ratio_matrix[anchor] < self.threshold_low
                )[0]
            elif strategy == 'hard':
                # 硬负样本：与anchor距离近但标签不同
                negative_candidates = np.where(
                    (self.synergy_ratio_matrix[anchor] < self.threshold_low) &
                    (self.synergy_ratio_matrix[anchor] > 0.01)  # 不要太简单的
                )[0]
            else:  # semi-hard
                # 半硬负样本：介于positive和margin之间
                anchor_pos_dist = 1 - self.synergy_ratio_matrix[anchor, positive]
                negative_candidates = np.where(
                    (self.synergy_ratio_matrix[anchor] < self.threshold_low) &
                    (1 - self.synergy_ratio_matrix[anchor] < anchor_pos_dist + 0.5)
                )[0]
            
            # 限制同run
            if run_ids is not None:
                negative_candidates = negative_candidates[
                    run_ids[negative_candidates] == anchor_run
                ]
            
            if len(negative_candidates) == 0:
                # 回退：任意低协同的同run checkpoint
                negative_candidates = np.where(
                    self.synergy_ratio_matrix[anchor] < self.threshold_low
                )[0]
                if run_ids is not None:
                    negative_candidates = negative_candidates[
                        run_ids[negative_candidates] == anchor_run
                    ]
            
            if len(negative_candidates) == 0:
                attempts += 1
                continue
            
            negative = np.random.choice(negative_candidates)
            
            triplets.append((anchor, positive, negative))
            attempts += 1
        
        logger.info(f"生成完成: {len(triplets)}个三元组")
        
        # 验证：检查是否所有三元组都来自同一个run
        if run_ids is not None and len(triplets) > 0:
            same_run_count = sum(
                1 for a, p, n in triplets 
                if run_ids[a] == run_ids[p] == run_ids[n]
            )
            logger.info(f"  验证: {same_run_count}/{len(triplets)} ({same_run_count/len(triplets)*100:.1f}%) 三元组来自同一run")
        
        return triplets
