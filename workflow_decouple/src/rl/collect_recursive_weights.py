"""
递归权重收集脚本

用法:
python -m workflow_decouple.src.rl.collect_recursive_weights \
    --policy data/rl_model_retrained/small_logistics_inspection/best_policy.pth \
    --encoder data/rl_model_retrained/small_logistics_inspection/encoder.pth \
    --data-root data/multi_seed_runs \
    --scenario-name small_logistics_inspection \
    --data-runs "42,43,44,45,46,47,48,49,50" \
    --max-depth 5 \
    --min-samples 50 \
    --output data/rl_model_retrained/small_logistics_inspection/weight_library.json
"""

import sys
import argparse
import logging
import json
from pathlib import Path
import torch

# 添加父目录到路径
parent_path = Path(__file__).parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

from rl.observation_encoder import ObservationEncoder
from rl.policy import MixedActionPolicy
from rl.recursive_weight_collector import RecursiveWeightCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='递归收集权重配置')
    
    # 模型参数
    parser.add_argument('--policy', type=str, required=True,
                       help='策略网络路径 (best_policy.pth)')
    parser.add_argument('--encoder', type=str, required=True,
                       help='观测编码器路径 (encoder.pth)')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, required=True,
                       help='multi_seed_runs根目录')
    parser.add_argument('--scenario-name', type=str, default='small_logistics_inspection',
                       help='场景名称')
    parser.add_argument('--data-runs', type=str, default='42,43,44,45,46,47,48,49,50',
                       help='数据run IDs（逗号分隔）')
    
    # 递归参数
    parser.add_argument('--max-depth', type=int, default=5,
                       help='最大递归深度')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='最小样本数（停止条件）')
    parser.add_argument('--min-variance', type=float, default=0.05,
                       help='最小方差（停止条件）')
    parser.add_argument('--split-strategy', type=str, default='binary',
                       choices=['binary', 'kmeans'],
                       help='分裂策略')
    
    # 输出参数
    parser.add_argument('--output', type=str, required=True,
                       help='输出权重库路径 (weight_library.json)')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备（cuda/cpu）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("🌳 递归权重收集脚本")
    print("="*80)
    
    # 解析数据目录
    data_root = Path(args.data_root) / args.scenario_name
    run_ids = [int(x.strip()) for x in args.data_runs.split(',')]
    data_dirs = [data_root / f'run_seed_{run_id}' for run_id in run_ids]
    
    # 检查目录是否存在
    missing_dirs = [d for d in data_dirs if not d.exists()]
    if missing_dirs:
        logger.error(f"以下目录不存在: {[str(d) for d in missing_dirs]}")
        return
    
    print(f"📁 场景: {args.scenario_name}")
    print(f"📁 数据runs: {run_ids}")
    print(f"📁 总目录数: {len(data_dirs)}\n")
    
    # 加载编码器
    print("⚙️  Step 1: 加载观测编码器...")
    encoder = ObservationEncoder(normalize=True)
    encoder_path = Path(args.encoder)
    if encoder_path.exists():
        encoder.load_stats(encoder_path)
        logger.info(f"Loaded encoder stats from {encoder_path}")
    else:
        logger.warning(f"Encoder file not found: {encoder_path}, will fit from data")
        encoder.fit_normalization(data_dirs)
    
    # 加载策略
    print("🧠 Step 2: 加载策略网络...")
    policy_path = Path(args.policy)
    if not policy_path.exists():
        logger.error(f"Policy file not found: {policy_path}")
        return

    # 从checkpoint加载策略参数（兼容纯 state_dict 与带 config 的checkpoint）
    checkpoint = torch.load(policy_path, map_location='cpu')

    state_dict = None
    config = None
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config')
    else:
        # BanditTrainer.save_results / best_policy.pth 直接保存的是 state_dict
        state_dict = checkpoint

    if config is not None:
        policy = MixedActionPolicy(
            obs_dim=config['obs_dim'],
            weight_dim=config['weight_dim'],
            hidden_dims=config.get('hidden_dims', [256, 256]),
            dropout=config.get('dropout', 0.1)
        )
    else:
        # 回退到当前 encoder 维度和默认 26 维权重
        policy = MixedActionPolicy(
            obs_dim=encoder.obs_dim,
            weight_dim=26,
            hidden_dims=[256, 256],
            dropout=0.1
        )

    policy.load_state_dict(state_dict)
    logger.info(f"Loaded policy from {policy_path}")
    
    # 初始化收集器
    print("🔧 Step 3: 初始化递归收集器...")
    collector = RecursiveWeightCollector(
        policy=policy,
        encoder=encoder,
        split_strategy=args.split_strategy,
        device=args.device
    )
    
    # 递归收集权重
    print(f"\n🚀 Step 4: 开始递归收集权重...")
    print(f"   ├─ 最大深度: {args.max_depth}")
    print(f"   ├─ 最小样本: {args.min_samples}")
    print(f"   ├─ 最小方差: {args.min_variance}")
    print(f"   └─ 分裂策略: {args.split_strategy}\n")
    
    weight_library = collector.collect_weights(
        data_dirs=data_dirs,
        max_depth=args.max_depth,
        min_samples=args.min_samples,
        min_variance=args.min_variance
    )
    
    # 保存权重库
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(weight_library, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 收集统计:")
    print(f"   ├─ 总节点数: {weight_library['metadata']['total_nodes']}")
    print(f"   ├─ 总样本数: {weight_library['metadata']['total_samples']}")
    print(f"   └─ 输出路径: {output_path}")
    
    print("\n" + "="*80)
    print(f"✅ 权重库收集完成!")
    print("="*80)


if __name__ == '__main__':
    main()

