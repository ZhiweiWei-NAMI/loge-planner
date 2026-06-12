"""
Bandit训练脚本

用法：
python train_bandit.py --data-root data/multi_seed_runs --output-dir data/bandit_results
"""

import sys
import argparse
import logging
from pathlib import Path
import torch

# 添加父目录到路径
parent_path = Path(__file__).parent.parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

from rl.observation_encoder import ObservationEncoder
from rl.policy import MixedActionPolicy
from rl.bandit_trainer import BanditTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='训练Bandit策略')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, required=True,
                       help='multi_seed_runs根目录')
    parser.add_argument('--scenario-name', type=str, default='large_mixed',
                       help='场景名称（如large_mixed, small_logistics_inspection等）')
    parser.add_argument('--train-runs', type=str, default='42,43,44,45,46,47,48',
                       help='训练run IDs（逗号分隔）')
    parser.add_argument('--val-runs', type=str, default='49,50',
                       help='验证run IDs（逗号分隔）')
    
    # 模型参数
    parser.add_argument('--obs-dim', type=int, default=120,
                       help='观测维度')
    parser.add_argument('--weight-dim', type=int, default=26,
                       help='权重维度（特征维度）')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256],
                       help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比例')
    
    # 训练参数
    parser.add_argument('--n-epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--baseline-momentum', type=float, default=0.9,
                       help='基线动量')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='熵正则化系数')
    parser.add_argument('--eval-every', type=int, default=5,
                       help='评估间隔')
    parser.add_argument('--early-stopping', type=int, default=20,
                       help='早停patience')
    
    # 聚类参数（新增：固定簇数）
    parser.add_argument('--n-clusters', type=int, default=30,
                       help='每个run的固定簇数（超参数，不再由策略采样）')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, 
                       default=None,
                       help='输出目录（默认: <data-root>/<scenario-name>/bandit_results）')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备（cuda/cpu）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("🎯 Bandit训练脚本")
    print("="*80)
    print(f"📋 场景名称: {args.scenario_name}")
    
    # 解析数据目录
    data_root = Path(args.data_root) / args.scenario_name  # 添加scenario子目录
    train_run_ids = [int(x.strip()) for x in args.train_runs.split(',')]
    val_run_ids = [int(x.strip()) for x in args.val_runs.split(',')]
    
    training_dirs = [data_root / f'run_seed_{run_id}' for run_id in train_run_ids]
    validation_dirs = [data_root / f'run_seed_{run_id}' for run_id in val_run_ids]
    
    # 检查目录是否存在
    missing_dirs = [d for d in training_dirs + validation_dirs if not d.exists()]
    if missing_dirs:
        logger.error(f"以下目录不存在: {[str(d) for d in missing_dirs]}")
        return
    
    print(f"📁 训练runs: {train_run_ids}")
    print(f"📁 验证runs: {val_run_ids}\n")
    
    # 初始化观测编码器
    print("⚙️  Step 1: 初始化观测编码器...")
    encoder = ObservationEncoder(normalize=True)
    encoder.fit_normalization(training_dirs)
    
    # 初始化策略
    print(f"🧠 Step 2: 初始化策略网络...")
    policy = MixedActionPolicy(
        obs_dim=encoder.obs_dim,
        weight_dim=args.weight_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    )
    
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"   ├─ 策略参数: {n_params:,}")
    print(f"   ├─ 观测维度: {encoder.obs_dim}")
    print(f"   ├─ 权重维度: {args.weight_dim}")
    print(f"   ├─ 固定簇数: {args.n_clusters} (超参数)")
    print(f"   └─ 注意: 策略仅训练特征权重，簇数不参与采样\n")
    
    # 初始化训练器
    print("🔧 Step 3: 初始化训练器...")
    trainer = BanditTrainer(
        policy=policy,
        encoder=encoder,
        n_clusters_fixed=args.n_clusters,  # 传递固定簇数
        learning_rate=args.lr,
        baseline_momentum=args.baseline_momentum,
        entropy_coef=args.entropy_coef,
        device=args.device
    )
    
    # 训练
    print(f"\n🚀 Step 4: 开始训练...\n")
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = data_root / 'bandit_results'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.train(
        training_dirs=training_dirs,
        n_epochs=args.n_epochs,
        eval_every=args.eval_every,
        validation_dirs=validation_dirs,
        early_stopping_patience=args.early_stopping,
        save_dir=output_dir
    )
    
    # 最终评估
    print(f"\n📊 Step 5: 最终评估...")
    
    train_reward = trainer.evaluate(training_dirs, verbose=False)
    print(f"   ├─ 训练集奖励: {train_reward:.4f}")
    
    if validation_dirs:
        val_reward = trainer.evaluate(validation_dirs, verbose=False)
        print(f"   └─ 验证集奖励: {val_reward:.4f}")
    
    print("\n" + "="*80)
    print(f"✅ 训练完成! 结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
