"""
Triplet Learning训练脚本

使用Oracle协同增益率构造Triplet样本，
训练Siamese Network学习协同关系。
"""

import sys
from pathlib import Path
import logging
import argparse
import torch
import numpy as np
import json

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_training.initial_data_loader import InitialDataLoader
from ml_training.checkpoint_extractor import CheckpointExtractor
from ml_training.synergy_evaluator import CheckpointSynergyEvaluator
from ml_training.deep_learning import (
    SiameseEncoder, ClusteringTrainer, SynergyPairGenerator
)
from sklearn.cluster import KMeans

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Triplet Learning训练（基于Oracle协同增益率）')
    
    # 数据参数
    default_data_dir = Path(__file__).resolve().parents[2] / "data" / "multi_seed_runs" / "large_mixed" / "merged_dataset"
    parser.add_argument('--data-dir', type=str,
                      default=str(default_data_dir),
                      help='数据目录（merged_dataset路径）')
    parser.add_argument('--train-runs', type=str, default='42,43,44,45,46,47,48',
                      help='训练run IDs（逗号分隔）')
    parser.add_argument('--val-runs', type=str, default='49,50',
                      help='验证run IDs（逗号分隔）')
    
    # 模型参数
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                      default=[64, 128, 64],
                      help='MLP隐藏层维度')
    parser.add_argument('--embed-dim', type=int, default=32,
                      help='Embedding维度')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout率')
    parser.add_argument('--n-clusters', type=int, default=15,
                      help='聚类簇数')
    
    # Triplet参数
    parser.add_argument('--n-triplets', type=int, default=20000,
                      help='Triplet样本数量')
    parser.add_argument('--margin', type=float, default=0.5,
                      help='Triplet Loss margin')
    parser.add_argument('--threshold-high', type=float, default=0.1,
                      help='正样本阈值（R>threshold为相似）')
    parser.add_argument('--threshold-low', type=float, default=0.05,
                      help='负样本阈值（R<threshold为不相似）')
    parser.add_argument('--triplet-strategy', type=str, default='hard',
                      choices=['random', 'hard', 'semi-hard'],
                      help='Triplet采样策略')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='权重衰减')
    parser.add_argument('--eval-every', type=int, default=5,
                      help='每N个epoch评估一次')
    parser.add_argument('--early-stopping', type=int, default=20,
                      help='早停patience')
    
    # 其他
    parser.add_argument('--device', type=str, default=None,
                      help='计算设备 (cuda/cpu)')
    parser.add_argument('--save-dir', type=str,
                      default=None,
                      help='结果保存目录（默认: <data-dir>/../triplet_learning_results）')
    parser.add_argument('--use-cache', action='store_true',
                      help='使用缓存的协同增益率矩阵')
    
    args = parser.parse_args()
    
    # 转换路径
    data_dir = Path(args.data_dir)
    
    # 设置保存目录
    if args.save_dir is None:
        save_dir = data_dir.parent / 'triplet_learning_results'
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析train和val runs
    train_runs = [int(x.strip()) for x in args.train_runs.split(',')]
    val_runs = [int(x.strip()) for x in args.val_runs.split(',')]
    
    logger.info("="*70)
    logger.info("Triplet Learning: 基于Oracle协同增益率")
    logger.info("="*70)
    logger.info(f"训练Runs: {train_runs}")
    logger.info(f"验证Runs: {val_runs}")
    
    # ========================================
    # Step 1: 加载数据和提取checkpoints
    # ========================================
    logger.info("\nStep 1: 加载初始数据和提取checkpoints")
    logger.info("-"*70)
    
    loader = InitialDataLoader(data_dir)
    loader.load_all()
    
    workflows_features = loader.get_all_workflows_with_agents()
    
    extractor = CheckpointExtractor()
    checkpoint_df = extractor.extract_all_checkpoints(workflows_features)
    
    logger.info(f"提取了 {len(checkpoint_df)} 个checkpoints")
    
    # ========================================
    # Step 2: 准备特征矩阵
    # ========================================
    logger.info("\nStep 2: 准备特征矩阵")
    logger.info("-"*70)
    
    feature_matrix = extractor.prepare_features_for_clustering(checkpoint_df)
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix_normalized = scaler.fit_transform(feature_matrix)
    
    logger.info(f"特征矩阵: {feature_matrix_normalized.shape}")
    
    # ========================================
    # Step 3: 生成Triplet样本（基于Oracle）
    # ========================================
    logger.info("\nStep 3: 生成Triplet样本（基于Oracle协同增益率）")
    logger.info("-"*70)
    
    cache_file = save_dir / 'synergy_ratio_cache.pkl' if args.use_cache else None
    
    pair_generator = SynergyPairGenerator(
        data_dir=data_dir,
        checkpoint_df=checkpoint_df,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        cache_file=cache_file
    )
    
    # 计算协同增益率矩阵
    pair_generator.compute_pairwise_synergy_ratios()
    
    # 生成triplets
    triplets = pair_generator.generate_triplets(
        n_triplets=args.n_triplets,
        strategy=args.triplet_strategy
    )
    
    logger.info(f"生成了 {len(triplets)} 个triplets")
    
    # ========================================
    # Step 4: 初始化Synergy Evaluator
    # ========================================
    logger.info("\nStep 4: 初始化Synergy Evaluator")
    logger.info("-"*70)
    
    synergy_evaluator = CheckpointSynergyEvaluator(data_dir)
    
    # ========================================
    # Step 5: 创建Siamese Encoder
    # ========================================
    logger.info("\nStep 5: 创建Siamese Encoder")
    logger.info("-"*70)
    
    input_dim = feature_matrix_normalized.shape[1]
    
    encoder = SiameseEncoder(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.embed_dim,
        dropout=args.dropout
    )
    
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"模型参数: {n_params:,}")
    
    # ========================================
    # Step 6: 基线性能（使用原始特征聚类）
    # ========================================
    logger.info("\nStep 6: 基线性能评估")
    logger.info("-"*70)
    
    from ml_training.hierarchical_clustering import get_cluster_statistics
    
    # 使用KMeans聚类（与训练时评估一致），每个run固定30个cluster
    n_clusters_per_run = 30
    logger.info(f"  使用KMeans聚类，每个run固定 {n_clusters_per_run} 个cluster")
    
    labels_baseline = np.zeros(len(checkpoint_df), dtype=int)
    run_ids = checkpoint_df['source_run_id'].unique()
    
    cluster_offset = 0
    for run_id in sorted(run_ids):
        run_mask = checkpoint_df['source_run_id'] == run_id
        run_features = feature_matrix_normalized[run_mask]
        
        n_samples = len(run_features)
        n_clusters_actual = min(n_clusters_per_run, n_samples)
        
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
        run_labels = kmeans.fit_predict(run_features)
        
        labels_baseline[run_mask] = run_labels + cluster_offset
        cluster_offset += n_clusters_actual
    
    # 输出统计
    stats = get_cluster_statistics(labels_baseline)
    logger.info(f"  基线聚类: {stats['n_clusters']} clusters, avg_size={stats['avg_cluster_size']:.1f}")
    
    # 按run分组评估（简洁模式）
    result_baseline = synergy_evaluator.evaluate_clustering_by_run(
        checkpoint_df=checkpoint_df,
        labels=labels_baseline,
        save_result=False,
        verbose=False
    )
    
    # 计算Sum Synergy Ratio
    baseline_cluster_scores = result_baseline.get('cluster_scores', [])
    baseline_synergy = sum(score['synergy_ratio'] for score in baseline_cluster_scores)
    
    # 对比Oracle（从metadata读取）
    comparison_baseline = synergy_evaluator.compare_with_oracle(result_baseline)
    baseline_relative = comparison_baseline['relative_performance']
    
    logger.info(f"基线 (层次聚类 on原始特征):")
    logger.info(f"  Sum Synergy Ratio: {baseline_synergy:.4f}")
    logger.info(f"  相对Oracle: {baseline_relative:.2%}")
    
    # ========================================
    # Step 7: 训练Triplet Network
    # ========================================
    logger.info("\nStep 7: 训练Triplet Network")
    logger.info("-"*70)
    
    trainer = ClusteringTrainer(
        encoder=encoder,
        synergy_evaluator=synergy_evaluator,
        checkpoint_df=checkpoint_df,
        features=feature_matrix_normalized,
        data_dir=data_dir,  # 用于加载Oracle阈值
        n_clusters=args.n_clusters,
        train_runs=train_runs,
        val_runs=val_runs,
        device=args.device
    )
    
    history = trainer.train_triplet(
        triplets=triplets,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        margin=args.margin,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping,
        save_dir=save_dir
    )
    
    # ========================================
    # Step 8: 最终评估
    # ========================================
    logger.info("\nStep 8: 最终评估")
    logger.info("-"*70)
    
    result_final = trainer.evaluate()
    
    # 计算Sum Synergy Ratio
    final_cluster_scores = result_final.get('cluster_scores', [])
    final_synergy = sum(score['synergy_ratio'] for score in final_cluster_scores)
    
    # 对比Oracle
    comparison_final = synergy_evaluator.compare_with_oracle(result_final)
    final_relative = comparison_final['relative_performance']
    
    logger.info(f"Triplet Learning:")
    logger.info(f"  Sum Synergy Ratio: {final_synergy:.4f}")
    logger.info(f"  相对Oracle: {final_relative:.2%}")
    
    # 性能提升
    improvement = (final_synergy - baseline_synergy) / baseline_synergy * 100
    logger.info(f"\n相对基线提升: {improvement:+.2f}%")
    
    # ========================================
    # Step 9: 保存结果
    # ========================================
    logger.info("\nStep 9: 保存结果")
    logger.info("-"*70)
    
    paths = trainer.save_results(save_dir)
    
    # 保存对比结果
    comparison = {
        'baseline': {
            'method': 'KMeans on Raw Features',
            'sum_synergy_ratio': float(baseline_synergy),
            'relative_to_oracle': float(baseline_relative),
        },
        'triplet_learning': {
            'method': 'Triplet Network (Oracle-guided)',
            'sum_synergy_ratio': float(final_synergy),
            'relative_to_oracle': float(final_relative),
            'improvement_over_baseline': float(improvement),
        },
        'oracle': {
            'sum_synergy_ratio': float(comparison_final['oracle_sum_synergy_ratio']),
        },
        'config': {
            'n_clusters': args.n_clusters,
            'embed_dim': args.embed_dim,
            'hidden_dims': args.hidden_dims,
            'n_triplets': len(triplets),
            'margin': args.margin,
            'threshold_high': args.threshold_high,
            'threshold_low': args.threshold_low,
            'triplet_strategy': args.triplet_strategy,
            'epochs': args.epochs,
            'lr': args.lr,
        }
    }
    
    comparison_path = save_dir / 'comparison_results.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"对比结果已保存: {comparison_path}")
    
    # ========================================
    # 总结
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("训练完成总结")
    logger.info("="*70)
    logger.info(f"基线方案: {baseline_synergy:.4f} ({baseline_relative:.2%})")
    logger.info(f"Triplet Learning: {final_synergy:.4f} ({final_relative:.2%})")
    logger.info(f"性能提升: {improvement:+.2f}%")
    logger.info(f"模型保存: {paths['model_path']}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
