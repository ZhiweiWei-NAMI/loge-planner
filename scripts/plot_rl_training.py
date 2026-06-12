#!/usr/bin/env python3
"""
绘制 RL 训练收敛曲线

读取各场景的 training_history.json，绘制:
1. 训练/验证 reward 曲线
2. 各场景对比图
3. 权重分布热力图
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 场景显示名称
SCENARIO_NAMES = {
    "small_logistics_inspection": "Small-Logistics",
    "small_comm_compute": "Small-Compute", 
    "medium_logistics_heavy": "Medium-Logistics",
    "medium_inspection_heavy": "Medium-Inspection",
    "large_mixed": "Large-Mixed"
}

# 颜色
COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']


def load_training_history(history_path: Path) -> Dict:
    """加载训练历史"""
    if not history_path.exists():
        return None
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_single_scenario(history: Dict, scenario: str, output_path: Path):
    """绘制单个场景的训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 训练 reward
    ax1 = axes[0]
    # 支持多种 key 格式
    rewards = history.get("train_rewards") or history.get("mean_reward") or history.get("rewards")
    epochs = history.get("epoch") or (range(1, len(rewards) + 1) if rewards else [])
    
    if rewards:
        ax1.plot(epochs, rewards, 'b-', linewidth=2, label='Train Reward')
        
        # 移动平均
        if len(rewards) > 5:
            window = 5
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(list(epochs)[window-1:], ma, 'r--', linewidth=2, alpha=0.8, label=f'MA-{window}')
        
        # 如果有标准差，绘制置信区间
        if "std_reward" in history:
            std = np.array(history["std_reward"])
            rewards_arr = np.array(rewards)
            ax1.fill_between(epochs, rewards_arr - std, rewards_arr + std, alpha=0.2, color='blue')
    
    if "val_rewards" in history:
        val_rewards = history["val_rewards"]
        val_epochs = history.get("val_epochs", range(1, len(val_rewards)+1))
        ax1.plot(val_epochs, val_rewards, 'g-o', markersize=4, linewidth=2, label='Validation')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'{SCENARIO_NAMES.get(scenario, scenario)} - Training Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 权重分布 (如果有)
    ax2 = axes[1]
    if "final_weights" in history:
        weights = history["final_weights"]
        if isinstance(weights, list):
            ax2.bar(range(len(weights)), weights, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Weight')
            ax2.set_title('Final Weight Distribution')
            ax2.set_ylim(0, 1)
            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    elif "weight_history" in history and len(history["weight_history"]) > 0:
        # 显示最后一个 epoch 的权重
        last_weights = history["weight_history"][-1]
        if isinstance(last_weights, list):
            ax2.bar(range(len(last_weights)), last_weights, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Weight')
            ax2.set_title('Final Weight Distribution')
            ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No weight data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def plot_comparison(all_histories: Dict[str, Dict], output_path: Path):
    """绘制所有场景对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练曲线对比
    ax1 = axes[0]
    for i, (scenario, history) in enumerate(all_histories.items()):
        if history is None:
            continue
        
        rewards = history.get("train_rewards") or history.get("mean_reward") or history.get("rewards")
        epochs = history.get("epoch") or (range(1, len(rewards) + 1) if rewards else [])
        std = history.get("std_reward")

        if rewards:
            label = SCENARIO_NAMES.get(scenario, scenario)
            color = COLORS[i % len(COLORS)]
            # 主曲线
            ax1.plot(epochs, rewards, color=color,
                    linewidth=2, label=label, alpha=0.9)

            # 阴影面积（标准差），如果有的话
            if std is not None and len(std) == len(rewards):
                epochs_arr = np.array(list(epochs))
                rewards_arr = np.array(rewards)
                std_arr = np.array(std)
                ax1.fill_between(
                    epochs_arr,
                    rewards_arr - std_arr,
                    rewards_arr + std_arr,
                    color=color,
                    alpha=0.15,
                )
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Reward', fontsize=12)
    ax1.set_title('RL Training Convergence Comparison', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 最终性能对比 (柱状图)
    ax2 = axes[1]
    scenarios = []
    final_rewards = []
    colors = []
    
    for i, (scenario, history) in enumerate(all_histories.items()):
        if history is None:
            continue
        
        rewards = history.get("train_rewards") or history.get("mean_reward") or history.get("rewards")
        if rewards and len(rewards) > 0:
            scenarios.append(SCENARIO_NAMES.get(scenario, scenario))
            # 取最后10个epoch的平均
            final_rewards.append(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))
            colors.append(COLORS[i % len(COLORS)])
    
    if scenarios:
        bars = ax2.bar(range(len(scenarios)), final_rewards, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.set_ylabel('Final Reward (avg last 10 epochs)', fontsize=12)
        ax2.set_title('Final Performance Comparison', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, final_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison: {output_path.name}")


def plot_weight_heatmap(all_histories: Dict[str, Dict], output_path: Path):
    """绘制权重热力图"""
    # 收集所有场景的最终权重
    weight_matrix = []
    scenario_labels = []
    
    for scenario, history in all_histories.items():
        if history is None:
            continue
        
        weights = None
        if "final_weights" in history:
            weights = history["final_weights"]
        elif "weight_history" in history and len(history["weight_history"]) > 0:
            weights = history["weight_history"][-1]
        
        if weights and isinstance(weights, list) and len(weights) == 26:
            weight_matrix.append(weights)
            scenario_labels.append(SCENARIO_NAMES.get(scenario, scenario))
    
    if not weight_matrix:
        print("  ⚠ No weight data for heatmap")
        return
    
    weight_matrix = np.array(weight_matrix)
    
    # 特征组标签
    feature_groups = {
        'Spatial': (0, 3),
        'Op-Type': (3, 11),
        'Require': (11, 16),
        'Resource': (16, 21),
        'QoS': (21, 26)
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(weight_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    # 标签
    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels)
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    ax.set_title('Weight Distribution Across Scenarios', fontsize=14)
    
    # 添加特征组分隔线和标签
    for group_name, (start, end) in feature_groups.items():
        if start > 0:
            ax.axvline(x=start-0.5, color='white', linewidth=2)
        ax.text((start + end) / 2 - 0.5, -0.7, group_name, ha='center', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Weight', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='绘制 RL 训练曲线')
    parser.add_argument('--data-root', type=str, required=True,
                       help='multi_seed_runs 根目录')
    parser.add_argument('--scenarios', type=str, nargs='+', required=True,
                       help='场景列表')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='图片输出目录')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("📈 Plot RL Training Curves")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Scenarios: {args.scenarios}")
    print()
    
    all_histories = {}
    
    for scenario in args.scenarios:
        print(f"Loading: {scenario}")
        
        # 尝试多个可能的路径
        history_paths = [
            data_root / scenario / "bandit_results" / "training_history.json",
            data_root / scenario / "rl_results" / "training_history.json",
            data_root / "all_scenarios_merged" / "bandit_results" / "training_history.json"
        ]
        
        history = None
        for hp in history_paths:
            history = load_training_history(hp)
            if history:
                print(f"  ✓ Loaded: {hp}")
                break
        
        if history is None:
            print(f"  ⚠ No training history found")
        
        all_histories[scenario] = history
        
        # 绘制单场景图
        if history:
            plot_single_scenario(
                history, scenario,
                output_dir / f"rl_training_{scenario}.pdf"
            )
    
    print()
    
    # 绘制对比图
    valid_histories = {k: v for k, v in all_histories.items() if v is not None}
    if valid_histories:
        plot_comparison(valid_histories, output_dir / "rl_training_comparison.pdf")
        plot_weight_heatmap(valid_histories, output_dir / "rl_weight_heatmap.pdf")
    else:
        print("⚠ No valid training histories to compare")
    
    print()
    print("="*60)
    print("✓ Plotting complete!")
    print("="*60)


if __name__ == "__main__":
    main()
