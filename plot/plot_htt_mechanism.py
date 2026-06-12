"""HtT (Hypothesis-to-Theory) Mechanism Analysis Visualization.

Generates IEEE paper-ready figures for HtT mechanism analysis:
1. Decision Boundary Visualization - Shows RL hypotheses and learned rule regions
2. Weight distribution heatmap across context conditions

Aligned with paper terminology:
- Context $S_{ctx}$: spatial variance σ_pos, temporal density ρ_τ
- Action (Hypothesis w): weight vector determining splitting priority
- Rules: "IF σ_pos > τ₁, THEN prioritize geometry" etc.

Style: Colorblind-friendly Seaborn colors, Times New Roman font.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

try:
    from .config import (
        PROJECT_ROOT,
        FIGURES_ROOT,
        FIGURE_DPI,
        SCENARIOS,
        SCENARIO_SHORT_NAMES,
    )
except ImportError:
    from config import (
        PROJECT_ROOT,
        FIGURES_ROOT,
        FIGURE_DPI,
        SCENARIOS,
        SCENARIO_SHORT_NAMES,
    )

logger = logging.getLogger(__name__)

# IEEE-style settings
IEEE_FONT = 'Times New Roman'
IEEE_FONTSIZE = 14

# Data paths
WORKFLOW_DECOUPLE_ROOT = PROJECT_ROOT / "workflow_decouple" / "data" / "multi_seed_runs"

# Feature names aligned with paper notation
FEATURE_NAMES = {
    'geo_var': '$\\sigma_{pos}$ (Spatial Variance)',
    'geo_mean': '$\\mu_{pos}$ (Spatial Mean)',
    'time_var': '$\\sigma_{\\tau}$ (Temporal Variance)',
    'time_mean': '$\\rho_{\\tau}$ (Temporal Density)',
    'task_var': '$\\sigma_{task}$ (Task Variance)',
    'task_diversity': 'Task Heterogeneity',
}

# Weight dimension categories (26 dimensions grouped) - aligned with paper
WEIGHT_CATEGORIES = {
    'geometry': list(range(0, 9)),     # Spatial/position features → "Prioritize Geometry"
    'time': list(range(9, 17)),        # Time features → "Prioritize Time Windows"
    'task': list(range(17, 26)),       # Task type features → "Prioritize Task Coupling"
}

# Strategy labels aligned with paper Rules
STRATEGY_LABELS = {
    'geometry': 'Prioritize Geometry\n(Rule A: $w=[0.9,0.1,0]$)',
    'time': 'Prioritize Time Windows\n(Rule B: $w=[0.1,0.9,0]$)',
    'task': 'Prioritize Task Coupling\n($w=[0,0.1,0.9]$)',
}


def apply_ieee_style():
    """Apply IEEE paper-ready style with Times New Roman."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [IEEE_FONT, 'DejaVu Serif'],
        'font.size': IEEE_FONTSIZE,
        'axes.labelsize': IEEE_FONTSIZE,
        'axes.titlesize': IEEE_FONTSIZE + 1,
        'xtick.labelsize': IEEE_FONTSIZE - 1,
        'ytick.labelsize': IEEE_FONTSIZE - 1,
        'legend.fontsize': IEEE_FONTSIZE - 1,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })


def load_weight_library(scenario: str) -> Optional[Dict]:
    """Load weight library for a scenario."""
    path = WORKFLOW_DECOUPLE_ROOT / scenario / "bandit_results" / "weight_library.json"
    if not path.exists():
        logger.warning(f"Weight library not found: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def get_weight_preference(weights: List[float]) -> str:
    """Determine weight preference (splitting strategy) from 26D weights.
    
    Based on paper terminology:
    - 'geometry': Prioritize spatial features for splitting (Rule A)
    - 'time': Prioritize temporal features for splitting (Rule B)
    - 'task': Prioritize task-type features for splitting
    
    Returns:
        'geometry', 'time', or 'task' based on dominant weight group
    """
    weights = np.array(weights)
    
    # Calculate mean weight for each category
    geometry_mean = np.mean(weights[WEIGHT_CATEGORIES['geometry']])
    time_mean = np.mean(weights[WEIGHT_CATEGORIES['time']])
    task_mean = np.mean(weights[WEIGHT_CATEGORIES['task']])
    
    # Determine dominant category (which feature group has highest weight)
    means = {'geometry': geometry_mean, 'time': time_mean, 'task': task_mean}
    return max(means, key=means.get)


def extract_node_features(library: Dict) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Extract context features and weight preferences from all tree nodes.
    
    Features aligned with paper notation:
    - X: σ_pos (geo_var) - Spatial variance of waypoints
    - Y: ρ_τ (time_mean) - Temporal density of deadlines
    
    Returns:
        features: (n_nodes, 2) array with [σ_pos, ρ_τ]
        preferences: List of splitting strategies ('geometry', 'time', 'task')
        nodes: List of node dictionaries
    """
    nodes = library.get('nodes', [])
    
    features = []
    preferences = []
    valid_nodes = []
    
    for node in nodes:
        context = node.get('context', {})
        weights = node.get('weights', [])
        
        if not context or not weights:
            continue
        
        # Extract context features (aligned with paper S_ctx)
        # σ_pos: spatial variance of waypoints
        sigma_pos = context.get('geo_var', 0)
        # ρ_τ: temporal density estimate
        rho_tau = context.get('time_mean', context.get('time_var', 0))
        
        features.append([sigma_pos, rho_tau])
        preferences.append(get_weight_preference(weights))
        valid_nodes.append(node)
    
    return np.array(features), preferences, valid_nodes


def plot_decision_boundary_single(
    scenario: str,
    ax: plt.Axes,
    show_legend: bool = True,
):
    """Plot decision boundary for a single scenario.
    
    Shows RL exploration samples (Hypotheses) and rule coverage regions (Theories).
    
    - Scatter points: Each point is an RL sample (context, weight) from experience buffer B
    - Point color: The splitting strategy chosen (which feature group was prioritized)
    - Background regions: Convex hull showing where each "rule" applies
    
    Args:
        scenario: Scenario name
        ax: Matplotlib axes
        show_legend: Whether to show legend
    """
    library = load_weight_library(scenario)
    if library is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    features, preferences, nodes = extract_node_features(library)
    
    if len(features) == 0:
        ax.text(0.5, 0.5, 'No valid nodes', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Color mapping for splitting strategies (aligned with paper Rules)
    colors = sns.color_palette('deep')
    pref_colors = {
        'geometry': colors[0],  # Blue - Rule A: Prioritize Geometry
        'time': colors[1],      # Orange - Rule B: Prioritize Time Windows
        'task': colors[2],      # Green - Prioritize Task Coupling
    }
    pref_labels = {
        'geometry': 'Rule A: Prioritize Geometry',
        'time': 'Rule B: Prioritize Time',
        'task': 'Prioritize Task Type',
    }
    
    # Draw convex hull regions for learned rule coverage.
    for pref in ['geometry', 'time', 'task']:
        mask = np.array([p == pref for p in preferences])
        if mask.sum() >= 3:  # Need at least 3 points for convex hull
            pref_features = features[mask]
            try:
                hull = ConvexHull(pref_features)
                hull_points = pref_features[hull.vertices]
                polygon = Polygon(hull_points, alpha=0.15, facecolor=pref_colors[pref],
                                 edgecolor=pref_colors[pref], linewidth=1.5, linestyle='--')
                ax.add_patch(polygon)
            except Exception:
                pass  # Skip if convex hull fails
    
    # Plot scatter points (RL hypotheses from experience buffer B)
    for pref in ['geometry', 'time', 'task']:
        mask = np.array([p == pref for p in preferences])
        if mask.sum() > 0:
            ax.scatter(
                features[mask, 0], features[mask, 1],
                c=[pref_colors[pref]], s=60, alpha=0.85,
                edgecolors='white', linewidth=0.8,
                label=pref_labels[pref], zorder=10,
                marker='o',
            )
    
    # Labels using paper notation
    ax.set_xlabel(FEATURE_NAMES['geo_var'], fontsize=IEEE_FONTSIZE)
    ax.set_ylabel(FEATURE_NAMES['time_mean'], fontsize=IEEE_FONTSIZE)
    
    short_name = SCENARIO_SHORT_NAMES.get(scenario, scenario)
    ax.set_title(f'{short_name}', fontsize=IEEE_FONTSIZE + 1, fontweight='bold')
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    if show_legend:
        ax.legend(loc='best', fontsize=IEEE_FONTSIZE - 2, framealpha=0.95)


def plot_decision_boundary_all(output_path: Path = None):
    """Plot decision boundary visualization for all scenarios.
    
    Creates a multi-panel figure showing RL exploration and rule coverage.
    """
    apply_ieee_style()
    
    if output_path is None:
        output_path = FIGURES_ROOT / "fig_htt_decision_boundary.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find scenarios with weight library
    valid_scenarios = []
    for scenario in SCENARIOS:
        lib_path = WORKFLOW_DECOUPLE_ROOT / scenario / "bandit_results" / "weight_library.json"
        if lib_path.exists():
            valid_scenarios.append(scenario)
    
    if not valid_scenarios:
        logger.warning("No weight libraries found for decision boundary plot")
        return
    
    n_scenarios = len(valid_scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    
    if n_scenarios == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, scenario in enumerate(valid_scenarios):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        plot_decision_boundary_single(scenario, ax, show_legend=(idx == 0))
    
    # Hide unused subplots
    for idx in range(n_scenarios, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_weight_heatmap_by_context(output_path: Path = None):
    """Plot weight heatmap showing how weights vary with context features.
    
    X-axis: Context feature ranges
    Y-axis: Weight dimensions (grouped)
    Color: Weight values
    """
    apply_ieee_style()
    
    if output_path is None:
        output_path = FIGURES_ROOT / "fig_htt_weight_heatmap.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all nodes across scenarios
    all_nodes = []
    for scenario in SCENARIOS:
        library = load_weight_library(scenario)
        if library:
            for node in library.get('nodes', []):
                node['scenario'] = scenario
                all_nodes.append(node)
    
    if not all_nodes:
        logger.warning("No data for weight heatmap")
        return
    
    # Sort nodes by geo_var (spatial dispersion)
    all_nodes.sort(key=lambda n: n.get('context', {}).get('geo_var', 0))
    
    # Create weight matrix
    n_nodes = min(20, len(all_nodes))  # Limit to 20 nodes for readability
    step = max(1, len(all_nodes) // n_nodes)
    selected_nodes = all_nodes[::step][:n_nodes]
    
    weight_matrix = np.array([n.get('weights', [0]*26) for n in selected_nodes])
    
    # Group weights by category
    weight_groups = {
        'Spatial (dim 0-8)': np.mean(weight_matrix[:, :9], axis=1),
        'Temporal (dim 9-16)': np.mean(weight_matrix[:, 9:17], axis=1),
        'Task (dim 17-25)': np.mean(weight_matrix[:, 17:], axis=1),
    }
    
    grouped_matrix = np.array(list(weight_groups.values())).T
    
    # X-axis labels: context features
    x_labels = []
    for n in selected_nodes:
        ctx = n.get('context', {})
        geo = ctx.get('geo_var', 0)
        task = ctx.get('task_diversity', 0)
        x_labels.append(f'{geo:.2f}\n{task:.2f}')
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    im = ax.imshow(grouped_matrix.T, aspect='auto', cmap='RdYlBu_r', vmin=0.3, vmax=0.7)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(weight_groups)))
    ax.set_yticklabels(list(weight_groups.keys()), fontsize=IEEE_FONTSIZE - 1)
    
    ax.set_xlabel('Context ($\\sigma_{geo}$ / $\\rho_{div}$)', fontsize=IEEE_FONTSIZE)
    ax.set_ylabel('Weight Group', fontsize=IEEE_FONTSIZE)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Avg. Weight', fontsize=IEEE_FONTSIZE - 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_rl_exploration_trajectory(output_path: Path = None):
    """Plot RL exploration trajectory showing how weights evolve during training.
    
    Shows the progression from random exploration to converged policy.
    """
    apply_ieee_style()
    
    if output_path is None:
        output_path = FIGURES_ROOT / "fig_htt_exploration.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load training history from a scenario
    scenario = "small_logistics_inspection"  # Default scenario
    history_path = WORKFLOW_DECOUPLE_ROOT / scenario / "bandit_results" / "training_history.json"
    
    if not history_path.exists():
        # Try other scenarios
        for s in SCENARIOS:
            history_path = WORKFLOW_DECOUPLE_ROOT / s / "bandit_results" / "training_history.json"
            if history_path.exists():
                scenario = s
                break
    
    if not history_path.exists():
        logger.warning("No training history found for exploration plot")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history.get('epoch', [])
    rewards = history.get('reward_mean', [])
    val_rewards = history.get('val_reward', [])
    
    if not epochs or not rewards:
        logger.warning("Empty training history")
        return
    
    colors = sns.color_palette('deep')
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    ax.plot(epochs, rewards, 'o-', color=colors[0], markersize=3, linewidth=1.5,
            label='Train Reward', markevery=max(1, len(epochs) // 15))
    
    if val_rewards and len(val_rewards) == len(epochs):
        ax.plot(epochs, val_rewards, 's-', color=colors[1], markersize=4, linewidth=1.5,
                label='Val Reward')
    
    ax.set_xlabel('Epoch', fontsize=IEEE_FONTSIZE)
    ax.set_ylabel('Reward', fontsize=IEEE_FONTSIZE)
    ax.set_title(f'RL Exploration: {SCENARIO_SHORT_NAMES.get(scenario, scenario)}',
                fontsize=IEEE_FONTSIZE + 1)
    
    ax.legend(loc='lower right', fontsize=IEEE_FONTSIZE - 1, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_fidelity_analysis(output_path: Path = None):
    """Plot fidelity analysis showing how well learned rules cover RL hypotheses.
    
    This demonstrates the HtT "Fidelity" property:
    - Left panel: RL samples (Hypotheses) vs learned Rules (Theories) with dual Y-axis
    - Right panel: Rule coverage rate (how well theories explain the data)
    """
    apply_ieee_style()
    
    if output_path is None:
        output_path = FIGURES_ROOT / "fig_htt_fidelity.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect statistics
    scenario_stats = []
    
    for scenario in SCENARIOS:
        library = load_weight_library(scenario)
        if not library:
            continue
        
        metadata = library.get('metadata', {})
        nodes = library.get('nodes', [])
        
        n_total = metadata.get('total_samples', 0)
        n_nodes = metadata.get('total_nodes', 0)
        n_leaf = sum(1 for n in nodes if n.get('is_leaf', False) or not n.get('children', []))
        
        # Calculate coverage: samples covered by leaf nodes / total samples
        leaf_samples = sum(n.get('n_samples', 0) for n in nodes 
                         if n.get('is_leaf', False) or not n.get('children', []))
        coverage = leaf_samples / max(n_total, 1)
        
        scenario_stats.append({
            'scenario': SCENARIO_SHORT_NAMES.get(scenario, scenario),
            'total_samples': n_total,
            'n_rules': n_leaf,
            'coverage': coverage,
        })
    
    if not scenario_stats:
        logger.warning("No data for fidelity analysis")
        return
    
    colors = sns.color_palette('deep')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    
    # Left: RL Hypotheses vs learned Rules (dual Y-axis)
    scenarios = [s['scenario'] for s in scenario_stats]
    n_rules = [s['n_rules'] for s in scenario_stats]
    n_samples = [s['total_samples'] for s in scenario_stats]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Primary Y-axis: RL samples (Hypotheses in buffer B)
    bars1 = ax1.bar(x - width/2, n_samples, width, label='RL Hypotheses $|\\mathcal{B}|$', 
                    color=colors[0], alpha=0.8)
    ax1.set_ylabel('# Hypotheses (RL Samples)', fontsize=IEEE_FONTSIZE, color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    
    # Secondary Y-axis: Number of rules (Theories in L)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, n_rules, width, label='Learned Rules $|\\mathcal{K}_{rules}|$',
                         color=colors[2], alpha=0.8)
    ax1_twin.set_ylabel('# Learned Rules', fontsize=IEEE_FONTSIZE, color=colors[2])
    ax1_twin.tick_params(axis='y', labelcolor=colors[2])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=IEEE_FONTSIZE - 1, rotation=15, ha='right')
    ax1.set_title('Hypothesis → Theory Compression', fontsize=IEEE_FONTSIZE + 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
               fontsize=IEEE_FONTSIZE - 2, framealpha=0.95)
    
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    
    # Right: Coverage rate (Fidelity metric)
    coverage = [s['coverage'] * 100 for s in scenario_stats]
    
    bars = ax2.bar(x, coverage, color=colors[1], alpha=0.8, edgecolor='white')
    
    # Add value labels on bars
    for i, (bar, cov) in enumerate(zip(bars, coverage)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{cov:.0f}%', ha='center', va='bottom', fontsize=IEEE_FONTSIZE - 2)
    
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Fidelity')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=IEEE_FONTSIZE - 1, rotation=15, ha='right')
    ax2.set_ylabel('Coverage Rate (%)', fontsize=IEEE_FONTSIZE)
    ax2.set_title('Rule Fidelity: $P(\\mathbf{w}|S,\\mathcal{K}_{rules})$', fontsize=IEEE_FONTSIZE + 1)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower right', fontsize=IEEE_FONTSIZE - 2, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_all_htt_mechanism_figures():
    """Generate all HtT mechanism analysis figures."""
    logger.info("Generating HtT mechanism analysis figures...")
    
    plot_decision_boundary_all()
    plot_weight_heatmap_by_context()
    plot_rl_exploration_trajectory()
    plot_fidelity_analysis()
    
    logger.info("HtT mechanism figures complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_all_htt_mechanism_figures()
