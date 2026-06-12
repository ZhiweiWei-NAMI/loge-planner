"""
Cluster Rule Mining for Multi-Agent Planning Post-processing

This script analyzes historical simulation data to extract rules for determining
whether a checkpoint should belong to a cluster based on execution success rate.

Data Sources:
- Input: ../data/offline_init (cluster grouping, checkpoint features)
- Planning: ../data/offline_plans (cluster assignment data)
- Results: ../data/results/aggregated_stats.csv and workflow_summary JSONs
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'workflow_decouple' / 'src'))

DATA_DIR = PROJECT_ROOT / 'data'
INIT_DIR = DATA_DIR / 'offline_init'
PLANS_DIR = DATA_DIR / 'offline_plans'
RESULTS_DIR = DATA_DIR / 'results'
OUTPUT_DIR = Path(__file__).parent / 'rules'

# =============================================================================
# Step 1: Data Alignment & Preprocessing
# =============================================================================

def parse_location(loc_str) -> Tuple[float, float, float]:
    """Parse location string to (x, y, z) tuple."""
    if pd.isna(loc_str):
        return (0.0, 0.0, 0.0)
    if isinstance(loc_str, (list, tuple)):
        return tuple(float(x) for x in loc_str[:3])
    try:
        import ast
        loc = ast.literal_eval(str(loc_str))
        return (float(loc[0]), float(loc[1]), float(loc[2]))
    except:
        return (0.0, 0.0, 0.0)


def load_aggregated_stats() -> pd.DataFrame:
    """Load aggregated execution statistics."""
    stats_file = RESULTS_DIR / 'aggregated_stats.csv'
    df = pd.read_csv(stats_file)
    print(f"[Step 1] Loaded aggregated_stats: {len(df)} scenario-method combinations")
    return df


def load_workflow_summary(result_file: Path) -> Dict:
    """Load a single workflow_summary JSON file."""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_result_index() -> pd.DataFrame:
    """
    Build index of all result files with their metadata.
    Extracts method, timestamp, and links to workflow_summary files.
    """
    result_files = list(RESULTS_DIR.glob('*_workflow_summary.json'))
    records = []
    
    for f in result_files:
        # Parse filename: Offline-{method}_{timestamp}_workflow_summary.json
        parts = f.stem.replace('_workflow_summary', '').split('_')
        if len(parts) >= 2:
            method_part = parts[0].replace('Offline-', '')
            timestamp = parts[1]
            
            # Load summary to get statistics
            summary = load_workflow_summary(f)
            stats = summary.get('statistics', {})
            
            records.append({
                'file_path': str(f),
                'method': method_part,
                'timestamp': timestamp,
                'total_workflows': summary.get('total_workflows', 0),
                'completed_workflows': stats.get('completed_workflows', 0),
                'failed_workflows': stats.get('failed_workflows', 0),
                'avg_completion_time': stats.get('average_completion_time', 0),
            })
    
    df = pd.DataFrame(records)
    if not df.empty:
        df['completion_rate'] = df['completed_workflows'] / df['total_workflows'].clip(lower=1)
    print(f"[Step 1] Built result index: {len(df)} execution runs")
    return df


def load_cluster_data(scenario: str, seed: int, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cluster grouping and checkpoint features for a specific scenario/seed/method.
    
    Returns:
        - checkpoint_df: Checkpoint features with cluster assignment
        - cluster_assignment_df: Cluster-level assignment data (agents, centers, etc.)
    """
    init_dir = INIT_DIR / scenario / f'run_seed_{seed}'
    plans_dir = PLANS_DIR / scenario / f'run_seed_{seed}'
    
    # Load checkpoint clusters
    cluster_file = init_dir / f'checkpoint_clusters_{method}.csv'
    if not cluster_file.exists():
        return None, None
    checkpoint_df = pd.read_csv(cluster_file)
    
    # Load cluster assignment
    assignment_file = plans_dir / f'cluster_assignment_{method}.csv'
    if assignment_file.exists():
        cluster_assignment_df = pd.read_csv(assignment_file)
    else:
        cluster_assignment_df = None
    
    return checkpoint_df, cluster_assignment_df


def align_data_sources() -> pd.DataFrame:
    """
    Align cluster info with execution results.
    Creates a master DataFrame linking clusters to execution outcomes.
    """
    print("\n" + "="*60)
    print("STEP 1: Data Alignment & Preprocessing")
    print("="*60)
    
    # Load aggregated stats
    agg_stats = load_aggregated_stats()
    
    # Define scenario-to-seed mapping based on directory structure
    scenarios = ['large_mixed', 'medium_inspection_heavy', 'medium_logistics_heavy',
                 'small_comm_compute', 'small_logistics_inspection']
    seeds = list(range(42, 51))  # seeds 42-50
    methods = ['kmeans', 'triplet']
    
    records = []
    
    for scenario in scenarios:
        for seed in seeds:
            for method in methods:
                checkpoint_df, cluster_assignment_df = load_cluster_data(scenario, seed, method)
                
                if checkpoint_df is None:
                    continue
                
                # Get completion rate from aggregated stats
                match = agg_stats[(agg_stats['scenario'] == scenario) & 
                                   (agg_stats['method'] == method)]
                if match.empty:
                    continue
                
                completion_rate = match['completion_rate_mean'].iloc[0]
                avg_time = match['avg_completion_time_mean'].iloc[0]
                
                # Compute cluster-level statistics
                cluster_stats = compute_cluster_level_stats(checkpoint_df, cluster_assignment_df)
                
                for cluster_id, c_stats in cluster_stats.items():
                    records.append({
                        'scenario': scenario,
                        'seed': seed,
                        'method': method,
                        'cluster_id': cluster_id,
                        'n_checkpoints': c_stats['n_checkpoints'],
                        'completion_rate': completion_rate,
                        'avg_completion_time': avg_time,
                        **c_stats  # Merge cluster-level features
                    })
    
    df = pd.DataFrame(records)
    print(f"[Step 1] Aligned data: {len(df)} cluster-level records")
    print(f"         Scenarios: {df['scenario'].nunique()}, Methods: {df['method'].nunique()}")
    
    return df


def compute_cluster_level_stats(checkpoint_df: pd.DataFrame, 
                                  cluster_assignment_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Compute statistics for each cluster from checkpoint data.
    
    Returns dict: {cluster_id: {feature_name: value, ...}}
    """
    cluster_stats = {}
    
    for cluster_id in checkpoint_df['cluster_group'].unique():
        cluster_checkpoints = checkpoint_df[checkpoint_df['cluster_group'] == cluster_id]
        n_ckpts = len(cluster_checkpoints)
        
        if n_ckpts == 0:
            continue
        
        # Parse locations
        locations = cluster_checkpoints['location'].apply(parse_location).tolist()
        xs = [loc[0] for loc in locations]
        ys = [loc[1] for loc in locations]
        zs = [loc[2] for loc in locations]
        
        # Spatial statistics
        dist_variance = np.var(xs) + np.var(ys) + np.var(zs) if n_ckpts > 1 else 0.0
        spatial_spread = np.sqrt(dist_variance)
        
        # Centroid
        centroid = (np.mean(xs), np.mean(ys), np.mean(zs))
        
        # Pairwise distance statistics
        if n_ckpts > 1:
            from scipy.spatial.distance import pdist
            coords = np.array(list(zip(xs, ys, zs)))
            pairwise_dists = pdist(coords)
            mean_pairwise_dist = np.mean(pairwise_dists) if len(pairwise_dists) > 0 else 0
            max_pairwise_dist = np.max(pairwise_dists) if len(pairwise_dists) > 0 else 0
        else:
            mean_pairwise_dist = 0
            max_pairwise_dist = 0
        
        # Task type homogeneity
        workflow_types = cluster_checkpoints['workflow_type'].value_counts()
        type_entropy = scipy_stats.entropy(workflow_types.values) if len(workflow_types) > 1 else 0
        dominant_type = workflow_types.index[0] if len(workflow_types) > 0 else 'Unknown'
        type_homogeneity = workflow_types.iloc[0] / n_ckpts if len(workflow_types) > 0 else 1.0
        
        # Operation type distribution
        if 'operation_type' in cluster_checkpoints.columns:
            op_types = cluster_checkpoints['operation_type'].value_counts()
            op_type_entropy = scipy_stats.entropy(op_types.values) if len(op_types) > 1 else 0
        else:
            op_type_entropy = 0
        
        # Payload weight statistics (for logistics)
        if 'payload_weight' in cluster_checkpoints.columns:
            payload_mean = cluster_checkpoints['payload_weight'].mean()
            payload_std = cluster_checkpoints['payload_weight'].std() if n_ckpts > 1 else 0
        else:
            payload_mean = 0
            payload_std = 0
        
        cluster_stats[cluster_id] = {
            'n_checkpoints': n_ckpts,
            'spatial_spread': spatial_spread,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'centroid_z': centroid[2],
            'mean_pairwise_dist': mean_pairwise_dist,
            'max_pairwise_dist': max_pairwise_dist,
            'type_entropy': type_entropy,
            'type_homogeneity': type_homogeneity,
            'dominant_type': dominant_type,
            'op_type_entropy': op_type_entropy,
            'payload_mean': payload_mean,
            'payload_std': payload_std,
        }
        
        # Add cluster assignment info if available
        if cluster_assignment_df is not None:
            assign_match = cluster_assignment_df[cluster_assignment_df['cluster_group'] == cluster_id]
            if not assign_match.empty:
                row = assign_match.iloc[0]
                cluster_stats[cluster_id]['earliest_start_time'] = row.get('earliest_start_time', 0)
                cluster_stats[cluster_id]['n_assigned_agents'] = len(str(row.get('assigned_agents', '')).split(';'))
    
    return cluster_stats


# =============================================================================
# Step 2: Metric Validation - Synergy Ratio Correlation
# =============================================================================

def compute_synergy_metrics(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute synergy-like metrics for clusters based on spatial relationships.
    
    Synergy Ratio approximation (without running full simulation):
    - Measures potential for task reassignment within cluster
    - Based on spatial compactness and task type compatibility
    """
    print("\n" + "="*60)
    print("STEP 2: Metric Validation - Synergy Correlation")
    print("="*60)
    
    # Compute synergy approximation metrics
    df = aligned_df.copy()
    
    # Synergy estimate 1: Spatial Compactness Score
    # Lower spread = higher potential synergy (easier to reassign)
    df['spatial_compactness'] = 1.0 / (1.0 + df['spatial_spread'] / 1000.0)
    
    # Synergy estimate 2: Type Homogeneity
    # Higher homogeneity = agents can more easily swap tasks
    df['type_synergy'] = df['type_homogeneity']
    
    # Synergy estimate 3: Combined synergy score
    df['synergy_score'] = (df['spatial_compactness'] + df['type_synergy']) / 2.0
    
    # Cluster density: checkpoints per unit spatial area
    df['cluster_density'] = df['n_checkpoints'] / (df['spatial_spread'].clip(lower=1.0) ** 2)
    
    return df


def analyze_synergy_completion_correlation(df: pd.DataFrame) -> Dict:
    """
    Analyze correlation between synergy metrics and completion rate.
    """
    print("\n--- Correlation Analysis ---")
    
    results = {}
    
    # Aggregate by scenario-method for correlation analysis
    agg_df = df.groupby(['scenario', 'method']).agg({
        'completion_rate': 'first',  # Same for all clusters in same run
        'synergy_score': 'mean',
        'spatial_compactness': 'mean',
        'type_synergy': 'mean',
        'cluster_density': 'mean',
        'n_checkpoints': 'mean',
        'spatial_spread': 'mean',
        'type_entropy': 'mean',
        'mean_pairwise_dist': 'mean',
    }).reset_index()
    
    metrics_to_test = [
        'synergy_score', 'spatial_compactness', 'type_synergy', 
        'cluster_density', 'spatial_spread', 'type_entropy', 
        'mean_pairwise_dist', 'n_checkpoints'
    ]
    
    print(f"\nCorrelation with Completion Rate (n={len(agg_df)} scenario-method pairs):")
    print("-" * 50)
    
    for metric in metrics_to_test:
        if metric not in agg_df.columns:
            continue
        
        valid = agg_df[[metric, 'completion_rate']].dropna()
        if len(valid) < 3:
            continue
        
        pearson_r, pearson_p = scipy_stats.pearsonr(valid[metric], valid['completion_rate'])
        spearman_r, spearman_p = scipy_stats.spearmanr(valid[metric], valid['completion_rate'])
        
        significance = "***" if pearson_p < 0.01 else ("**" if pearson_p < 0.05 else ("*" if pearson_p < 0.1 else ""))
        
        print(f"  {metric:25s}: Pearson r={pearson_r:+.4f} (p={pearson_p:.4f}){significance}")
        print(f"  {' ':25s}  Spearman ρ={spearman_r:+.4f} (p={spearman_p:.4f})")
        
        results[metric] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'significant': pearson_p < 0.05
        }
    
    # Insight summary
    print("\n--- Key Insights ---")
    significant_positive = [m for m, v in results.items() if v['significant'] and v['pearson_r'] > 0]
    significant_negative = [m for m, v in results.items() if v['significant'] and v['pearson_r'] < 0]
    
    if significant_positive:
        print(f"  ✓ Positive correlations: {', '.join(significant_positive)}")
    if significant_negative:
        print(f"  ✗ Negative correlations: {', '.join(significant_negative)}")
    if not significant_positive and not significant_negative:
        print("  ⚠ No significant correlations found at p<0.05 level")
        print("  → Need to design new metrics (path conflict, resource contention)")
    
    return results


# =============================================================================
# Step 3: Feature Mining - High Completion Rate Cluster Analysis
# =============================================================================

def analyze_high_completion_clusters(df: pd.DataFrame) -> Dict:
    """
    Analyze characteristics of clusters from high completion rate runs.
    """
    print("\n" + "="*60)
    print("STEP 3: Feature Mining - High vs Low Completion Analysis")
    print("="*60)
    
    # Split into high and low completion rate groups
    median_completion = df['completion_rate'].median()
    high_completion = df[df['completion_rate'] >= median_completion]
    low_completion = df[df['completion_rate'] < median_completion]
    
    print(f"\nMedian completion rate: {median_completion:.4f}")
    print(f"High completion clusters: {len(high_completion)}")
    print(f"Low completion clusters: {len(low_completion)}")
    
    # Compare feature distributions
    features_to_compare = [
        'n_checkpoints', 'spatial_spread', 'mean_pairwise_dist',
        'type_homogeneity', 'type_entropy', 'cluster_density',
        'synergy_score', 'payload_mean'
    ]
    
    comparison_results = {}
    print("\n--- Feature Comparison (High vs Low Completion) ---")
    print("-" * 70)
    print(f"{'Feature':25s} | {'High Mean':>12s} | {'Low Mean':>12s} | {'Diff %':>10s} | {'t-stat':>8s}")
    print("-" * 70)
    
    for feature in features_to_compare:
        if feature not in df.columns:
            continue
        
        high_vals = high_completion[feature].dropna()
        low_vals = low_completion[feature].dropna()
        
        if len(high_vals) < 2 or len(low_vals) < 2:
            continue
        
        high_mean = high_vals.mean()
        low_mean = low_vals.mean()
        diff_pct = ((high_mean - low_mean) / max(abs(low_mean), 1e-6)) * 100
        
        # t-test
        t_stat, p_val = scipy_stats.ttest_ind(high_vals, low_vals)
        significance = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
        
        print(f"{feature:25s} | {high_mean:12.4f} | {low_mean:12.4f} | {diff_pct:+10.1f}% | {t_stat:+8.2f}{significance}")
        
        comparison_results[feature] = {
            'high_mean': high_mean,
            'low_mean': low_mean,
            'diff_percent': diff_pct,
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'direction': 'higher' if diff_pct > 0 else 'lower'
        }
    
    # Identify discriminative features
    print("\n--- Discriminative Features (p < 0.05) ---")
    discriminative = {k: v for k, v in comparison_results.items() if v['significant']}
    
    if discriminative:
        for feat, vals in discriminative.items():
            direction = "HIGHER" if vals['direction'] == 'higher' else "LOWER"
            print(f"  • {feat}: {direction} in high-completion clusters ({vals['diff_percent']:+.1f}%)")
    else:
        print("  No statistically significant differences found")
    
    return {
        'median_completion': median_completion,
        'feature_comparison': comparison_results,
        'discriminative_features': list(discriminative.keys())
    }


def design_new_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Design new metrics for better prediction of cluster quality.
    
    New metrics focus on:
    - Path conflict potential
    - Resource contention density
    - Temporal overlap
    """
    print("\n--- Designing New Metrics ---")
    
    df = df.copy()
    
    # Metric 1: Spatial Concentration Index
    # High concentration in small area may cause conflicts
    df['spatial_concentration'] = df['n_checkpoints'] / (df['max_pairwise_dist'].clip(lower=1.0))
    
    # Metric 2: Type Diversity Score (inverse of homogeneity)
    # Mixed types may need diverse agent capabilities
    df['type_diversity'] = 1.0 - df['type_homogeneity']
    
    # Metric 3: Cluster Size Efficiency
    # Optimal cluster size range (not too small, not too large)
    optimal_size = 30  # Based on typical assignment patterns
    df['size_efficiency'] = 1.0 / (1.0 + abs(df['n_checkpoints'] - optimal_size) / optimal_size)
    
    # Metric 4: Spatial Balance Score
    # Low variance in z-coordinate often indicates ground-level efficiency
    df['altitude_variance'] = df.apply(
        lambda r: 0 if pd.isna(r.get('centroid_z')) else abs(r['centroid_z'] - 50), axis=1
    )
    
    print(f"  Added metrics: spatial_concentration, type_diversity, size_efficiency, altitude_variance")
    
    return df


# =============================================================================
# Step 4: Rule Generation
# =============================================================================

def generate_cluster_assignment_rules(df: pd.DataFrame, 
                                        feature_analysis: Dict,
                                        correlation_results: Dict) -> List[Dict]:
    """
    Generate rules for checkpoint-to-cluster assignment based on analysis.
    
    Rule format: {
        'rule_id': str,
        'condition': str (human-readable),
        'feature_name': str,
        'threshold': float,
        'operator': str,
        'confidence': float,
        'effect_on_completion': str
    }
    """
    print("\n" + "="*60)
    print("STEP 4: Rule Generation")
    print("="*60)
    
    rules = []
    rule_id = 1
    
    # Rule generation from discriminative features
    discriminative = feature_analysis.get('discriminative_features', [])
    comparison = feature_analysis.get('feature_comparison', {})
    
    for feature in discriminative:
        if feature not in comparison:
            continue
        
        feat_info = comparison[feature]
        high_mean = feat_info['high_mean']
        low_mean = feat_info['low_mean']
        
        # Calculate threshold (midpoint between high and low)
        threshold = (high_mean + low_mean) / 2
        
        # Determine operator based on which direction is better
        if feat_info['direction'] == 'higher':
            operator = '>='
            condition = f"{feature} >= {threshold:.4f}"
        else:
            operator = '<='
            condition = f"{feature} <= {threshold:.4f}"
        
        # Confidence based on p-value
        p_val = feat_info['p_value']
        confidence = 1.0 - min(p_val, 0.5)  # Map p-value to confidence
        
        rules.append({
            'rule_id': f'R{rule_id:03d}',
            'feature_name': feature,
            'condition': condition,
            'threshold': threshold,
            'operator': operator,
            'confidence': round(confidence, 3),
            'effect': 'positive' if feat_info['direction'] == 'higher' else 'negative',
            'diff_percent': feat_info['diff_percent'],
            't_statistic': feat_info['t_stat'],
            'p_value': feat_info['p_value']
        })
        rule_id += 1
    
    # Add correlation-based rules
    for metric, corr_info in correlation_results.items():
        if not corr_info['significant']:
            continue
        
        r = corr_info['pearson_r']
        
        # Determine optimal direction
        if r > 0:
            condition = f"Prefer higher {metric}"
            effect = 'positive_correlation'
        else:
            condition = f"Prefer lower {metric}"
            effect = 'negative_correlation'
        
        rules.append({
            'rule_id': f'R{rule_id:03d}',
            'feature_name': metric,
            'condition': condition,
            'threshold': None,  # Correlation-based, no fixed threshold
            'operator': 'maximize' if r > 0 else 'minimize',
            'confidence': round(abs(r), 3),
            'effect': effect,
            'pearson_r': r,
            'p_value': corr_info['pearson_p']
        })
        rule_id += 1
    
    print(f"Generated {len(rules)} rules")
    for rule in rules:
        print(f"  [{rule['rule_id']}] {rule['condition']} (conf={rule['confidence']:.2f})")
    
    return rules


def generate_checkpoint_scoring_function(rules: List[Dict], df: pd.DataFrame) -> Dict:
    """
    Generate a scoring function for checkpoint-cluster assignment.
    
    The score predicts the quality contribution of adding a checkpoint to a cluster.
    """
    print("\n--- Generating Scoring Function ---")
    
    # Compute feature statistics for normalization
    feature_stats = {}
    for rule in rules:
        feat = rule['feature_name']
        if feat in df.columns:
            feature_stats[feat] = {
                'mean': float(df[feat].mean()),
                'std': float(df[feat].std()),
                'min': float(df[feat].min()),
                'max': float(df[feat].max())
            }
    
    # Create scoring weights based on confidence
    weights = {}
    for rule in rules:
        feat = rule['feature_name']
        weight = rule['confidence']
        if rule.get('operator') == 'minimize' or rule.get('effect') == 'negative':
            weight = -weight
        weights[feat] = weight
    
    scoring_function = {
        'version': '1.0',
        'description': 'Checkpoint-to-Cluster scoring based on mined rules',
        'weights': weights,
        'normalization': feature_stats,
        'usage': 'score = sum(weight[f] * (value[f] - mean[f]) / std[f] for f in features)'
    }
    
    print(f"  Scoring function with {len(weights)} weighted features")
    
    return scoring_function


# =============================================================================
# Step 5: Output - Generate JSON Rules
# =============================================================================

def validate_rules_on_historical_data(rules: List[Dict], df: pd.DataFrame) -> Dict:
    """
    Backtest rules on historical data to validate effectiveness.
    """
    print("\n--- Rule Validation on Historical Data ---")
    
    validation_results = {}
    
    for rule in rules:
        feat = rule['feature_name']
        if feat not in df.columns:
            continue
        
        threshold = rule.get('threshold')
        operator = rule.get('operator')
        
        if threshold is None:
            # Correlation-based rule - use median split
            threshold = df[feat].median()
            operator = '>=' if 'maximize' in str(operator) else '<='
        
        # Split data based on rule
        if operator in ['>=', 'maximize']:
            satisfies = df[feat] >= threshold
        else:
            satisfies = df[feat] <= threshold
        
        # Compare completion rates
        satisfying_completion = df[satisfies]['completion_rate'].mean()
        violating_completion = df[~satisfies]['completion_rate'].mean()
        
        improvement = satisfying_completion - violating_completion
        
        validation_results[rule['rule_id']] = {
            'rule_id': rule['rule_id'],
            'feature': feat,
            'n_satisfying': int(satisfies.sum()),
            'n_violating': int((~satisfies).sum()),
            'avg_completion_satisfying': round(satisfying_completion, 4),
            'avg_completion_violating': round(violating_completion, 4),
            'improvement': round(improvement, 4),
            'effective': improvement > 0
        }
        
        status = "✓ Effective" if improvement > 0 else "✗ Ineffective"
        print(f"  [{rule['rule_id']}] {feat}: {status} (Δ={improvement:+.4f})")
    
    return validation_results


def save_rules_to_json(rules: List[Dict], 
                        scoring_function: Dict,
                        validation: Dict,
                        metadata: Dict,
                        scenario: str = None):
    """Save rules and related data to JSON file.
    
    Args:
        rules: List of rule dictionaries
        scoring_function: Scoring function configuration
        validation: Validation results
        metadata: Metadata dictionary
        scenario: If provided, save scenario-specific rules; otherwise save general rules
    """
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output = {
        'metadata': metadata,
        'rules': rules,
        'scoring_function': scoring_function,
        'validation': validation
    }
    
    if scenario:
        output_file = OUTPUT_DIR / f'cluster_rules_{scenario}.json'
    else:
        output_file = OUTPUT_DIR / 'cluster_assignment_rules.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n[Step 5] Rules saved to: {output_file}")
    
    return output_file


# =============================================================================
# Main Execution
# =============================================================================

# All scenarios for scenario-specific rule mining
ALL_SCENARIOS = [
    'large_mixed', 
    'medium_inspection_heavy', 
    'medium_logistics_heavy',
    'small_comm_compute', 
    'small_logistics_inspection'
]


def mine_rules_for_scenario(scenario_df: pd.DataFrame, scenario: str) -> Dict:
    """
    Mine rules for a specific scenario.
    
    Args:
        scenario_df: DataFrame containing only data for this scenario
        scenario: Scenario name
    
    Returns:
        Dictionary with rules, scoring function, validation, and output file path
    """
    print(f"\n{'='*60}")
    print(f"MINING RULES FOR SCENARIO: {scenario}")
    print(f"{'='*60}")
    print(f"Records: {len(scenario_df)}")
    
    if len(scenario_df) < 10:
        print(f"WARNING: Too few records ({len(scenario_df)}) for scenario {scenario}, skipping.")
        return None
    
    # Step 2: Synergy Metrics & Correlation
    df_with_synergy = compute_synergy_metrics(scenario_df)
    correlation_results = analyze_synergy_completion_correlation(df_with_synergy)
    
    # Step 3: Feature Mining
    df_with_new_metrics = design_new_metrics(df_with_synergy)
    feature_analysis = analyze_high_completion_clusters(df_with_new_metrics)
    
    # Step 4: Rule Generation
    rules = generate_cluster_assignment_rules(
        df_with_new_metrics, 
        feature_analysis, 
        correlation_results
    )
    
    if not rules:
        print(f"WARNING: No rules generated for scenario {scenario}")
        return None
    
    scoring_function = generate_checkpoint_scoring_function(rules, df_with_new_metrics)
    
    # Validation
    validation = validate_rules_on_historical_data(rules, df_with_new_metrics)
    
    # Step 5: Output - scenario-specific file
    metadata = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'scenario': scenario,
        'data_sources': {
            'offline_init': str(INIT_DIR),
            'offline_plans': str(PLANS_DIR),
            'results': str(RESULTS_DIR)
        },
        'n_records_analyzed': len(df_with_new_metrics),
        'n_rules_generated': len(rules),
        'effective_rules': sum(1 for v in validation.values() if v['effective'])
    }
    
    output_file = save_rules_to_json(rules, scoring_function, validation, metadata, scenario=scenario)
    
    print(f"[{scenario}] Rules: {len(rules)}, Effective: {metadata['effective_rules']}")
    
    return {
        'scenario': scenario,
        'df_with_metrics': df_with_new_metrics,
        'correlation_results': correlation_results,
        'feature_analysis': feature_analysis,
        'rules': rules,
        'validation': validation,
        'output_file': output_file
    }


def main(scenario_specific: bool = True):
    """Main execution pipeline.
    
    Args:
        scenario_specific: If True, mine rules for each scenario separately.
                           If False, mine general rules across all scenarios.
    """
    print("="*60)
    print("CLUSTER RULE MINING FOR MULTI-AGENT PLANNING")
    print("="*60)
    print(f"Mode: {'Scenario-Specific' if scenario_specific else 'General'}")
    
    # Step 1: Data Alignment
    aligned_df = align_data_sources()
    
    if aligned_df.empty:
        print("ERROR: No aligned data available. Check data directories.")
        return
    
    results = {}
    
    if scenario_specific:
        # Mine rules for each scenario separately
        scenarios = aligned_df['scenario'].unique()
        print(f"\nFound {len(scenarios)} scenarios: {list(scenarios)}")
        
        for scenario in scenarios:
            scenario_df = aligned_df[aligned_df['scenario'] == scenario].copy()
            scenario_result = mine_rules_for_scenario(scenario_df, scenario)
            if scenario_result:
                results[scenario] = scenario_result
        
        # Summary
        print("\n" + "="*60)
        print("SCENARIO-SPECIFIC RULE MINING SUMMARY")
        print("="*60)
        for scenario, result in results.items():
            n_rules = len(result['rules'])
            n_effective = result['validation']
            n_eff = sum(1 for v in n_effective.values() if v['effective'])
            print(f"  {scenario:30s}: {n_rules:3d} rules, {n_eff:3d} effective")
        print(f"\nOutput directory: {OUTPUT_DIR}")
        
    else:
        # Original behavior: mine general rules across all scenarios
        # Step 2: Synergy Metrics & Correlation
        df_with_synergy = compute_synergy_metrics(aligned_df)
        correlation_results = analyze_synergy_completion_correlation(df_with_synergy)
        
        # Step 3: Feature Mining
        df_with_new_metrics = design_new_metrics(df_with_synergy)
        feature_analysis = analyze_high_completion_clusters(df_with_new_metrics)
        
        # Step 4: Rule Generation
        rules = generate_cluster_assignment_rules(
            df_with_new_metrics, 
            feature_analysis, 
            correlation_results
        )
        
        scoring_function = generate_checkpoint_scoring_function(rules, df_with_new_metrics)
        
        # Validation
        validation = validate_rules_on_historical_data(rules, df_with_new_metrics)
        
        # Step 5: Output
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'scenario': 'general',
            'data_sources': {
                'offline_init': str(INIT_DIR),
                'offline_plans': str(PLANS_DIR),
                'results': str(RESULTS_DIR)
            },
            'n_records_analyzed': len(df_with_new_metrics),
            'n_rules_generated': len(rules),
            'effective_rules': sum(1 for v in validation.values() if v['effective'])
        }
        
        output_file = save_rules_to_json(rules, scoring_function, validation, metadata)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total cluster records analyzed: {len(df_with_new_metrics)}")
        print(f"Rules generated: {len(rules)}")
        print(f"Effective rules (validated): {metadata['effective_rules']}/{len(rules)}")
        print(f"Output file: {output_file}")
        
        results = {
            'aligned_df': aligned_df,
            'df_with_metrics': df_with_new_metrics,
            'correlation_results': correlation_results,
            'feature_analysis': feature_analysis,
            'rules': rules,
            'validation': validation
        }
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mine cluster assignment rules')
    parser.add_argument('--general', action='store_true', 
                        help='Mine general rules instead of scenario-specific')
    args = parser.parse_args()
    
    results = main(scenario_specific=not args.general)
