"""Stage 1: Precluster checkpoints for a given scenario config.

This script reuses the tested workflow_decouple pipeline to:
1) Run a single data-collection simulation from a scenario YAML.
2) Run spatial-temporal preprocessing and checkpoint clustering.
3) Produce a workflow_checkpoint_clustering.csv for downstream planning.

It is designed to work entirely inside the LoGe planner project, treating
workflow_decouple as a reusable library.
"""

import argparse
import ast
import json
import logging
import sys
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.simulation.environment.topology_manager import TopologyManager

logger = logging.getLogger(__name__)


def _add_workflow_decouple_to_path(current_file: Path) -> Path:
    """Add workflow_decouple/src to sys.path and return its path.

    The directory layout is assumed to be:
      loge-planner/
        src/
          offline_pipeline/stage1_precluster_from_config.py  (this file)
        workflow_decouple/
          src/
    """
    src_root = current_file.parent.parent           # .../loge-planner/src
    project_root = src_root.parent                  # .../loge-planner
    wf_decouple_src = project_root / "workflow_decouple" / "src"

    if not wf_decouple_src.exists():
        raise FileNotFoundError(f"workflow_decouple/src not found at: {wf_decouple_src}")

    if str(wf_decouple_src) not in sys.path:
        sys.path.insert(0, str(wf_decouple_src))

    return wf_decouple_src


def _resolve_under_project(path_str: str, project_root: Path) -> Path:
    """Resolve a path which can be absolute or project-root-relative."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / path_str


def _resolve_rules_config_for_scenario(
    scenario_name: str,
    project_root: Path,
    rules_config: Path | None = None,
) -> Path:
    if rules_config is not None:
        resolved = rules_config.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Rules config not found: {resolved}")
        return resolved
    rules_dir = project_root / "analysis" / "rules"
    scenario_rules_file = rules_dir / f"rule_kmeans_config_{scenario_name}.yaml"
    default_rules_file = rules_dir / "rule_kmeans_config_default.yaml"
    if scenario_rules_file.exists():
        return scenario_rules_file
    if default_rules_file.exists():
        return default_rules_file
    raise FileNotFoundError(
        f"No rule config found. Expected: {scenario_rules_file} or {default_rules_file}"
    )


def _apply_htt_rules_postprocessing(
    checkpoint_df: pd.DataFrame, 
    labels: np.ndarray,
    scenario_name: str = "large_mixed"
) -> np.ndarray:
    """Apply HtT (Hypotheses-to-Theories) rule-based post-processing to cluster assignments.
    
    This function refines initial cluster assignments based on mined rules that
    correlate with high completion rates. It may:
    - Split large clusters that violate size rules
    - Merge nearby small clusters with compatible types
    - Reassign checkpoints to better-matching clusters
    
    Args:
        checkpoint_df: DataFrame with checkpoint data
        labels: Current cluster assignments
        scenario_name: Scenario name for loading scenario-specific rules
    
    Returns:
        Refined cluster labels
    """
    import json
    from pathlib import Path
    from scipy.spatial.distance import cdist
    
    labels = labels.copy()
    n_checkpoints = len(labels)
    
    # Load scenario-specific rules first, then use general rules if needed.
    rules_dir = Path(__file__).parent.parent.parent / "analysis" / "rules"
    scenario_rules_file = rules_dir / f"cluster_rules_{scenario_name}.json"
    general_rules_file = rules_dir / "cluster_assignment_rules.json"
    
    if scenario_rules_file.exists():
        rules_file = scenario_rules_file
        logger.info("Loading scenario-specific HtT rules for: %s", scenario_name)
    elif general_rules_file.exists():
        rules_file = general_rules_file
        logger.warning("Scenario-specific rules not found for %s, using general rules", scenario_name)
    else:
        logger.warning("No HtT rules file found, skipping post-processing")
        return labels
    
    with open(rules_file, 'r') as f:
        rules_data = json.load(f)
    
    # Log which scenario the rules were mined for
    rule_scenario = rules_data.get('metadata', {}).get('scenario', 'unknown')
    logger.info("Loaded rules from scenario: %s (file: %s)", rule_scenario, rules_file.name)
    
    rules = rules_data.get('rules', [])
    
    # Parse rules - only extract rules that actually exist (no implicit defaults)
    # Key insight: No rule = not discriminative = should NOT apply any operation for that feature
    active_rules = {}
    for rule in rules:
        feat = rule.get('feature_name')
        op = rule.get('operator', '')
        threshold = rule.get('threshold')
        
        # Only consider rules with thresholds (not preference rules like "Prefer higher X")
        if threshold is not None and op in ['<=', '>=']:
            active_rules[feat] = {
                'operator': op,
                'threshold': threshold,
                'prefer': 'lower' if op == '<=' else 'higher',
                'diff_pct': abs(rule.get('diff_percent', 0))
            }
    
    logger.info("Active rules for scenario %s: %s", scenario_name, list(active_rules.keys()))
    
    # Only apply size-based transformations if n_checkpoints rule exists
    apply_size_transform = 'n_checkpoints' in active_rules
    if apply_size_transform:
        prefer_smaller = active_rules['n_checkpoints']['prefer'] == 'lower'
        size_threshold = active_rules['n_checkpoints']['threshold']
        logger.info("Applying n_checkpoints rule: %s %.1f (prefer_smaller=%s)",
                    active_rules['n_checkpoints']['operator'], size_threshold, prefer_smaller)
    else:
        prefer_smaller = None
        size_threshold = None
        logger.info("No n_checkpoints rule - skipping size-based transformations")
    
    # Only apply spatial-based transformations if spatial_spread rule exists
    apply_spatial_transform = 'spatial_spread' in active_rules
    if apply_spatial_transform:
        spread_threshold = active_rules['spatial_spread']['threshold']
        prefer_lower_spread = active_rules['spatial_spread']['prefer'] == 'lower'
        logger.info("Applying spatial_spread rule: %s %.1f",
                    active_rules['spatial_spread']['operator'], spread_threshold)
    else:
        logger.info("No spatial_spread rule - skipping spatial-based transformations")
    
    # Parse locations
    def parse_location(loc):
        if isinstance(loc, str):
            import ast
            try:
                loc = ast.literal_eval(loc)
            except:
                return np.array([0.0, 0.0, 0.0])
        return np.array([float(loc[0]), float(loc[1]), float(loc[2])])
    
    locations = np.array([parse_location(loc) for loc in checkpoint_df['location']])
    workflow_types = checkpoint_df['workflow_type'].values
    
    # Track modifications
    splits_count = 0
    merges_count = 0
    reassigns_count = 0
    
    unique_clusters = np.unique(labels)
    max_new_cluster_id = labels.max() + 1
    
    # =========================================================================
    # Strategy Determination: Vote for Split vs Merge based on active rules
    # =========================================================================
    split_votes = 0
    merge_votes = 0
    vote_reasons = []

    # Features that vote for SPLIT if prefer='lower' (e.g., smaller spread is better)
    # or prefer='higher' (e.g., higher density is better)
    for feat, rule in active_rules.items():
        is_split = False
        is_merge = False
        
        if feat == 'n_checkpoints':
            if rule['prefer'] == 'lower': is_split = True  # n_checkpoints <= X
            else: is_merge = True
        elif feat == 'spatial_spread':
            if rule['prefer'] == 'lower': is_split = True  # spread <= X (compact)
            else: is_merge = True  # spread >= X (large area)
        elif feat == 'mean_pairwise_dist':
            if rule['prefer'] == 'lower': is_split = True
            else: is_merge = True
        elif feat == 'cluster_density':
            if rule['prefer'] == 'higher': is_split = True  # density >= X (dense/compact)
            else: is_merge = True  # density <= X (sparse/large)
            
        if is_split:
            split_votes += 1
            vote_reasons.append(f"{feat}(split)")
        elif is_merge:
            merge_votes += 1
            vote_reasons.append(f"{feat}(merge)")

    # Decide strategy
    strategy = "none"
    if split_votes > merge_votes:
        strategy = "split"
    elif merge_votes > split_votes:
        strategy = "merge"
    elif split_votes > 0:
        strategy = "split" # Default to split if tied but active
        
    logger.info("Strategy Vote: Split=%d, Merge=%d. Winner: %s. Reasons: %s", 
                split_votes, merge_votes, strategy, vote_reasons)

    # =========================================================================
    # Execution: Apply Split or Merge logic
    # =========================================================================
    
    if strategy == "split":
        # Determine thresholds from rules or defaults
        max_size = int(active_rules['n_checkpoints']['threshold']) if 'n_checkpoints' in active_rules else 10
        # If spatial_spread rule exists and votes split, use it. Otherwise use a loose default.
        max_spread = active_rules['spatial_spread']['threshold'] if 'spatial_spread' in active_rules and active_rules['spatial_spread']['prefer'] == 'lower' else float('inf')
        
        logger.info("Executing SPLIT strategy: max_size=%d, max_spread=%.1f", max_size, max_spread)
        
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            cluster_size = mask.sum()
            cluster_locs = locations[mask]
            
            # Check if needs splitting
            # Condition: Size too big OR Spread too big
            # Calculate spread (max distance between any two points)
            current_spread = 0
            if len(cluster_locs) > 1:
                from scipy.spatial.distance import pdist
                current_spread = np.max(pdist(cluster_locs)) if len(cluster_locs) > 1 else 0
            
            should_split = False
            if cluster_size > max_size:
                should_split = True
            elif current_spread > max_spread:
                should_split = True
            
            if should_split:
                # Estimate number of subclusters
                n_sub_size = int(np.ceil(cluster_size / max_size)) if max_size > 0 else 1
                if max_spread != float('inf') and max_spread > 0:
                    n_sub_spread = int(np.ceil(current_spread / max_spread))
                else:
                    n_sub_spread = 1
                n_subclusters = max(n_sub_size, n_sub_spread, 2)  # At least 2 if splitting
                # Never request more clusters than samples in this cluster
                n_subclusters = min(n_subclusters, cluster_size)
                if n_subclusters < 2:
                    # Not enough points to split safely
                    continue
                
                from sklearn.cluster import KMeans
                mini_km = KMeans(n_clusters=n_subclusters, random_state=42, n_init=3)
                sub_labels = mini_km.fit_predict(cluster_locs)
                
                indices = np.where(mask)[0]
                for i, idx in enumerate(indices):
                    if sub_labels[i] > 0:
                        labels[idx] = max_new_cluster_id + sub_labels[i] - 1
                
                max_new_cluster_id += n_subclusters - 1
                splits_count += 1
                logger.debug("  Split cluster %d (size %d, spread %.1f) into %d sub-clusters", 
                            cluster_id, cluster_size, current_spread, n_subclusters)

    elif strategy == "merge":
        # Determine thresholds
        min_size = int(active_rules['n_checkpoints']['threshold']) if 'n_checkpoints' in active_rules else 3
        # If spatial_spread rule exists and votes merge (spread >= X), it means we want clusters AT LEAST X spread.
        # But merging based on spread is tricky. We mainly use size as trigger, and density as guidance.
        # For simplicity, we stick to min_size trigger, but could use density.
        
        logger.info("Executing MERGE strategy: min_size=%d", min_size)
        
        small_clusters = []
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            if mask.sum() < min_size:
                cluster_locs = locations[mask]
                centroid = cluster_locs.mean(axis=0) if len(cluster_locs) > 0 else np.zeros(3)
                small_clusters.append({
                    'id': cluster_id,
                    'size': mask.sum(),
                    'centroid': centroid,
                    'indices': np.where(mask)[0]
                })
        
        merged = set()
        for sc in small_clusters:
            if sc['id'] in merged:
                continue
            
            best_target = None
            best_dist = float('inf')
            
            for cluster_id in unique_clusters:
                if cluster_id == sc['id'] or cluster_id in merged:
                    continue
                
                mask = labels == cluster_id
                target_locs = locations[mask]
                if len(target_locs) == 0:
                    continue
                
                target_centroid = target_locs.mean(axis=0)
                dist = np.linalg.norm(sc['centroid'] - target_centroid)
                
                if dist < best_dist:
                    best_dist = dist
                    best_target = cluster_id
            
            if best_target is not None:
                for idx in sc['indices']:
                    labels[idx] = best_target
                merged.add(sc['id'])
                merges_count += 1
                logger.debug("  Merged cluster %d (size %d) into cluster %d",
                            sc['id'], sc['size'], best_target)
    
    # =========================================================================
    # Rule 2: Reassign outlier checkpoints to better clusters (type homogeneity)
    # =========================================================================
    unique_clusters = np.unique(labels)
    
    # Compute cluster centroids and dominant types
    cluster_info = {}
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_locs = locations[mask]
        cluster_types = workflow_types[mask]
        
        centroid = cluster_locs.mean(axis=0)
        type_counts = pd.Series(cluster_types).value_counts()
        dominant_type = type_counts.index[0] if len(type_counts) > 0 else None
        
        cluster_info[cluster_id] = {
            'centroid': centroid,
            'dominant_type': dominant_type,
            'size': mask.sum()
        }
    
    # For each checkpoint, check if it's an outlier in its cluster
    for i in range(n_checkpoints):
        current_cluster = labels[i]
        current_type = workflow_types[i]
        current_loc = locations[i]
        
        info = cluster_info[current_cluster]
        
        # Check if this checkpoint's type differs from cluster's dominant type
        if info['dominant_type'] and current_type != info['dominant_type']:
            # Find a better cluster with matching type and closer distance
            best_cluster = current_cluster
            best_score = float('inf')
            
            for cid, cinfo in cluster_info.items():
                if cid == current_cluster:
                    continue
                
                # Must match type
                if cinfo['dominant_type'] != current_type:
                    continue
                
                # Check size constraint
                current_max_size = 50
                if strategy == "split":
                    current_max_size = int(active_rules['n_checkpoints']['threshold']) if 'n_checkpoints' in active_rules else 10
                
                if cinfo['size'] >= current_max_size:
                    continue
                
                # Compute distance
                dist = np.linalg.norm(current_loc - cinfo['centroid'])
                
                # Score: prefer closer clusters with same type
                score = dist
                
                if score < best_score:
                    best_score = score
                    best_cluster = cid
            
            if best_cluster != current_cluster:
                # Reassign
                labels[i] = best_cluster
                # Update cluster info
                cluster_info[current_cluster]['size'] -= 1
                cluster_info[best_cluster]['size'] += 1
                reassigns_count += 1
    
    logger.info("HtT post-processing: %d splits, %d merges, %d reassignments",
                splits_count, merges_count, reassigns_count)
    
    checkpoint_clustered = checkpoint_df.copy()
    checkpoint_clustered["cluster_group"] = labels
    labels = _merge_inspection_checkpoints_by_location(checkpoint_clustered, labels)
    return labels


def _merge_inspection_checkpoints_by_location(checkpoint_df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Post-process clustering to merge same-location checkpoints.
    
    This function ensures that move_to_* and corresponding operation checkpoints at
    the same location are assigned to the same cluster, preventing unfair advantage
    from redundant agent assignment in Stage 2.
    
    Handles:
    - inspection / move_to_inspection
    - communication / move_to_communication  
    - compute / move_to_compute
    
    Args:
        checkpoint_df: DataFrame with checkpoint data including 'operation_type' and 'location'
        labels: Current cluster assignments (will be modified)
    
    Returns:
        Updated labels with checkpoints merged by location
    """
    labels = labels.copy()
    
    # Identify all operation types that should be merged by location
    # - inspection/communication/compute: move_to_* and action at same location
    # - logistics: transport/delivery/completion at same delivery location
    mergeable_ops = [
        'inspection', 'move_to_inspection',
        'communication', 'move_to_communication',
        'compute', 'move_to_compute',
        'transport', 'delivery', 'completion',  # logistics delivery location
    ]
    
    is_mergeable = checkpoint_df['operation_type'].str.lower().isin(mergeable_ops)
    
    if not is_mergeable.any():
        return labels
    
    # Group by location
    merge_df = checkpoint_df[is_mergeable].copy()
    merge_df['label'] = labels[is_mergeable]
    merge_df['loc_str'] = merge_df['location'].astype(str)
    
    # For each location, find the most common cluster assignment
    merged_count = 0
    for loc_str, group in merge_df.groupby('loc_str'):
        if len(group) <= 1:
            continue
        
        # Get most common cluster for this location
        cluster_counts = Counter(group['label'])
        target_cluster = cluster_counts.most_common(1)[0][0]
        
        # Check if there are multiple clusters
        if len(cluster_counts) > 1:
            merged_count += len(group) - cluster_counts[target_cluster]
            # Assign all checkpoints at this location to the target cluster
            indices = group.index.tolist()
            labels[indices] = target_cluster
    
    if merged_count > 0:
        logger.info(
            "Merged %d checkpoints to same clusters (by location) for fair comparison",
            merged_count
        )
    
    return labels


def run_precluster(
    seed: int,
    scenario_name: str,
    output_root: Path,
    generation_config: Path,
    uav_types_config: Path,
    method: str,
    simulation_time: int,
    skip_simulation: bool = False,
    triplet_model_path: Path | None = None,
    rl_model_dir: Path | None = None,
    n_clusters_per_run: int | None = None,
    topology_file: Path | None = None,
    snap_to_graph: bool = False,
    rules_config: Path | None = None,
) -> Path:
    """Run Stage 1: scenario generation + single-run simulation + init-based clustering.

    Returns the path to the checkpoint_clusters_<method>.csv file under run_seed_<seed>.
    """
    current_file = Path(__file__).resolve()
    src_root = current_file.parent.parent
    project_root = src_root.parent

    # Ensure workflow_decouple src is importable
    wf_decouple_src = _add_workflow_decouple_to_path(current_file)

    # Lazy imports after sys.path is updated
    from run_dynamic_collection import run_data_collection
    from scenario_generator import generate_and_save_scenario
    from ml_training.initial_data_loader import load_initial_data
    from ml_training.checkpoint_extractor import extract_checkpoints_from_initial_data

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / f"run_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Stage 1: Precluster from generated scenario (init-based)")
    logger.info("Seed: %s", seed)
    logger.info("Scenario name: %s", scenario_name)
    logger.info("Output root: %s", output_root)
    logger.info("Run directory: %s", run_dir)
    logger.info("Generation config: %s", generation_config)
    logger.info("UAV types config: %s", uav_types_config)
    logger.info("Clustering method: %s", method)
    logger.info("Skip simulation: %s", skip_simulation)
    logger.info("Rules config override: %s", rules_config if rules_config is not None else "auto")
    logger.info("workflow_decouple src: %s", wf_decouple_src)
    logger.info("=" * 70)

    topology_json_path: Optional[Path] = topology_file
    if topology_json_path is not None and not topology_json_path.exists():
        try:
            with generation_config.open("r", encoding="utf-8") as f:
                gen_cfg = yaml.safe_load(f) or {}
        except Exception:
            gen_cfg = {}

        sim_space = gen_cfg.get("simulation_space", {}) or {}
        bounds = {
            "x_min": float(sim_space.get("x_min", 0.0)),
            "x_max": float(sim_space.get("x_max", 5000.0)),
            "y_min": float(sim_space.get("y_min", 0.0)),
            "y_max": float(sim_space.get("y_max", 5000.0)),
            "z_min": float(sim_space.get("z_min", 0.0)),
            "z_max": float(sim_space.get("z_max", 200.0)),
        }

        topo = TopologyManager(bounds=bounds, scale_factor=1.0)
        topo.generate_grid_topology(
            n_vertiports=40,
            n_waypoints=120,
            n_charging=10,
            k_neighbors=5,
            seed=seed,
        )

        topology_json_path.parent.mkdir(parents=True, exist_ok=True)
        topo.save_to_file(str(topology_json_path))

        nodes_csv = topology_json_path.with_name(f"{topology_json_path.stem}_nodes.csv")
        edges_csv = topology_json_path.with_name(f"{topology_json_path.stem}_edges.csv")
        topo.export_to_csv(str(nodes_csv), str(edges_csv))

        traffic_csv = topology_json_path.with_name(f"{topology_json_path.stem}_traffic.csv")
        with traffic_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["u", "v", "t_start", "t_end", "speed_factor", "blocked"])
            for (u, v), edge in topo.edges.items():
                writer.writerow([u, v, 0.0, 1e9, 1.0, 0])

        logger.info("Generated topology JSON and CSV at %s", topology_json_path)

    # 1) Optional scenario generation + simulation + data collection
    if not skip_simulation:
        # Generate scenario.yaml using workflow_decouple's ScenarioGenerator
        scenario_output = run_dir / "scenario.yaml"
        logger.info("Generating scenario.yaml using scenario_generation.yaml and seed...")
        generate_and_save_scenario(
            scenario_size=None,
            output_path=str(scenario_output),
            uav_types_config=str(uav_types_config),
            generation_config=str(generation_config),
            scenario_name=scenario_name,
            seed=seed,
            topology_file=str(topology_json_path) if topology_json_path is not None else None,
            snap_to_graph=bool(snap_to_graph or topology_json_path is not None),
        )

        logger.info("Running data collection simulation...")
        run_data_collection(
            scenario_config_path=str(scenario_output),
            simulation_time=simulation_time,
            output_dir=str(run_dir),
        )
        logger.info("Data collection finished. Files should exist under: %s", run_dir)
    else:
        logger.info(
            "Skipping simulation step. Expecting existing workflow_init_info.csv, "
            "agent_snapshots.csv and agent_id_mapping.csv under: %s",
            run_dir,
        )

    # 2) Extract checkpoints from initial data (workflow_init_info + initial agent states)
    logger.info("Extracting initial checkpoints and feature matrix from %s", run_dir)
    workflow_df, agent_df, workflows_features = load_initial_data(run_dir)
    logger.info("Loaded %d workflows and %d agents", len(workflow_df), len(agent_df))

    checkpoint_df, feature_matrix = extract_checkpoints_from_initial_data(workflows_features)

    # Ensure source_run_id column exists because downstream pipelines read it.
    if "source_run_id" not in checkpoint_df.columns:
        checkpoint_df["source_run_id"] = seed

    # Save raw init checkpoints for inspection
    checkpoint_init_file = run_dir / "checkpoint_init.csv"
    checkpoint_df.to_csv(checkpoint_init_file, index=False)
    logger.info("Saved init checkpoints to: %s", checkpoint_init_file)

    # 3) Init-based clustering (no global trajectory, no threshold search)
    method = method.lower()

    # ------------------------------------------------------------------
    # Baseline: KMeans on 26-d checkpoint features
    # ------------------------------------------------------------------
    if method == "kmeans":
        logger.info("Running KMeans clustering on init-based checkpoint features...")

        # 默认使用固定簇数，后续会被 rule_kmeans 替代
        if n_clusters_per_run is None:
            n_clusters_per_run = 30

        run_ids = checkpoint_df["source_run_id"].values
        labels = np.zeros(len(checkpoint_df), dtype=int)
        cluster_offset = 0
        logger.info("Using n_clusters_per_run (kmeans): %d", n_clusters_per_run)


        for run_id in sorted(np.unique(run_ids)):
            run_mask = run_ids == run_id
            run_features = feature_matrix[run_mask]

            n_samples = len(run_features)
            if n_samples == 0:
                logger.warning("Run %s has no checkpoints, skipping.", run_id)
                continue

            n_clusters_actual = min(n_clusters_per_run, n_samples)
            logger.info("  Run %s: %d samples, using %d clusters", run_id, n_samples, n_clusters_actual)

            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=seed, n_init=5)
            run_labels = kmeans.fit_predict(run_features)

            labels[run_mask] = run_labels + cluster_offset
            cluster_offset += n_clusters_actual

    # ------------------------------------------------------------------
    # Triplet model: Siamese embedding + KMeans per run
    # ------------------------------------------------------------------
    elif method == "triplet":
        if triplet_model_path is None:
            raise ValueError("triplet_model_path must be provided when method='triplet'")

        from ml_training.deep_learning.siamese_encoder import SiameseEncoder

        if not Path(triplet_model_path).exists():
            raise FileNotFoundError(f"Triplet model not found: {triplet_model_path}")

        logger.info("Running Triplet-based clustering (Siamese embedding + KMeans)...")
        logger.info("Triplet model path: %s", triplet_model_path)

        # Standardize features (match training-time preprocessing strategy)
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(feature_matrix)

        # Load encoder (CPU is enough for offline Stage1)
        encoder = SiameseEncoder.load(str(triplet_model_path))
        encoder.eval()

        with torch.no_grad():
            embeddings = encoder.get_embedding(features_norm)
        embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)

        run_ids = checkpoint_df["source_run_id"].values
        labels = np.zeros(len(checkpoint_df), dtype=int)
        cluster_offset = 0
        # Use parameter or default to 30
        if n_clusters_per_run is None:
            n_clusters_per_run = 30
        logger.info("Using n_clusters_per_run: %d", n_clusters_per_run)

        for run_id in sorted(np.unique(run_ids)):
            run_mask = run_ids == run_id
            run_embeddings = embeddings_np[run_mask]

            n_samples = len(run_embeddings)
            if n_samples == 0:
                logger.warning("Run %s has no checkpoints, skipping.", run_id)
                continue

            n_clusters_actual = min(n_clusters_per_run, n_samples)
            logger.info("  Run %s: %d samples, using %d clusters", run_id, n_samples, n_clusters_actual)

            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=seed, n_init=5)
            run_labels = kmeans.fit_predict(run_embeddings)

            labels[run_mask] = run_labels + cluster_offset
            cluster_offset += n_clusters_actual

    # ------------------------------------------------------------------
    # RL model: ObservationEncoder + MixedActionPolicy + weighted KMeans
    # ------------------------------------------------------------------
    elif method == "rl":
        if rl_model_dir is None:
            raise ValueError("rl_model_dir must be provided when method='rl'")

        from rl.observation_encoder import ObservationEncoder
        from rl.policy import MixedActionPolicy
        from rl.weighted_kmeans import kmeans_with_weights

        rl_model_dir = Path(rl_model_dir)
        if not rl_model_dir.exists():
            raise FileNotFoundError(f"RL model directory not found: {rl_model_dir}")

        policy_path = rl_model_dir / "policy.pth"
        if not policy_path.exists():
            raise FileNotFoundError(f"RL policy not found in {rl_model_dir} (expected best_policy.pth or policy.pth)")

        encoder_stats_path = rl_model_dir / "encoder_stats.pth"
        if not encoder_stats_path.exists():
            logger.warning("encoder_stats.pth not found in %s, proceeding without normalization", rl_model_dir)

        logger.info("Running RL-based clustering (weighted KMeans)...")
        logger.info("RL model dir: %s", rl_model_dir)
        logger.info("Using policy checkpoint: %s", policy_path)

        # Initialize encoder and load normalization stats (if available)
        encoder = ObservationEncoder(normalize=True)
        if encoder_stats_path.exists():
            encoder.load_stats(encoder_stats_path)

        # Encode observation from the same run_dir used above
        obs = encoder.encode(run_dir)

        # Initialize policy and load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = MixedActionPolicy(
            obs_dim=encoder.obs_dim,
            weight_dim=feature_matrix.shape[1],
            hidden_dims=[256, 256],
            dropout=0.1,
        )
        state_dict = torch.load(policy_path, map_location=device)
        policy.load_state_dict(state_dict)
        policy.to(device)
        obs = obs.to(device)

        # 采样权重和簇数（policy自主控制）
        with torch.no_grad():
            weights_tensor, n_clusters, _ = policy.sample(obs, deterministic=True)

        weights_np = weights_tensor.detach().cpu().numpy()
        logger.info("  Policy sampled n_clusters: %d (from policy, not fixed)", n_clusters)
        
        # 如果命令行指定了n_clusters_per_run，警告忽略
        if n_clusters_per_run is not None:
            logger.warning(
                "--n-clusters-per-run=%d is specified but will be IGNORED. "
                "Policy autonomously controls cluster count (sampled: %d)",
                n_clusters_per_run, n_clusters
            )

        run_ids = checkpoint_df["source_run_id"].values
        labels = kmeans_with_weights(
            features=feature_matrix,
            run_ids=run_ids,
            weights=weights_np,
            n_clusters_per_run=n_clusters,
            random_state=seed,
            n_init=5,
        )

    # ------------------------------------------------------------------
    # HtT variants: base clustering + HtT rule-based post-processing
    # ------------------------------------------------------------------
    elif method in ("kmeans_htt", "rl_htt", "triplet_htt"):
        # Determine base method
        base_method = method.replace("_htt", "")
        logger.info("Running HtT variant: base=%s + HtT post-processing", base_method)
        
        # Run base clustering first (recursive call with skip_simulation=True to avoid re-running sim)
        if base_method == "kmeans":
            # KMeans baseline
            run_ids = checkpoint_df["source_run_id"].values
            labels = np.zeros(len(checkpoint_df), dtype=int)
            cluster_offset = 0
            if n_clusters_per_run is None:
                n_clusters_per_run = 50
            logger.info("Using n_clusters_per_run (kmeans): %d", n_clusters_per_run)
            
            for run_id in sorted(np.unique(run_ids)):
                run_mask = run_ids == run_id
                run_features = feature_matrix[run_mask]
                
                n_samples = len(run_features)
                if n_samples == 0:
                    continue
                
                n_clusters_actual = min(n_clusters_per_run, n_samples)
                kmeans = KMeans(n_clusters=n_clusters_actual, random_state=seed, n_init=5)
                run_labels = kmeans.fit_predict(run_features)
                
                labels[run_mask] = run_labels + cluster_offset
                cluster_offset += n_clusters_actual
        
        elif base_method == "triplet":
            if triplet_model_path is None:
                raise ValueError("triplet_model_path required for triplet_htt")
            
            from ml_training.deep_learning.siamese_encoder import SiameseEncoder
            
            scaler = StandardScaler()
            features_norm = scaler.fit_transform(feature_matrix)
            
            encoder = SiameseEncoder.load(str(triplet_model_path))
            encoder.eval()
            
            with torch.no_grad():
                embeddings = encoder.get_embedding(features_norm)
            embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
            
            run_ids = checkpoint_df["source_run_id"].values
            labels = np.zeros(len(checkpoint_df), dtype=int)
            cluster_offset = 0
            if n_clusters_per_run is None:
                n_clusters_per_run = 50
            
            for run_id in sorted(np.unique(run_ids)):
                run_mask = run_ids == run_id
                run_embeddings = embeddings_np[run_mask]
                
                n_samples = len(run_embeddings)
                if n_samples == 0:
                    continue
                
                n_clusters_actual = min(n_clusters_per_run, n_samples)
                kmeans = KMeans(n_clusters=n_clusters_actual, random_state=seed, n_init=5)
                run_labels = kmeans.fit_predict(run_embeddings)
                
                labels[run_mask] = run_labels + cluster_offset
                cluster_offset += n_clusters_actual
        
        elif base_method == "rl":
            if rl_model_dir is None:
                raise ValueError("rl_model_dir required for rl_htt")
            
            from rl.observation_encoder import ObservationEncoder
            from rl.policy import MixedActionPolicy
            from rl.weighted_kmeans import kmeans_with_weights
            
            rl_model_dir_path = Path(rl_model_dir)
            policy_path = rl_model_dir_path / "policy.pth"
            encoder_stats_path = rl_model_dir_path / "encoder_stats.pth"
            
            encoder = ObservationEncoder(normalize=True)
            if encoder_stats_path.exists():
                encoder.load_stats(encoder_stats_path)
            
            obs = encoder.encode(run_dir)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = MixedActionPolicy(
                obs_dim=encoder.obs_dim,
                weight_dim=feature_matrix.shape[1],
                hidden_dims=[256, 256],
                dropout=0.1,
            )
            state_dict = torch.load(policy_path, map_location=device)
            policy.load_state_dict(state_dict)
            policy.to(device)
            obs = obs.to(device)
            
            with torch.no_grad():
                weights_tensor, n_clusters, _ = policy.sample(obs, deterministic=True)
            
            weights_np = weights_tensor.detach().cpu().numpy()
            
            run_ids = checkpoint_df["source_run_id"].values
            labels = kmeans_with_weights(
                features=feature_matrix,
                run_ids=run_ids,
                weights=weights_np,
                n_clusters_per_run=n_clusters,
                random_state=seed,
                n_init=5,
            )
        
        # Apply HtT post-processing
        labels = _apply_htt_rules_postprocessing(checkpoint_df, labels, scenario_name)

    # ------------------------------------------------------------------
    # rule_kmeans: Rule-based weight selection + KMeans
    # ------------------------------------------------------------------
    elif method == "rule_kmeans":
        from rule_library.rule_based_weight_selector import RuleBasedWeightSelector

        config_path = str(
            _resolve_rules_config_for_scenario(
                scenario_name=scenario_name,
                project_root=project_root,
                rules_config=rules_config,
            )
        )

        logger.info("Using rule_kmeans with config: %s", config_path)

        selector = RuleBasedWeightSelector(config_path)
        ctx = selector.summarize_context(checkpoint_df, feature_matrix)
        weights, n_clusters_rule = selector.select_weights_and_k(ctx)

        # 如果命令行指定了 n_clusters_per_run，使用命令行值
        if n_clusters_per_run is not None:
            logger.info("Overriding rule-based n_clusters=%d with CLI value=%d",
                       n_clusters_rule, n_clusters_per_run)
            n_clusters_rule = n_clusters_per_run

        # 加权特征
        features_weighted = feature_matrix * weights

        run_ids = checkpoint_df["source_run_id"].values
        labels = np.zeros(len(checkpoint_df), dtype=int)
        cluster_offset = 0

        for run_id in sorted(np.unique(run_ids)):
            run_mask = run_ids == run_id
            run_features = features_weighted[run_mask]

            n_samples = len(run_features)
            if n_samples == 0:
                continue

            n_clusters_actual = min(n_clusters_rule, n_samples)
            logger.info("  Run %s: %d samples, using %d clusters", run_id, n_samples, n_clusters_actual)

            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=seed, n_init=5)
            run_labels = kmeans.fit_predict(run_features)

            labels[run_mask] = run_labels + cluster_offset
            cluster_offset += n_clusters_actual

    # ------------------------------------------------------------------
    # htt_recursive: 论文描述的递归HtT分解树
    # 在每个节点根据上下文查询规则库选择权重，进行spectral bisection
    # ------------------------------------------------------------------
    elif method == "htt_recursive":
        from rule_library.rule_based_weight_selector import RuleBasedWeightSelector
        from rule_library.recursive_htt_clusterer import RecursiveHtTClusterer

        config_path = str(
            _resolve_rules_config_for_scenario(
                scenario_name=scenario_name,
                project_root=project_root,
                rules_config=rules_config,
            )
        )

        logger.info("Using htt_recursive with config: %s", config_path)

        # 初始化权重选择器和递归聚类器
        selector = RuleBasedWeightSelector(config_path)
        
        # 从规则配置获取递归参数（如果有的话）
        # 通过max_depth/min_samples/min_variance/max_leaf_size控制树的结构和叶子cluster大小
        recursive_params = selector.config.get('recursive_params', {})
        max_depth = recursive_params.get('max_depth', 6)
        min_samples = recursive_params.get('min_samples', 15)
        min_variance = recursive_params.get('min_variance', 0.01)
        bisection_method = recursive_params.get('bisection_method', 'kmeans')
        max_leaf_size = recursive_params.get('max_leaf_size', None)
        spatial_var_threshold = recursive_params.get('spatial_var_threshold', None)
        max_workflows_per_cluster = recursive_params.get('max_workflows_per_cluster', None)
        causal_anchor_lambda = recursive_params.get('causal_anchor_lambda', 0.0)
        disable_adaptive_min_samples = bool(recursive_params.get('disable_adaptive_min_samples', False))

        clusterer = RecursiveHtTClusterer(
            weight_selector=selector,
            max_depth=max_depth,
            min_samples=min_samples,
            min_variance=min_variance,
            bisection_method=bisection_method,
            max_leaf_size=max_leaf_size,
            spatial_var_threshold=spatial_var_threshold,
            max_workflows_per_cluster=max_workflows_per_cluster,
            causal_anchor_lambda=causal_anchor_lambda,
        )

        # 按run分别处理
        run_ids = checkpoint_df["source_run_id"].values
        labels = np.zeros(len(checkpoint_df), dtype=int)
        cluster_offset = 0

        for run_id in sorted(np.unique(run_ids)):
            run_mask = run_ids == run_id
            run_checkpoint_df = checkpoint_df[run_mask].reset_index(drop=True)
            run_features = feature_matrix[run_mask]

            n_samples = len(run_features)
            if n_samples == 0:
                continue

            # 自适应策略：根据workflow数量调整参数
            n_workflows = run_checkpoint_df['workflow_id'].nunique() if 'workflow_id' in run_checkpoint_df.columns else n_samples // 5
            
            if n_workflows < 40 and not disable_adaptive_min_samples:
                # 小规模场景：适度调整参数，但仍尊重 config 下限
                adaptive_min_samples = max(min_samples, n_samples // 6)
                adaptive_max_depth = max(max_depth, 3)
                logger.info("  Run %s: %d samples, %d workflows (SMALL) -> adaptive min_samples=%d, max_depth=%d",
                           run_id, n_samples, n_workflows, adaptive_min_samples, adaptive_max_depth)
                
                # 重新创建clusterer使用自适应参数
                adaptive_clusterer = RecursiveHtTClusterer(
                    weight_selector=selector,
                    max_depth=adaptive_max_depth,
                    min_samples=adaptive_min_samples,
                    min_variance=min_variance,
                    bisection_method=bisection_method,
                    max_leaf_size=max_leaf_size,
                    spatial_var_threshold=spatial_var_threshold,
                    max_workflows_per_cluster=max_workflows_per_cluster,
                    causal_anchor_lambda=causal_anchor_lambda,
                )
                run_labels = adaptive_clusterer.fit_predict(run_checkpoint_df, run_features)
                stats = adaptive_clusterer.get_tree_stats()
            else:
                # 中大规模场景：使用默认参数
                logger.info("  Run %s: %d samples, %d workflows, applying recursive HtT decomposition", 
                           run_id, n_samples, n_workflows)
                run_labels = clusterer.fit_predict(run_checkpoint_df, run_features)
                stats = clusterer.get_tree_stats()
            
            logger.info("    Tree stats: %d clusters, %d nodes, max_depth=%d",
                       stats['n_clusters'], stats['total_nodes'], stats['max_depth_reached'])

            labels[run_mask] = run_labels + cluster_offset
            cluster_offset += stats['n_clusters']

    # ------------------------------------------------------------------
    # hungarian: 基于workflow第一个checkpoint到agent距离的贪婪匹配
    # 每个workflow分配给最近的agent，同一workflow的所有checkpoints属于同一cluster
    # ------------------------------------------------------------------
    elif method == "hungarian":
        logger.info("Using hungarian baseline: greedy workflow-agent matching by first checkpoint distance")
        
        # 读取agent位置
        agent_snapshot_file = run_dir / "agent_snapshots.csv"
        if not agent_snapshot_file.exists():
            raise FileNotFoundError(f"Agent snapshot file not found: {agent_snapshot_file}")
        agent_df = pd.read_csv(agent_snapshot_file)
        
        # 获取每个agent的初始位置 (最早时间点)
        agent_positions = {}
        min_time = agent_df['time'].min()
        for _, row in agent_df[agent_df['time'] == min_time].iterrows():
            agent_id = row['agent_id']
            # 位置在agent_state列中
            agent_state = ast.literal_eval(row['agent_state']) if isinstance(row['agent_state'], str) else row['agent_state']
            pos = agent_state.get('position', (0, 0, 0))
            agent_positions[agent_id] = np.array([pos[0], pos[1], pos[2]])
        
        # 获取每个workflow的第一个checkpoint位置
        workflow_first_ckp = {}
        for _, row in checkpoint_df.iterrows():
            wf_id = row['workflow_id']
            if wf_id not in workflow_first_ckp:
                loc = ast.literal_eval(row['location']) if isinstance(row['location'], str) else row['location']
                workflow_first_ckp[wf_id] = np.array([loc[0], loc[1], loc[2]])
        
        # 贪婪匹配：每个workflow分配给最近的agent
        workflow_to_agent = {}
        agent_list = list(agent_positions.keys())
        
        for wf_id, wf_pos in workflow_first_ckp.items():
            best_agent = None
            best_dist = float('inf')
            for agent_id, agent_pos in agent_positions.items():
                dist = np.linalg.norm(wf_pos - agent_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_agent = agent_id
            workflow_to_agent[wf_id] = best_agent
        
        # 为每个agent创建一个cluster
        agent_to_cluster = {agent: i for i, agent in enumerate(agent_list)}
        
        # 分配labels：同一workflow的所有checkpoints属于同一cluster
        labels = np.zeros(len(checkpoint_df), dtype=int)
        for idx, row in checkpoint_df.iterrows():
            wf_id = row['workflow_id']
            agent = workflow_to_agent.get(wf_id, agent_list[0])
            labels[idx] = agent_to_cluster.get(agent, 0)
        
        logger.info("  Hungarian: %d workflows -> %d agents (clusters)", 
                   len(workflow_first_ckp), len(set(labels)))

    else:
        raise ValueError(
            f"Unsupported clustering method: {method} (supported: 'kmeans', 'triplet', 'rl', 'rule_kmeans', 'htt_recursive', 'kmeans_htt', 'triplet_htt', 'rl_htt', 'hungarian')"
        )

    checkpoint_clustered = checkpoint_df.copy()
    checkpoint_clustered["cluster_group"] = labels

    # ============================================================================
    # POST-PROCESSING DISABLED FOR PERFORMANCE COMPARISON
    # ============================================================================
    # Post-processing: merge same-location inspection checkpoints to same cluster
    # This ensures fair comparison across all methods (prevents KMeans from getting
    # unfair advantage due to splitting move_to_inspection and inspection into different clusters)
    # labels = _merge_inspection_checkpoints_by_location(checkpoint_clustered, labels)
    
    # Renumber cluster IDs to be consecutive (0, 1, 2, ..., N-1)
    # This ensures accurate cluster count statistics after merging
    unique_labels = np.unique(labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_mapping[l] for l in labels])
    
    checkpoint_clustered["cluster_group"] = labels
    # ============================================================================

    cluster_file = run_dir / f"checkpoint_clusters_{method}.csv"
    checkpoint_clustered.to_csv(cluster_file, index=False)
    logger.info("Saved init-based clustering result to: %s", cluster_file)

    return cluster_file


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    current_file = Path(__file__).resolve()
    src_root = current_file.parent.parent
    project_root = src_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: Generate scenario from scenario_generation.yaml and seed, "
            "run a single simulation, and perform init-based checkpoint clustering."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used for scenario generation and to label run_seed_<seed> directory.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="large_mixed",
        help="Scenario name defined in workflow_decouple/config/scenario_generation.yaml (e.g., large_mixed).",
    )
    parser.add_argument(
        "--generation-config",
        type=str,
        default="workflow_decouple/config/scenario_generation.yaml",
        help="Path to scenario_generation.yaml (absolute or project-root-relative).",
    )
    parser.add_argument(
        "--uav-types-config",
        type=str,
        default="workflow_decouple/config/uav_types.yaml",
        help="Path to uav_types.yaml (absolute or project-root-relative).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/offline_init",
        help="Root directory to store run_seed_<seed> directory (relative to project root by default).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="kmeans",
        choices=["kmeans", "triplet", "rl", "rule_kmeans", "htt_recursive", "kmeans_htt", "rl_htt", "triplet_htt", "hungarian"],
        help="Clustering method to use on init-based checkpoint features.",
    )
    parser.add_argument(
        "--triplet-model-path",
        type=str,
        default="workflow_decouple/data/multi_seed_runs/triplet_learning_results/siamese_encoder.pth",
        help="Path to trained Triplet Siamese encoder (used when --method=triplet).",
    )
    parser.add_argument(
        "--rl-model-dir",
        type=str,
        default="workflow_decouple/data/multi_seed_runs/bandit_results",
        help="Directory containing RL bandit results (used when --method=rl).",
    )
    parser.add_argument(
        "--n-clusters-per-run",
        type=int,
        default=None,
        help=(
            "Number of clusters per run. "
            "For kmeans: overrides config scenario values (default: 30 if not specified). "
            "For kmeans/triplet: fixed hyperparameter for per-run clustering."
        ),
    )
    parser.add_argument(
        "--simulation-time",
        type=int,
        default=1800,
        help="Simulation time in seconds for data collection.",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help=(
            "Skip scenario generation and simulation; expect workflow_init_info.csv, "
            "agent_snapshots.csv, and agent_id_mapping.csv to already exist under run_seed_<seed>."
        ),
    )
    parser.add_argument(
        "--topology-file",
        type=str,
        default=None,
        help=(
            "Optional path to topology JSON. If provided and missing, a grid topology will be "
            "generated using simulation_space bounds and exported to JSON/CSV."
        ),
    )
    parser.add_argument(
        "--snap-to-graph",
        action="store_true",
        help=(
            "Enable snap-to-graph behavior in scenario generation (positions snapped to nearest "
            "topology nodes when topology is available)."
        ),
    )
    parser.add_argument(
        "--rules-config",
        type=str,
        default=None,
        help="Optional path to a rule_kmeans_config YAML used to override scenario/default rule lookup.",
    )

    args = parser.parse_args()

    seed = args.seed
    scenario_name = args.scenario_name

    generation_cfg = _resolve_under_project(args.generation_config, project_root)
    uav_types_cfg = _resolve_under_project(args.uav_types_config, project_root)
    output_root = _resolve_under_project(args.output_root, project_root)
    triplet_model_path = _resolve_under_project(args.triplet_model_path, project_root)
    rl_model_dir = _resolve_under_project(args.rl_model_dir, project_root)

    topo_path: Optional[Path] = None
    if args.topology_file is not None:
        topo_path = _resolve_under_project(args.topology_file, project_root)
    rules_config_path: Optional[Path] = None
    if args.rules_config:
        rules_config_path = _resolve_under_project(args.rules_config, project_root)

    cluster_file = run_precluster(
        seed=seed,
        scenario_name=scenario_name,
        output_root=output_root,
        generation_config=generation_cfg,
        uav_types_config=uav_types_cfg,
        method=args.method,
        simulation_time=args.simulation_time,
        skip_simulation=args.skip_simulation,
        triplet_model_path=triplet_model_path,
        rl_model_dir=rl_model_dir,
        n_clusters_per_run=args.n_clusters_per_run,
        topology_file=topo_path,
        snap_to_graph=args.snap_to_graph,
        rules_config=rules_config_path,
    )

    print("\n=== Stage 1 Completed ===")
    print(f"Init-based clustering file: {cluster_file}")


if __name__ == "__main__":
    main()
