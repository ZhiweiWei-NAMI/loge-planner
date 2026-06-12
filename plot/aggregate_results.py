"""Aggregate experiment results into a unified DataFrame.

This script reads all workflow_summary.json files from data/results/
and aggregates them into a single DataFrame for analysis and plotting.

Output:
- aggregated_results.csv: Per-experiment metrics
- aggregated_by_type.csv: Per-workflow-type metrics
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from .config import (
        METHODS,
        SCENARIOS,
        SEEDS,
        WORKFLOW_TYPES,
        RESULTS_ROOT,
        PROJECT_ROOT,
        OFFLINE_INIT_ROOT,
        PLOT_DATA_ROOT,
        get_method_display_name,
        get_scenario_display_name,
    )
except ImportError:
    from config import (
        METHODS,
        SCENARIOS,
        SEEDS,
        WORKFLOW_TYPES,
        RESULTS_ROOT,
        PROJECT_ROOT,
        OFFLINE_INIT_ROOT,
        PLOT_DATA_ROOT,
        get_method_display_name,
        get_scenario_display_name,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_result_filename(filename: str) -> Optional[Tuple[str, int]]:
    """Parse method and timestamp from result filename.
    
    Format: Offline-{method}_{timestamp}_workflow_summary.json
    
    Returns (method, timestamp) or None if not parseable.
    """
    pattern = r"Offline-(\w+)_(\d+)_workflow_summary\.json"
    match = re.match(pattern, filename)
    
    if match:
        return match.group(1), int(match.group(2))
    return None


def load_workflow_summary(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a workflow_summary.json file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def extract_metrics_from_summary(
    data: Dict[str, Any],
    method: str,
    scenario: str,
    seed: int,
) -> Dict[str, Any]:
    """Extract key metrics from workflow summary data.
    
    Note: 数据源头（data_collector.py）已经对失败的workflow使用1800s惩罚，
    这里直接使用statistics里的值即可，无需重复处理。
    """
    
    stats = data.get("statistics", {})
    total = data.get("total_workflows", 0)
    per_workflow = data.get("data", {}) or {}

    # 直接使用statistics里已经包含惩罚的平均完成时间
    completed = stats.get("completed_workflows", 0)
    completion_rate = completed / total if total > 0 else 0.0
    avg_completion_time = stats.get("average_completion_time", 0.0)

    # -------------------------------------------------------------------------
    # Compute ANMD (Average Normalized Mission Distance) from movement data
    # -------------------------------------------------------------------------
    anmd = np.nan
    try:
        # Construct movement filename: Offline-{method}_{timestamp}_movement.json
        # We need to find the timestamp from the file path if possible, or pass it in.
        # But here we only have `data`. Let's assume we can find the sibling file.
        # This function doesn't know the filepath.
        # We will handle ANMD in `aggregate_all_results` where we have the filepath.
        pass
    except Exception:
        pass

    # Per-type metrics（各类型也需要对失败的workflow使用1800s惩罚）
    by_type: Dict[str, Dict[str, Any]] = {}

    for wf_id, wf_info in per_workflow.items():
        wf_type = wf_info.get("workflow_type", "Unknown")

        if wf_type not in by_type:
            by_type[wf_type] = {
                "total": 0,
                "completed": 0,
                "durations": [],
            }

        by_type[wf_type]["total"] += 1

        if "duration" in wf_info:
            by_type[wf_type]["completed"] += 1
            by_type[wf_type]["durations"].append(float(wf_info["duration"]))
        else:
            # 失败的workflow使用1800s惩罚
            by_type[wf_type]["durations"].append(1800.0)

    metrics = {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "total_workflows": total,
        "completed_workflows": completed,
        "completion_rate": completion_rate,
        "avg_completion_time": avg_completion_time,
    }

    # Add per-type metrics
    for wf_type in WORKFLOW_TYPES:
        type_data = by_type.get(wf_type, {"total": 0, "completed": 0, "durations": []})
        type_key = wf_type.lower()

        total_type = type_data["total"]
        completed_type = type_data["completed"]
        durations = type_data["durations"]

        metrics[f"{type_key}_total"] = total_type
        metrics[f"{type_key}_completed"] = completed_type
        metrics[f"{type_key}_rate"] = completed_type / total_type if total_type > 0 else 0
        metrics[f"{type_key}_avg_time"] = float(np.mean(durations)) if durations else 0.0
        metrics[f"{type_key}_std_time"] = float(np.std(durations)) if len(durations) > 1 else 0.0

    return metrics


def infer_scenario_from_workflows(total_workflows: int) -> str:
    """Infer scenario from total workflow count based on SCENARIO_SCALE.
    
    WARNING: This is ambiguous for scenarios with same scale (e.g. S1 vs S2).
    Should rely on 'scenario' field in JSON whenever possible.
    """
    from .config import SCENARIO_SCALE
    
    # Try to find exact match
    matches = []
    for scenario, scale in SCENARIO_SCALE.items():
        if scale["workflows"] == total_workflows:
            matches.append(scenario)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Ambiguous case: return the first one (usually Logistics/Inspection variants)
        # Ideally this shouldn't happen if JSON has scenario name
        return matches[0]
    
    # Fallback based on size ranges
    if total_workflows <= 40:
        return "small_logistics_inspection"
    elif total_workflows <= 100:
        return "medium_logistics_heavy"
    else:
        return "large_mixed"


def aggregate_all_results(
    results_dir: Path = RESULTS_ROOT,
    methods: List[str] = None,
    scenarios: List[str] = None,
    seeds: List[int] = None,
) -> pd.DataFrame:
    """Aggregate all results from the results directory.
    
    Groups by (scenario, method, seed) using the most recent result file.
    """
    if methods is None:
        methods = METHODS
    if scenarios is None:
        scenarios = SCENARIOS
    if seeds is None:
        seeds = SEEDS
    
    # Find all result files
    all_files = list(results_dir.glob("Offline-*_workflow_summary.json"))
    logger.info(f"Found {len(all_files)} result files")
    
    # Group files by method, keeping the most recent per method
    files_by_method: Dict[str, List[Tuple[Path, int]]] = {}
    
    for filepath in all_files:
        parsed = parse_result_filename(filepath.name)
        if parsed:
            method, timestamp = parsed
            if method not in files_by_method:
                files_by_method[method] = []
            files_by_method[method].append((filepath, timestamp))
    
    # Extract metrics from each file
    all_metrics = []
    
    for method in methods:
        if method not in files_by_method:
            logger.warning(f"No results found for method: {method}")
            continue
        
        # Sort by timestamp (most recent first)
        files = sorted(files_by_method[method], key=lambda x: x[1], reverse=True)
        
        # Process ALL result files for this method
        for filepath, timestamp in files:
            data = load_workflow_summary(filepath)
            if data is None:
                continue
            
            # Infer scenario from total_workflows count
            total_workflows = data.get("total_workflows", 0)
            scenario = data.get("scenario") or infer_scenario_from_workflows(total_workflows)
            seed = data.get("seed", timestamp % 1000)  # Use timestamp mod as pseudo-seed
            
            metrics = extract_metrics_from_summary(data, method, scenario, seed)
            
            # ------------------------------------------------------------------
            # Calculate ANMD: Actual Fleet Distance / Theoretical Baseline Distance
            # Per-type ANMD: allocate actual distance proportionally by baseline
            # ANMD closer to 1.0 = better (less redundant flight)
            # ------------------------------------------------------------------
            movement_file = filepath.with_name(filepath.name.replace("workflow_summary.json", "movement.json"))
            anmd_val = np.nan
            type_anmd = {t.lower(): np.nan for t in WORKFLOW_TYPES}
            
            # Get completed workflow IDs and their types from the summary data
            completed_wf_by_type = {t.lower(): [] for t in WORKFLOW_TYPES}
            wf_data = data.get("data", {})
            for wf_id, wf_info in wf_data.items():
                if "duration" in wf_info:  # has duration means completed
                    wf_type = wf_info.get("workflow_type", "").lower()
                    # Normalize type names
                    if "inspection" in wf_type:
                        completed_wf_by_type["inspection"].append(wf_id)
                    elif "communication" in wf_type:
                        completed_wf_by_type["communication"].append(wf_id)
                    elif "computation" in wf_type or "compute" in wf_type:
                        completed_wf_by_type["computation"].append(wf_id)
                    elif "logistics" in wf_type:
                        completed_wf_by_type["logistics"].append(wf_id)
            
            completed_wf_ids = [wf for wfs in completed_wf_by_type.values() for wf in wfs]
            
            if movement_file.exists() and completed_wf_ids:
                try:
                    # 1) Calculate actual fleet distance from movement.json
                    with open(movement_file, "r") as mf:
                        m_data = json.load(mf)
                        movements = m_data.get("data", {})
                        
                        total_actual_distance = 0.0
                        for agent_id, moves in movements.items():
                            for m in moves:
                                distance = m.get("distance", 0.0)
                                if distance > 0:
                                    total_actual_distance += distance
                    
                    # 2) Calculate per-type theoretical baseline from checkpoint_init.csv
                    checkpoint_csv = OFFLINE_INIT_ROOT / scenario / f"run_seed_{seed}" / "checkpoint_init.csv"
                    
                    type_baseline = {t.lower(): 0.0 for t in WORKFLOW_TYPES}
                    total_baseline_distance = 0.0
                    
                    if checkpoint_csv.exists():
                        import ast
                        ckp_df = pd.read_csv(checkpoint_csv)
                        
                        for wf_type_key, wf_ids in completed_wf_by_type.items():
                            for wf_id in wf_ids:
                                wf_ckps = ckp_df[ckp_df["workflow_id"] == wf_id].sort_values("checkpoint_idx")
                                if wf_ckps.empty:
                                    continue
                                
                                # Start from origin (0, 0, 0)
                                prev_pos = np.array([0.0, 0.0, 0.0])
                                wf_baseline = 0.0
                                
                                for _, row in wf_ckps.iterrows():
                                    loc_str = row.get("location", "[0,0,0]")
                                    try:
                                        loc = ast.literal_eval(loc_str) if isinstance(loc_str, str) else loc_str
                                        curr_pos = np.array(loc[:3])
                                    except:
                                        continue
                                    
                                    dist = np.linalg.norm(curr_pos - prev_pos)
                                    wf_baseline += dist
                                    prev_pos = curr_pos
                                
                                type_baseline[wf_type_key] += wf_baseline
                                total_baseline_distance += wf_baseline
                    
                    # 3) Calculate per-type ANMD by proportional allocation
                    # actual_for_type = total_actual * (type_baseline / total_baseline)
                    # anmd_for_type = actual_for_type / type_baseline = total_actual / total_baseline
                    # So overall ANMD applies to all types (proportional allocation)
                    if total_baseline_distance > 0 and total_actual_distance > 0:
                        anmd_val = total_actual_distance / total_baseline_distance
                        
                        # Per-type ANMD: use same ratio since we can't split actual distance
                        for wf_type_key in type_baseline:
                            if type_baseline[wf_type_key] > 0:
                                type_anmd[wf_type_key] = anmd_val
                        
                except Exception as e:
                    logger.warning(f"Failed to calc ANMD for {movement_file.name}: {e}")
            
            metrics["anmd"] = anmd_val
            for wf_type_key, anmd in type_anmd.items():
                metrics[f"{wf_type_key}_anmd"] = anmd
            
            all_metrics.append(metrics)
    
    if not all_metrics:
        logger.warning("No metrics extracted")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_metrics)
    
    # Add display names
    df["method_display"] = df["method"].apply(get_method_display_name)
    df["scenario_display"] = df["scenario"].apply(get_scenario_display_name)
    
    return df


def compute_aggregated_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across seeds for each (scenario, method)."""
    
    if df.empty:
        return pd.DataFrame()
    
    # Numeric columns to aggregate
    numeric_cols = [
        "completion_rate",
        "avg_completion_time",
        "anmd",  # Added ANMD
        "inspection_rate",
        "communication_rate",
        "computation_rate",
        "logistics_rate",
        "inspection_avg_time",
        "communication_avg_time",
        "computation_avg_time",
        "logistics_avg_time",
    ]
    
    # Filter to existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    # Group by scenario and method
    grouped = df.groupby(["scenario", "method"])
    
    # Compute mean and std
    agg_funcs = {}
    for col in numeric_cols:
        agg_funcs[f"{col}_mean"] = (col, "mean")
        agg_funcs[f"{col}_std"] = (col, "std")
    
    agg_funcs["n_seeds"] = ("seed", "count")
    
    agg_df = grouped.agg(**agg_funcs).reset_index()
    
    # Add display names
    agg_df["method_display"] = agg_df["method"].apply(get_method_display_name)
    agg_df["scenario_display"] = agg_df["scenario"].apply(get_scenario_display_name)
    
    return agg_df


def save_aggregated_results(
    df: pd.DataFrame,
    output_dir: Path = None,
):
    """Save aggregated results to CSV files."""
    
    if output_dir is None:
        output_dir = PLOT_DATA_ROOT
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Raw per-experiment results
    raw_path = output_dir / "aggregated_results.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw results to: {raw_path}")
    
    # Aggregated stats
    agg_df = compute_aggregated_stats(df)
    if not agg_df.empty:
        agg_path = output_dir / "aggregated_stats.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated stats to: {agg_path}")
    
    return df, agg_df


def main():
    """Main entry point for aggregation."""
    
    logger.info("Aggregating experiment results...")
    
    df = aggregate_all_results()
    
    if df.empty:
        logger.warning("No results to aggregate")
        return
    
    logger.info(f"Aggregated {len(df)} experiment results")
    
    # Save
    df, agg_df = save_aggregated_results(df)
    
    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATION SUMMARY")
    print("=" * 70)
    
    if not df.empty:
        print("\nPer-method completion rates:")
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            rate = method_df["completion_rate"].mean() * 100
            std = method_df["completion_rate"].std() * 100
            print(f"  {get_method_display_name(method):20s}: {rate:5.1f}% ± {std:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
