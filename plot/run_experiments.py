"""Batch runner for all experiment combinations.

This script runs the full experiment grid:
- All methods x All scenarios x All seeds

It handles:
1. Stage 1: Generate scenario + precluster (per scenario x seed)
2. Stage 2: Cluster assignment (per scenario x seed x method)
3. Stage 3: PDDL plan generation (per scenario x seed x method)
4. Simulation: Run offline simulation (per scenario x seed x method)

Usage:
    python -m plot.run_experiments --scenarios all --methods all --seeds all
    python -m plot.run_experiments --scenarios large_mixed --methods rule_kmeans kmeans --seeds 42
"""

import argparse
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import pandas as pd

from .config import (
    METHODS,
    SCENARIOS, 
    SEEDS,
    PROJECT_ROOT,
    RESULTS_ROOT,
    OFFLINE_INIT_ROOT,
    OFFLINE_PLANS_ROOT,
    PLOT_DATA_ROOT,
    get_method_display_name,
    get_scenario_display_name,
)

# Build PYTHONPATH for subprocess calls
# Include airfogsim src and workflow_decouple src
AIRFOGSIM_SRC = PROJECT_ROOT.parent / "src"
WORKFLOW_DECOUPLE_SRC = PROJECT_ROOT / "workflow_decouple" / "src"

# Model paths for different methods
RL_MODEL_DIR = PROJECT_ROOT / "workflow_decouple/data/multi_seed_runs/bandit_results"
TRIPLET_MODEL_PATH = PROJECT_ROOT / "workflow_decouple/data/multi_seed_runs/triplet_learning_results/siamese_encoder.pth"

def get_subprocess_env() -> dict:
    """Get environment variables for subprocess calls with correct PYTHONPATH."""
    env = os.environ.copy()
    
    # Build PYTHONPATH
    paths = [
        str(AIRFOGSIM_SRC),
        str(WORKFLOW_DECOUPLE_SRC),
        str(PROJECT_ROOT),
    ]
    
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    
    env["PYTHONPATH"] = os.pathsep.join(paths)
    
    return env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clear_results_directory() -> None:
    """Clear all files in the results directory before running experiments."""
    if not RESULTS_ROOT.exists():
        logger.info(f"Results directory does not exist: {RESULTS_ROOT}")
        return
    
    logger.info(f"Clearing results directory: {RESULTS_ROOT}")
    
    # Remove all files in the directory
    for item in RESULTS_ROOT.iterdir():
        if item.is_file():
            item.unlink()
            logger.debug(f"Removed file: {item.name}")
        elif item.is_dir():
            shutil.rmtree(item)
            logger.debug(f"Removed directory: {item.name}")
    
    logger.info("Results directory cleared successfully")


def clear_old_cluster_files(scenario: str, seed: int, methods: List[str] = None) -> None:
    """Clear old cluster files for a specific scenario and seed.
    
    Args:
        scenario: Scenario name
        seed: Seed number
        methods: List of methods to clear cluster files for. If None, clears all.
    """
    run_dir = OFFLINE_INIT_ROOT / scenario / f"run_seed_{seed}"
    if not run_dir.exists():
        return
    
    if methods is None:
        # Clear all checkpoint_clusters_*.csv files
        pattern = "checkpoint_clusters_*.csv"
    else:
        # Clear only specified methods
        for method in methods:
            cluster_file = run_dir / f"checkpoint_clusters_{method}.csv"
            if cluster_file.exists():
                cluster_file.unlink()
                logger.info(f"Cleared old cluster file: {cluster_file.name}")
        return
    
    # Clear all matching files
    for cluster_file in run_dir.glob(pattern):
        cluster_file.unlink()
        logger.info(f"Cleared old cluster file: {cluster_file.name}")


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    scenario: str
    method: str
    seed: int
    success: bool
    duration_seconds: float
    result_file: Optional[Path] = None
    error_message: Optional[str] = None
    # Stage-3 planning latency (solver time only, excluding simulation)
    planning_time_seconds: float = 0.0


def run_stage1(scenario: str, seed: int, method: str) -> bool:
    """Run Stage 1: scenario generation + preclustering.
    
    This generates the initial scenario.yaml and checkpoint files.
    
    Args:
        scenario: Scenario name
        seed: Random seed
        method: Clustering method
    """
    logger.info(f"[Stage 1] Generating scenario: {scenario}, method: {method}, seed: {seed}")
    
    # 每个scenario使用独立的目录，避免不同scenario互相覆盖
    # 目录结构: offline_init/{scenario}/run_seed_{seed}/
    scenario_root = OFFLINE_INIT_ROOT / scenario
    run_dir = scenario_root / f"run_seed_{seed}"
    cluster_file = run_dir / f"checkpoint_clusters_{method}.csv"
    scenario_file = run_dir / "scenario.yaml"
    
    # 检查 scenario.yaml 是否存在且匹配当前 scenario
    scenario_matches = False
    if scenario_file.exists():
        try:
            import yaml
            # 使用文件锁和重试防止并发读取损坏
            scenario_config = None
            for attempt in range(3):
                try:
                    with open(scenario_file, 'r', encoding='utf-8') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            content = f.read()
                            if '\x00' in content:
                                raise ValueError("File corrupted")
                            scenario_config = yaml.safe_load(content)
                            break
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.1 * (attempt + 1))
                    else:
                        raise
            if scenario_config is None:
                raise ValueError("Failed to load scenario config")
            # 检查 scenario description 或 workflow 数量是否匹配预期
            from .config import SCENARIO_SCALE
            expected_scale = SCENARIO_SCALE.get(scenario, {})
            actual_workflows = len(scenario_config.get('workflows', []))
            expected_workflows = expected_scale.get('workflows', 0)
            
            scenario_matches = (actual_workflows == expected_workflows)
            if not scenario_matches:
                logger.warning(f"Scenario mismatch: expected {expected_workflows} workflows, found {actual_workflows}. Will regenerate.")
        except Exception as e:
            logger.warning(f"Failed to verify scenario: {e}. Will regenerate.")
    
    
    # 强制重新聚类：删除该 method 的旧 cluster 文件
    if cluster_file.exists():
        cluster_file.unlink()
    
    # 如果 scenario 不匹配，删除旧数据强制重新生成
    if not scenario_matches and run_dir.exists():
        # 删除场景和数据文件
        for old_file in ['scenario.yaml', 'workflow_init_info.csv', 'agent_snapshots.csv', 
                         'agent_id_mapping.csv', 'processed_task_spatial_temporal_coupling.csv']:
            old_path = run_dir / old_file
            if old_path.exists():
                old_path.unlink()
                logger.info(f"  Removed: {old_file}")
        # 删除所有聚类文件（因为是基于旧 scenario 的）
        for cluster_csv in run_dir.glob('checkpoint_clusters_*.csv'):
            cluster_csv.unlink()
            logger.info(f"  Removed: {cluster_csv.name}")
        # 删除旧的 PDDL plans（也是基于旧 scenario 的）
        plan_dir = OFFLINE_PLANS_ROOT / scenario / f"seed_{seed}"
        if plan_dir.exists():
            for plan_file in plan_dir.glob('pddl_plan_*.json'):
                plan_file.unlink()
                logger.info(f"  Removed plan: {plan_file.name}")
    
    # Stage1输出到scenario特定的根目录: offline_init/{scenario}/
    cmd = [
        sys.executable, "-m", "src.offline_pipeline.stage1_precluster_from_config",
        "--seed", str(seed),
        "--scenario-name", scenario,
        "--method", method,
        "--output-root", str(scenario_root),
        "--simulation-time", "1800",  # 30 minutes to ensure all agents with delayed start can execute
    ]
    
    # If scenario files already exist and match, skip simulation (only do clustering)
    if scenario_matches:
        cmd.append("--skip-simulation")
        logger.info(f"  Scenario exists and matches, skipping simulation for {method}")
    
    # Add method-specific parameters
    if method == "rl":
        cmd.extend([
            "--rl-model-dir", str(RL_MODEL_DIR),
            # n_clusters now sampled from policy, not fixed
        ])
    elif method == "triplet":
        cmd.extend([
            "--triplet-model-path", str(TRIPLET_MODEL_PATH),
            "--n-clusters-per-run", "50",  # Triplet使用30个簇（与KMeans相同）
        ])
    elif method == "kmeans":
        cmd.extend([
            "--n-clusters-per-run", "50",  # Fixed hyperparameter for KMeans baseline
        ])
    # HtT variants use the same model paths as their base methods
    elif method == "kmeans_htt":
        cmd.extend([
            "--n-clusters-per-run", "50",
        ])
    elif method == "triplet_htt":
        cmd.extend([
            "--triplet-model-path", str(TRIPLET_MODEL_PATH),
            "--n-clusters-per-run", "50",
        ])
    elif method == "rl_htt":
        cmd.extend([
            "--rl-model-dir", str(RL_MODEL_DIR),
        ])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=900,  # 15 min timeout
            env=get_subprocess_env(),
        )
        
        if result.returncode != 0:
            logger.error(f"Stage 1 failed: {result.stderr[-2000:]}")  # Last 2000 chars
            return False
        
        logger.info(f"[Stage 1] Completed for {scenario}, {method}, seed {seed}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Stage 1 timeout for {scenario}, seed {seed}")
        return False
    except Exception as e:
        logger.error(f"Stage 1 error: {e}")
        return False


def run_stage2(scenario: str, seed: int, method: str) -> bool:
    """Run Stage 2: cluster assignment."""
    logger.info(f"[Stage 2] Cluster assignment: {scenario}, {method}, seed {seed}")
    
    # Stage2从scenario特定的目录读取和输出
    # 输入: offline_init/{scenario}/run_seed_{seed}/
    # 输出: offline_plans/{scenario}/run_seed_{seed}/
    scenario_init_root = OFFLINE_INIT_ROOT / scenario
    scenario_plan_root = OFFLINE_PLANS_ROOT / scenario
    scenario_plan_root.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "src.offline_pipeline.stage2_plan_from_clusters",
        "--seed", str(seed),
        "--cluster-method", method,
        "--input-root", str(scenario_init_root),
        "--output-root", str(scenario_plan_root),
        "--cooldown", "0.0",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            env=get_subprocess_env(),
        )
        
        if result.returncode != 0:
            logger.error(f"Stage 2 failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Stage 2 error: {e}")
        return False


# Global flag for topo-aware mode (set from main)
TOPO_AWARE_MODE = False


def run_stage3(scenario: str, seed: int, method: str) -> bool:
    """Run Stage 3: PDDL plan generation."""
    logger.info(f"[Stage 3] PDDL planning: {scenario}, {method}, seed {seed}")
    
    # Stage3从 scenario特定的目录读取和输出
    # 输入scenario.yaml: offline_init/{scenario}/run_seed_{seed}/scenario.yaml
    # 输入cluster assignments: offline_plans/{scenario}/run_seed_{seed}/cluster_assignments_{method}.csv
    # 输出PDDL plan: offline_plans/{scenario}/run_seed_{seed}/pddl_plan_{method}.json
    scenario_init_root = OFFLINE_INIT_ROOT / scenario
    scenario_plan_root = OFFLINE_PLANS_ROOT / scenario
    
    # Select appropriate Stage 3 script based on method
    if method == "monolithic":
        module = "src.offline_pipeline.stage3_plan_monolithic_pddl"
        cmd = [
            sys.executable, "-m", module,
            "--seed", str(seed),
            "--input-root", str(scenario_init_root),
            "--output-root", str(scenario_plan_root),
        ]
    else:
        module = "src.offline_pipeline.stage3_plan_with_pddl"
        cmd = [
            sys.executable, "-m", module,
            "--seed", str(seed),
            "--cluster-method", method,
            "--input-root", str(scenario_init_root),
            "--output-root", str(scenario_plan_root),
            "--export-edge-routes",  # Always enable edge route generation for path planning
        ]
        # Add topo-aware flag if enabled (uses time-varying speed for path optimization)
        if TOPO_AWARE_MODE:
            cmd.append("--topo-aware")
            logger.info(f"[Stage 3] Topo-aware mode enabled (time-varying path optimization)")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            env=get_subprocess_env(),
        )
        
        if result.returncode != 0:
            logger.error(f"Stage 3 failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Stage 3 error: {e}")
        return False


def run_simulation(scenario: str, seed: int, method: str) -> Optional[Path]:
    """Run offline simulation and return result file path."""
    logger.info(f"[Simulation] Running: {scenario}, {method}, seed {seed}")
    
    # Simulation读取scenario配置和PDDL plan
    # scenario.yaml: offline_init/{scenario}/run_seed_{seed}/scenario.yaml
    # PDDL plan: offline_plans/{scenario}/run_seed_{seed}/pddl_plan_{method}.json
    # 输出: data/results/Offline-{method}_{timestamp}_workflow_summary.json
    scenario_init_root = OFFLINE_INIT_ROOT / scenario
    scenario_plan_root = OFFLINE_PLANS_ROOT / scenario
    
    cmd = [
        sys.executable, "run_offline_simulation.py",
        "--method", method,
        "--seed", str(seed),
        "--scenario", scenario,
        "--init-root", str(scenario_init_root),
        "--plan-root", str(scenario_plan_root),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min timeout
            env=get_subprocess_env(),
        )
        
        if result.returncode != 0:
            logger.error(f"Simulation failed: {result.stderr}")
            return None
        
        # Find the result file (most recent with matching method)
        pattern = f"Offline-{method}_*_workflow_summary.json"
        result_files = sorted(
            RESULTS_ROOT.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        if result_files:
            result_file = result_files[0]
            # Inject scenario, seed, and planning_time into the result JSON
            try:
                import json
                with open(result_file, 'r') as f:
                    data = json.load(f)
                data['scenario'] = scenario
                data['seed'] = seed
                # planning_times is passed via closure from run_experiment
                with open(result_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to inject scenario into result: {e}")
            return result_file
        
        logger.warning(f"No result file found for {method}")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error(f"Simulation timeout for {scenario}, {method}, seed {seed}")
        return None
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return None


def run_experiment(scenario: str, method: str, seed: int) -> ExperimentResult:
    """Run a single experiment (all stages + simulation).
    
    Args:
        scenario: Scenario name
        method: Method name
        seed: Random seed
    """
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info(f"Experiment: {get_scenario_display_name(scenario)} | "
                f"{get_method_display_name(method)} | seed={seed}")
    logger.info("=" * 70)
    
    # 使用scenario特定的目录
    init_dir = OFFLINE_INIT_ROOT / scenario / f"run_seed_{seed}"
    scenario_file = init_dir / "scenario.yaml"
    
    # Track planning latency for each stage
    planning_times = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0}
    
    # For clustering methods, check if cluster file exists
    if method not in ["monolithic"]:
        # Run Stage 1 to generate scenario and cluster file
        t1 = time.time()
        if not run_stage1(scenario, seed, method):
            return ExperimentResult(
                scenario=scenario,
                method=method,
                seed=seed,
                success=False,
                duration_seconds=time.time() - start_time,
                error_message="Stage 1 failed",
            )
        planning_times["stage1"] = time.time() - t1
    else:
        # For monolithic, just need scenario.yaml
        # Generate scenario using kmeans method (just for scenario generation)
        t1 = time.time()
        if not run_stage1(scenario, seed, "kmeans"):
            return ExperimentResult(
                scenario=scenario,
                method=method,
                seed=seed,
                success=False,
                duration_seconds=time.time() - start_time,
                error_message="Stage 1 (scenario generation) failed",
            )
        planning_times["stage1"] = time.time() - t1
    
    # Run Stage 2 for clustering methods
    if method not in ["monolithic"]:
        t2 = time.time()
        if not run_stage2(scenario, seed, method):
            return ExperimentResult(
                scenario=scenario,
                method=method,
                seed=seed,
                success=False,
                duration_seconds=time.time() - start_time,
                error_message="Stage 2 failed",
            )
        planning_times["stage2"] = time.time() - t2
    
    # Run Stage 3 (PDDL + geometric planning) -- this is the true planning latency we care about
    t3 = time.time()
    if not run_stage3(scenario, seed, method):
        stage3_time = time.time() - t3
        return ExperimentResult(
            scenario=scenario,
            method=method,
            seed=seed,
            success=False,
            duration_seconds=time.time() - start_time,
            error_message="Stage 3 failed",
            planning_time_seconds=stage3_time,
        )
    planning_times["stage3"] = time.time() - t3
    
    # Define planning latency strictly as Stage 3 solver time
    planning_latency = planning_times["stage3"]
    logger.info(
        f"[Planning] Stage-3 planning latency: {planning_latency:.2f}s "
        f"(S1:{planning_times['stage1']:.2f}s, S2:{planning_times['stage2']:.2f}s, S3:{planning_times['stage3']:.2f}s)"
    )
    
    # Run simulation
    result_file = run_simulation(scenario, seed, method)
    
    duration = time.time() - start_time
    
    if result_file is None:
        return ExperimentResult(
            scenario=scenario,
            method=method,
            seed=seed,
            success=False,
            duration_seconds=duration,
            error_message="Simulation failed",
        )
    
    logger.info(f"Experiment completed in {duration:.1f}s: {result_file}")
    
    return ExperimentResult(
        scenario=scenario,
        method=method,
        seed=seed,
        success=True,
        duration_seconds=duration,
        result_file=result_file,
        planning_time_seconds=planning_latency,
    )


def run_aggregate_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run aggregate_results.py and return the aggregated DataFrames."""
    logger.info("=" * 70)
    logger.info("Running results aggregation...")
    logger.info("=" * 70)
    
    cmd = [sys.executable, "-m", "plot.aggregate_results"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            env=get_subprocess_env(),
        )
        
        if result.returncode != 0:
            logger.error(f"Aggregation failed: {result.stderr}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Load the generated CSV files
        agg_results_path = PLOT_DATA_ROOT / "aggregated_results.csv"
        agg_stats_path = PLOT_DATA_ROOT / "aggregated_stats.csv"
        
        agg_df = pd.read_csv(agg_results_path) if agg_results_path.exists() else pd.DataFrame()
        stats_df = pd.read_csv(agg_stats_path) if agg_stats_path.exists() else pd.DataFrame()
        
        logger.info("Aggregation completed successfully")
        return agg_df, stats_df
        
    except Exception as e:
        logger.error(f"Aggregation error: {e}")
        return pd.DataFrame(), pd.DataFrame()


def run_coupling_analysis() -> pd.DataFrame:
    """Run analyze_coupling_vs_completion.py and return the correlation DataFrame."""
    logger.info("=" * 70)
    logger.info("Running coupling-completion correlation analysis...")
    logger.info("=" * 70)
    
    cmd = [sys.executable, "-m", "plot.analyze_coupling_vs_completion"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            env=get_subprocess_env(),
        )
        
        if result.returncode != 0:
            logger.error(f"Coupling analysis failed: {result.stderr}")
            return pd.DataFrame()
        
        # Load the generated correlation CSV
        corr_path = PLOT_DATA_ROOT / "coupling_completion_corr.csv"
        corr_df = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()
        
        logger.info("Coupling analysis completed successfully")
        return corr_df
        
    except Exception as e:
        logger.error(f"Coupling analysis error: {e}")
        return pd.DataFrame()


def compute_cluster_stats() -> pd.DataFrame:
    """Compute per-(scenario, method) cluster count statistics from Stage 1 outputs."""
    rows = []

    for scenario in SCENARIOS:
        for method in METHODS:
            # Only clustering-based methods have checkpoint cluster files
            if method in ["monolithic"]:
                continue

            counts = []
            for seed in SEEDS:
                run_dir = OFFLINE_INIT_ROOT / scenario / f"run_seed_{seed}"
                cluster_file = run_dir / f"checkpoint_clusters_{method}.csv"
                if not cluster_file.exists():
                    continue

                try:
                    df = pd.read_csv(cluster_file)
                except Exception:
                    continue

                if "cluster_group" in df.columns:
                    n_clusters = int(df["cluster_group"].nunique())
                elif "cluster_id" in df.columns:
                    n_clusters = int(df["cluster_id"].nunique())
                else:
                    continue

                counts.append(n_clusters)

            if counts:
                s = pd.Series(counts, dtype=float)
                rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "n_clusters_mean": float(s.mean()),
                        "n_clusters_std": float(s.std()),
                        "n_runs": len(counts),
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def print_analysis_summary(
    stats_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    cluster_stats_df: Optional[pd.DataFrame] = None,
) -> None:
    """Print formatted tables of aggregated statistics and correlations."""
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    if not stats_df.empty:
        # Check required columns
        required_cols = ["scenario", "method", "completion_rate_mean", "completion_rate_std", 
                        "avg_completion_time_mean", "avg_completion_time_std"]
        
        if all(col in stats_df.columns for col in required_cols):
            # Group by scenario and display separately
            scenarios = stats_df["scenario"].unique()
            
            for scenario in scenarios:
                scenario_data = stats_df[stats_df["scenario"] == scenario].copy()
                
                print(f"\n【{get_scenario_display_name(scenario)}】")
                print("-" * 80)
                
                # Format data for display
                display_rows = []
                scenario_cluster = None
                if cluster_stats_df is not None and not cluster_stats_df.empty:
                    scenario_cluster = cluster_stats_df[cluster_stats_df["scenario"] == scenario]

                for _, row in scenario_data.iterrows():
                    method_name = get_method_display_name(row['method'])
                    comp_rate = row['completion_rate_mean'] * 100
                    comp_std = row['completion_rate_std'] * 100
                    latency = row['avg_completion_time_mean']
                    latency_std = row['avg_completion_time_std']

                    avg_clusters_str = ""
                    if scenario_cluster is not None and not scenario_cluster.empty:
                        cs = scenario_cluster[scenario_cluster["method"] == row["method"]]
                        if not cs.empty:
                            mean_val = cs["n_clusters_mean"].iloc[0]
                            std_val = cs["n_clusters_std"].iloc[0]
                            avg_clusters_str = f"{mean_val:.1f}±{std_val:.1f}"

                    display_rows.append({
                        'Method': method_name,
                        'Completion Rate': f"{comp_rate:.1f}±{comp_std:.1f}%",
                        'Avg Latency (s)': f"{latency:.1f}±{latency_std:.1f}",
                        'Avg #Clusters': avg_clusters_str,
                    })
                
                display_df = pd.DataFrame(display_rows)
                print(display_df.to_string(index=False))
        else:
            # Fallback: show what we have
            print(f"\nAvailable columns: {stats_df.columns.tolist()}")
            print(stats_df.to_string(index=False))
    
    # Skip correlation analysis output
    print("\n" + "=" * 80)
    print("(Coupling analysis skipped - no coupling data available)")
    print("=" * 80)
    
    print("\n" + "=" * 80)

def run_all_experiments(
    scenarios: List[str],
    methods: List[str],
    seeds: List[int],
    max_workers: int = 4,
) -> List[ExperimentResult]:
    """Run all experiment combinations in parallel.
    
    To avoid concurrent writes to shared scenario files, experiments are organized as:
    - Sequential by (scenario, seed): ensures only one process generates scenario files
    - Parallel by method within each (scenario, seed): safe because methods only differ in clustering
    
    Args:
        scenarios: List of scenarios to run
        methods: List of methods to run
        seeds: List of seeds to run
        max_workers: Maximum number of parallel workers (default: 4)
    """
    
    total = len(scenarios) * len(methods) * len(seeds)
    logger.info(f"Running {total} experiments with {max_workers} parallel workers: "
                f"{len(scenarios)} scenarios x {len(methods)} methods x {len(seeds)} seeds")
    logger.info(f"Note: Running sequentially by (scenario, seed) to avoid file conflicts")
    
    # Build experiment tasks grouped by (scenario, seed)
    # This prevents concurrent writes to shared scenario files
    task_groups = []
    for scenario in scenarios:
        for seed in seeds:
            group = [(scenario, method, seed) for method in methods]
            task_groups.append(group)
    
    results = []
    completed = 0
    lock = Lock()  # For thread-safe logging
    
    # Run experiments: sequential by (scenario, seed), parallel by method
    for group in task_groups:
        # Each group is a list of (scenario, method, seed) for the same (scenario, seed)
        # Run methods in this group in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(run_experiment, scenario, method, seed): (scenario, method, seed)
                for scenario, method, seed in group
            }
            
            # Process completed tasks in this group
            for future in as_completed(future_to_task):
                scenario, method, seed = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    with lock:
                        completed += 1
                        status = "✓" if result.success else "✗"
                        logger.info(f"[{completed}/{total}] {status} "
                                   f"{scenario}/{method}/seed{seed}")
                except Exception as e:
                    with lock:
                        completed += 1
                        logger.error(f"[{completed}/{total}] ✗ "
                                    f"{scenario}/{method}/seed{seed} - Exception: {e}")
                    results.append(ExperimentResult(
                        scenario=scenario,
                        method=method,
                        seed=seed,
                        success=False,
                        duration_seconds=0.0,
                        error_message=str(e),
                    ))
    
    # Summary
    success_count = sum(1 for r in results if r.success)
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT SUMMARY: {success_count}/{total} succeeded")
    logger.info("=" * 70)
    
    # Save experiment log
    log_file = PLOT_DATA_ROOT / "experiment_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": total,
        "successful": success_count,
        "failed": total - success_count,
        "results": [
            {
                "scenario": r.scenario,
                "method": r.method,
                "seed": r.seed,
                "success": r.success,
                "duration_seconds": r.duration_seconds,
                "planning_time_seconds": r.planning_time_seconds,
                "result_file": str(r.result_file) if r.result_file else None,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"Experiment log saved to: {log_file}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch experiments for baseline comparison"
    )
    
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Scenarios to run (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+", 
        default=["all"],
        help="Methods to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["all"],
        help="Seeds to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without running",
    )
    parser.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip clearing results directory before running",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--topo-aware",
        action="store_true",
        help="Enable topology-aware speed estimation for MoveToTask",
    )
    return parser.parse_args()


def main():
    global TOPO_AWARE_MODE
    args = parse_args()
    
    # Set topo-aware mode
    TOPO_AWARE_MODE = args.topo_aware
    if TOPO_AWARE_MODE:
        logger.info("Topo-aware mode ENABLED: speed limits will be estimated from topology")
    
    # Parse scenarios
    if "all" in args.scenarios:
        scenarios = SCENARIOS
    else:
        scenarios = [s for s in args.scenarios if s in SCENARIOS]
    
    # Parse methods
    if "all" in args.methods:
        methods = METHODS
    else:
        methods = [m for m in args.methods if m in METHODS]
    
    # Parse seeds
    if "all" in args.seeds:
        seeds = SEEDS
    else:
        seeds = [int(s) for s in args.seeds]
    
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Seeds: {seeds}")
    
    if args.dry_run:
        total = len(scenarios) * len(methods) * len(seeds)
        logger.info(f"[DRY RUN] Would run {total} experiments")
        for scenario in scenarios:
            for method in methods:
                for seed in seeds:
                    logger.info(f"  - {scenario} / {method} / seed={seed}")
        return
    
    # Clear results directory before running (unless skipped)
    if not args.skip_clear:
        clear_results_directory()
    else:
        logger.info("Skipping results directory clearing (--skip-clear)")
    
    # Run all experiments
    run_all_experiments(scenarios, methods, seeds, max_workers=args.workers)
    
    # Run post-experiment analysis
    logger.info("\n" + "=" * 70)
    logger.info("Running post-experiment analysis...")
    logger.info("=" * 70)
    
    # Aggregate results
    agg_df, stats_df = run_aggregate_analysis()
    
    # Analyze coupling vs completion (disabled - too slow)
    # corr_df = run_coupling_analysis()
    corr_df = pd.DataFrame()

    # Compute cluster statistics from Stage 1 outputs
    cluster_stats_df = compute_cluster_stats()
    
    # Print formatted summary (including average cluster counts)
    print_analysis_summary(stats_df, corr_df, cluster_stats_df)
    

if __name__ == "__main__":
    main()
