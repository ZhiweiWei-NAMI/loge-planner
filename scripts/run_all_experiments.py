#!/usr/bin/env python3
"""Run all experiments for paper comparison.

This script runs the complete experiment pipeline:
1. Generate scenario data for all scenarios and seeds
2. Run Stage 1-3 for each method
3. Replay offline simulation for each method
4. Collect and aggregate results

Methods compared:
- htt_recursive: Our HtT recursive tree method
- hungarian: Hungarian greedy assignment baseline
- kmeans: Pure KMeans baseline
- triplet: Triplet learning + KMeans baseline

Usage:
    python scripts/run_all_experiments.py --scenarios all --methods all --seeds 42,43,44
    python scripts/run_all_experiments.py --scenarios small_logistics_inspection --methods kmeans htt_recursive
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "workflow_decouple" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SCENARIOS = [
    "small_logistics_inspection",
    "small_comm_compute",
    "medium_logistics_heavy",
    "medium_inspection_heavy",
    "large_mixed",
]

METHODS = [
    "htt_recursive",
    "rule_kmeans",
    "rl",
    "hungarian",
    "kmeans",
    "triplet",
    "monolithic_global_pddl",
    "graph_mapf_priority",
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# Paths
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = DATA_ROOT / "results"
OFFLINE_PLANS_ROOT = DATA_ROOT / "offline_plans"
MULTI_SEED_ROOT = PROJECT_ROOT / "workflow_decouple" / "data" / "multi_seed_runs"


def get_env() -> dict:
    """Get environment with PYTHONPATH set."""
    env = os.environ.copy()
    paths = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "workflow_decouple" / "src"),
        str(PROJECT_ROOT.parent / "src"),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_command(cmd: List[str], cwd: Path = None, timeout: int = 3600) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            env=get_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, f"Error: {result.stderr}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout expired"
    except Exception as e:
        return False, str(e)


def ensure_scenario_data(scenario: str, seed: int) -> bool:
    """Ensure scenario data exists for the given scenario and seed."""
    run_dir = MULTI_SEED_ROOT / scenario / f"run_seed_{seed}"
    scenario_yaml = run_dir / "scenario.yaml"
    
    if scenario_yaml.exists():
        logger.info(f"Scenario data exists: {scenario}/seed_{seed}")
        return True
    
    logger.info(f"Generating scenario data: {scenario}/seed_{seed}")
    cmd = [
        sys.executable, "-m", "src.run_multi_seed_pipeline",
        "--scenario", scenario,
        "--seeds", str(seed),
        "--skip-clustering",
    ]
    success, output = run_command(cmd, cwd=PROJECT_ROOT / "workflow_decouple")
    if not success:
        logger.error(f"Failed to generate scenario data: {output}")
    return success


def run_stage1_clustering(scenario: str, seed: int, method: str) -> bool:
    """Run Stage 1 clustering for a specific method."""
    run_dir_root = MULTI_SEED_ROOT / scenario
    run_dir = run_dir_root / f"run_seed_{seed}"
    output_csv = run_dir / f"checkpoint_clusters_{method}.csv"

    if output_csv.exists():
        logger.info(f"Stage 1 output exists: {output_csv.name}")
        return True

    logger.info(f"Running Stage 1 ({method}): {scenario}/seed_{seed}")

    if method in {"monolithic_global_pddl", "graph_mapf_priority"}:
        logger.info("Method %s does not require Stage 1 clustering", method)
        return True

    if method not in {"kmeans", "triplet", "rule_kmeans", "rl", "htt_recursive", "hungarian"}:
        logger.error(f"Unknown method: {method}")
        return False

    cmd = [
        sys.executable, "-m", "src.offline_pipeline.stage1_precluster_from_config",
        "--seed", str(seed),
        "--scenario-name", scenario,
        "--method", method,
        "--output-root", str(run_dir_root),
        "--skip-simulation",
    ]

    success, output = run_command(cmd, cwd=PROJECT_ROOT)
    if not success:
        logger.error(f"Stage 1 failed: {output}")
    return success


def run_stage2_assignment(scenario: str, seed: int, method: str) -> bool:
    """Run Stage 2 cluster-to-agent assignment."""
    if method in {"monolithic_global_pddl", "graph_mapf_priority"}:
        logger.info("Method %s does not require Stage 2 assignment", method)
        return True

    run_dir = MULTI_SEED_ROOT / scenario / f"run_seed_{seed}"
    cluster_csv = run_dir / f"checkpoint_clusters_{method}.csv"
    assignment_csv = OFFLINE_PLANS_ROOT / f"{scenario}_{method}" / f"run_seed_{seed}" / f"cluster_assignment_{method}.csv"

    if assignment_csv.exists():
        logger.info(f"Stage 2 output exists: {assignment_csv.name}")
        return True

    if not cluster_csv.exists():
        logger.warning(f"Cluster CSV not found: {cluster_csv}")
        return False

    logger.info(f"Running Stage 2 ({method}): {scenario}/seed_{seed}")
    cmd = [
        sys.executable, "-m", "src.offline_pipeline.stage2_plan_from_clusters",
        "--seed", str(seed),
        "--cluster-method", method,
        "--input-root", str(MULTI_SEED_ROOT / scenario),
        "--output-root", str(OFFLINE_PLANS_ROOT / f"{scenario}_{method}"),
    ]

    success, output = run_command(cmd, cwd=PROJECT_ROOT)
    if not success:
        logger.error(f"Stage 2 failed: {output}")
    return success


def run_stage3_pddl(scenario: str, seed: int, method: str) -> bool:
    """Run Stage 3 PDDL planning."""
    # Output directory
    plan_dir = OFFLINE_PLANS_ROOT / f"{scenario}_{method}" / f"run_seed_{seed}"
    plan_file = plan_dir / f"pddl_plan_{method}.json"
    
    if plan_file.exists():
        logger.info(f"Stage 3 output exists: {plan_file.name}")
        return True
    
    logger.info(f"Running Stage 3 ({method}): {scenario}/seed_{seed}")

    plan_dir.mkdir(parents=True, exist_ok=True)

    if method in {"monolithic_global_pddl", "graph_mapf_priority"}:
        cmd = [
            sys.executable, "-m", f"src.baselines.{method}",
            "--seed", str(seed),
            "--scenario", scenario,
            "--input-root", str(MULTI_SEED_ROOT / scenario),
            "--output-root", str(OFFLINE_PLANS_ROOT / f"{scenario}_{method}"),
        ]
    else:
        cmd = [
            sys.executable, "-m", "src.offline_pipeline.stage3_plan_with_pddl",
            "--seed", str(seed),
            "--cluster-method", method,
            "--input-root", str(MULTI_SEED_ROOT / scenario),
            "--output-root", str(OFFLINE_PLANS_ROOT / f"{scenario}_{method}"),
            "--export-edge-routes",
        ]

    success, output = run_command(cmd, cwd=PROJECT_ROOT, timeout=1800)
    if not success:
        logger.error(f"Stage 3 failed: {output}")
    return success


def find_existing_simulation_result(scenario: str, seed: int, method: str) -> Optional[Path]:
    """Find a workflow summary JSON already matching the scenario/seed/method."""
    for candidate in RESULTS_ROOT.glob(f"Offline-{method}_*_workflow_summary.json"):
        try:
            with candidate.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        if payload.get("scenario") == scenario and int(payload.get("seed", -1)) == seed:
            return candidate
    return None


def run_offline_simulation_replay(scenario: str, seed: int, method: str) -> bool:
    """Replay simulation with an offline plan."""
    plan_root = OFFLINE_PLANS_ROOT / f"{scenario}_{method}"
    plan_dir = plan_root / f"run_seed_{seed}"
    plan_file = plan_dir / f"pddl_plan_{method}.json"
    
    if not plan_file.exists():
        logger.warning(f"Plan file not found: {plan_file}")
        return False
    
    existing = find_existing_simulation_result(scenario, seed, method)
    if existing:
        logger.info(f"Simulation result exists: {existing.name}")
        return True
    
    logger.info(f"Running simulation ({method}): {scenario}/seed_{seed}")
    
    init_root = MULTI_SEED_ROOT / scenario
    
    cmd = [
        sys.executable, "run_offline_simulation.py",
        "--method", method,
        "--seed", str(seed),
        "--scenario", scenario,
        "--init-root", str(init_root),
        "--plan-root", str(plan_root),
    ]
    
    success, output = run_command(cmd, cwd=PROJECT_ROOT, timeout=3600)
    if not success:
        logger.error(f"Simulation failed: {output}")
    return success


def run_single_experiment(scenario: str, seed: int, method: str) -> Dict[str, Any]:
    """Run complete experiment for one scenario/seed/method combination."""
    result = {
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "timestamp": datetime.now().isoformat(),
        "stages": {},
    }
    
    # Stage 0: Ensure scenario data
    if not ensure_scenario_data(scenario, seed):
        result["stages"]["data_generation"] = "failed"
        return result
    result["stages"]["data_generation"] = "success"
    
    # Stage 1: Clustering
    if not run_stage1_clustering(scenario, seed, method):
        result["stages"]["stage1_clustering"] = "failed"
        return result
    result["stages"]["stage1_clustering"] = "success"
    
    # Stage 2: Assignment
    if not run_stage2_assignment(scenario, seed, method):
        result["stages"]["stage2_assignment"] = "failed"
        return result
    result["stages"]["stage2_assignment"] = "success"
    
    # Stage 3: PDDL Planning
    if not run_stage3_pddl(scenario, seed, method):
        result["stages"]["stage3_pddl"] = "failed"
        return result
    result["stages"]["stage3_pddl"] = "success"
    
    # Stage 4: Offline simulation replay
    if not run_offline_simulation_replay(scenario, seed, method):
        result["stages"]["stage4_simulation"] = "failed"
        return result
    result["stages"]["stage4_simulation"] = "success"
    
    result["status"] = "complete"
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                       help="Scenarios to run (or 'all')")
    parser.add_argument("--methods", nargs="+", default=["all"],
                       help="Methods to run (or 'all')")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46",
                       help="Comma-separated seeds")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Parse arguments
    scenarios = SCENARIOS if "all" in args.scenarios else args.scenarios
    methods = METHODS if "all" in args.methods else args.methods
    seeds = [int(s) for s in args.seeds.split(",")]
    
    logger.info(f"Running experiments:")
    logger.info(f"  Scenarios: {scenarios}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Seeds: {seeds}")
    
    # Ensure results directory exists
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    experiments = [
        (scenario, seed, method)
        for scenario in scenarios
        for seed in seeds
        for method in methods
    ]
    
    logger.info(f"Total experiments: {len(experiments)}")
    
    results = []
    start_time = time.time()
    
    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single_experiment, s, sd, m): (s, sd, m)
                for s, sd, m in experiments
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(f"Completed: {result['scenario']}/{result['method']}/seed{result['seed']}")
    else:
        for scenario, seed, method in experiments:
            result = run_single_experiment(scenario, seed, method)
            results.append(result)
            logger.info(f"Completed: {result['scenario']}/{result['method']}/seed{result['seed']}")
    
    elapsed = time.time() - start_time
    logger.info(f"All experiments completed in {elapsed:.1f}s")
    
    # Save results
    output_file = args.output or str(RESULTS_ROOT / "experiment_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    success_count = sum(1 for r in results if r.get("status") == "complete")
    logger.info(f"Success: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
