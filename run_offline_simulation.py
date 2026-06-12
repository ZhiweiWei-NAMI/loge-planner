#!/usr/bin/env python3
"""
Offline Simulation Entry Point
==============================

This script runs the AirFogSim simulation using pre-computed offline plans.
All methods are executed completely offline; no model API is called during simulation.

Supported Methods:
------------------
1. CLUSTERING-BASED (分类+PDDL):
   - kmeans: Pure KMeans on 26-dim features
   - htt_recursive: LoGe hierarchical decomposition planner
   - triplet: Siamese embedding + KMeans
   
2. NON-CLUSTERING BASELINES (不分类):
   - monolithic: Greedy assignment + PDDL (no clustering)

Usage:
------
# Run main method
python run_offline_simulation.py --method htt_recursive

# Run baseline comparisons
python run_offline_simulation.py --method kmeans
python run_offline_simulation.py --method monolithic

# Run all methods for comparison
python run_offline_simulation.py --method all

Prerequisites:
--------------
For clustering methods (kmeans, htt_recursive, triplet):
  python -m src.offline_pipeline.stage1_precluster_from_config --seed 42 --method <method>
  python -m src.offline_pipeline.stage2_plan_from_clusters --seed 42 --method <method>
  python -m src.offline_pipeline.stage3_plan_with_pddl --seed 42 --method <method>

For monolithic baseline:
  python -m src.offline_pipeline.stage3_plan_monolithic_pddl --seed 42

@author: TCCN Auto Planner
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Setup project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent  # airfogsim-project
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# All public methods.
# hungarian: greedy sequential assignment.
CLUSTERING_METHODS = [
    'htt_recursive',
    'rule_kmeans',
    'kmeans',
    'rl',
    'triplet',
    'hungarian',
    'graph_mapf_priority',
    'monolithic_global_pddl',
]
# HtT (Hypotheses-to-Theories) rule-based post-processing variants
HTT_METHODS = ['kmeans_htt', 'rl_htt', 'triplet_htt']
NON_CLUSTERING_METHODS = ['monolithic']
ALL_METHODS = CLUSTERING_METHODS + HTT_METHODS + NON_CLUSTERING_METHODS


def find_offline_plan_path(seed: int, method: str, plan_root: Path = None, scenario: str = None) -> Path:
    """Construct the default offline plan path based on seed and method."""
    if plan_root is None:
        if scenario:
            plan_root = SCRIPT_DIR / "data" / "offline_plans" / scenario
        else:
            plan_root = SCRIPT_DIR / "data" / "offline_plans"
    plan_dir = plan_root / f"run_seed_{seed}"
    plan_path = plan_dir / f"pddl_plan_{method}.json"
    return plan_path


def run_single_method(method: str, seed: int, config_dir: Path, plan_root: Path = None, scenario: str = None) -> dict:
    """Run simulation for a single method and return summary."""
    from src.simulation.examples.comprehensive_example import run_single_experiment
    
    offline_plan_path = find_offline_plan_path(seed, method, plan_root, scenario)
    
    if not offline_plan_path.exists():
        logger.error(f"Plan not found for method '{method}': {offline_plan_path}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running method: {method}")
    logger.info(f"Plan file: {offline_plan_path}")
    logger.info(f"{'='*60}")
    
    summary = run_single_experiment(
        method_name=f"Offline-{method}",
        enable_online_model=False,
        planning_mode='offline',
        offline_plan_path=str(offline_plan_path),
        config_dir=str(config_dir)
    )
    
    # Inject scenario name into summary if available
    if summary and scenario:
        summary['scenario'] = scenario
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run AirFogSim simulation with offline plans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--method', type=str, default='rule_kmeans',
        choices=ALL_METHODS + ['all'],
        help='Planning method to run (default: rule_kmeans). Use "all" to run all methods.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed used to locate the offline plan directory (default: 42)'
    )
    parser.add_argument(
        '--scenario', type=str, default=None,
        help='Scenario name (e.g., large_mixed, small_logistics_inspection). Used to locate scenario-specific directories.'
    )
    parser.add_argument(
        '--init-root', type=str, default=None,
        help='Root directory containing scenario.yaml (overrides default data/offline_init)'
    )
    parser.add_argument(
        '--plan-root', type=str, default=None,
        help='Root directory containing PDDL plans (overrides default data/offline_plans)'
    )
    parser.add_argument(
        '--offline-plan', type=str, default=None,
        help='Explicit path to offline plan JSON (overrides --method)'
    )
    parser.add_argument(
        '--replan-plan', type=str, default=None,
        help='Optional path to replan PDDL plan JSON for second-phase replay (not yet fully automated)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import simulation components
    try:
        from src.simulation.examples.comprehensive_example import run_single_experiment
    except ImportError as e:
        logger.error(f"Failed to import simulation modules: {e}")
        logger.error("Make sure you are running from the LoGe planner project directory")
        sys.exit(1)

    # Config directory
    if args.init_root:
        init_root = Path(args.init_root)
    elif args.scenario:
        init_root = SCRIPT_DIR / "data" / "offline_init" / args.scenario
    else:
        init_root = SCRIPT_DIR / "data" / "offline_init"
    
    config_dir = init_root / f"run_seed_{args.seed}"
    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        logger.error("Please run Stage 1 first to generate scenario.yaml")
        sys.exit(1)
    
    # Plan root directory
    if args.plan_root:
        plan_root = Path(args.plan_root)
    else:
        plan_root = None

    logger.info("="*80)
    logger.info("Offline Simulation")
    logger.info("="*80)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Config Directory: {config_dir}")
    
    # Handle explicit plan path
    if args.offline_plan:
        offline_plan_path = Path(args.offline_plan)
        if not offline_plan_path.exists():
            logger.error(f"Plan file not found: {offline_plan_path}")
            sys.exit(1)
        
        method_name = offline_plan_path.stem.replace("pddl_plan_", "")
        logger.info(f"Using explicit plan: {offline_plan_path}")
        
        try:
            summary = run_single_experiment(
                method_name=f"Offline-{method_name}",
                enable_online_model=False,
                planning_mode='offline',
                offline_plan_path=str(offline_plan_path),
                config_dir=str(config_dir)
            )
            logger.info(f"\nCompleted: {summary.get('total_agents', 'N/A')} agents, "
                       f"{summary.get('total_workflows', 'N/A')} workflows")
            return 0
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Determine which methods to run
    if args.method == 'all':
        methods_to_run = ALL_METHODS
    else:
        methods_to_run = [args.method]

    logger.info(f"Methods to run: {methods_to_run}")
    logger.info("="*80)

    # Run each method
    results = {}
    for method in methods_to_run:
        try:
            summary = run_single_method(method, args.seed, config_dir, plan_root, args.scenario)
            if summary:
                results[method] = summary
                logger.info(f"✓ {method}: {summary.get('total_agents', 'N/A')} agents, "
                           f"{summary.get('total_workflows', 'N/A')} workflows")
            else:
                logger.warning(f"✗ {method}: Plan file not found")
        except Exception as e:
            logger.error(f"✗ {method}: Failed - {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SIMULATION SUMMARY")
    logger.info("="*80)
    
    for method, summary in results.items():
        logger.info(f"  {method:15s}: {summary.get('total_agents', 'N/A'):3d} agents, "
                   f"{summary.get('total_workflows', 'N/A'):3d} workflows, "
                   f"{summary.get('simulation_time', 'N/A'):6.0f}s")
    
    if not results:
        logger.error("No methods completed successfully!")
        return 1
    
    logger.info("="*80)
    logger.info(f"Results saved to: data/results/")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
