"""Plot module for experiment analysis and visualization.

This module provides:
- config.py: Global configuration for methods, scenarios, and plotting
- run_experiments.py: Batch runner for all experiment combinations
- aggregate_results.py: Result aggregation and statistics

Usage:
    # Run experiments
    python -m plot.run_experiments --scenarios all --methods all --seeds all
    
    # Aggregate results
    python -m plot.aggregate_results
"""

from .config import (
    METHODS,
    METHODS_FOR_PLOT,
    SCENARIOS,
    SEEDS,
    WORKFLOW_TYPES,
    METHOD_NAMES,
    SCENARIO_NAMES,
    get_method_display_name,
    get_scenario_display_name,
)

__all__ = [
    "METHODS",
    "METHODS_FOR_PLOT",
    "SCENARIOS", 
    "SEEDS",
    "WORKFLOW_TYPES",
    "METHOD_NAMES",
    "SCENARIO_NAMES",
    "get_method_display_name",
    "get_scenario_display_name",
]
