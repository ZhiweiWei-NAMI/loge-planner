"""Global configuration for experiments and plotting.

This file contains all mappings for method names, scenario names,
and plotting styles used across the experiment pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = DATA_ROOT / "results"
OFFLINE_INIT_ROOT = DATA_ROOT / "offline_init"
OFFLINE_PLANS_ROOT = DATA_ROOT / "offline_plans"
FIGURES_ROOT = PROJECT_ROOT / "plot" / "figures"
REPRODUCIBILITY_ROOT = PROJECT_ROOT / "reproducibility"
PLOT_DATA_ROOT = Path(os.environ.get("SCIS_PLOT_DATA_ROOT", REPRODUCIBILITY_ROOT / "plot_data"))


# =============================================================================
# Method Configuration
# =============================================================================

# Method ID -> Display name for paper figures (English)
METHOD_NAMES: Dict[str, str] = {
    # Our method (HiNS Planner)
    "htt_recursive": "LoGe-Planner (Ours)",
    "rule_kmeans": "Rule-KMeans",
    # Baselines
    "hungarian": "Hungarian-Greedy",
    "kmeans": "KMeans",
    "triplet": "DML-KMeans",
    # Additional experimental methods
    "rl": "RL-KMeans",
    "monolithic": "Greedy-NN",
    "kmeans_htt": "KMeans-HtT",
    "rl_htt": "RL-KMeans-HtT",
    "triplet_htt": "Triplet-KMeans-HtT",
}

# Method ID -> Short name for legends
METHOD_SHORT_NAMES: Dict[str, str] = {
    "htt_recursive": "HiNS",
    "rule_kmeans": "Rule-KM",
    "hungarian": "Hungarian",
    "kmeans": "KMeans",
    "triplet": "DML-KM",
    "rl": "RL",
    "monolithic": "Greedy",
    "kmeans_htt": "KM-HtT",
    "rl_htt": "RL-HtT",
    "triplet_htt": "Tri-HtT",
}

# Method ID -> Description
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "htt_recursive": "HiNS hierarchical neighborhood search planner (recursive tree)",
    "rule_kmeans": "Rule-based weighted KMeans clustering",
    "hungarian": "Hungarian algorithm based greedy assignment",
    "kmeans": "Pure KMeans on 26-dim features",
    "triplet": "Deep metric learning (DML) embeddings + KMeans",
    "rl": "RL-based weighted KMeans on 26-dim features",
    "monolithic": "Greedy nearest-neighbor assignment (no clustering)",
    "kmeans_htt": "KMeans + HtT rule-based post-processing",
    "rl_htt": "RL-KMeans + HtT rule-based post-processing",
    "triplet_htt": "Triplet-KMeans + HtT rule-based post-processing",
}

# List of methods to run (in order for paper comparison)
# Order: Our method first, then baselines by complexity
METHODS: List[str] = [
    "htt_recursive",  # Our method (HiNS-Planner)
    "rule_kmeans",    # Rule-based weighted KMeans (excluded from plots/tables)
    "kmeans",         # Baseline 1: Pure KMeans
    "triplet",        # Baseline 2: DML-KMeans
]

# Methods to include in paper figures (excludes rule_kmeans which is internal)
METHODS_FOR_PLOT: List[str] = [
    "htt_recursive",
    "kmeans",
    "triplet",
]


# =============================================================================
# Scenario Configuration
# =============================================================================

# Scenario ID -> Display name for paper figures (English)
SCENARIO_NAMES: Dict[str, str] = {
    "small_logistics_inspection": "S1: Small Mixed",
    "small_comm_compute": "S2: Small Comm-Comp",
    "medium_logistics_heavy": "S3: Medium Logistics",
    "medium_inspection_heavy": "S4: Medium Inspection",
    "large_mixed": "S5: Large Mixed",
}

# Scenario ID -> Short name for legends
SCENARIO_SHORT_NAMES: Dict[str, str] = {
    "small_logistics_inspection": "S1",
    "small_comm_compute": "S2",
    "medium_logistics_heavy": "S3",
    "medium_inspection_heavy": "S4",
    "large_mixed": "S5",
}

# Scenario ID -> Difficulty level (for ordering)
SCENARIO_DIFFICULTY: Dict[str, int] = {
    "small_logistics_inspection": 1,
    "small_comm_compute": 1,
    "medium_logistics_heavy": 2,
    "medium_inspection_heavy": 2,
    "large_mixed": 3,
}

# Scenario ID -> Expected scale (agents, workflows)
SCENARIO_SCALE: Dict[str, Dict[str, int]] = {
    "small_logistics_inspection": {"agents": 30, "workflows": 30},
    "small_comm_compute": {"agents": 30, "workflows": 30},
    "medium_logistics_heavy": {"agents": 80, "workflows": 80},
    "medium_inspection_heavy": {"agents": 80, "workflows": 80},
    "large_mixed": {"agents": 150, "workflows": 150},
}

# List of scenarios to run (in order of difficulty)
SCENARIOS: List[str] = [
    "small_logistics_inspection",
    "small_comm_compute",
    "medium_logistics_heavy",
    "medium_inspection_heavy",
    "large_mixed",
]


# =============================================================================
# Experiment Configuration
# =============================================================================

# Random seeds for reproducibility (9 seeds as requested)
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50]

# Workflow types to track
WORKFLOW_TYPES: List[str] = [
    "Inspection",
    "Communication", 
    "Computation",
    "Logistics",
]


# =============================================================================
# Plotting Configuration (Paper-ready style)
# =============================================================================

# Color palette for methods (colorblind-friendly)
METHOD_COLORS: Dict[str, str] = {
    "htt_recursive": "#2ecc71", # Green (our method)
    "rule_kmeans": "#16a085",   # Teal
    "hungarian": "#e74c3c",     # Red
    "kmeans": "#3498db",        # Blue
    "triplet": "#1abc9c",       # Turquoise
    "rl": "#f39c12",            # Orange
    "monolithic": "#c0392b",    # Dark Red
    "kmeans_htt": "#5dade2",    # Light Blue
    "rl_htt": "#eb984e",        # Light Orange
    "triplet_htt": "#48c9b0",   # Light Turquoise
}

# Marker styles for methods
METHOD_MARKERS: Dict[str, str] = {
    "htt_recursive": "o", # Circle (our method)
    "rule_kmeans": "P",   # Plus filled
    "hungarian": "^",     # Triangle up
    "kmeans": "s",        # Square
    "triplet": "<",       # Triangle left
    "rl": "v",            # Triangle down
    "monolithic": "^",    # Triangle up
    "kmeans_htt": "p",    # Pentagon
    "rl_htt": "h",        # Hexagon
    "triplet_htt": "*",   # Star
}

# Bar hatching patterns for B&W printing (use fine, single-character patterns)
METHOD_HATCHES: Dict[str, str] = {
    "htt_recursive": "",   # Solid (our method - no hatch)
    "rule_kmeans": "/",   # Diagonal
    "hungarian": "\\",   # Back diagonal
    "kmeans": "x",        # Cross
    "triplet": ".",       # Dots
    "rl": "-",            # Horizontal
    "monolithic": "*",    # Star-like
}

# Line styles for methods
METHOD_LINESTYLES: Dict[str, str] = {
    "htt_recursive": "-",  # Solid (our method)
    "rule_kmeans": "-",    # Solid
    "hungarian": "--",     # Dashed
    "kmeans": ":",         # Dotted
    "triplet": "-.",       # Dash-dot
    "rl": "--",            # Dashed
    "monolithic": "-",     # Solid
    "kmeans_htt": "-",     # Solid
    "rl_htt": "-",         # Solid
    "triplet_htt": "-",    # Solid
}

# Colors for workflow types
WORKFLOW_COLORS: Dict[str, str] = {
    "Inspection": "#e74c3c",      # Red
    "Communication": "#3498db",   # Blue
    "Computation": "#9b59b6",     # Purple
    "Logistics": "#f39c12",       # Orange
}

# Figure size presets (width, height in inches)
FIGURE_SIZES: Dict[str, tuple] = {
    "single_column": (4.0, 2.9),   # IEEE single column
    "double_column": (7.8, 3.6),   # IEEE double column
    "square": (3.5, 3.5),          # Square
    "wide": (7.8, 3.0),            # Wide format
}

# Font sizes for paper
FONT_SIZES: Dict[str, int] = {
    "title": 15,
    "label": 14,
    "tick": 13,
    "legend": 12,
    "annotation": 12,
}

# DPI for figures
FIGURE_DPI: int = 300

# Default matplotlib style settings
PLOT_STYLE: Dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_method_display_name(method_id: str) -> str:
    """Get display name for a method."""
    return METHOD_NAMES.get(method_id, method_id)


def get_scenario_display_name(scenario_id: str) -> str:
    """Get display name for a scenario."""
    return SCENARIO_NAMES.get(scenario_id, scenario_id)


def get_method_color(method_id: str) -> str:
    """Get color for a method."""
    return METHOD_COLORS.get(method_id, "#808080")


def get_workflow_color(workflow_type: str) -> str:
    """Get color for a workflow type."""
    return WORKFLOW_COLORS.get(workflow_type, "#808080")


def get_scenarios_sorted_by_difficulty() -> List[str]:
    """Get scenarios sorted by difficulty level."""
    return sorted(SCENARIOS, key=lambda s: SCENARIO_DIFFICULTY.get(s, 0))
