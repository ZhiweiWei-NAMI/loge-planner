"""Offline pipeline package for TCCN Auto Planner.

This package implements a complete offline planning pipeline:

Stage 1: Clustering (stage1_precluster_from_config.py)
  - kmeans: Pure KMeans on 26-dim features
  - triplet: Siamese embedding + KMeans
  - htt_recursive: rule-based hierarchical decomposition

Stage 2: Cluster Assignment (stage2_plan_from_clusters.py)
  - Assigns clusters to agents based on proximity and cooldown

Stage 3: PDDL Planning (multiple options)
  - stage3_plan_with_pddl.py: Clustering-based PDDL planning
  - stage3_plan_monolithic_pddl.py: No clustering, greedy + PDDL

Runtime: All methods use OfflineTaskLoader for simulation.
"""

from .runtime_loader import OfflineTaskLoader

__all__ = ["OfflineTaskLoader"]
