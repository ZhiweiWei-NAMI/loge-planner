"""Baseline exports with lazy loading to avoid module run warnings."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HungarianGreedyAssigner",
    "run_graph_mapf_priority",
    "run_monolithic_global_pddl",
]


def __getattr__(name: str) -> Any:
    if name == "HungarianGreedyAssigner":
        return import_module(".hungarian_greedy", __name__).HungarianGreedyAssigner
    if name == "run_graph_mapf_priority":
        return import_module(".graph_mapf_priority", __name__).run_baseline
    if name == "run_monolithic_global_pddl":
        return import_module(".monolithic_global_pddl", __name__).run_baseline
    raise AttributeError(name)
