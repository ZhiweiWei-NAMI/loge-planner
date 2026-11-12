"""Scenario loading / summarisation for LLM prompts.

The decomposition prompt receives a compact, deterministic summary of the
scenario plus the full checkpoint table.  Nothing is sampled or truncated
silently: if the checkpoint table is larger than the caller's budget the CLI
fails instead of sending a partial instance to the model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_scenario_document(path: Path) -> Dict[str, Any]:
    """Load ``scenario.yaml`` (Stage 1 output) or a JSON scenario description."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        doc = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        doc = json.loads(text)
    else:
        raise ValueError(f"Unsupported scenario format: {path.suffix} (use .yaml/.yml/.json)")
    if not isinstance(doc, dict):
        raise ValueError(f"Scenario document must be a mapping, got {type(doc).__name__}")
    return doc


def _component_names(agent: Dict[str, Any]) -> List[str]:
    props = agent.get("properties") or {}
    comps = props.get("components") or []
    return [str(c) for c in comps]


def summarize_scenario(
    doc: Dict[str, Any],
    workflow_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the scenario block embedded in the user prompt.

    ``workflow_rows`` (rows of ``workflow_init_info.csv``) takes precedence for
    workflow counts because it is the table Stage 2/3 actually consume.
    """
    agents = doc.get("agents") or []
    agent_types = Counter(
        str(a.get("uav_type") or a.get("type") or "unknown") for a in agents if isinstance(a, dict)
    )
    component_hist: Counter = Counter()
    for agent in agents:
        if isinstance(agent, dict):
            component_hist.update(_component_names(agent))

    if workflow_rows:
        workflow_types = Counter(str(r.get("workflow_type", "unknown")) for r in workflow_rows)
        workflow_ids = [str(r.get("workflow_id")) for r in workflow_rows]
        start_times = []
        for row in workflow_rows:
            try:
                start_times.append(float(row.get("start_time", 0.0)))
            except (TypeError, ValueError):
                pass
        workflow_count = len(workflow_ids)
    else:
        workflows = doc.get("workflows") or []
        workflow_types = Counter(
            str(w.get("type", "unknown")) for w in workflows if isinstance(w, dict)
        )
        workflow_count = len(workflows)
        start_times = []

    simulation = doc.get("simulation") or doc.get("environment") or {}
    horizon = simulation.get("duration") if isinstance(simulation, dict) else None

    summary: Dict[str, Any] = {
        "name": str(doc.get("scenario_name") or doc.get("name") or "unnamed_scenario"),
        "description": str(doc.get("scenario_description") or doc.get("description") or ""),
        "horizon_s": horizon,
        "n_agents": len(agents),
        "agent_types": dict(sorted(agent_types.items())),
        "agent_components": dict(sorted(component_hist.items())),
        "n_workflows": workflow_count,
        "workflow_types": dict(sorted(workflow_types.items())),
    }
    if start_times:
        summary["workflow_start_time_range_s"] = [min(start_times), max(start_times)]
    return summary
