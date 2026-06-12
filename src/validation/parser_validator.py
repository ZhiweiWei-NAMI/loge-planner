from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set


ALLOWED_TASK_TYPES = {"communication", "computation", "inspection", "logistics"}


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_schema(payload: Dict[str, Any]) -> ValidationReport:
    errors: List[str] = []
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        errors.append("payload.workflows must be a list")
        return ValidationReport(valid=False, errors=errors)

    workflow_ids: Set[str] = set()
    for index, workflow in enumerate(workflows):
        if not isinstance(workflow, dict):
            errors.append(f"workflow[{index}] must be an object")
            continue
        workflow_id = str(workflow.get("id", "")).strip()
        if not workflow_id:
            errors.append(f"workflow[{index}] missing id")
        elif workflow_id in workflow_ids:
            errors.append(f"workflow id duplicated: {workflow_id}")
        else:
            workflow_ids.add(workflow_id)
        workflow_type = str(workflow.get("type", "")).lower()
        if workflow_type not in ALLOWED_TASK_TYPES:
            errors.append(f"workflow[{index}] invalid type: {workflow_type}")
        if "properties" not in workflow:
            errors.append(f"workflow[{index}] missing properties")
    return ValidationReport(valid=not errors, errors=errors)


def validate_dependency_dag(payload: Dict[str, Any]) -> ValidationReport:
    workflows = payload.get("workflows", [])
    workflow_ids = {str(workflow.get("id")) for workflow in workflows if workflow.get("id") is not None}
    adjacency: Dict[str, List[str]] = {workflow_id: [] for workflow_id in workflow_ids}
    errors: List[str] = []

    for workflow in workflows:
        workflow_id = str(workflow.get("id"))
        dependencies = workflow.get("depends_on", []) or []
        if not isinstance(dependencies, list):
            errors.append(f"{workflow_id}.depends_on must be a list")
            continue
        for predecessor in dependencies:
            predecessor = str(predecessor)
            if predecessor not in workflow_ids:
                errors.append(f"{workflow_id} depends on unknown workflow {predecessor}")
            else:
                adjacency.setdefault(predecessor, []).append(workflow_id)

    seen: Set[str] = set()
    active: Set[str] = set()

    def dfs(node: str) -> None:
        if node in active:
            errors.append(f"dependency cycle detected at {node}")
            return
        if node in seen:
            return
        seen.add(node)
        active.add(node)
        for succ in adjacency.get(node, []):
            dfs(succ)
        active.remove(node)

    for workflow_id in workflow_ids:
        dfs(workflow_id)
    return ValidationReport(valid=not errors, errors=errors)


def validate_resource_deadline_sanity(payload: Dict[str, Any]) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    for workflow in payload.get("workflows", []):
        workflow_id = str(workflow.get("id"))
        props = workflow.get("properties", {}) or {}
        deadline = props.get("max_end_to_end_latency_s", props.get("deadline_s", None))
        if deadline is not None:
            try:
                if float(deadline) <= 0:
                    errors.append(f"{workflow_id} deadline must be positive")
            except Exception:
                errors.append(f"{workflow_id} deadline is not numeric")
        if "data_size_mb" in props:
            try:
                if float(props["data_size_mb"]) <= 0:
                    warnings.append(f"{workflow_id} has non-positive data_size_mb")
            except Exception:
                errors.append(f"{workflow_id} data_size_mb is not numeric")
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings)
