from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.pddl.solve_problem import validate_pddl_files
from src.validation.parser_validator import (
    ValidationReport,
    validate_dependency_dag,
    validate_resource_deadline_sanity,
    validate_schema,
)


@dataclass
class GateResult:
    accepted: bool
    schema: Dict[str, Any]
    dag: Dict[str, Any]
    resource_deadline: Dict[str, Any]
    pddl: Dict[str, Any]


def validate_parser_payload(
    payload: Dict[str, Any],
    domain_file: Optional[Path] = None,
    problem_file: Optional[Path] = None,
) -> GateResult:
    schema_report = validate_schema(payload)
    dag_report = validate_dependency_dag(payload)
    resource_report = validate_resource_deadline_sanity(payload)

    pddl_result = {"valid": True, "errors": []}
    if domain_file is not None and problem_file is not None:
        valid, _, _ = validate_pddl_files(domain_file, problem_file)
        pddl_result["valid"] = bool(valid)
        if not valid:
            pddl_result["errors"].append("pddl validation failed")

    accepted = all([
        schema_report.valid,
        dag_report.valid,
        resource_report.valid,
        pddl_result["valid"],
    ])
    return GateResult(
        accepted=accepted,
        schema=schema_report.to_dict(),
        dag=dag_report.to_dict(),
        resource_deadline=resource_report.to_dict(),
        pddl=pddl_result,
    )


def corrupt_payload(payload: Dict[str, Any], corruption: str) -> Dict[str, Any]:
    import copy

    corrupted = copy.deepcopy(payload)
    workflows = corrupted.get("workflows", [])
    if not workflows:
        return corrupted

    first = workflows[0]
    if corruption == "drop_dependency":
        for workflow in workflows:
            if workflow.get("depends_on"):
                workflow["depends_on"] = workflow["depends_on"][1:]
                break
    elif corruption == "spurious_edge":
        if len(workflows) >= 2:
            workflows[1].setdefault("depends_on", []).append(str(workflows[0]["id"]))
            workflows[0].setdefault("depends_on", []).append(str(workflows[1]["id"]))
    elif corruption == "swap_task_type":
        first["type"] = "invalid_type"
    elif corruption == "deadline_corruption":
        first.setdefault("properties", {})["max_end_to_end_latency_s"] = -1
    elif corruption == "missing_task":
        corrupted["workflows"] = workflows[1:]
    elif corruption == "duplicate_task":
        corrupted["workflows"].append(copy.deepcopy(first))
    return corrupted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plan validation gate")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=False)
    parser.add_argument("--problem", type=Path, required=False)
    parser.add_argument("--corruption", type=str, required=False)
    args = parser.parse_args()

    with args.payload.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if args.corruption:
        payload = corrupt_payload(payload, args.corruption)
    result = validate_parser_payload(payload, domain_file=args.domain, problem_file=args.problem)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
