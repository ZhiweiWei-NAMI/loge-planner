"""Prompt construction for LLM macro-task decomposition.

The prompts are intentionally explicit: the model is told the exact JSON
protocol, the hard causality constraints, and the fields it may use.  The user
message always carries the *complete* checkpoint table -- the decomposition is
never computed from a sample of the instance.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Optional

SYSTEM_PROMPT = """\
You are the macro-task decomposition module of LoGe Planner v1.1.0, a multi-UAV
workflow scheduling system for low-altitude logistics, inspection,
communication, and computation missions.

INPUT
You receive (a) a scenario summary and (b) a flat list of workflow checkpoints.
A checkpoint is one atomic operation of one workflow: its state name, operation
type, 3-D location, required simulator components, successor states, and its
workload attributes (payload, bandwidth, latency budget, compute demand,
inspection points).

TASK
Partition all checkpoints into macro tasks ("clusters") such that a downstream
cluster-to-agent assignment and PDDL grounding can operate on macro-task
granularity instead of individual checkpoints.

HARD CONSTRAINTS (a response violating any of them is rejected)
1. Every checkpoint belongs to exactly one macro task. No omission, no
   duplicate. Reference a checkpoint only by the exact pair
   (workflow_id, checkpoint_idx) copied from the input list.
2. Preserve causal order inside each workflow: if checkpoint i must precede
   checkpoint j of the same workflow (checkpoint_idx i < j), then the macro task
   of i must have a cluster_group less than or equal to that of j.
3. cluster_group values must be integers starting at 0, unique across macro
   tasks.
4. Never invent workflow ids, checkpoint indices, locations, or operation
   types. Only reuse values that appear in the input.

DECOMPOSITION GUIDELINES
- Keep a macro task spatially coherent: group checkpoints that are close in the
  3-D space so that one agent can serve them without long detours.
- Keep a macro task operationally coherent: group checkpoints whose required
  components are compatible (e.g. logistics pickup+transport+delivery,
  inspection move+inspect, communication move+communicate,
  computation move+compute).
- Balance the load: avoid a single macro task that swallows most of the
  instance, and avoid macro tasks of a single trivial checkpoint when a nearby
  compatible checkpoint exists.
- Respect the requested macro-task budget given in the user message.
- Checkpoints of different workflows may share a macro task when they are
  spatially close and component-compatible.

OUTPUT
Return exactly one JSON object and nothing else: no prose, no markdown fence,
no trailing commentary. Schema:

{
  "schema_version": "1.0",
  "scenario": "<scenario name copied from the input>",
  "macro_tasks": [
    {
      "cluster_group": 0,
      "operation": "<one of the input operation_type values, or 'mixed'>",
      "causal_anchor": "<optional 'workflow_id:state_name' anchor for this macro task>",
      "checkpoints": [
        {"workflow_id": "<workflow id>", "checkpoint_idx": <integer>}
      ],
      "rationale": "<one short sentence, <= 30 words>"
    }
  ]
}

Every input checkpoint must appear exactly once across all macro tasks.
"""


def _format_location(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(f"{float(v):.2f}" for v in value) + ")"
    return str(value)


def _format_sequence(value: Any) -> str:
    if value in (None, "", "nan"):
        return "-"
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value) if value else "-"
    return str(value)


def _format_number(value: Any) -> str:
    if value in (None, "", "nan"):
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def render_checkpoint_line(index: int, row: Dict[str, Any]) -> str:
    """Render one checkpoint as a single compact, unambiguous line."""
    parts = [
        f"[{index}]",
        f"wf={row.get('workflow_id')}",
        f"ckpt={row.get('checkpoint_idx')}",
        f"state={row.get('state_name', '-')}",
        f"op={row.get('operation_type', '-')}",
        f"loc={_format_location(row.get('location', '-'))}",
        f"req={_format_sequence(row.get('requirements'))}",
        f"next={_format_sequence(row.get('next_states'))}",
        f"payload_kg={_format_number(row.get('payload_weight'))}",
        f"comm_mb={_format_number(row.get('comm_bandwidth_mb'))}",
        f"lat_s={_format_number(row.get('max_latency_s'))}",
        f"gflops={_format_number(row.get('compute_gflops'))}",
        f"inspect_pts={_format_number(row.get('n_inspection_points'))}",
    ]
    return " ".join(parts)


def build_user_prompt(
    scenario_summary: Dict[str, Any],
    checkpoint_rows: Iterable[Dict[str, Any]],
    *,
    min_macro_tasks: Optional[int] = None,
    max_macro_tasks: Optional[int] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    """Build the user message: scenario block + the full checkpoint table."""
    rows = list(checkpoint_rows)
    budget = []
    if min_macro_tasks is not None:
        budget.append(f"min_macro_tasks={min_macro_tasks}")
    if max_macro_tasks is not None:
        budget.append(f"max_macro_tasks={max_macro_tasks}")
    budget_text = ", ".join(budget) if budget else "unspecified (use your judgement)"

    lines: List[str] = [
        "SCENARIO",
        f"name: {scenario_summary.get('name')}",
        f"description: {scenario_summary.get('description') or '-'}",
        f"horizon_s: {scenario_summary.get('horizon_s')}",
        f"agents: total={scenario_summary.get('n_agents')} "
        f"types={scenario_summary.get('agent_types')} "
        f"components={scenario_summary.get('agent_components')}",
        f"workflows: total={scenario_summary.get('n_workflows')} "
        f"types={scenario_summary.get('workflow_types')}",
    ]
    if scenario_summary.get("workflow_start_time_range_s"):
        lines.append(
            f"workflow_start_time_range_s: {scenario_summary['workflow_start_time_range_s']}"
        )
    lines.extend(
        [
            "",
            f"MACRO-TASK BUDGET: {budget_text}",
            "",
            f"CHECKPOINTS ({len(rows)} rows, ordered by workflow then checkpoint index)",
            "format: [i] wf=<workflow_id> ckpt=<int> state=<name> op=<operation_type> "
            "loc=(x,y,z) req=<a|b> next=<states> payload_kg=<f> comm_mb=<f> lat_s=<f> "
            "gflops=<f> inspect_pts=<f>",
        ]
    )
    for i, row in enumerate(rows):
        lines.append(render_checkpoint_line(i, row))
    if extra_instructions:
        lines.extend(["", "ADDITIONAL INSTRUCTIONS", str(extra_instructions).strip()])
    lines.extend(["", "Return only the JSON object described in the system message."])
    return "\n".join(lines)


def count_checkpoint_rows(prompt: str) -> int:
    """Count rendered checkpoint lines in a prompt (used for the dry-run check)."""
    return sum(1 for line in prompt.splitlines() if line.startswith("[") and " wf=" in line)
