"""JSON protocol for LLM macro-task decomposition (protocol ``1.0``).

The decomposition output is the only artefact Stage 2/3 consume.  It must be
validated before it is allowed to influence planning, therefore this module
implements an explicit, strict protocol instead of trusting free-form text.

Wire format (exactly one JSON object)::

    {
      "schema_version": "1.0",
      "scenario": "<scenario name, optional echo>",
      "macro_tasks": [
        {
          "cluster_group": 0,
          "operation": "logistics",
          "causal_anchor": "workflow_1:pickup",
          "checkpoints": [
            {"workflow_id": "workflow_1", "checkpoint_idx": 0}
          ],
          "rationale": "optional short justification"
        }
      ]
    }

Hard validation (always fatal):

1. ``macro_tasks`` is a non-empty list of objects.
2. ``cluster_group`` is an integer >= 0 and unique across macro tasks.
3. Every referenced ``(workflow_id, checkpoint_idx)`` exists in the input table.
4. Every input checkpoint is covered exactly once (no omission, no duplicate).

Soft validation (fatal only in strict mode, see ``--allow-causal-violation``):

5. Causal order per workflow: for checkpoints of one workflow sorted by
   ``checkpoint_idx``, ``cluster_group`` must be non-decreasing.
6. Macro-task budget ``[min_macro_tasks, max_macro_tasks]`` when supplied.
7. ``operation`` should be one of the input ``operation_type`` values or
   ``"mixed"``.
8. ``workflow_ids`` (if supplied) must match the derived workflow set.

On success :func:`apply_decomposition_to_csv` writes
``checkpoint_clusters_<method>.csv`` -- the original checkpoint table plus a
renumbered ``cluster_group`` column -- which is exactly what
``stage2_plan_from_clusters`` expects (``--cluster-method <method>``).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SUPPORTED_PROTOCOL_VERSIONS = ("1.0",)

# Operation labels that are always acceptable in ``macro_task.operation``.
EXTRA_ALLOWED_OPERATIONS = {"mixed", "composite", "other"}


class DecompositionError(RuntimeError):
    """Base class for decomposition failures."""


class DecompositionParseError(DecompositionError):
    """The model output was not a single valid JSON object."""


class DecompositionValidationError(DecompositionError):
    """The JSON object parsed but violates the decomposition protocol."""

    def __init__(self, message: str, errors: Sequence[str], warnings: Sequence[str] = ()):
        self.errors = list(errors)
        self.warnings = list(warnings)
        detail = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"{message}\n{detail}")


@dataclass(frozen=True)
class CheckpointRef:
    workflow_id: str
    checkpoint_idx: int

    def as_dict(self) -> Dict[str, Any]:
        return {"workflow_id": self.workflow_id, "checkpoint_idx": self.checkpoint_idx}

    def key(self) -> Tuple[str, int]:
        return (self.workflow_id, self.checkpoint_idx)


@dataclass
class MacroTask:
    cluster_group: int
    operation: str
    checkpoints: List[CheckpointRef]
    causal_anchor: Optional[str] = None
    rationale: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_group": self.cluster_group,
            "operation": self.operation,
            "causal_anchor": self.causal_anchor,
            "checkpoints": [c.as_dict() for c in self.checkpoints],
            "rationale": self.rationale,
        }


@dataclass
class Decomposition:
    macro_tasks: List[MacroTask]
    schema_version: str = "1.0"
    scenario: Optional[str] = None

    @property
    def n_clusters(self) -> int:
        return len(self.macro_tasks)

    def cluster_of(self) -> Dict[Tuple[str, int], int]:
        mapping: Dict[Tuple[str, int], int] = {}
        for task in self.macro_tasks:
            for ref in task.checkpoints:
                mapping[ref.key()] = task.cluster_group
        return mapping

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "macro_tasks": [t.as_dict() for t in self.macro_tasks],
        }


@dataclass
class ValidationReport:
    n_checkpoints: int
    n_macro_tasks: int
    n_workflows: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    causal_violations: List[Dict[str, Any]] = field(default_factory=list)
    operation_histogram: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "n_checkpoints": self.n_checkpoints,
            "n_macro_tasks": self.n_macro_tasks,
            "n_workflows": self.n_workflows,
            "errors": self.errors,
            "warnings": self.warnings,
            "causal_violations": self.causal_violations,
            "operation_histogram": self.operation_histogram,
        }


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer index")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return int(float(str(value).strip()))


def load_checkpoint_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Load a checkpoint table (``checkpoint_init.csv`` schema) as plain dicts."""
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Checkpoint table not found: {csv_path}")
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"workflow_id", "checkpoint_idx"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        for raw in reader:
            row = dict(raw)
            row["workflow_id"] = str(raw["workflow_id"])
            row["checkpoint_idx"] = _as_int(raw["checkpoint_idx"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Checkpoint table is empty: {csv_path}")
    return rows


def parse_locations(rows: Iterable[Dict[str, Any]]) -> None:
    """Best-effort sanity check of the ``location`` column (no mutation)."""
    for row in rows:
        loc = row.get("location")
        if loc in (None, ""):
            continue
        if isinstance(loc, str):
            try:
                parsed = ast.literal_eval(loc)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Unparsable location {loc!r} in checkpoint row: {exc}") from exc
            if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
                raise ValueError(f"location must be a 2- or 3-element sequence, got {loc!r}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first top-level JSON object from model output.

    Accepts a bare object or a fenced ```json block.  Anything else is an
    error: we never silently repair a malformed response.
    """
    if text is None:
        raise DecompositionParseError("model returned no text")
    stripped = text.strip()
    if not stripped:
        raise DecompositionParseError("model returned an empty response")

    candidates: List[str] = [stripped]
    if "```" in stripped:
        for chunk in stripped.split("```"):
            body = chunk.strip()
            if body.lower().startswith("json"):
                body = body[4:].strip()
            if body.startswith("{") and body.endswith("}"):
                candidates.append(body)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            raise DecompositionParseError(
                f"expected a JSON object, got {type(payload).__name__}"
            )
        return payload
    raise DecompositionParseError(
        "response is not valid JSON (first 400 chars): " + stripped[:400]
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _cluster_index(value: Any, label: str, errors: List[str]) -> Optional[int]:
    if isinstance(value, bool):
        errors.append(f"{label}: cluster_group must be an integer, got boolean")
        return None
    if isinstance(value, int):
        idx = value
    elif isinstance(value, float) and float(value).is_integer():
        idx = int(value)
    else:
        errors.append(f"{label}: cluster_group must be an integer, got {value!r}")
        return None
    if idx < 0:
        errors.append(f"{label}: cluster_group must be >= 0, got {idx}")
        return None
    return idx


def validate_decomposition(
    payload: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    strict_causal: bool = True,
    min_macro_tasks: Optional[int] = None,
    max_macro_tasks: Optional[int] = None,
) -> Tuple[Decomposition, ValidationReport]:
    """Validate a parsed payload against the checkpoint table.

    Raises :class:`DecompositionValidationError` when a fatal rule is violated.
    """
    errors: List[str] = []
    warnings: List[str] = []
    causal_violations: List[Dict[str, Any]] = []

    if not isinstance(payload, dict):
        raise DecompositionValidationError(
            "payload is not a JSON object", ["payload must be a JSON object"]
        )

    schema_version = payload.get("schema_version", "1.0")
    if schema_version not in SUPPORTED_PROTOCOL_VERSIONS:
        errors.append(
            f"unsupported schema_version {schema_version!r}; "
            f"supported: {list(SUPPORTED_PROTOCOL_VERSIONS)}"
        )

    macro_tasks_raw = payload.get("macro_tasks")
    if not isinstance(macro_tasks_raw, list) or not macro_tasks_raw:
        raise DecompositionValidationError(
            "invalid decomposition",
            errors + ["'macro_tasks' must be a non-empty list"],
            warnings,
        )

    row_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["workflow_id"]), int(row["checkpoint_idx"]))
        if key in row_by_key:
            raise ValueError(f"duplicate checkpoint key in input table: {key}")
        row_by_key[key] = row

    seen: Dict[Tuple[str, int], int] = {}
    macro_tasks: List[MacroTask] = []
    derived_workflows: List[set] = []  # parallel to macro_tasks_raw
    used_clusters: set = set()
    operation_histogram: Dict[str, int] = {}

    for ti, task_raw in enumerate(macro_tasks_raw):
        label = f"macro_tasks[{ti}]"
        derived_workflows.append(set())
        if not isinstance(task_raw, dict):
            errors.append(f"{label}: must be an object")
            continue

        cluster = _cluster_index(task_raw.get("cluster_group"), label, errors)
        if cluster is not None:
            if cluster in used_clusters:
                errors.append(f"{label}: duplicate cluster_group {cluster}")
            used_clusters.add(cluster)

        operation = task_raw.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            warnings.append(f"{label}: 'operation' missing; defaulting to 'mixed'")
            operation = "mixed"
        operation_histogram[operation] = operation_histogram.get(operation, 0) + 1

        ckpt_raw = task_raw.get("checkpoints")
        if not isinstance(ckpt_raw, list) or not ckpt_raw:
            errors.append(f"{label}: 'checkpoints' must be a non-empty list")
            continue

        refs: List[CheckpointRef] = []
        for ci, ref_raw in enumerate(ckpt_raw):
            if not isinstance(ref_raw, dict):
                errors.append(f"{label}.checkpoints[{ci}]: must be an object")
                continue
            wf = ref_raw.get("workflow_id")
            idx_raw = ref_raw.get("checkpoint_idx")
            if not isinstance(wf, str) or not wf:
                errors.append(f"{label}.checkpoints[{ci}]: missing 'workflow_id'")
                continue
            try:
                idx = _as_int(idx_raw)
            except (TypeError, ValueError):
                errors.append(
                    f"{label}.checkpoints[{ci}]: 'checkpoint_idx' must be an integer, "
                    f"got {idx_raw!r}"
                )
                continue
            ref = CheckpointRef(wf, idx)
            if ref.key() not in row_by_key:
                errors.append(
                    f"{label}.checkpoints[{ci}]: unknown checkpoint {ref.key()} "
                    f"(not present in the input table)"
                )
                continue
            if ref.key() in seen:
                errors.append(
                    f"{label}.checkpoints[{ci}]: checkpoint {ref.key()} already "
                    f"assigned to cluster_group {seen[ref.key()]}"
                )
                continue
            seen[ref.key()] = cluster if cluster is not None else -1
            refs.append(ref)

        derived_workflows[ti] = {ref.workflow_id for ref in refs}

        causal_anchor = task_raw.get("causal_anchor")
        if causal_anchor is not None and not isinstance(causal_anchor, str):
            warnings.append(f"{label}: 'causal_anchor' is not a string; ignored")
            causal_anchor = None

        rationale = task_raw.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            rationale = None

        if refs and cluster is not None:
            macro_tasks.append(
                MacroTask(
                    cluster_group=cluster,
                    operation=operation,
                    checkpoints=refs,
                    causal_anchor=causal_anchor,
                    rationale=rationale,
                )
            )

    missing = [k for k in row_by_key if k not in seen]
    if missing:
        preview = ", ".join(f"{wf}:{idx}" for wf, idx in missing[:8])
        errors.append(
            f"{len(missing)} checkpoint(s) not assigned to any macro task: {preview}"
            + (" ..." if len(missing) > 8 else "")
        )

    # Causal order per workflow.
    by_workflow: Dict[str, List[Tuple[int, int]]] = {}
    for key, cluster in seen.items():
        by_workflow.setdefault(key[0], []).append((key[1], cluster))
    for wf_id, seq in by_workflow.items():
        seq.sort(key=lambda item: item[0])
        for (prev_idx, prev_cluster), (next_idx, next_cluster) in zip(seq, seq[1:]):
            if next_cluster < prev_cluster:
                causal_violations.append(
                    {
                        "workflow_id": wf_id,
                        "checkpoint_idx": next_idx,
                        "cluster_group": next_cluster,
                        "predecessor_checkpoint_idx": prev_idx,
                        "predecessor_cluster_group": prev_cluster,
                    }
                )
    if causal_violations:
        message = (
            f"{len(causal_violations)} causal-order violation(s): a checkpoint was "
            f"placed in an earlier macro task than its predecessor "
            f"(example: {causal_violations[0]})"
        )
        (errors if strict_causal else warnings).append(message)

    n_tasks = len(macro_tasks)
    if min_macro_tasks is not None and n_tasks < min_macro_tasks:
        errors.append(
            f"decomposition produced {n_tasks} macro task(s), fewer than the "
            f"requested minimum {min_macro_tasks}"
        )
    if max_macro_tasks is not None and n_tasks > max_macro_tasks:
        errors.append(
            f"decomposition produced {n_tasks} macro task(s), more than the "
            f"requested maximum {max_macro_tasks}"
        )

    input_operations = {
        str(row.get("operation_type", "")).strip()
        for row in rows
        if str(row.get("operation_type", "")).strip()
    }
    allowed = input_operations | EXTRA_ALLOWED_OPERATIONS
    for task in macro_tasks:
        if task.operation not in allowed:
            warnings.append(
                f"macro task {task.cluster_group}: operation {task.operation!r} is not "
                f"an input operation_type; accepted as free-form label"
            )

    # ``workflow_ids`` is informational; if the model supplied it, check it.
    for ti, task_raw in enumerate(macro_tasks_raw):
        if not isinstance(task_raw, dict):
            continue
        declared = task_raw.get("workflow_ids")
        if declared is None:
            continue
        if not isinstance(declared, list) or not all(isinstance(x, str) for x in declared):
            warnings.append(f"macro_tasks[{ti}]: 'workflow_ids' must be a list of strings")
            continue
        derived = derived_workflows[ti] if ti < len(derived_workflows) else set()
        if set(declared) != derived:
            warnings.append(
                f"macro_tasks[{ti}]: 'workflow_ids' {sorted(set(declared))} != derived "
                f"{sorted(derived)}"
            )

    report = ValidationReport(
        n_checkpoints=len(row_by_key),
        n_macro_tasks=n_tasks,
        n_workflows=len(by_workflow),
        errors=errors,
        warnings=warnings,
        causal_violations=causal_violations,
        operation_histogram=operation_histogram,
    )
    if errors:
        raise DecompositionValidationError("decomposition failed validation", errors, warnings)

    decomposition = Decomposition(
        macro_tasks=macro_tasks,
        schema_version=str(schema_version),
        scenario=payload.get("scenario") if isinstance(payload.get("scenario"), str) else None,
    )
    return decomposition, report


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def renumber_clusters(
    decomposition: Decomposition, rows: Sequence[Dict[str, Any]]
) -> Decomposition:
    """Renumber cluster ids 0..N-1 by first appearance in the checkpoint table.

    Stage 2/3 group by ``cluster_group`` and sort by earliest start time, so the
    numbering itself carries no semantics; a deterministic renumbering makes
    runs reproducible even if the model emits sparse ids.
    """
    order: Dict[int, int] = {}
    cluster_of = decomposition.cluster_of()
    for row in rows:
        key = (str(row["workflow_id"]), int(row["checkpoint_idx"]))
        cluster = cluster_of.get(key)
        if cluster is not None and cluster not in order:
            order[cluster] = len(order)
    tasks = [
        MacroTask(
            cluster_group=order.get(task.cluster_group, task.cluster_group),
            operation=task.operation,
            checkpoints=task.checkpoints,
            causal_anchor=task.causal_anchor,
            rationale=task.rationale,
        )
        for task in decomposition.macro_tasks
    ]
    tasks.sort(key=lambda t: t.cluster_group)
    return Decomposition(
        macro_tasks=tasks,
        schema_version=decomposition.schema_version,
        scenario=decomposition.scenario,
    )


def apply_decomposition_to_csv(
    checkpoint_csv: Path,
    decomposition: Decomposition,
    output_csv: Path,
) -> int:
    """Write ``checkpoint_clusters_<method>.csv`` for Stage 2/3.

    Returns the number of clusters written.  Raises if any checkpoint row has no
    cluster assignment (never emits a partially-clustered table).
    """
    import pandas as pd

    checkpoint_csv = Path(checkpoint_csv)
    df = pd.read_csv(checkpoint_csv)
    if "workflow_id" not in df.columns or "checkpoint_idx" not in df.columns:
        raise ValueError(
            f"{checkpoint_csv} must contain 'workflow_id' and 'checkpoint_idx'; "
            f"found {list(df.columns)}"
        )

    mapping = decomposition.cluster_of()
    cluster_col: List[int] = []
    missing: List[Tuple[str, int]] = []
    for wf, idx in zip(df["workflow_id"].astype(str), df["checkpoint_idx"]):
        key = (wf, _as_int(idx))
        if key not in mapping:
            missing.append(key)
            cluster_col.append(-1)
        else:
            cluster_col.append(int(mapping[key]))
    if missing:
        raise DecompositionValidationError(
            "cannot emit clustered checkpoint table",
            [f"{len(missing)} row(s) without cluster_group: {missing[:8]}"],
        )

    out = df.copy()
    out["cluster_group"] = cluster_col
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return int(out["cluster_group"].nunique())


def decomposition_metadata(
    decomposition: Decomposition,
    report: ValidationReport,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the auditable metadata blob stored next to the emitted CSV."""
    payload: Dict[str, Any] = {
        "schema_version": decomposition.schema_version,
        "scenario": decomposition.scenario,
        "n_macro_tasks": decomposition.n_clusters,
        "validation": report.as_dict(),
        "macro_tasks": [
            {
                "cluster_group": task.cluster_group,
                "operation": task.operation,
                "causal_anchor": task.causal_anchor,
                "n_checkpoints": len(task.checkpoints),
                "workflow_ids": sorted({c.workflow_id for c in task.checkpoints}),
                "rationale": task.rationale,
            }
            for task in decomposition.macro_tasks
        ],
    }
    if extra:
        payload["run"] = extra
    return payload
