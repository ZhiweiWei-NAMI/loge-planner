"""CLI: OpenAI-compatible macro-task decomposition for LoGe Planner.

Produces ``checkpoint_clusters_<method>.csv`` -- the exact input Stage 2
(``stage2_plan_from_clusters``) and Stage 3 (``stage3_plan_with_pddl``) expect
via ``--cluster-method <method>``.

Examples
--------
Dry run (no API call, prints the exact prompt that would be sent)::

    python -m src.agent_llm.decompose_cli \\
        --run-dir data/offline_init/large_mixed/run_seed_42 \\
        --model gpt-4o-mini --dry-run

Cloud call and in-place cluster file for Stage 2/3::

    export OPENAI_API_KEY=...
    python -m src.agent_llm.decompose_cli \\
        --run-dir data/offline_init/large_mixed/run_seed_42 \\
        --model gpt-4o-mini \\
        --base-url https://api.openai.com/v1 \\
        --min-macro-tasks 8 --max-macro-tasks 60 \\
        --cluster-method llm --inplace

Re-validate / re-emit a saved decomposition without calling the API::

    python -m src.agent_llm.decompose_cli \\
        --run-dir data/offline_init/large_mixed/run_seed_42 \\
        --from-json data/offline_init/large_mixed/run_seed_42/llm_decomposition/llm_decomposition_gpt-4o-mini.json \\
        --cluster-method llm --inplace

The CLI validates the decomposition and exits non-zero without writing a
cluster CSV when it encounters API failures, non-JSON responses, missing
checkpoints, duplicate assignments, or causal-order violations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent_llm import PLANNER_VERSION
from src.agent_llm.openai_client import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    LLMCallConfig,
    LLMRequestError,
    OpenAICompatibleClient,
)
from src.agent_llm.prompts import SYSTEM_PROMPT, build_user_prompt, count_checkpoint_rows
from src.agent_llm.protocol import (
    DecompositionError,
    DecompositionValidationError,
    apply_decomposition_to_csv,
    decomposition_metadata,
    extract_json_object,
    load_checkpoint_rows,
    renumber_clusters,
    validate_decomposition,
)
from src.agent_llm.scenario_io import load_scenario_document, summarize_scenario


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or "model"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_workflow_rows(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if path is None or not Path(path).exists():
        return None
    import csv

    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Decompose workflow checkpoints into macro tasks with an "
            "OpenAI-compatible chat model and emit checkpoint_clusters_<method>.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Stage 1 run directory containing checkpoint_init.csv / workflow_init_info.csv / scenario.yaml.",
    )
    p.add_argument("--checkpoint-csv", type=Path, default=None, help="Explicit checkpoint table (overrides --run-dir).")
    p.add_argument("--scenario-file", type=Path, default=None, help="Explicit scenario .yaml/.yml/.json (overrides --run-dir).")
    p.add_argument("--workflow-init-csv", type=Path, default=None, help="Explicit workflow_init_info.csv (optional).")
    p.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Hard limit on checkpoint rows sent to the model; exceeding it is an error (no silent truncation).",
    )
    # Model / endpoint
    p.add_argument("--model", type=str, default=None, help="Model id, e.g. gpt-4o-mini or a compatible gateway model.")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    p.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV, help="Environment variable holding the API key.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--json-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="'auto' tries response_format=json_object and retries once without it if rejected.",
    )
    p.add_argument(
        "--extra-instructions",
        type=str,
        default=None,
        help="Optional additional instruction appended to the user prompt (or use --extra-instructions-file).",
    )
    p.add_argument("--extra-instructions-file", type=Path, default=None)
    # Decomposition budget / validation
    p.add_argument("--min-macro-tasks", type=int, default=None)
    p.add_argument("--max-macro-tasks", type=int, default=None)
    p.add_argument(
        "--allow-causal-violation",
        action="store_true",
        help="Downgrade causal-order violations from error to warning (not recommended).",
    )
    # Outputs
    p.add_argument("--cluster-method", type=str, default="llm", help="Method name used in checkpoint_clusters_<method>.csv.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <run-dir>/llm_decomposition).")
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Also write checkpoint_clusters_<method>.csv into the Stage 1 run directory (needed by Stage 2/3).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the prompts and exit without calling the API.")
    p.add_argument("--from-json", type=Path, default=None, help="Validate/emit a saved decomposition JSON instead of calling the API.")
    return p


def _resolve_inputs(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir not found: {run_dir}")
        checkpoint = args.checkpoint_csv or (run_dir / "checkpoint_init.csv")
        scenario = args.scenario_file or (run_dir / "scenario.yaml")
        workflow_init = args.workflow_init_csv or (run_dir / "workflow_init_info.csv")
    else:
        if args.checkpoint_csv is None:
            raise ValueError("provide --run-dir or --checkpoint-csv")
        run_dir = None
        checkpoint = Path(args.checkpoint_csv)
        scenario = args.scenario_file
        workflow_init = args.workflow_init_csv
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"checkpoint table not found: {checkpoint} "
            f"(run Stage 1 first, or pass --checkpoint-csv)"
        )
    if scenario is not None and not Path(scenario).exists():
        raise FileNotFoundError(f"scenario file not found: {scenario}")
    return {
        "run_dir": run_dir,
        "checkpoint": Path(checkpoint),
        "scenario": Path(scenario) if scenario is not None else None,
        "workflow_init": Path(workflow_init) if workflow_init is not None else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        paths = _resolve_inputs(args)
        rows = load_checkpoint_rows(paths["checkpoint"])  # type: ignore[arg-type]
        if args.max_checkpoints is not None and len(rows) > args.max_checkpoints:
            raise ValueError(
                f"{len(rows)} checkpoints exceed --max-checkpoints {args.max_checkpoints}; "
                f"refusing to send a truncated instance to the model"
            )

        workflow_rows = _read_workflow_rows(paths["workflow_init"])
        if paths["scenario"] is not None:
            scenario_doc = load_scenario_document(paths["scenario"])  # type: ignore[arg-type]
        else:
            scenario_doc = {
                "scenario_name": Path(paths["checkpoint"]).parent.name,
                "scenario_description": "(scenario file not provided)",
            }
        summary = summarize_scenario(scenario_doc, workflow_rows)

        extra_instructions = args.extra_instructions
        if args.extra_instructions_file is not None:
            extra_instructions = Path(args.extra_instructions_file).read_text(encoding="utf-8")
        user_prompt = build_user_prompt(
            summary,
            rows,
            min_macro_tasks=args.min_macro_tasks,
            max_macro_tasks=args.max_macro_tasks,
            extra_instructions=extra_instructions,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        output_dir = Path(args.output_dir) if args.output_dir else (
            (Path(paths["run_dir"]) / "llm_decomposition") if paths["run_dir"] else Path("llm_decomposition")
        )
        method_slug = _slug(args.cluster_method)
        model_slug = _slug(args.model or "from-json")

        if args.dry_run:
            print("=== DRY RUN (no API call) ===")
            print(f"planner_version: {PLANNER_VERSION}")
            print(f"checkpoint_csv: {paths['checkpoint']}")
            print(f"scenario_file: {paths['scenario']}")
            print(f"n_checkpoints: {len(rows)}")
            print(f"n_workflows: {summary.get('n_workflows')}")
            print(f"rendered_checkpoint_lines: {count_checkpoint_rows(user_prompt)}")
            print(f"system_prompt_chars: {len(SYSTEM_PROMPT)}")
            print(f"user_prompt_chars: {len(user_prompt)}")
            print(f"approx_prompt_tokens: {int((len(SYSTEM_PROMPT) + len(user_prompt)) / 4)}")
            print(f"planned_cluster_csv: {output_dir / f'checkpoint_clusters_{method_slug}.csv'}")
            print("--- user prompt (first 2000 chars) ---")
            print(user_prompt[:2000])
            return 0

        metadata: Dict[str, Any] = {
            "planner_version": PLANNER_VERSION,
            "generated_at": _now(),
            "cluster_method": args.cluster_method,
            "checkpoint_csv": str(paths["checkpoint"]),
            "scenario_file": str(paths["scenario"]) if paths["scenario"] else None,
            "n_checkpoints": len(rows),
            "scenario_summary": summary,
            "prompt_sha256": _sha256(SYSTEM_PROMPT + "\n" + user_prompt),
            "prompt_chars": {"system": len(SYSTEM_PROMPT), "user": len(user_prompt)},
            "validation_policy": {
                "strict_causal": not args.allow_causal_violation,
                "min_macro_tasks": args.min_macro_tasks,
                "max_macro_tasks": args.max_macro_tasks,
            },
        }

        if args.from_json is not None:
            raw = Path(args.from_json).read_text(encoding="utf-8")
            payload = extract_json_object(raw)
            metadata["call"] = {"mode": "from_json", "path": str(args.from_json)}
        else:
            if not args.model:
                raise ValueError("--model is required for a cloud decomposition call")
            cfg = LLMCallConfig(
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
                json_mode=args.json_mode,
            )
            client = OpenAICompatibleClient(cfg)
            result = client.chat(messages)
            metadata["call"] = {"mode": "cloud", **result.as_dict()}
            payload = extract_json_object(result.text)

        decomposition, report = validate_decomposition(
            payload,
            rows,
            strict_causal=not args.allow_causal_violation,
            min_macro_tasks=args.min_macro_tasks,
            max_macro_tasks=args.max_macro_tasks,
        )
        decomposition = renumber_clusters(decomposition, rows)

        output_dir.mkdir(parents=True, exist_ok=True)
        decomposition_json = output_dir / f"llm_decomposition_{model_slug}.json"
        decomposition_json.write_text(
            json.dumps(decomposition.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        cluster_csv = output_dir / f"checkpoint_clusters_{method_slug}.csv"
        n_clusters = apply_decomposition_to_csv(paths["checkpoint"], decomposition, cluster_csv)  # type: ignore[arg-type]

        inplace_csv: Optional[Path] = None
        if args.inplace:
            if paths["run_dir"] is None:
                raise ValueError("--inplace requires --run-dir")
            inplace_csv = Path(paths["run_dir"]) / f"checkpoint_clusters_{method_slug}.csv"
            apply_decomposition_to_csv(paths["checkpoint"], decomposition, inplace_csv)  # type: ignore[arg-type]

        metadata["n_macro_tasks"] = n_clusters
        metadata["outputs"] = {
            "decomposition_json": str(decomposition_json),
            "checkpoint_clusters_csv": str(cluster_csv),
            "inplace_checkpoint_clusters_csv": str(inplace_csv) if inplace_csv else None,
        }
        meta_json = output_dir / f"llm_call_metadata_{model_slug}.json"
        meta_json.write_text(
            json.dumps(decomposition_metadata(decomposition, report, extra=metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("=== LLM decomposition completed ===")
        print(f"macro_tasks: {n_clusters}")
        print(f"checkpoints: {len(rows)}")
        if report.warnings:
            print(f"warnings: {len(report.warnings)}")
            for w in report.warnings[:5]:
                print(f"  - {w}")
        print(f"decomposition_json: {decomposition_json}")
        print(f"metadata_json: {meta_json}")
        print(f"checkpoint_clusters_csv: {cluster_csv}")
        if inplace_csv is not None:
            print(f"inplace_checkpoint_clusters_csv: {inplace_csv}")
        print("\nNext steps (Stage 2 -> Stage 3):")
        print(
            f"  python3 -m src.offline_pipeline.stage2_plan_from_clusters "
            f"--seed <seed> --cluster-method {args.cluster_method}"
        )
        print(
            f"  python3 -m src.offline_pipeline.stage3_plan_with_pddl "
            f"--seed <seed> --cluster-method {args.cluster_method}"
        )
        return 0
    except (DecompositionError, LLMRequestError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "No decomposition CSV was written; fix the input/credentials and rerun.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
