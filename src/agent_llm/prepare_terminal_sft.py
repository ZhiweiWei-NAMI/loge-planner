"""Build terminal-monitor SFT JSONL from window CSV (or the labelled fixture).

The generated examples use the *same* rendering and system prompt as
:mod:`src.agent_llm.terminal_llm`, so train-time and inference-time prompts are
identical by construction.

Leakage policy
--------------
``--split`` defaults to ``auto``:

* CSV **with** a ``split`` column -> only ``train`` rows are exported (leak-safe
  default), so a later evaluation on the ``test`` rows of the same CSV is clean.
* CSV **without** a ``split`` column -> hard error: the caller must either add a
  ``split`` column or pass ``--split all`` to *declare* the file training-only.
  ``--split all`` prints a warning because training on ``all`` and evaluating on
  the same CSV is train/test leakage.
* Fixture -> training-only by construction: it is generated with ``--seed`` while
  ``terminal_monitor_eval`` draws its fixture test set with ``seed + 101``.

Example::

    python -m src.agent_llm.prepare_terminal_sft \\
        --windows-csv examples/llm/terminal_windows_example.csv \\
        --split train \\
        --output-jsonl /tmp/terminal_sft.jsonl

    # training-only file without a 'split' column (explicit declaration)
    python -m src.agent_llm.prepare_terminal_sft \\
        --windows-csv my_training_only_windows.csv --split all \\
        --output-jsonl /tmp/terminal_sft.jsonl

    # labelled fixture with L1-L3 degradation augmentation
    python -m src.agent_llm.prepare_terminal_sft \\
        --fixture-samples-per-class 40 --levels L0 L1 L2 L3 \\
        --output-jsonl /tmp/terminal_sft_fixture.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.agent_llm.terminal_data import (
    EVENT_CLASSES,
    WindowDataset,
    apply_observation_degradation,
    load_window_csv,
    synthesize_fixture_windows,
)
from src.agent_llm.terminal_llm import SYSTEM_PROMPT, build_messages


def _rows_for(dataset: WindowDataset, levels: Optional[List[str]], seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    level_list = levels or ["L0"]
    for level in level_list:
        for i in range(len(dataset)):
            window = dataset.windows[i]
            if level != "L0":
                window = apply_observation_degradation(window, level, rng)
            label = dataset.labels[i]
            rows.append(
                {
                    "sample_id": f"{dataset.sample_ids[i]}|{level}",
                    "level": level,
                    "window": window,
                    "label": EVENT_CLASSES[int(label)],
                }
            )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert window CSV / fixture into terminal-monitor SFT chat JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--windows-csv", type=Path, default=None, help="Real window CSV (long or wide format).")
    p.add_argument("--fixture-samples-per-class", type=int, default=None, help="Use the labelled fixture instead of a CSV.")
    p.add_argument("--levels", nargs="+", default=None, help="Degradation levels to include (L0 keeps clean).")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split",
        choices=["auto", "train", "test", "all"],
        default="auto",
        help=(
            "Which rows to export. 'auto' = 'train' when the CSV has a 'split' "
            "column (leak-safe), and an error when it has none. 'all' must be "
            "passed explicitly to declare a training-only file; training on 'all' "
            "and evaluating on the same CSV is leakage."
        ),
    )
    p.add_argument("--include-system", action="store_true", default=True, help="Include the system message (recommended).")
    p.add_argument("--no-system", dest="include_system", action="store_false")
    args = p.parse_args(argv)

    if (args.windows_csv is None) == (args.fixture_samples_per_class is None):
        p.error("provide exactly one of --windows-csv or --fixture-samples-per-class")

    split_used: str
    if args.windows_csv is not None:
        dataset = load_window_csv(args.windows_csv)
        if args.split == "auto":
            if dataset.splits is None:
                p.error(
                    f"{args.windows_csv} has no 'split' column. Add one, or pass "
                    f"--split all to declare this file TRAINING-ONLY (never evaluate "
                    f"on the same CSV afterwards)."
                )
            split_used = "train"
            print(
                "note: 'split' column detected -> exporting only split=train "
                "(leak-safe default; pass --split all only for a training-only file)."
            )
        else:
            split_used = args.split

        if split_used in ("train", "test"):
            if dataset.splits is None:
                p.error(f"--split {split_used} requires a 'split' column in {args.windows_csv}")
            dataset = dataset.split_by(split_used)
            if split_used == "test":
                print(
                    "WARNING: --split test trains on the held-out test rows; any "
                    "evaluation on this CSV is leaked."
                )
        else:  # "all"
            print(
                "WARNING: --split all exports every row, including any 'test' split. "
                "Treat the result as TRAINING-ONLY data; do NOT evaluate on this "
                "same CSV (train/test leakage)."
            )
    else:
        if args.split not in ("auto", "all"):
            p.error("fixture data has no 'split' column; use --split auto or --split all")
        dataset = synthesize_fixture_windows(args.fixture_samples_per_class, args.seed)
        split_used = "fixture_train_only"
        print(
            f"note: fixture generated with seed={args.seed}; terminal_monitor_eval "
            f"draws its fixture test set with seed={args.seed + 101}, so this data is "
            f"training-only relative to that evaluation."
        )

    rows = _rows_for(dataset, args.levels, args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            messages = build_messages(row["window"])
            if not args.include_system:
                messages = messages[1:]
            messages.append({"role": "assistant", "content": row["label"]})
            fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} example(s) to {args.output_jsonl}")
    print(f"source: {dataset.source}")
    print(f"split: {split_used}")
    print(f"levels: {args.levels or ['L0']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
