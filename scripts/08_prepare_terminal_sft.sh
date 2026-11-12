#!/bin/bash
# Build terminal-monitor SFT JSONL from a real window CSV (or the labelled fixture).
#
# The produced examples use exactly the same prompt rendering as
# src/agent_llm/terminal_llm.py, so training and inference prompts match.
#
# --split auto (default) exports only split=train when the CSV has a 'split'
# column and errors when it does not; pass --split all only to declare a
# TRAINING-ONLY file (never evaluate on the same CSV afterwards).
#
# Examples:
#   bash scripts/08_prepare_terminal_sft.sh \
#       --windows-csv examples/llm/terminal_windows_example.csv --split train \
#       --output-jsonl /tmp/terminal_sft.jsonl
#   bash scripts/08_prepare_terminal_sft.sh \
#       --windows-csv my_training_only_windows.csv --split all \
#       --output-jsonl /tmp/terminal_sft.jsonl
#   bash scripts/08_prepare_terminal_sft.sh \
#       --fixture-samples-per-class 40 --levels L0 L1 L2 L3 \
#       --output-jsonl /tmp/terminal_sft_fixture.jsonl
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
exec python3 -u -m src.agent_llm.prepare_terminal_sft "$@"
