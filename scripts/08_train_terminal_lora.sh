#!/bin/bash
# LoRA SFT for the terminal-monitor Qwen2.5-0.5B model on external JSONL data.
#
# Requires: pip install -r requirements-llm.txt
# The adapter directory is later passed to:
#   bash scripts/07_run_terminal_monitor_eval.sh --adapter-dir <adapter_dir> --data-csv <windows.csv>
#
# Examples:
#   bash scripts/08_train_terminal_lora.sh --train-jsonl examples/llm/terminal_sft_example.jsonl \
#       --output-dir /tmp/qwen_terminal_lora --base-model Qwen/Qwen2.5-0.5B-Instruct \
#       --epochs 3 --batch-size 4 --lr 2e-4
#   bash scripts/08_train_terminal_lora.sh --train-jsonl examples/llm/terminal_sft_example.jsonl \
#       --output-dir /tmp/qwen_terminal_lora --validate-only     # no model download
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
exec python3 -u -m src.agent_llm.train_lora "$@"
