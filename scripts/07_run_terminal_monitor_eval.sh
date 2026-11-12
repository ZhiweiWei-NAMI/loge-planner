#!/bin/bash
# Terminal semantic monitor evaluation (L0-L3).
#
# Non-LLM baselines are always reported under explicit baseline ids
# (baseline_rule_threshold / baseline_mlp / baseline_gbdt).
# --adapter-dir really loads Qwen2.5-0.5B + PEFT/LoRA and scores its generations.
#
# Examples:
#   bash scripts/07_run_terminal_monitor_eval.sh --levels L0 L1 L2 L3 --samples-per-class 80
#   bash scripts/07_run_terminal_monitor_eval.sh --data-csv data/terminal/windows.csv \
#       --adapter-dir /tmp/qwen_terminal_lora --include-base-model --output-json data/terminal/eval.json
#   bash scripts/07_run_terminal_monitor_eval.sh --latency-replay-json data/terminal/measured_latency.json
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
exec python3 -u -m src.validation.terminal_monitor_eval "$@"
