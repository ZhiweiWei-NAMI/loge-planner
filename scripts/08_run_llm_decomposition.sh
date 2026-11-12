#!/bin/bash
# LLM macro-task decomposition (real OpenAI-compatible cloud call).
#
# The emitted checkpoint_clusters_<method>.csv is the exact input consumed by
#   src/offline_pipeline/stage2_plan_from_clusters.py  (--cluster-method <method>)
#   src/offline_pipeline/stage3_plan_with_pddl.py      (--cluster-method <method>)
#
# Examples:
#   export OPENAI_API_KEY=...
#   bash scripts/08_run_llm_decomposition.sh --run-dir data/offline_init/large_mixed/run_seed_42 \
#       --model gpt-4o-mini --min-macro-tasks 8 --max-macro-tasks 60 --inplace
#   bash scripts/08_run_llm_decomposition.sh --run-dir data/offline_init/large_mixed/run_seed_42 \
#       --model gpt-4o-mini --dry-run          # prompt preview, no API call
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
exec python3 -u -m src.agent_llm.decompose_cli "$@"
