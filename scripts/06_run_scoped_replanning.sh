#!/bin/bash
# Scoped replanning benchmark: real PDDL (unified_planning + Fast Downward) + real A*.
#
#   ./scripts/06_run_scoped_replanning.sh                 # bundled demo fixtures
#   ./scripts/06_run_scoped_replanning.sh --demo --artifacts-dir tmp/artifacts
#   ./scripts/06_run_scoped_replanning.sh \
#       --graph examples/replanning/graph_small.json \
#       --snapshot examples/replanning/snapshot_small.json \
#       --alert examples/replanning/alerts/demo_detour.json
#
# Exit code is non-zero when any case misses its declared expectation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
if [ "$#" -eq 0 ]; then
    set -- --demo
fi
exec python3 -m src.validation.scoped_replan_benchmark "$@"
