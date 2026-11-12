#!/bin/bash
# Scoped replanning benchmark: real PDDL (unified_planning + Fast Downward) + real A*.
#
# Bundled fixtures (5 cases, each checked against its declared status):
#   ./scripts/06_run_scoped_replanning.sh --demo
#
# Batch evaluation on deterministically generated benchmark instances
# (fixed (scenario, seed) -> identical instance; S1 small .. S5 large):
#   ./scripts/06_run_scoped_replanning.sh \
#       --scenarios S1 S2 S3 S4 S5 --seeds 42 43 44 45 46 --cases-per-seed 20
#   ./scripts/06_run_scoped_replanning.sh \
#       --scenarios S1 S2 --seeds 42 --cases-per-seed 2 \
#       --artifacts-dir tmp/batch --output-json tmp/batch/summary.json
#
# Batch evaluation on your own schedule files
# (<dir>/<scenario>_seed<seed>.json with graph + snapshot + alerts):
#   ./scripts/06_run_scoped_replanning.sh \
#       --scenarios S1 S2 --seeds 42 --cases-per-seed 5 --schedule-dir data/schedules
#
# External inputs:
#   ./scripts/06_run_scoped_replanning.sh \
#       --graph examples/replanning/graph_small.json \
#       --snapshot examples/replanning/snapshot_small.json \
#       --alert examples/replanning/alerts/demo_detour.json
#
# Exit code is non-zero when any case errors out or misses its declared
# expectation in --demo mode.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
if [ "$#" -eq 0 ]; then
    set -- --demo
fi
exec python3 -m src.validation.scoped_replan_benchmark "$@"
