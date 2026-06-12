#!/bin/bash
# Script 5: Aggregate replay results
# Summarizes local result files when experiment outputs exist

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKFLOW_DECOUPLE="$PROJECT_ROOT/workflow_decouple"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Step 5: Aggregate Results"
echo "=========================================="

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/workflow_decouple/src:${PYTHONPATH:-}"

RESULTS_DIR="$PROJECT_ROOT/data/results"
echo "Input: $RESULTS_DIR"
echo ""

# Run aggregation script
python -m plot.aggregate_results

echo ""
echo "=========================================="
echo "Aggregation complete!"
echo "=========================================="
