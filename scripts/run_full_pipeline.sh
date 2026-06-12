#!/bin/bash
# Master script: Run the complete experiment pipeline
# This script runs all steps in sequence

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "HtT-Recursive Experiment Pipeline"
echo "============================================================"
echo ""
echo "This script will run the complete experiment pipeline:"
echo "  1. Generate scenario data (if not exists)"
echo "  2. Run Stage 1 clustering (all methods)"
echo "  3. Run Stage 3 PDDL planning (all methods)"
echo "  4. Replay offline simulation (all methods)"
echo "  5. Aggregate local results"
echo ""

# Configuration - can be overridden via environment variables
export START_SEED="${START_SEED:-42}"
export N_SEEDS="${N_SEEDS:-5}"

echo "Configuration:"
echo "  Seeds: $START_SEED to $((START_SEED + N_SEEDS - 1)) ($N_SEEDS total)"
echo ""

# Ask for confirmation
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

START_TIME=$(date +%s)

# Step 1: Generate data
echo ""
echo "============================================================"
echo "Running Step 1/5: Generate Data"
echo "============================================================"
bash scripts/01_generate_data.sh

# Step 2: Run clustering
echo ""
echo "============================================================"
echo "Running Step 2/5: Clustering"
echo "============================================================"
bash scripts/02_run_clustering.sh

# Step 3: Run PDDL planning
echo ""
echo "============================================================"
echo "Running Step 3/5: PDDL Planning"
echo "============================================================"
bash scripts/03_run_pddl_planning.sh

# Step 4: Run simulation
echo ""
echo "============================================================"
echo "Running Step 4/5: Simulation"
echo "============================================================"
bash scripts/04_run_simulation.sh

# Step 5: Plot results
echo ""
echo "============================================================"
echo "Running Step 5/5: Aggregate Results"
echo "============================================================"
bash scripts/05_plot_results.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo ""
echo "Results: data/results/"
echo "Figures: plot/figures/"
echo "============================================================"
