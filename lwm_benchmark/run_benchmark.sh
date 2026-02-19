#!/bin/bash

# Benchmark LWM Models
# Usage: ./run_benchmark.sh [task] [scenarios...]

TASK=${1:-"los"}
SCENARIOS=${2:-"O1_3p5B"}

if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi


echo "Running Benchmark for Task: $TASK"
echo "Scenarios: $SCENARIOS"

# Define Model Checkpoints
# Adjust these paths to where your actual checkpoints are trained/saved
BASE_CKPT="lwm/model_weights.pth"
CA_CKPT="lwm_ca/model_weights_ca_e2e.pth"
AXIAL_CKPT="lwm_axial/model_weights_rope_ddp.pth"
PHYSICS_CKPT="lwm_physics/model_weights_physics_ddp.pth"

# Resolve project root (one level up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run Benchmark
/home/audbhav22/.conda/envs/lwm/bin/python3 lwm_benchmark/benchmark.py \
    --task "$TASK" \
    --scenarios $SCENARIOS \
    --models base ca axial physics \
    --checkpoints "$BASE_CKPT" "$CA_CKPT" "$AXIAL_CKPT" "$PHYSICS_CKPT" \
    --input-types cls_emb channel_emb \
    --split-ratios 0.001  \
    --n-trials 3 \
    --epochs 50 \
    --batch-size 1024 \
    --save-csv "results/benchmark_comparison_${TASK}.csv" \
    "$@"

# Plot Results
echo "Plotting results..."
/home/audbhav22/.conda/envs/lwm/bin/python3 lwm_benchmark/plot_results.py \
    --csv "results/benchmark_comparison_${TASK}.csv" \
    --out "results/benchmark_comparison_${TASK}.png" \
    --title "Benchmark Results: ${TASK}"
