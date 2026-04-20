#!/bin/bash
# Evaluation launch script for OpenMythos
#
# Usage:
#   ./scripts/run_eval.sh [rdt|baseline|compare] [checkpoint_path]
#
# Examples:
#   ./scripts/run_eval.sh rdt outputs/rdt_1.5b/checkpoints/final_checkpoint
#   ./scripts/run_eval.sh baseline outputs/dense_baseline/checkpoints/final_checkpoint
#   ./scripts/run_eval.sh compare

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

EVAL_TYPE="${1:-compare}"
RDT_CHECKPOINT="${2:-outputs/rdt_1.5b/checkpoints/final_checkpoint}"
BASELINE_CHECKPOINT="${3:-outputs/dense_baseline/checkpoints/final_checkpoint}"

echo "=============================================="
echo "OpenMythos Evaluation"
echo "=============================================="
echo "Evaluation type: $EVAL_TYPE"
echo ""

# Create eval output directory
mkdir -p outputs/eval

if [ "$EVAL_TYPE" = "rdt" ]; then
    echo "Evaluating RDT model..."
    python -c "
import torch
from open_mythos import MythosConfig, OpenMythos
from training.checkpointing import load_checkpoint

config = MythosConfig(
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    n_experts=128,
    n_shared_experts=2,
    n_experts_per_tok=4,
    expert_dim=512,
    prelude_layers=4,
    coda_layers=4,
    max_loop_iters=16,
)

model = OpenMythos(config).cuda()
load_checkpoint('$RDT_CHECKPOINT', model, device='cuda')
print('RDT model loaded')

# Run evaluation
from eval.evaluate import evaluate_rdt
from data.dataloader import get_validation_split

val_data = get_validation_split()
results = evaluate_rdt(model, val_data, None, vocab_size=32000, device='cuda')
print(results)
"

elif [ "$EVAL_TYPE" = "baseline" ]; then
    echo "Evaluating dense baseline..."
    python -c "
import torch
from open_mythos.baseline import DenseTransformer, dense_config_from_rdt
from open_mythos.main import MythosConfig
from training.checkpointing import load_checkpoint

rdt_cfg = MythosConfig(dim=2048, n_heads=16, n_kv_heads=4)
config = dense_config_from_rdt(rdt_cfg, n_layers=10)

model = DenseTransformer(config).cuda()
load_checkpoint('$BASELINE_CHECKPOINT', model, device='cuda')
print('Baseline model loaded')

from eval.evaluate import evaluate_baseline
from data.dataloader import get_validation_split

val_data = get_validation_split()
results = evaluate_baseline(model, val_data, vocab_size=32000, device='cuda')
print(results)
"

elif [ "$EVAL_TYPE" = "compare" ]; then
    echo "Comparing RDT vs Baseline..."
    echo ""
    echo "NOTE: Run rdt and baseline evaluations first, then compare results."
    echo "This script provides the comparison framework."
    
    python -c "
# Comparison would require loading both models and running both evals
# See eval/evaluate.py for the full comparison logic
print('To run comparison:')
print('  1. First evaluate RDT: ./scripts/run_eval.sh rdt')
print('  2. Then evaluate baseline: ./scripts/run_eval.sh baseline')
print('  3. Then compare results')
"

else
    echo "Unknown evaluation type: $EVAL_TYPE"
    echo "Usage: ./scripts/run_eval.sh [rdt|baseline|compare]"
    exit 1
fi

echo ""
echo "Evaluation complete!"