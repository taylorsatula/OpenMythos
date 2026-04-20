#!/bin/bash
# Training launch script for OpenMythos.
#
# Usage:
#   ./scripts/run_training.sh [rdt|baseline|tiny] [extra args]
#
# Examples:
#   ./scripts/run_training.sh rdt
#   ./scripts/run_training.sh rdt --resume outputs/rdt_1.5b/checkpoints/checkpoint_step_5000.pt
#   ./scripts/run_training.sh baseline
#   ./scripts/run_training.sh tiny           # ~15-min end-to-end pipeline sanity run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

MODEL_TYPE="${1:-rdt}"

echo "=============================================="
echo "OpenMythos Training"
echo "=============================================="
echo "Model: $MODEL_TYPE"
echo "Project root: $PROJECT_ROOT"
echo ""

if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: No NVIDIA GPU detected. Training will be unusable on CPU at this scale."
fi

python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')" 2>/dev/null || echo "WARNING: wandb not installed"

# Note: CUDA_LAUNCH_BLOCKING is intentionally NOT set — it serializes kernel
# launches and cuts throughput by ~3×. Only enable it for debugging.

if [ "$MODEL_TYPE" = "rdt" ]; then
    echo "Starting RDT training..."
    python -m training.train \
        --model rdt \
        --output_dir "outputs" \
        --data_dir "data/tokenized_shards" \
        --total_steps 29000 \
        --warmup_steps 2000 \
        --micro_batch 32 \
        --grad_accum 8 \
        --max_seq_len 2048 \
        --max_loops 8 \
        --lr_muon 0.02 \
        --lr_adamw 3e-4 \
        --eval_interval 500 \
        --checkpoint_interval 2500 \
        "${@:2}"
elif [ "$MODEL_TYPE" = "tiny" ]; then
    echo "Starting tiny RDT sanity run (~15 min target)..."
    python -m training.train \
        --model rdt \
        --config tiny_rdt \
        --output_dir "outputs" \
        --data_dir "data/tokenized_shards" \
        --total_steps 50 \
        --warmup_steps 5 \
        --micro_batch 2 \
        --grad_accum 2 \
        --max_seq_len 2048 \
        --max_loops 2 \
        --lr_muon 0.02 \
        --lr_adamw 3e-4 \
        --eval_interval 25 \
        --checkpoint_interval 25 \
        --no_compile \
        "${@:2}"
elif [ "$MODEL_TYPE" = "baseline" ]; then
    echo "Starting dense baseline training..."
    python -m training.train \
        --model dense \
        --output_dir "outputs" \
        --data_dir "data/tokenized_shards" \
        --total_steps 29000 \
        --warmup_steps 2000 \
        --micro_batch 32 \
        --grad_accum 8 \
        --max_seq_len 2048 \
        --lr_muon 0.02 \
        --lr_adamw 3e-4 \
        --eval_interval 500 \
        --checkpoint_interval 2500 \
        "${@:2}"
else
    echo "Unknown model type: $MODEL_TYPE"
    echo "Usage: ./scripts/run_training.sh [rdt|baseline|tiny]"
    exit 1
fi

echo ""
echo "Training complete!"
