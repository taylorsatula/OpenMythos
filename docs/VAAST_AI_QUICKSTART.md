# OpenMythos Training - Vast.AI Quick Start Guide

## Instance Requirements

- **GPU**: NVIDIA A100-80GB (recommended) or A100-40GB
- **RAM**: 16GB+ 
- **Disk**: 200GB+ (170GB for datasets + checkpoints)
- **OS**: Ubuntu 20.04+ or Debian 12

## Quick Start

### 1. Initial Setup (One-time)

SSH into your Vast.AI instance and run:

```bash
# Download and run setup script
wget https://your-repo-url/scripts/setup_vastai.sh
bash setup_vastai.sh

# Activate environment
source ~/miniconda3/bin/activate mythos

# Login to services
wandb login
huggingface-cli login  # if using private datasets
```

### 2. Download Datasets

```bash
# Download and prepare all datasets (~9B tokens, may take several hours)
bash scripts/download_datasets.sh

# Or for faster setup with only synthetic data (for testing)
python -c "from data.synthetic import generate_dataset; generate_dataset()"
```

### 3. Run Training

```bash
# Activate environment
source ~/miniconda3/bin/activate mythos
cd /root/mythosmini

# Train RDT model
python -m training.train \
    --model rdt \
    --output_dir outputs/rdt_1.5b \
    --total_steps 29000 \
    --batch_size 64 \
    --max_seq_len 2048 \
    --max_loops 8 \
    --lr 3e-4

# Train dense baseline (for comparison)
python -m training.train \
    --model dense \
    --output_dir outputs/dense_baseline \
    --total_steps 29000
```

### 4. Monitor Training

```bash
# View logs
tail -f outputs/rdt_1.5b/train.log

# TensorBoard
tensorboard --logdir outputs/

# Weights & Biases dashboard
wandb sync
```

## Dataset Links

If downloading manually:

| Dataset | Link | Filter |
|---------|------|--------|
| FineWeb-Edu | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | score >= 3 |
| OpenWebMath | https://huggingface.co/datasets/open-web-math/open-web-math | none |
| The Stack v2 | https://huggingface.co/datasets/bigcode/the-stack | Python, Rust, TS, C; >50 lines |
| Wikipedia | https://huggingface.co/datasets/wikimedia/wikipedia | none |
| arXiv | https://huggingface.co/datasets/arxiv_dataset | abstracts + intros |

## Environment Variables

```bash
# Optional environment variables for setup script
export REPO_URL=https://github.com/your-repo/openmythos.git
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export CUDA_VERSION=12.4
export PYTHON_VERSION=3.11
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python -m training.train --batch_size 32

# Use gradient checkpointing (coming soon)
```

### Slow tokenization
```bash
# Use more workers
python data/prepare_data.py --num_workers 8
```

### Dataset download fails
```bash
# Retry with HF mirror
export HF_ENDPOINT=https://hf-mirror.com
```

## Expected Training Time

- **RDT 1.5B @ 15B tokens**: ~72 hours on A100-80GB
- **Dense Baseline @ 15B tokens**: ~48 hours on A100-80GB
- **Combined (both models + evals)**: ~150 hours

## Cost Estimate (A100-80GB @ $1.50/hr)

- RDT training: ~$108
- Baseline training: ~$72  
- Evaluation: ~$15
- **Total**: ~$195

## Files

```
mythosmini/
├── open_mythos/       # Model implementation
│   ├── main.py       # Core RDT model
│   └── baseline.py   # Dense baseline
├── training/          # Training harness
│   ├── train.py      # Main training loop
│   ├── losses.py     # Composite loss
│   ├── curriculum.py # Loop depth schedule
│   └── checkpointing.py
├── eval/              # Evaluation suite
│   ├── perplexity.py
│   ├── act_profile.py
│   ├── expert_util.py
│   └── arithmetic.py
├── data/              # Data pipeline
│   ├── prepare_data.py
│   ├── dataloader.py
│   └── synthetic.py
├── configs/           # Model configs
│   ├── rdt_1.5b.py
│   └── baseline.py
├── scripts/
│   ├── setup_vastai.sh      # Initial setup
│   ├── download_datasets.sh # Dataset download
│   ├── run_training.sh      # Training launcher
│   └── run_eval.sh          # Eval launcher
└── utils/
    ├── count_params.py
    ├── count_flops.py
    └── logging.py
```