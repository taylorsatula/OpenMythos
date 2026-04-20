#!/bin/bash
# =============================================================================
# OpenMythos Training Setup Script for Vast.AI (A100-80GB)
# =============================================================================
# Run this script on a fresh Vast.AI instance with an A100 GPU.
# 
# Usage:
#   bash setup_vastai.sh
#
# This script will:
#   1. Install system dependencies
#   2. Install PyTorch with CUDA 12.x support
#   3. Install Python dependencies
#   4. Clone/fetch the repository
#   5. Download and prepare datasets
#   6. Verify the installation
# =============================================================================

set -e

# Configuration
REPO_URL="${REPO_URL:-https://github.com/kyegomez/OpenMythos.git}"
REPO_DIR="${REPO_DIR:-/root/OpenMythos}"
PROJECT_DIR="${PROJECT_DIR:-/root/mythosmini}"
BRANCH="${BRANCH:-main}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Step 1: System Updates and Dependencies
# =============================================================================
log_info "Step 1: Installing system dependencies..."

apt-get update
apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    pkg-config \
    && apt-get clean

# =============================================================================
# Step 2: Install CUDA Toolkit (if not already present)
# =============================================================================
log_info "Step 2: Checking CUDA installation..."

if ! command -v nvcc &> /dev/null; then
    log_info "Installing CUDA ${CUDA_VERSION}..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit-${CUDA_VERSION}
    rm cuda-keyring_1.1-1_all.deb
    
    export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH
    
    log_warn "Add CUDA to PATH by adding to ~/.bashrc:"
    echo "export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
else
    log_info "CUDA already installed: $(nvcc --version | grep release)"
fi

# =============================================================================
# Step 3: Install Miniconda
# =============================================================================
log_info "Step 3: Installing Miniconda..."

if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi

export PATH="$HOME/miniconda3/bin:$PATH"

# =============================================================================
# Step 4: Create Python Environment
# =============================================================================
log_info "Step 4: Creating Python ${PYTHON_VERSION} environment..."

if [ ! -d "$HOME/miniconda3/envs/mythos" ]; then
    conda create -n mythos python=${PYTHON_VERSION} -y
fi

source "$HOME/miniconda3/bin/activate" mythos

# =============================================================================
# Step 5: Install PyTorch with CUDA Support
# =============================================================================
log_info "Step 5: Installing PyTorch with CUDA ${CUDA_VERSION} support..."

pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# =============================================================================
# Step 6: Install Python Dependencies
# =============================================================================
log_info "Step 6: Installing Python dependencies..."

# Core ML dependencies
pip install \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipython

# Hugging Face ecosystem
pip install \
    transformers \
    datasets \
    accelerate \
    huggingface_hub \
    tokenizers

# Training utilities
pip install \
    wandb \
    tensorboard \
    tqdm \
    pyyaml \
    omegaconf \
    hydra-core

# Data processing
pip install \
    pyarrow \
    pandas \
    fastparquet

# =============================================================================
# Step 7: Clone/Update Repository
# =============================================================================
log_info "Step 7: Setting up repository..."

if [ -d "$PROJECT_DIR" ]; then
    log_info "Repository already exists at $PROJECT_DIR"
    cd "$PROJECT_DIR"
    git pull origin ${BRANCH}
else
    # Clone OpenMythos
    git clone ${REPO_URL} "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    git checkout ${BRANCH}
fi

# =============================================================================
# Step 8: Install Project in Editable Mode
# =============================================================================
log_info "Step 8: Installing project in editable mode..."

cd "$PROJECT_DIR"
pip install -e .

# =============================================================================
# Step 9: Download Datasets
# =============================================================================
log_info "Step 9: Downloading and preparing datasets..."

mkdir -p "$PROJECT_DIR/data/tokenized_shards"
mkdir -p "$PROJECT_DIR/data/synthetic"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/outputs"

# Login to HuggingFace (required for some datasets)
if [ -n "$HF_TOKEN" ]; then
    log_info "Logging into HuggingFace..."
    pip install huggingface_hub
    huggingface-cli login --token "$HF_TOKEN"
fi

# Download FineWeb-Edu
log_info "Downloading FineWeb-Edu (this may take a while)..."
python3 -c "
from datasets import load_dataset
import os

os.makedirs('$PROJECT_DIR/data/raw/fineweb_edu', exist_ok=True)

# FineWeb-Edu with score >= 3
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='default', split='train')
ds = ds.filter(lambda x: x.get('score', 0) >= 3, batched=False)
print(f'FineWeb-Edu filtered: {len(ds)} examples')
"

# Download OpenWebMath
log_info "Downloading OpenWebMath..."
python3 -c "
from datasets import load_dataset
import os

os.makedirs('$PROJECT_DIR/data/raw/open_web_math', exist_ok=True)

ds = load_dataset('open-web-math/open-web-math', split='train')
print(f'OpenWebMath: {len(ds)} examples')
"

# Download The Stack v2 (filtered to Python, Rust, TypeScript, C)
log_info "Downloading The Stack v2 (filtered)..."
python3 -c "
from datasets import load_dataset
import os

os.makedirs('$PROJECT_DIR/data/raw/the_stack', exist_ok=True)

ds = load_dataset('bigcode/the-stack', name='data', split='train')
# Filter by language
ds = ds.filter(lambda x: x.get('language', '') in ['Python', 'Rust', 'TypeScript', 'C'], batched=False)
# Filter by length (>50 lines)
ds = ds.filter(lambda x: x.get('length', 0) > 50, batched=False)
print(f'The Stack filtered: {len(ds)} examples')
"

# Download Wikipedia
log_info "Downloading Wikipedia..."
python3 -c "
from datasets import load_dataset
import os

os.makedirs('$PROJECT_DIR/data/raw/wikipedia', exist_ok=True)

ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
print(f'Wikipedia: {len(ds)} examples')
"

# Generate synthetic arithmetic data
log_info "Generating synthetic arithmetic data..."
cd "$PROJECT_DIR"
python -c "
from data.synthetic import generate_dataset
generate_dataset(output_dir='data/synthetic')
print('Synthetic arithmetic data generated')
"

# =============================================================================
# Step 10: Verify Installation
# =============================================================================
log_info "Step 10: Verifying installation..."

cd "$PROJECT_DIR"

python -c "
import torch
from open_mythos import MythosConfig, OpenMythos

print('='*60)
print('Installation Verification')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Test model creation
config = MythosConfig(
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    n_experts=8,
    n_shared_experts=2,
    n_experts_per_tok=2,
    expert_dim=64,
    prelude_layers=2,
    coda_layers=2,
    max_loop_iters=4,
)
model = OpenMythos(config)
print(f'Model created: {sum(p.numel() for p in model.parameters()):,} parameters')

# Test forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 32))
logits, ponder = model(input_ids, n_loops=2)
print(f'Forward pass successful: logits={logits.shape}, ponder={ponder.shape}')

print('='*60)
print('Installation verification PASSED')
print('='*60)
"

# =============================================================================
# Step 11: Create Convenience Scripts
# =============================================================================
log_info "Step 11: Creating convenience scripts..."

# Create a simple train launcher
cat > "$PROJECT_DIR/train.sh" << 'EOF'
#!/bin/bash
source $HOME/miniconda3/bin/activate mythos
cd $(dirname "$0")
python -m training.train "$@"
EOF
chmod +x "$PROJECT_DIR/train.sh"

# Create a simple eval launcher
cat > "$PROJECT_DIR/eval.sh" << 'EOF'
#!/bin/bash
source $HOME/miniconda3/bin/activate mythos
cd $(dirname "$0")
python -c "from eval.evaluate import *; print('Eval module loaded')"
EOF
chmod +x "$PROJECT_DIR/eval.sh"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate conda environment: source ~/miniconda3/bin/activate mythos"
echo "  2. Login to WandB: wandb login"
echo "  3. Login to HuggingFace: huggingface-cli login (if using private datasets)"
echo "  4. Run training: ./train.sh --model rdt --total_steps 29000"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $PROJECT_DIR/data"
echo "Checkpoints: $PROJECT_DIR/checkpoints"
echo ""
echo "For interactive training monitoring:"
echo "  tensorboard --logdir outputs/"
echo ""
echo "============================================================"