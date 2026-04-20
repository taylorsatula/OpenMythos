#!/bin/bash
# =============================================================================
# Dataset Download and Tokenization Script
# =============================================================================
# Downloads, filters, tokenizes, and prepares the 15B token dataset.
#
# Usage:
#   bash download_datasets.sh
#
# Datasets:
#   - FineWeb-Edu (score >= 3): ~9B tokens (60%)
#   - OpenWebMath: ~1.5B tokens (10%)
#   - The Stack v2 (Python, Rust, TS, C): ~2.25B tokens (15%)
#   - Wikipedia + WikiBooks: ~1.5B tokens (10%)
#   - arXiv (abstracts + intros): ~0.75B tokens (5%)
# =============================================================================

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-/root/mythosmini}"
TOKENIZER="${TOKENIZER:-mistralai/Mistral-7B-v0.1}"
SEQ_LEN="${SEQ_LEN:-2048}"
VAL_SPLIT="${VAL_SPLIT:-0.001}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check environment
if [ ! -d "$PROJECT_DIR" ]; then
    log_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

source $HOME/miniconda3/bin/activate mythos
cd "$PROJECT_DIR"

log_info "Starting dataset download and preparation..."
echo "  Project: $PROJECT_DIR"
echo "  Tokenizer: $TOKENIZER"
echo "  Seq length: $SEQ_LEN"
echo "  Val split: ${VAL_SPLIT}"
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

download_fineweb_edu() {
    log_info "Downloading FineWeb-Edu..."
    
    mkdir -p "$PROJECT_DIR/data/raw/fineweb_edu"
    mkdir -p "$PROJECT_DIR/data/tokenized/fineweb_edu"
    
    python3 << 'PYEOF'
from datasets import load_dataset
import os

print("  Loading FineWeb-Edu dataset...")
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='default', split='train')
print(f"  Total examples before filtering: {len(ds)}")

print("  Filtering for score >= 3...")
ds_filtered = ds.filter(lambda x: x.get('score', 0) >= 3, batched=False)
print(f"  Examples after filtering: {len(ds_filtered)}")

# Save filtered dataset
output_dir = os.path.join(os.environ.get('PROJECT_DIR', '/root/mythosmini'), 'data/raw/fineweb_edu')
ds_filtered.to_json(f"{output_dir}/train.jsonl", orient='records', lines=True)
print(f"  Saved to {output_dir}/train.jsonl")
PYEOF
    
    log_info "FineWeb-Edu download complete"
}

download_open_web_math() {
    log_info "Downloading OpenWebMath..."
    
    mkdir -p "$PROJECT_DIR/data/raw/open_web_math"
    
    python3 << 'PYEOF'
from datasets import load_dataset
import os

print("  Loading OpenWebMath dataset...")
ds = load_dataset('open-web-math/open-web-math', split='train')
print(f"  Total examples: {len(ds)}")

# Save
output_dir = os.path.join(os.environ.get('PROJECT_DIR', '/root/mythosmini'), 'data/raw/open_web_math')
ds.to_json(f"{output_dir}/train.jsonl", orient='records', lines=True)
print(f"  Saved to {output_dir}/train.jsonl")
PYEOF
    
    log_info "OpenWebMath download complete"
}

download_the_stack() {
    log_info "Downloading The Stack v2 (this may take a while)..."
    
    mkdir -p "$PROJECT_DIR/data/raw/the_stack"
    
    python3 << 'PYEOF'
from datasets import load_dataset
import os

print("  Loading The Stack dataset...")
# Load only the data we need
ds = load_dataset('bigcode/the-stack', name='data', split='train')
print(f"  Total examples before filtering: {len(ds)}")

print("  Filtering for Python, Rust, TypeScript, C...")
LANGUAGES = ['Python', 'Rust', 'TypeScript', 'C']
ds_filtered = ds.filter(lambda x: x.get('language', '') in LANGUAGES, batched=False)
print(f"  After language filter: {len(ds_filtered)}")

print("  Filtering for length > 50...")
ds_filtered = ds_filtered.filter(lambda x: x.get('length', 0) > 50, batched=False)
print(f"  After length filter: {len(ds_filtered)}")

# Save
output_dir = os.path.join(os.environ.get('PROJECT_DIR', '/root/mythosmini'), 'data/raw/the_stack')
ds_filtered.to_json(f"{output_dir}/train.jsonl", orient='records', lines=True)
print(f"  Saved to {output_dir}/train.jsonl")
PYEOF
    
    log_info "The Stack download complete"
}

download_wikipedia() {
    log_info "Downloading Wikipedia..."
    
    mkdir -p "$PROJECT_DIR/data/raw/wikipedia"
    
    python3 << 'PYEOF'
from datasets import load_dataset
import os

print("  Loading Wikipedia dataset...")
ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
print(f"  Total examples: {len(ds)}")

# Save
output_dir = os.path.join(os.environ.get('PROJECT_DIR', '/root/mythosmini'), 'data/raw/wikipedia')
ds.to_json(f"{output_dir}/train.jsonl", orient='records', lines=True)
print(f"  Saved to {output_dir}/train.jsonl")
PYEOF
    
    log_info "Wikipedia download complete"
}

download_arxiv() {
    log_info "Downloading arXiv..."
    
    mkdir -p "$PROJECT_DIR/data/raw/arxiv"
    
    python3 << 'PYEOF'
from datasets import load_dataset
import os

print("  Loading arXiv dataset...")
ds = load_dataset('arxiv_dataset', split='train')
print(f"  Total examples: {len(ds)}")

# arXiv doesn't have abstracts+intros split in the HF dataset
# We'll use the full text and extract abstract + intro later
output_dir = os.path.join(os.environ.get('PROJECT_DIR', '/root/mythosmini'), 'data/raw/arxiv')
ds.to_json(f"{output_dir}/train.jsonl", orient='records', lines=True)
print(f"  Saved to {output_dir}/train.jsonl")
PYEOF
    
    log_info "arXiv download complete"
}

tokenize_and_pack() {
    log_info "Tokenizing and packing datasets..."
    
    python3 << 'PYEOF'
import os
import json
import hashlib
from functools import partial
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

PROJECT_DIR = os.environ.get('PROJECT_DIR', '/root/mythosmini')
TOKENIZER_NAME = os.environ.get('TOKENIZER', 'mistralai/Mistral-7B-v0.1')
SEQ_LEN = int(os.environ.get('SEQ_LEN', 2048'))
VAL_SPLIT = float(os.environ.get('VAL_SPLIT', 0.001'))

print(f"  Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def deduplicate_documents(documents):
    """Deduplicate using SHA-256 of first 1000 chars."""
    seen = set()
    for doc in documents:
        hash_key = hashlib.sha256(doc[:1000].encode()).hexdigest()
        if hash_key not in seen:
            seen.add(hash_key)
            yield doc

def tokenize_and_pack(input_file, output_dir, name, proportion):
    """Tokenize a JSONL file and pack into sequences."""
    print(f"\n  Processing {name}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read documents
    documents = []
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc=f"    Reading {name}"):
            data = json.loads(line)
            if 'text' in data:
                documents.append(data['text'])
            elif 'content' in data:
                documents.append(data['content'])
    
    print(f"    {len(documents)} documents loaded")
    
    # Deduplicate
    documents = list(deduplicate_documents(documents))
    print(f"    {len(documents)} after deduplication")
    
    # Tokenize
    print(f"    Tokenizing (seq_len={SEQ_LEN})...")
    all_tokens = []
    for doc in tqdm(documents, desc=f"    Tokenizing {name}"):
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        
        # Pack into sequences
        for i in range(0, len(tokens), SEQ_LEN):
            chunk = tokens[i:i + SEQ_LEN]
            if len(chunk) == SEQ_LEN:
                all_tokens.append(chunk)
    
    print(f"    Packed into {len(all_tokens)} sequences")
    
    # Save as tensor shards
    shard_size = 10000  # sequences per shard
    for i in range(0, len(all_tokens), shard_size):
        shard = all_tokens[i:i + shard_size]
        shard_tensor = torch.tensor(shard, dtype=torch.int32)
        shard_path = os.path.join(output_dir, f"{name}_shard_{i//shard_size:04d}.pt")
        torch.save(shard_tensor, shard_path)
        print(f"    Saved {shard_path}")
    
    return len(all_tokens)

# Process each dataset
token_counts = {}

datasets = [
    ("fineweb_edu", f"{PROJECT_DIR}/data/raw/fineweb_edu/train.jsonl", f"{PROJECT_DIR}/data/tokenized/fineweb_edu", 0.60),
    ("open_web_math", f"{PROJECT_DIR}/data/raw/open_web_math/train.jsonl", f"{PROJECT_DIR}/data/tokenized/open_web_math", 0.10),
    ("the_stack", f"{PROJECT_DIR}/data/raw/the_stack/train.jsonl", f"{PROJECT_DIR}/data/tokenized/the_stack", 0.15),
    ("wikipedia", f"{PROJECT_DIR}/data/raw/wikipedia/train.jsonl", f"{PROJECT_DIR}/data/tokenized/wikipedia", 0.10),
]

for name, input_file, output_dir, proportion in datasets:
    if os.path.exists(input_file):
        count = tokenize_and_pack(input_file, output_dir, name, proportion)
        token_counts[name] = count * SEQ_LEN
    else:
        print(f"  Skipping {name} (file not found: {input_file})")

print(f"\n  Token counts by dataset:")
total = 0
for name, count in token_counts.items():
    print(f"    {name}: {count:,} tokens ({count/1e9:.2f}B)")
    total += count
print(f"    TOTAL: {total:,} tokens ({total/1e9:.2f}B)")
print(f"    Target: 15B tokens")
PYEOF
    
    log_info "Tokenization complete"
}

create_validation_split() {
    log_info "Creating validation split..."
    
    python3 << 'PYEOF'
import os
import torch
from pathlib import Path

PROJECT_DIR = os.environ.get('PROJECT_DIR', '/root/mythosmini')
VAL_SPLIT = float(os.environ.get('VAL_SPLIT', 0.001'))

tokenized_dir = Path(f"{PROJECT_DIR}/data/tokenized")
val_dir = Path(f"{PROJECT_DIR}/data/tokenized_shards/val")
train_dir = Path(f"{PROJECT_DIR}/data/tokenized_shards/train")

val_dir.mkdir(parents=True, exist_ok=True)
train_dir.mkdir(parents=True, exist_ok=True)

print(f"  Creating {VAL_SPLIT*100:.1%} validation split...")

# Collect all shards
all_shards = list(tokenized_dir.glob("*/train.jsonl"))
all_shards = []
for subdir in tokenized_dir.iterdir():
    if subdir.is_dir() and (subdir / "train.jsonl").exists():
        all_shards.append(subdir / "train.jsonl")

# Simple random split
import random
random.seed(42)

val_count = 0
train_count = 0

for shard_file in all_shards:
    # Each file is treated as one "document" for split purposes
    if random.random() < VAL_SPLIT:
        # Copy to val
        import shutil
        shutil.copy(shard_file, val_dir / shard_file.name)
        val_count += 1
    else:
        import shutil
        shutil.copy(shard_file, train_dir / shard_file.name)
        train_count += 1

print(f"  Val files: {val_count}")
print(f"  Train files: {train_count}")

# Create a manifest
manifest = {
    "val_files": val_count,
    "train_files": train_count,
    "seq_len": int(os.environ.get('SEQ_LEN', 2048')),
    "tokenizer": os.environ.get('TOKENIZER', 'mistralai/Mistral-7B-v0.1'),
}

import json
with open(f"{PROJECT_DIR}/data/tokenized_shards/manifest.json", 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"  Manifest saved to {PROJECT_DIR}/data/tokenized_shards/manifest.json")
PYEOF
    
    log_info "Validation split created"
}

# =============================================================================
# Main Execution
# =============================================================================

log_info "=========================================="
log_info "Dataset Download and Preparation"
log_info "=========================================="
echo ""

# Download datasets
download_fineweb_edu &
download_open_web_math &
download_the_stack &
download_wikipedia &
download_arxiv &

wait

# Tokenize and pack
tokenize_and_pack

# Create validation split
create_validation_split

echo ""
log_info "=========================================="
log_info "Dataset Preparation Complete!"
log_info "=========================================="
echo ""
echo "Data location: $PROJECT_DIR/data/tokenized_shards/"
echo ""
echo "Next steps:"
echo "  1. Verify data: ls -la $PROJECT_DIR/data/tokenized_shards/"
echo "  2. Run training: ./train.sh --model rdt"
echo ""