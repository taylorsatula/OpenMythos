#!/usr/bin/env python3
"""Perplexity evaluation for RDT and baseline."""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on a single batch.
    
    Args:
        model: The language model
        input_ids: Input token IDs (B, T)
        targets: Target token IDs (B, T)
        vocab_size: Vocabulary size
        device: Device to compute on
    
    Returns:
        Perplexity value
    """
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'recurrent'):
            logits, _ = model(input_ids, n_loops=8)
        else:
            logits = model(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_targets.view(-1),
            reduction="mean",
        )
    
    # Do not flip back to train mode here — the caller owns model mode.
    return math.exp(loss.item())


def evaluate_perplexity_at_loops(
    model,
    val_data: DataLoader,
    vocab_size: int,
    loop_counts: list = None,
    num_batches: int = 50,
    device: str = "cuda",
) -> Dict[int, float]:
    """
    Evaluate perplexity at multiple loop counts.
    
    This tests depth extrapolation: if perplexity improves at higher loop counts,
    the model has learned to use additional computation effectively.
    
    Args:
        model: The RDT model
        val_data: Validation DataLoader
        vocab_size: Vocabulary size
        loop_counts: List of loop counts to evaluate (default: [4, 8, 12, 16])
        num_batches: Number of batches to evaluate
        device: Device
    
    Returns:
        Dictionary mapping loop_count -> perplexity
    """
    if loop_counts is None:
        loop_counts = [4, 8, 12, 16]
    
    results = {}
    
    model.eval()
    for n_loops in loop_counts:
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                if batch_idx >= num_batches:
                    break
                
                if isinstance(batch, (tuple, list)):
                    input_ids, targets = batch[0], batch[1]
                else:
                    input_ids = batch
                    targets = batch
                
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                logits, _ = model(input_ids, n_loops=n_loops)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = targets[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_targets.view(-1),
                    reduction="mean",
                )
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / count
        perplexity = math.exp(avg_loss)
        results[n_loops] = perplexity
        print(f"  loops={n_loops}: perplexity={perplexity:.2f}")
    
    # Do not flip back to train mode here — the caller owns model mode.
    return results


def check_depth_extrapolation(perplexities: Dict[int, float]) -> bool:
    """
    Check if the model shows depth extrapolation.
    
    Depth extrapolation means perplexity at loop=12 or loop=16 is better
    (lower) than perplexity at loop=8.
    
    Args:
        perplexities: Dict mapping loop_count -> perplexity
    
    Returns:
        True if depth extrapolation is observed
    """
    if 8 not in perplexities:
        return False
    
    base_ppl = perplexities[8]
    
    for loops in [12, 16]:
        if loops in perplexities and perplexities[loops] < base_ppl:
            return True
    
    return False


if __name__ == "__main__":
    print("Perplexity evaluation utilities")
    print("=" * 40)
    print("Functions: compute_perplexity, evaluate_perplexity_at_loops,")
    print("          check_depth_extrapolation")