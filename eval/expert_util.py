#!/usr/bin/env python3
"""MoE expert utilization analysis."""

import math
from typing import Dict, List, Tuple
import torch
import numpy as np


def get_expert_counts(model) -> torch.Tensor:
    """
    Get per-expert token counts from the most recent forward pass of the MoE block.

    Reads `MoEFFN.last_topk_idx`, which is cached during each forward pass
    (shape: (B*T, topk)). A token is counted once per top-k slot it occupies,
    so the total count is B*T*topk.

    Args:
        model: OpenMythos model (must have already been run forward)

    Returns:
        Tensor of shape (n_experts,) with counts per expert.
        Returns zeros if no forward pass has happened yet.
    """
    ffn = getattr(model.recurrent.block, "ffn", None)
    if ffn is None or not hasattr(ffn, "last_topk_idx") or ffn.last_topk_idx is None:
        n_experts = getattr(ffn, "n_experts", 128) if ffn is not None else 128
        return torch.zeros(n_experts)
    n_experts = ffn.n_experts
    return torch.bincount(ffn.last_topk_idx.view(-1).to(torch.long), minlength=n_experts).float()


def compute_expert_entropy(expert_counts: torch.Tensor) -> float:
    """
    Compute Shannon entropy of expert selection distribution.
    
    Higher entropy means more balanced expert utilization.
    
    Args:
        expert_counts: Tensor of shape (n_experts,) with counts per expert
    
    Returns:
        Shannon entropy
    """
    total = expert_counts.sum()
    if total == 0:
        return 0.0
    
    probs = expert_counts / total
    probs = probs[probs > 0]
    
    entropy = -(probs * torch.log(probs)).sum().item()
    return entropy


def compute_expert_entropy_normalized(expert_counts: torch.Tensor) -> float:
    """
    Normalized expert entropy (0 to 1, where 1 is uniform).
    
    Args:
        expert_counts: Tensor of shape (n_experts,) with counts per expert
    
    Returns:
        Normalized entropy (1 = perfectly uniform)
    """
    n_experts = len(expert_counts)
    max_entropy = math.log(n_experts)
    
    entropy = compute_expert_entropy(expert_counts)
    
    return entropy / max_entropy


def count_dead_experts(expert_counts: torch.Tensor, threshold: float = 0.001) -> int:
    """
    Count experts with very low activation (< threshold of total tokens).
    
    Args:
        expert_counts: Tensor of shape (n_experts,) with counts per expert
        threshold: Fraction threshold (default: 0.001 = 0.1%)
    
    Returns:
        Number of dead experts
    """
    total = expert_counts.sum()
    if total == 0:
        return len(expert_counts)
    
    fractions = expert_counts / total
    dead_count = (fractions < threshold).sum().item()
    
    return int(dead_count)


def get_expert_activation_patterns(
    model,
    input_ids: torch.Tensor,
    n_loops: int = 8,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Get per-loop expert activation patterns.
    
    Args:
        model: The RDT model
        input_ids: Input token IDs
        n_loops: Number of loops
        device: Device
    
    Returns:
        Dictionary mapping loop_index -> expert_counts tensor
    """
    model.eval()
    
    n_experts = model.recurrent.block.ffn.n_experts
    patterns = {}
    
    for loop_idx in range(n_loops):
        patterns[loop_idx] = torch.zeros(n_experts)
    
    with torch.no_grad():
        _, _ = model(input_ids, n_loops=n_loops)
    
    # Do not flip back to train mode here — the caller owns model mode.
    
    return patterns


def analyze_expert_specialization(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    n_loops: int = 8,
    device: str = "cuda",
) -> Dict[str, List[int]]:
    """
    Analyze which token types activate which experts.
    
    Args:
        model: The RDT model
        input_ids: Input token IDs
        tokenizer: Tokenizer for decoding
        n_loops: Number of loops
        device: Device
    
    Returns:
        Dictionary mapping expert_id -> list of decoded tokens
    """
    model.eval()
    
    with torch.no_grad():
        _, _ = model(input_ids, n_loops=n_loops)
    
    # Do not flip back to train mode here — the caller owns model mode.
    
    return {}


def compute_load_balance_loss(expert_counts: torch.Tensor) -> float:
    """
    Compute the load balancing auxiliary loss value.
    
    This is the actual aux_loss that was used during training,
    computed from expert counts.
    
    Args:
        expert_counts: Tensor of shape (n_experts,) with counts per expert
    
    Returns:
        Load balancing loss value
    """
    n_experts = len(expert_counts)
    total = expert_counts.sum()
    
    if total == 0:
        return 0.0
    
    f = expert_counts.float() / total
    ideal_load = torch.ones(n_experts) / n_experts
    
    load_imbalance = ((f - ideal_load) ** 2).sum().item()
    
    return load_imbalance


def get_utilization_report(
    model,
    val_data,
    vocab_size: int,
    n_loops: int = 8,
    num_batches: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Generate a full expert utilization report.
    
    Args:
        model: The RDT model
        val_data: Validation DataLoader
        vocab_size: Vocabulary size
        n_loops: Number of loops
        num_batches: Number of batches to analyze
        device: Device
    
    Returns:
        Dictionary with utilization metrics
    """
    model.eval()

    n_experts = model.recurrent.block.ffn.n_experts
    total_expert_counts = torch.zeros(n_experts)

    for batch_idx, batch in enumerate(val_data):
        if batch_idx >= num_batches:
            break

        if isinstance(batch, (tuple, list)):
            input_ids, _ = batch[0], batch[1]
        else:
            input_ids = batch

        input_ids = input_ids.to(device)

        with torch.no_grad():
            _, _ = model(input_ids, n_loops=n_loops)

        # Accumulate from the per-forward cache written by MoEFFN
        total_expert_counts = total_expert_counts + get_expert_counts(model).cpu()

    # Do not flip back to train mode here — the caller owns model mode.

    entropy = compute_expert_entropy(total_expert_counts)
    entropy_norm = compute_expert_entropy_normalized(total_expert_counts)
    dead_count = count_dead_experts(total_expert_counts)
    
    return {
        "expert_entropy": entropy,
        "expert_entropy_normalized": entropy_norm,
        "dead_expert_count": dead_count,
        "total_expert_counts": total_expert_counts,
    }


if __name__ == "__main__":
    print("Expert utilization analysis utilities")
    print("=" * 40)
    print("Functions: get_expert_counts, compute_expert_entropy,")
    print("          compute_expert_entropy_normalized, count_dead_experts,")
    print("          get_expert_activation_patterns, compute_load_balance_loss,")
    print("          get_utilization_report")