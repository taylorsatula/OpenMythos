#!/usr/bin/env python3
"""ACT halting profile analysis."""

from typing import Dict, List, Tuple
import torch
import numpy as np


def get_halt_distribution(
    model,
    input_ids: torch.Tensor,
    n_loops: int = 8,
    device: str = "cuda",
) -> Dict[int, int]:
    """
    Get the distribution of halt steps across positions.
    
    Args:
        model: The RDT model
        input_ids: Input token IDs (B, T)
        n_loops: Number of loops to run
        device: Device
    
    Returns:
        Dictionary mapping halt_step -> count of positions that halted at that step
    """
    input_ids = input_ids.to(device)
    
    model.eval()
    with torch.no_grad():
        _, ponder_cost = model(input_ids, n_loops=n_loops)
    
    # Do not flip back to train mode here — the caller owns model mode.
    
    halt_steps = []
    for p in ponder_cost:
        for prob in p:
            cumulative = 0.0
            for step in range(1, n_loops + 1):
                cumulative += prob.item()
                if cumulative >= 0.99:
                    halt_steps.append(step)
                    break
            else:
                halt_steps.append(n_loops)
    
    distribution = {}
    for step in range(1, n_loops + 1):
        distribution[step] = halt_steps.count(step)
    
    return distribution


def compute_mean_halt_step(ponder_cost: torch.Tensor) -> float:
    """
    Compute the mean halt step across all positions.
    
    The halt step is inferred from the cumulative ponder cost.
    
    Args:
        ponder_cost: Cumulative ponder cost tensor (B, T)
    
    Returns:
        Mean halt step
    """
    halt_steps = []
    for p in ponder_cost:
        cumulative = 0.0
        for step in range(1, 17):
            cumulative += p[0].item()
            if cumulative >= 0.99:
                halt_steps.append(step)
                break
        else:
            halt_steps.append(16)
    
    return np.mean(halt_steps)


def analyze_halt_by_token_type(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    n_loops: int = 8,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Analyze halting behavior by token type.
    
    Token types:
    - Punctuation: tokens that are punctuation characters
    - Function words: common short words
    - Content words: nouns, verbs, adjectives
    - Code tokens: identifiers and operators (for code data)
    
    Args:
        model: The RDT model
        input_ids: Input token IDs (B, T)
        tokenizer: Tokenizer for decoding
        n_loops: Number of loops to run
        device: Device
    
    Returns:
        Dictionary mapping token_type -> list of halt steps
    """
    input_ids = input_ids.to(device)
    
    model.eval()
    with torch.no_grad():
        _, ponder_cost = model(input_ids, n_loops=n_loops)
    # Do not flip back to train mode here — the caller owns model mode.
    
    decoded = tokenizer.decode(input_ids[0])
    
    categories = {
        "punctuation": [],
        "function_words": [],
        "content_words": [],
        "code_tokens": [],
    }
    
    function_word_tokens = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                           "have", "has", "had", "do", "does", "did", "will", "would", "could",
                           "should", "may", "might", "must", "shall", "can", "to", "of", "in",
                           "for", "on", "with", "at", "by", "from", "as", "into", "through"}
    
    punctuation_chars = ".,!?;:\"'()[]{}"
    
    for i, (token_id, p) in enumerate(zip(input_ids[0], ponder_cost[0])):
        token_str = tokenizer.decode([token_id.item()]).strip()
        
        cumulative = 0.0
        halt_step = n_loops
        for step in range(1, n_loops + 1):
            cumulative += p.item()
            if cumulative >= 0.99:
                halt_step = step
                break
        
        if not token_str:
            continue
        
        is_punct = any(c in punctuation_chars for c in token_str)
        is_func = token_str.lower() in function_word_tokens
        is_code = not token_str.isalpha() and not is_punct
        
        if is_punct:
            categories["punctuation"].append(halt_step)
        elif is_func:
            categories["function_words"].append(halt_step)
        elif is_code:
            categories["code_tokens"].append(halt_step)
        else:
            categories["content_words"].append(halt_step)
    
    return categories


def check_act_collapse(
    model,
    input_ids: torch.Tensor,
    n_loops: int = 8,
    threshold: float = 0.95,
    device: str = "cuda",
) -> bool:
    """
    Check if ACT has collapsed (most positions halt at step 1).
    
    Args:
        model: The RDT model
        input_ids: Input token IDs
        n_loops: Number of loops
        threshold: Fraction of positions that must halt at step 1 for collapse
        device: Device
    
    Returns:
        True if ACT has collapsed
    """
    input_ids = input_ids.to(device)
    
    model.eval()
    with torch.no_grad():
        _, ponder_cost = model(input_ids, n_loops=n_loops)
    # Do not flip back to train mode here — the caller owns model mode.
    
    total_positions = ponder_cost.numel()
    halted_at_1 = (ponder_cost < 0.5).sum().item()
    
    fraction_at_1 = halted_at_1 / total_positions
    
    return fraction_at_1 > threshold


if __name__ == "__main__":
    print("ACT profile analysis utilities")
    print("=" * 40)
    print("Functions: get_halt_distribution, compute_mean_halt_step,")
    print("          analyze_halt_by_token_type, check_act_collapse")