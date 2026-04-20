#!/usr/bin/env python3
"""Main evaluation orchestrator."""

import time
from typing import Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from open_mythos import OpenMythos, MythosConfig
from open_mythos.baseline import DenseTransformer

from eval.perplexity import (
    compute_perplexity,
    evaluate_perplexity_at_loops,
    check_depth_extrapolation,
)
from eval.act_profile import (
    get_halt_distribution,
    compute_mean_halt_step,
    check_act_collapse,
)
from eval.expert_util import (
    compute_expert_entropy,
    count_dead_experts,
)
from eval.arithmetic import (
    evaluate_arithmetic,
    evaluate_depth_extrapolation,
)


def evaluate_rdt(
    model: OpenMythos,
    val_data: DataLoader,
    tokenizer,
    vocab_size: int = 32000,
    loop_counts: list = None,
    num_batches: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Run full evaluation suite on RDT model.
    
    Args:
        model: The RDT model
        val_data: Validation DataLoader
        tokenizer: Tokenizer
        vocab_size: Vocabulary size
        loop_counts: Loop counts to evaluate for perplexity
        num_batches: Number of batches for perplexity evaluation
        device: Device
    
    Returns:
        Dictionary with all evaluation metrics
    """
    if loop_counts is None:
        loop_counts = [4, 8, 12, 16]
    
    results = {}
    results["timestamp"] = time.time()
    
    print("\n" + "=" * 60)
    print("RDT Evaluation")
    print("=" * 60)
    
    print("\n[1/4] Perplexity at loop counts...")
    perplexities = evaluate_perplexity_at_loops(
        model, val_data, vocab_size, loop_counts, num_batches, device
    )
    results["perplexities"] = perplexities
    results["depth_extrapolation"] = check_depth_extrapolation(perplexities)
    print(f"  Depth extrapolation: {results['depth_extrapolation']}")
    
    print("\n[2/4] ACT profile analysis...")
    sample_input = torch.randint(0, vocab_size, (4, 256), device=device)
    halt_dist = get_halt_distribution(model, sample_input, n_loops=16, device=device)
    results["halt_distribution"] = halt_dist
    
    with torch.no_grad():
        _, ponder_cost = model(sample_input, n_loops=8)
    mean_halt = compute_mean_halt_step(ponder_cost)
    results["mean_halt_step"] = mean_halt
    print(f"  Mean halt step: {mean_halt:.2f}")
    
    act_collapsed = check_act_collapse(model, sample_input, n_loops=8, device=device)
    results["act_collapsed"] = act_collapsed
    print(f"  ACT collapsed: {act_collapsed}")
    
    print("\n[3/4] Expert utilization...")
    try:
        from eval.expert_util import get_expert_counts
        # Counts come from MoEFFN.last_topk_idx, populated by the sample_input forward above.
        expert_counts = get_expert_counts(model)
        entropy = compute_expert_entropy(expert_counts)
        dead_count = count_dead_experts(expert_counts)
        results["expert_entropy"] = entropy
        results["dead_expert_count"] = dead_count
        print(f"  Expert entropy: {entropy:.2f}")
        print(f"  Dead experts: {dead_count}")
    except AttributeError:
        print("  Expert counts not available (run a forward pass first)")
        results["expert_entropy"] = None
        results["dead_expert_count"] = None
    
    print("\n[4/4] Arithmetic evaluation...")
    arithmetic_data = load_synthetic_data(depth=10, data_dir="data/synthetic")
    if arithmetic_data:
        questions = [q for q, _ in arithmetic_data[:100]]
        arith_metrics = evaluate_arithmetic(model, questions, tokenizer, device=device)
        results["arithmetic_accuracy"] = arith_metrics["accuracy"]
        print(f"  Arithmetic accuracy (K=10): {arith_metrics['accuracy']:.2%}")
    else:
        print("  No arithmetic data found")
        results["arithmetic_accuracy"] = None
    
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Perplexity @ 8 loops: {perplexities.get(8, 'N/A')}")
    print(f"  Perplexity @ 12 loops: {perplexities.get(12, 'N/A')}")
    print(f"  Perplexity @ 16 loops: {perplexities.get(16, 'N/A')}")
    print(f"  Depth extrapolation: {results['depth_extrapolation']}")
    print(f"  Mean halt step: {results['mean_halt_step']:.2f}")
    print(f"  Expert entropy: {results['expert_entropy']:.2f if results['expert_entropy'] else 'N/A'}")
    print(f"  Dead expert count: {results['dead_expert_count']}")
    print(f"  ACT collapsed: {results['act_collapsed']}")
    print("=" * 60)
    
    return results


def evaluate_baseline(
    model: DenseTransformer,
    val_data: DataLoader,
    vocab_size: int = 32000,
    num_batches: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Run evaluation suite on dense baseline model.
    
    Args:
        model: The dense baseline model
        val_data: Validation DataLoader
        vocab_size: Vocabulary size
        num_batches: Number of batches for evaluation
        device: Device
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {}
    results["timestamp"] = time.time()
    
    print("\n" + "=" * 60)
    print("Dense Baseline Evaluation")
    print("=" * 60)
    
    print("\n[1/2] Perplexity...")
    total_loss = 0.0
    count = 0
    
    model.eval()
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
            
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_targets.view(-1),
            )
            total_loss += loss.item()
            count += 1
    
    model.train()
    
    avg_loss = total_loss / count
    perplexity = __import__('math').exp(avg_loss)
    results["perplexity"] = perplexity
    print(f"  Perplexity: {perplexity:.2f}")
    
    print("\n[2/2] Arithmetic evaluation...")
    arithmetic_data = load_synthetic_data(depth=10, data_dir="data/synthetic")
    if arithmetic_data:
        from eval.arithmetic import evaluate_arithmetic, extract_answer
        from open_mythos.tokenizer import MythosTokenizer
        tok = MythosTokenizer()
        questions = [q for q, _ in arithmetic_data[:100]]
        arith_metrics = evaluate_arithmetic(model, questions, tok, device=device)
        results["arithmetic_accuracy"] = arith_metrics["accuracy"]
        print(f"  Arithmetic accuracy (K=10): {arith_metrics['accuracy']:.2%}")
    
    print("=" * 60)
    return results


def compare_models(
    rdt_results: Dict,
    baseline_results: Dict,
) -> Dict:
    """
    Compare RDT and baseline results.
    
    Args:
        rdt_results: Results from evaluate_rdt
        baseline_results: Results from evaluate_baseline
    
    Returns:
        Comparison metrics
    """
    comparison = {}
    
    rdt_ppl = rdt_results.get("perplexities", {}).get(8, float('inf'))
    baseline_ppl = baseline_results.get("perplexity", float('inf'))
    
    comparison["perplexity_rdt"] = rdt_ppl
    comparison["perplexity_baseline"] = baseline_ppl
    comparison["perplexity_ratio"] = rdt_ppl / baseline_ppl if baseline_ppl > 0 else float('inf')
    comparison["rdt_beats_baseline"] = rdt_ppl < baseline_ppl
    
    comparison["depth_extrapolation"] = rdt_results.get("depth_extrapolation", False)
    comparison["arithmetic_rdt"] = rdt_results.get("arithmetic_accuracy")
    comparison["arithmetic_baseline"] = baseline_results.get("arithmetic_accuracy")
    
    return comparison


if __name__ == "__main__":
    print("Evaluation orchestrator")
    print("=" * 40)
    print("Functions: evaluate_rdt, evaluate_baseline, compare_models")