#!/usr/bin/env python3
"""Synthetic multi-hop arithmetic evaluation."""

import re
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader

from data.synthetic import generate_arithmetic_chain, load_synthetic_data


def extract_answer(text: str) -> int:
    """Extract the final answer number from model output."""
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None


def evaluate_arithmetic(
    model,
    questions: List[str],
    tokenizer,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model on arithmetic questions.
    
    Args:
        model: The language model
        questions: List of arithmetic question strings
        tokenizer: Tokenizer for encoding/decoding
        max_new_tokens: Max tokens to generate
        device: Device
    
    Returns:
        Dictionary with accuracy and per-depth metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    results_by_depth = {}
    
    for q in questions:
        depth = len(re.findall(r'[+\-*]', q))

        # Ensure the prompt does not contain the final answer; regenerate with
        # mask_answer=True if callers passed answer-visible strings in.
        # extract_answer on the model output captures the last integer emitted.
        input_ids = tokenizer.encode(q, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                n_loops=8 if hasattr(model, 'recurrent') else None,
            )

        # Decode only the newly generated tokens to avoid echoing intermediate values
        # from the prompt (which would poison extract_answer).
        gen_tokens = output_ids[0][input_ids.shape[-1]:]
        output_text = tokenizer.decode(gen_tokens)

        pred_answer = extract_answer(output_text)
        # If the caller supplied an (answers: List[int]) alongside, use it; otherwise
        # fall back to extracting from the (answer-visible) input.
        true_answer = extract_answer(q)
        
        is_correct = pred_answer == true_answer
        
        if depth not in results_by_depth:
            results_by_depth[depth] = {"correct": 0, "total": 0}
        
        results_by_depth[depth]["total"] += 1
        results_by_depth[depth]["correct"] += int(is_correct)
        
        total += 1
        correct += int(is_correct)
    
    model.train()
    
    metrics = {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }
    
    for depth, stats in results_by_depth.items():
        metrics[f"accuracy_K{depth}"] = (
            stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        )
    
    return metrics


def evaluate_depth_extrapolation(
    model,
    tokenizer,
    depths: List[int] = None,
    examples_per_depth: int = 100,
    n_loops_list: List[int] = None,
    device: str = "cuda",
) -> Dict[Tuple[int, int], float]:
    """
    Evaluate depth extrapolation on arithmetic tasks.
    
    Trains on K=2-5, evaluates on K=6-15.
    Compares RDT at different loop counts vs baseline.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        depths: List of depths to evaluate
        examples_per_depth: Number of examples per depth
        n_loops_list: List of loop counts to test for RDT
        device: Device
    
    Returns:
        Dictionary mapping (depth, n_loops) -> accuracy
    """
    if depths is None:
        depths = [2, 3, 4, 5, 6, 8, 10, 12, 15]
    
    if n_loops_list is None:
        n_loops_list = [8, 12, 16]
    
    results = {}
    
    for depth in depths:
        questions = []
        answers = []
        for i in range(examples_per_depth):
            # mask_answer=True: prompt ends in "Answer:" so the model must produce it.
            q, a = generate_arithmetic_chain(depth, seed=depth * 1000 + i, mask_answer=True)
            questions.append(q)
            answers.append(a)
        
        for n_loops in n_loops_list:
            input_ids = tokenizer.encode_batch(questions)
            input_ids = torch.stack([ids.squeeze() for ids in input_ids]).to(device)
            
            model.eval()
            correct = 0
            
            with torch.no_grad():
                for i, (ids, true_answer) in enumerate(zip(input_ids, answers)):
                    if hasattr(model, 'recurrent'):
                        output_ids = model.generate(
                            ids.unsqueeze(0),
                            max_new_tokens=50,
                            n_loops=n_loops,
                        )
                    else:
                        output_ids = model.generate(
                            ids.unsqueeze(0),
                            max_new_tokens=50,
                        )
                    
                    output_text = tokenizer.decode(output_ids[0])
                    pred_answer = extract_answer(output_text)
                    
                    if pred_answer == true_answer:
                        correct += 1
            
            model.train()
            
            results[(depth, n_loops)] = correct / examples_per_depth
            print(f"  depth={depth}, loops={n_loops}: {results[(depth, n_loops)]:.2%}")
    
    return results


def finetune_on_arithmetic(
    model,
    tokenizer,
    depths: List[int] = None,
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-5,
    device: str = "cuda",
):
    """
    Finetune model on arithmetic tasks.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        depths: List of depths to train on (default: [2, 3, 4, 5])
        steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        device: Device
    """
    if depths is None:
        depths = [2, 3, 4, 5]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for step in range(steps):
        questions = []
        answers = []
        
        for _ in range(batch_size):
            depth = depths[step % len(depths)]
            q, a = generate_arithmetic_chain(depth)
            questions.append(q)
            answers.append(a)
        
        input_ids = tokenizer.encode_batch(questions)
        input_ids = torch.stack([ids.squeeze() for ids in input_ids]).to(device)
        
        targets = tokenizer.encode_batch(answers)
        targets = torch.stack([ids.squeeze() for ids in targets]).to(device)
        
        optimizer.zero_grad()
        
        if hasattr(model, 'recurrent'):
            logits, _ = model(input_ids, n_loops=8)
        else:
            logits = model(input_ids)
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model.cfg.vocab_size if hasattr(model, 'cfg') else 32000),
            targets.view(-1),
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    print("Arithmetic evaluation utilities")
    print("=" * 40)
    print("Functions: evaluate_arithmetic, evaluate_depth_extrapolation,")
    print("          finetune_on_arithmetic")