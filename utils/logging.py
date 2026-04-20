#!/usr/bin/env python3
"""Weights & Biases logging utilities for the OpenMythos training harness."""

import os
from typing import Mapping, Optional

import torch


def setup_wandb(
    project: str = "openmythos-training",
    name: Optional[str] = None,
    config: Optional[dict] = None,
    entity: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    """
    Initialize a wandb run. Returns the run object, or None if wandb is unavailable
    or WANDB_API_KEY is unset (in which case all subsequent log_* calls are no-ops).

    If `resume_id` is provided, the run is resumed via wandb's resume='must' mode.
    """
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed; run tracking disabled.")
        return None

    if not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set; run tracking disabled.")
        return None

    init_kwargs = dict(project=project, name=name, config=config, entity=entity)
    if resume_id is not None:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "must"

    wandb.init(**init_kwargs)
    return wandb.run


def _count_dead_experts(model, threshold_frac: float = 0.001) -> int:
    """
    Fraction of experts that received fewer than `threshold_frac` of routing
    slots in the most recent forward pass. Reads `last_topk_idx` cached by
    MoEFFN.forward.
    """
    try:
        ffn = model.recurrent.block.ffn
        topk = ffn.last_topk_idx
    except AttributeError:
        return 0
    if topk is None:
        return 0
    n_experts = ffn.n_experts
    counts = torch.bincount(topk.view(-1), minlength=n_experts).float()
    total = counts.sum().clamp(min=1)
    frac = counts / total
    return int((frac < threshold_frac).sum().item())


def _mean_halt_step(ponder_cost: Optional[torch.Tensor]) -> float:
    """
    Mean halt step across positions from a ponder_cost tensor (B, T).
    Ponder cost ≈ expected halt step under ACT.
    """
    if ponder_cost is None:
        return 0.0
    return float(ponder_cost.detach().float().mean().item())


def log_training_step(
    run,
    step: int,
    loss_dict: dict,
    grad_norms_per_group: Mapping[str, float],
    total_grad_norm: float,
    lrs_per_group: Mapping[str, float],
    n_loops: int,
    tokens_seen: int,
    spectral_radius_max: float,
    model,
    ponder_cost: Optional[torch.Tensor] = None,
):
    """Log per-optimizer-step training metrics to wandb."""
    if run is None:
        return
    try:
        import wandb  # noqa: F401
    except ImportError:
        return

    dead_count = _count_dead_experts(model)
    halt_mean = _mean_halt_step(ponder_cost)

    metrics = {
        "train/loss": loss_dict.get("total", 0.0),
        "train/lm_loss": loss_dict.get("lm", 0.0),
        "train/moe_aux_loss": loss_dict.get("moe_aux", 0.0),
        "train/moe_z_loss": loss_dict.get("moe_z", 0.0),
        "train/act_ponder_cost": loss_dict.get("act_ponder", 0.0),
        "train/grad_norm/total": total_grad_norm,
        "train/n_loops": n_loops,
        "train/tokens_seen": tokens_seen,
        "train/spectral_radius_max": spectral_radius_max,
        "train/moe_dead_expert_count": dead_count,
        "train/act_halt_mean": halt_mean,
    }
    for g, v in grad_norms_per_group.items():
        metrics[f"train/grad_norm/{g}"] = v
    for g, v in lrs_per_group.items():
        metrics[f"train/lr/{g}"] = v

    run.log(metrics, step=step)


def log_eval_metrics(run, step: int, metrics: dict, prefix: str = "eval"):
    """Log evaluation metrics with a prefix."""
    if run is None:
        return
    try:
        import wandb  # noqa: F401
    except ImportError:
        return
    run.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)


def log_depth_extrapolation(run, step: int, perplexities: dict):
    """Log perplexity at multiple loop counts."""
    if run is None:
        return
    try:
        import wandb  # noqa: F401
    except ImportError:
        return
    metrics = {f"eval/perplexity_loop{loops}": ppl for loops, ppl in perplexities.items()}
    base = perplexities.get(8, float("inf"))
    metrics["eval/depth_extrapolation"] = any(
        perplexities.get(k, float("inf")) < base for k in (12, 16)
    )
    run.log(metrics, step=step)


def log_act_profile(
    run,
    step: int,
    mean_halt_step: float,
    halt_distribution: Optional[dict] = None,
    expert_entropy: Optional[float] = None,
    dead_expert_count: Optional[int] = None,
):
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return
    metrics = {"eval/act_mean_halt_step": mean_halt_step}
    if expert_entropy is not None:
        metrics["eval/expert_entropy"] = expert_entropy
    if dead_expert_count is not None:
        metrics["eval/dead_expert_count"] = dead_expert_count
    run.log(metrics, step=step)
    if halt_distribution:
        run.log({"eval/halt_distribution": wandb.Histogram(halt_distribution)}, step=step)


def finish_run(run):
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except ImportError:
        pass
