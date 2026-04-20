#!/usr/bin/env python3
"""
Checkpoint save/resume for the OpenMythos training harness.

Saves:
    - model state dict (rank-0 only under DDP/FSDP)
    - both optimizer state dicts (Muon + AdamW)
    - both scheduler state dicts
    - curriculum state dict
    - RNG states for torch, cuda, numpy, python random
    - step, tokens_seen, n_loops, loss_history
    - MythosConfig used to construct the model
    - wandb run id (for resume='must')
"""

import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler],
    curriculum,
    step: int,
    n_loops: int,
    loss_history: list,
    tokens_seen: int,
    config,
    wandb_run_id: Optional[str],
    checkpoint_dir: str,
    checkpoint_name: Optional[str] = None,
) -> Optional[str]:
    """Save a full training checkpoint. Returns path, or None if not rank 0."""
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_distributed and torch.distributed.get_rank() != 0:
        return None

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint_name or f"checkpoint_step_{step}.pt"
    path = ckpt_dir / name

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [o.state_dict() for o in optimizers],
        "scheduler_state_dicts": [s.state_dict() for s in schedulers],
        "curriculum_state_dict": curriculum.state_dict() if curriculum is not None else None,
        "step": step,
        "n_loops": n_loops,
        "loss_history": loss_history,
        "tokens_seen": tokens_seen,
        "config": config,
        "wandb_run_id": wandb_run_id,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")
    return str(path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler],
    curriculum,
    device: str = "cuda",
) -> dict:
    """
    Load a checkpoint and restore full training state. Returns metadata dict
    with keys: step, n_loops, loss_history, tokens_seen, wandb_run_id, config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False is required because the checkpoint pickles MythosConfig
    # and the various RNG state blobs. Only load checkpoints from trusted sources.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    for opt, sd in zip(optimizers, ckpt["optimizer_state_dicts"]):
        opt.load_state_dict(sd)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    for sch, sd in zip(schedulers, ckpt["scheduler_state_dicts"]):
        sch.load_state_dict(sd)

    if curriculum is not None and ckpt.get("curriculum_state_dict") is not None:
        curriculum.load_state_dict(ckpt["curriculum_state_dict"])

    rng = ckpt.get("rng_state", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])

    meta = {
        "step": ckpt.get("step", 0),
        "n_loops": ckpt.get("n_loops", 8),
        "loss_history": ckpt.get("loss_history", []),
        "tokens_seen": ckpt.get("tokens_seen", 0),
        "wandb_run_id": ckpt.get("wandb_run_id"),
        "config": ckpt.get("config"),
    }
    print(
        f"Checkpoint loaded: {path}\n"
        f"  step={meta['step']} n_loops={meta['n_loops']} "
        f"tokens_seen={meta['tokens_seen']:,} "
        f"loss_history_len={len(meta['loss_history'])}"
    )
    return meta


def rotate_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    preserve_names: Sequence[str] = ("final_checkpoint.pt",),
) -> List[str]:
    """
    Delete old step-numbered checkpoints, keeping only the most recent
    `keep_last_n`. Named checkpoints in `preserve_names` are never deleted.

    Returns the list of deleted paths.
    """
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    ckpts = sorted(
        d.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    to_delete = ckpts[:-keep_last_n] if len(ckpts) > keep_last_n else []
    deleted = []
    for p in to_delete:
        if p.name in preserve_names:
            continue
        p.unlink()
        deleted.append(str(p))
    return deleted


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    ckpts = sorted(
        d.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    return [
        str(p)
        for p in sorted(
            d.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
    ]
