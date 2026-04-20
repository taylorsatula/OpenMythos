#!/usr/bin/env python3
"""
OpenMythos training harness.

Entry points:
    train_rdt(...)           — recurrent-depth transformer
    train_dense_baseline(...) — FLOP-matched dense baseline

Optimizer: Muon (2D transformer matrices + MoE experts) + AdamW (embeddings,
norms, 1D tensors), four groups with the recurrent-group LR halved in both.

Step unit: one `optimizer.step()` (i.e. one gradient-accumulated update).
All intervals (total_steps, warmup_steps, eval_interval, checkpoint_interval)
are in optimizer-update units.
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from open_mythos import MythosConfig, OpenMythos
from open_mythos.baseline import DenseTransformer, dense_config_from_rdt
from open_mythos.tokenizer import MythosTokenizer

from training.checkpointing import (
    get_latest_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from training.curriculum import CurriculumScheduler
from training.losses import composite_loss_rdt
from training.muon import Muon
from utils.logging import (
    finish_run,
    log_eval_metrics,
    log_training_step,
    setup_wandb,
)


# ---------------------------------------------------------------------------
# Parameter partitioning
# ---------------------------------------------------------------------------


RECURRENT_KEYS = ("recurrent.", "injection.", "act.", "lora.")


def _is_recurrent(name: str) -> bool:
    return any(k in name for k in RECURRENT_KEYS)


def build_optimizers(
    model: OpenMythos,
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    wd_default: float = 0.1,
    wd_recurrent: float = 0.05,
) -> Tuple[Muon, torch.optim.AdamW, dict]:
    """
    Partition parameters into four groups and build Muon + AdamW optimizers.

    Returns:
        (muon, adamw, groups_dict) where groups_dict maps group name
        ('muon_default', 'muon_recurrent', 'adamw_default', 'adamw_recurrent')
        to a list of (name, param) pairs — useful for per-group logging.
    """
    muon_default: List[Tuple[str, nn.Parameter]] = []
    muon_recurrent: List[Tuple[str, nn.Parameter]] = []
    adamw_default: List[Tuple[str, nn.Parameter]] = []
    adamw_recurrent: List[Tuple[str, nn.Parameter]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_muon = model.muon_param_predicate(name, param)
        is_recurrent = _is_recurrent(name)
        if is_muon:
            (muon_recurrent if is_recurrent else muon_default).append((name, param))
        else:
            (adamw_recurrent if is_recurrent else adamw_default).append((name, param))

    muon = Muon(
        [
            {"params": [p for _, p in muon_default], "lr": lr_muon, "weight_decay": wd_default},
            {"params": [p for _, p in muon_recurrent], "lr": lr_muon * 0.5, "weight_decay": wd_recurrent},
        ],
        momentum=0.95,
        nesterov=True,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": [p for _, p in adamw_default], "lr": lr_adamw, "weight_decay": wd_default},
            {"params": [p for _, p in adamw_recurrent], "lr": lr_adamw * 0.5, "weight_decay": wd_recurrent},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    groups = {
        "muon_default": muon_default,
        "muon_recurrent": muon_recurrent,
        "adamw_default": adamw_default,
        "adamw_recurrent": adamw_recurrent,
    }
    return muon, adamw, groups


def build_schedulers(
    muon: torch.optim.Optimizer,
    adamw: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
):
    """Shared warmup-then-cosine schedule across both optimizers."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return (
        torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda),
        torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda),
    )


# ---------------------------------------------------------------------------
# Gradient norm helpers
# ---------------------------------------------------------------------------


def _per_group_grad_norm(params: List[Tuple[str, nn.Parameter]]) -> float:
    """L2 norm across a group's gradient tensors. Does not modify grads."""
    tensors = [p.grad for _, p in params if p.grad is not None]
    if not tensors:
        return 0.0
    # torch.nn.utils.clip_grad_norm_ with max_norm=inf computes norm without clipping
    total = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(t.detach()) for t in tensors])
    )
    return float(total.item())


# ---------------------------------------------------------------------------
# Validation perplexity (fast, called every eval_interval)
# ---------------------------------------------------------------------------


def _evaluate_perplexity(
    model,
    val_iter,
    vocab_size: int,
    n_loops: int,
    device: str,
    num_batches: int = 20,
) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(val_iter):
            if i >= num_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids, n_loops=n_loops)
                logits = out[0] if isinstance(out, tuple) else out
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = targets[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_targets.view(-1),
                )
            total += loss.item()
            count += 1
    if was_training:
        model.train()
    if count == 0:
        return float("inf")
    return math.exp(total / count)


# ---------------------------------------------------------------------------
# RDT training loop
# ---------------------------------------------------------------------------


def train_rdt(
    config: MythosConfig,
    *,
    output_dir: str = "outputs/rdt_1.5b",
    data_dir: str = "data/tokenized_shards",
    total_steps: int = 29000,          # optimizer updates
    warmup_steps: int = 2000,          # optimizer updates
    grad_accum: int = 8,
    micro_batch: int = 32,
    max_seq_len: int = 2048,
    max_loops_train: int = 8,
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    grad_clip: float = 1.0,
    eval_interval: int = 500,
    checkpoint_interval: int = 2500,
    log_interval: int = 10,
    wandb_project: str = "openmythos-training",
    resume_from: Optional[str] = None,
    device: str = "cuda",
    use_torch_compile: bool = True,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_path / "checkpoints"

    # 1. Tokenizer → vocab_size override
    tokenizer = MythosTokenizer()
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = max_seq_len

    # 2. Model
    model = OpenMythos(config).to(device)
    if use_torch_compile and device.startswith("cuda"):
        # Compile Prelude and Coda blocks individually. Do NOT compile the
        # recurrent block — dynamic n_loops and Python-level MoE dispatch
        # trigger recompiles or fallbacks.
        for i in range(len(model.prelude)):
            model.prelude[i] = torch.compile(model.prelude[i], mode="reduce-overhead")
        for i in range(len(model.coda)):
            model.coda[i] = torch.compile(model.coda[i], mode="reduce-overhead")

    muon, adamw, groups = build_optimizers(model, lr_muon=lr_muon, lr_adamw=lr_adamw)
    sched_muon, sched_adamw = build_schedulers(muon, adamw, warmup_steps, total_steps)
    curriculum = CurriculumScheduler(
        total_steps=total_steps,
        max_loops_train=max_loops_train,
        max_loop_iters=config.max_loop_iters,
    )

    start_step = 0
    loss_history: list = []
    tokens_seen = 0
    wandb_resume_id: Optional[str] = None

    if resume_from:
        meta = load_checkpoint(resume_from, model, [muon, adamw], [sched_muon, sched_adamw],
                               curriculum, device)
        start_step = meta["step"]
        loss_history = meta["loss_history"]
        tokens_seen = meta["tokens_seen"]
        wandb_resume_id = meta.get("wandb_run_id")

    run = setup_wandb(
        project=wandb_project,
        name=f"rdt_{int(time.time())}",
        config={
            "model": "rdt_1.5b",
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "max_seq_len": max_seq_len,
            "max_loops_train": max_loops_train,
            "lr_muon": lr_muon,
            "lr_adamw": lr_adamw,
            "vocab_size": config.vocab_size,
            "optimizer": "muon+adamw",
        },
        resume_id=wandb_resume_id,
    )

    # 3. Data
    from data.dataloader import create_dataloader

    def _fresh_train_iter():
        return iter(create_dataloader(data_dir, micro_batch, max_seq_len, split="train"))

    try:
        train_iter = _fresh_train_iter()
    except FileNotFoundError as e:
        print(f"ERROR: training data not found at {data_dir}. "
              f"Run `python -m data.prepare_data` first.")
        raise

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining RDT")
    print(f"  Parameters:          {n_params/1e9:.3f}B")
    print(f"  Vocab size:          {config.vocab_size:,}")
    print(f"  Total steps:         {total_steps:,} (optimizer updates)")
    print(f"  Warmup steps:        {warmup_steps:,}")
    print(f"  micro_batch:         {micro_batch}")
    print(f"  grad_accum:          {grad_accum}")
    print(f"  tokens/step:         {micro_batch*grad_accum*max_seq_len:,}")
    print(f"  target tokens:       {total_steps*micro_batch*grad_accum*max_seq_len/1e9:.1f}B")
    print()

    model.train()
    step = start_step
    while step < total_steps:
        n_loops = curriculum.step(step)
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        loss_accum = 0.0
        last_loss_dict = {}
        last_ponder_cost: Optional[torch.Tensor] = None
        for _ in range(grad_accum):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = _fresh_train_iter()
                input_ids, targets = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tokens_seen += input_ids.numel()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ponder_cost = model(input_ids, n_loops=n_loops)
                loss, loss_dict = composite_loss_rdt(
                    model, logits, ponder_cost, targets, config.vocab_size
                )
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss_dict["total"]
            last_loss_dict = loss_dict
            last_ponder_cost = ponder_cost.detach()

        loss_accum /= grad_accum  # report the average

        # Per-group grad norms (pre-clip) for diagnostic logging
        grad_norms = {name: _per_group_grad_norm(params) for name, params in groups.items()}

        # Global clip (applies to both Muon and AdamW params)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()

        muon.step()
        adamw.step()
        sched_muon.step()
        sched_adamw.step()
        step += 1

        # Spectral-radius safety gate
        with torch.no_grad():
            spectral = model.recurrent.injection.get_A().abs().max().item()
        if spectral >= 1.0:
            raise RuntimeError(
                f"LTI stability violated: ρ(A)={spectral:.6f} at step {step}. "
                f"This should be architecturally impossible."
            )

        loss_history.append(loss_accum)
        if len(loss_history) > 200:
            loss_history = loss_history[-200:]

        if step % log_interval == 0:
            lrs = {
                "muon_default": muon.param_groups[0]["lr"],
                "muon_recurrent": muon.param_groups[1]["lr"],
                "adamw_default": adamw.param_groups[0]["lr"],
                "adamw_recurrent": adamw.param_groups[1]["lr"],
            }
            print(
                f"step {step:>6}/{total_steps} | "
                f"loss {last_loss_dict['total']:.4f} | "
                f"lm {last_loss_dict['lm']:.4f} | "
                f"n_loops {n_loops} | "
                f"grad {total_grad_norm:.2f} | "
                f"ρ(A) {spectral:.4f} | "
                f"tokens {tokens_seen/1e9:.2f}B"
            )
            log_training_step(
                run, step, last_loss_dict, grad_norms, total_grad_norm, lrs,
                n_loops, tokens_seen, spectral, model, ponder_cost=last_ponder_cost,
            )

        if step % eval_interval == 0:
            val_iter = iter(create_dataloader(data_dir, micro_batch, max_seq_len, split="val"))
            val_ppl = _evaluate_perplexity(model, val_iter, config.vocab_size, n_loops, device)
            log_eval_metrics(run, step, {f"perplexity_loop{n_loops}": val_ppl})
            print(f"  [eval] step {step} perplexity@loop{n_loops} = {val_ppl:.2f}")

        if step % checkpoint_interval == 0:
            save_checkpoint(
                model, [muon, adamw], [sched_muon, sched_adamw], curriculum,
                step, n_loops, loss_history, tokens_seen, config,
                run.id if run is not None else None,
                checkpoint_dir=str(ckpt_dir),
            )
            rotate_checkpoints(str(ckpt_dir), keep_last_n=5)

    save_checkpoint(
        model, [muon, adamw], [sched_muon, sched_adamw], curriculum,
        step, n_loops, loss_history, tokens_seen, config,
        run.id if run is not None else None,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_name="final_checkpoint.pt",
    )
    finish_run(run)

    print(f"\nTraining complete.")
    print(f"  final step:    {step}")
    print(f"  total tokens:  {tokens_seen:,}")
    return model, {"step": step, "tokens_seen": tokens_seen, "loss_history": loss_history}


# ---------------------------------------------------------------------------
# Dense baseline
# ---------------------------------------------------------------------------


def train_dense_baseline(
    cfg,
    *,
    output_dir: str = "outputs/dense_baseline",
    data_dir: str = "data/tokenized_shards",
    total_steps: int = 29000,
    warmup_steps: int = 2000,
    grad_accum: int = 8,
    micro_batch: int = 32,
    max_seq_len: int = 2048,
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    grad_clip: float = 1.0,
    eval_interval: int = 500,
    checkpoint_interval: int = 2500,
    log_interval: int = 10,
    wandb_project: str = "openmythos-training",
    resume_from: Optional[str] = None,
    device: str = "cuda",
    use_torch_compile: bool = True,
):
    """
    FLOP-matched dense baseline training. Uses the same Muon+AdamW hybrid as the
    RDT (Muon on 2D transformer matrices, AdamW on embeddings/norms), but a
    single non-recurrent group per optimizer (no 0.5× LR split).
    """
    from open_mythos.baseline import DenseConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_path / "checkpoints"

    tokenizer = MythosTokenizer()
    if isinstance(cfg, MythosConfig):
        config = dense_config_from_rdt(cfg, n_layers=10)
    else:
        config = cfg
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = max_seq_len

    model = DenseTransformer(config).to(device)
    if use_torch_compile and device.startswith("cuda"):
        for i in range(len(model.blocks)):
            model.blocks[i] = torch.compile(model.blocks[i], mode="reduce-overhead")

    # Partition: 2D transformer matrices → Muon, everything else → AdamW.
    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim == 2 and "embed" not in name and "head.weight" not in name
                and ".norm" not in name):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon = Muon(
        [{"params": muon_params, "lr": lr_muon, "weight_decay": 0.1}],
        momentum=0.95, nesterov=True,
    )
    adamw = torch.optim.AdamW(
        [{"params": adamw_params, "lr": lr_adamw, "weight_decay": 0.1}],
        betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available(),
    )
    sched_muon, sched_adamw = build_schedulers(muon, adamw, warmup_steps, total_steps)

    from data.dataloader import create_dataloader

    def _fresh_train_iter():
        return iter(create_dataloader(data_dir, micro_batch, max_seq_len, split="train"))

    train_iter = _fresh_train_iter()

    start_step, loss_history, tokens_seen, wandb_resume_id = 0, [], 0, None
    if resume_from:
        meta = load_checkpoint(resume_from, model, [muon, adamw], [sched_muon, sched_adamw],
                               None, device)
        start_step = meta["step"]
        loss_history = meta["loss_history"]
        tokens_seen = meta["tokens_seen"]
        wandb_resume_id = meta.get("wandb_run_id")

    run = setup_wandb(
        project=wandb_project,
        name=f"baseline_{int(time.time())}",
        config={
            "model": "dense_baseline",
            "n_layers": config.n_layers,
            "total_steps": total_steps,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "max_seq_len": max_seq_len,
            "lr_muon": lr_muon,
            "lr_adamw": lr_adamw,
            "vocab_size": config.vocab_size,
            "optimizer": "muon+adamw",
        },
        resume_id=wandb_resume_id,
    )

    print(f"\nTraining Dense Baseline ({config.n_layers} layers)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    model.train()
    step = start_step
    while step < total_steps:
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        loss_val = 0.0
        for _ in range(grad_accum):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = _fresh_train_iter()
                input_ids, targets = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tokens_seen += input_ids.numel()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = targets[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, config.vocab_size), shift_targets.view(-1),
                )
                loss = loss / grad_accum
            loss.backward()
            loss_val += loss.item()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        muon.step()
        adamw.step()
        sched_muon.step()
        sched_adamw.step()
        step += 1

        loss_history.append(loss_val)
        if len(loss_history) > 200:
            loss_history = loss_history[-200:]

        if step % log_interval == 0:
            print(f"step {step:>6}/{total_steps} | loss {loss_val:.4f} | "
                  f"grad {total_grad_norm:.2f} | tokens {tokens_seen/1e9:.2f}B")
            if run is not None:
                run.log({
                    "train/loss": loss_val,
                    "train/grad_norm/total": total_grad_norm,
                    "train/lr/muon": muon.param_groups[0]["lr"],
                    "train/lr/adamw": adamw.param_groups[0]["lr"],
                    "train/tokens_seen": tokens_seen,
                }, step=step)

        if step % checkpoint_interval == 0:
            save_checkpoint(
                model, [muon, adamw], [sched_muon, sched_adamw], None,
                step, 0, loss_history, tokens_seen, config,
                run.id if run is not None else None,
                checkpoint_dir=str(ckpt_dir),
            )
            rotate_checkpoints(str(ckpt_dir), keep_last_n=5)

    save_checkpoint(
        model, [muon, adamw], [sched_muon, sched_adamw], None,
        step, 0, loss_history, tokens_seen, config,
        run.id if run is not None else None,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_name="final_checkpoint.pt",
    )
    finish_run(run)
    return model, {"step": step, "tokens_seen": tokens_seen}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train OpenMythos RDT or dense baseline")
    parser.add_argument("--model", choices=["rdt", "dense"], default="rdt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default="data/tokenized_shards")
    parser.add_argument("--total_steps", type=int, default=29000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--micro_batch", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_loops", type=int, default=8)
    parser.add_argument("--lr_muon", type=float, default=0.02)
    parser.add_argument("--lr_adamw", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=2500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    use_compile = not args.no_compile

    if args.model == "rdt":
        from configs.rdt_1_5b import RDT_1_5B_CONFIG
        train_rdt(
            RDT_1_5B_CONFIG,
            output_dir=f"{args.output_dir}/rdt_1.5b",
            data_dir=args.data_dir,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            grad_accum=args.grad_accum,
            micro_batch=args.micro_batch,
            max_seq_len=args.max_seq_len,
            max_loops_train=args.max_loops,
            lr_muon=args.lr_muon,
            lr_adamw=args.lr_adamw,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume,
            device=args.device,
            use_torch_compile=use_compile,
        )
    else:
        from configs.baseline import DENSE_BASELINE_CONFIG
        train_dense_baseline(
            DENSE_BASELINE_CONFIG,
            output_dir=f"{args.output_dir}/dense_baseline",
            data_dir=args.data_dir,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            grad_accum=args.grad_accum,
            micro_batch=args.micro_batch,
            max_seq_len=args.max_seq_len,
            lr_muon=args.lr_muon,
            lr_adamw=args.lr_adamw,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume,
            device=args.device,
            use_torch_compile=use_compile,
        )


if __name__ == "__main__":
    main()
