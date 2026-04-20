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
from typing import Any, Dict, List, Optional, Tuple

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
from tools.mythos_admin_runtime import AdminRuntime
from utils.logging import (
    cost_metrics,
    expected_lr,
    finish_run,
    log_act_profile,
    log_decoded_sample,
    log_depth_extrapolation,
    log_eval_metrics,
    log_expert_norms,
    log_synthetic_arithmetic,
    log_token_id_histogram,
    log_training_step,
    microbatch_loss_cv,
    per_layer_grad_norms,
    run_generation_probe,
    setup_wandb,
    system_metrics,
    timing_metrics,
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
# Diagnostic helpers
# ---------------------------------------------------------------------------


def _set_attn_diagnostics(model, enabled: bool) -> None:
    """Flip the attention-logit sampling flag on every attention module that
    supports it. GQAttention carries a `diagnostics` attribute; when True the
    forward pass computes an extra QK matmul under no_grad to report the
    pre-softmax abs-max. Cost is one matmul per attention block per forward,
    so we gate this to log-interval steps only."""
    try:
        for block in model.prelude:
            if hasattr(block.attn, "diagnostics"):
                block.attn.diagnostics = enabled
        if hasattr(model.recurrent.block.attn, "diagnostics"):
            model.recurrent.block.attn.diagnostics = enabled
        for block in model.coda:
            if hasattr(block.attn, "diagnostics"):
                block.attn.diagnostics = enabled
    except AttributeError:
        pass


def _evaluate_synthetic_arithmetic(
    model,
    tokenizer,
    device: str,
    depths=(2, 4, 8, 12),
    loop_counts=(8, 12, 16),
    n_per_depth: int = 20,
) -> dict:
    """
    Run a small synthetic multi-hop arithmetic probe. Returns
    {(depth, n_loops): exact_match_accuracy}. Expensive enough to throttle
    (runs model.generate per prompt), so called at checkpoint cadence.
    """
    from data.synthetic import generate_arithmetic_chain

    import re
    def _extract_answer(text: str):
        nums = re.findall(r"-?\d+", text)
        return int(nums[-1]) if nums else None

    was_training = model.training
    model.eval()
    results: dict = {}
    try:
        with torch.no_grad():
            for d in depths:
                prompts = []
                answers = []
                for i in range(n_per_depth):
                    p, a = generate_arithmetic_chain(d, seed=d * 9973 + i, mask_answer=True)
                    prompts.append(p)
                    answers.append(a)
                for nl in loop_counts:
                    correct = 0
                    for p, a in zip(prompts, answers):
                        ids = torch.tensor(tokenizer.encode(p), dtype=torch.long,
                                           device=device).unsqueeze(0)
                        out = model.generate(ids, max_new_tokens=12, n_loops=nl,
                                             temperature=1.0, top_k=0)
                        gen = tokenizer.decode(out[0][ids.shape[-1]:].tolist())
                        if _extract_answer(gen) == a:
                            correct += 1
                    results[(d, nl)] = correct / max(1, len(prompts))
    finally:
        if was_training:
            model.train()
    return results


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
    gpu_hourly_rate: float = 3.50,
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
    resumed_admin_state: Optional[dict] = None

    if resume_from:
        meta = load_checkpoint(resume_from, model, [muon, adamw], [sched_muon, sched_adamw],
                               curriculum, device)
        start_step = meta["step"]
        loss_history = meta["loss_history"]
        tokens_seen = meta["tokens_seen"]
        wandb_resume_id = meta.get("wandb_run_id")
        resumed_admin_state = meta.get("admin_state")

    # 3. Admin runtime — owns control plane (config/commands/status/audit)
    # and local event streams (metrics/incidents/generations). Must exist
    # before the loop so commands can land from step 0.
    run_start_wall = time.time()
    admin = AdminRuntime(
        output_path,
        run_start_wall=run_start_wall,
        gpu_hourly_rate=gpu_hourly_rate,
        max_loop_iters=config.max_loop_iters,
    )
    admin.seed_config_if_missing({
        "grad_clip": grad_clip,
        "gpu_hourly_rate": gpu_hourly_rate,
    })
    admin.load_state(resumed_admin_state)
    admin.register_sigterm()
    fixed_prompts = admin.load_prompts()

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
    if run is not None:
        try:
            print(f"  wandb run URL: {run.get_url()}")
        except Exception:
            pass

    # 4. Data — request metadata for shard provenance.
    from data.dataloader import create_dataloader

    def _fresh_train_iter():
        return iter(create_dataloader(data_dir, micro_batch, max_seq_len,
                                       split="train", return_meta=True))

    try:
        train_iter = _fresh_train_iter()
    except FileNotFoundError:
        print(f"ERROR: training data not found at {data_dir}. "
              f"Run `python -m data.prepare_data` first.")
        raise

    # Structural invariants at training start. These must hold end-to-end;
    # if any of them changes later, something is very wrong and we'd rather
    # crash than keep training on a silently-corrupted model.
    model.assert_weight_tying()
    invariant_param_count = model.unique_param_count()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining RDT")
    print(f"  Unique params:       {invariant_param_count/1e9:.3f}B (tying-aware)")
    print(f"  Total params raw:    {n_params/1e9:.3f}B (double-counts tied head)")
    print(f"  Vocab size:          {config.vocab_size:,}")
    print(f"  Total steps:         {total_steps:,} (optimizer updates)")
    print(f"  Warmup steps:        {warmup_steps:,}")
    print(f"  micro_batch:         {micro_batch}")
    print(f"  grad_accum:          {grad_accum}")
    print(f"  tokens/step:         {micro_batch*grad_accum*max_seq_len:,}")
    print(f"  target tokens:       {total_steps*micro_batch*grad_accum*max_seq_len/1e9:.1f}B")
    print(f"  admin dir:           {admin.admin_dir}")
    print(f"  events dir:          {admin.events_dir}")
    print()

    # Intervals for the deeper diagnostic surfaces. See HANDOFF §7.1.
    expert_norm_interval = max(log_interval, 100)         # per-expert weight/momentum norms
    decoded_sample_interval = max(log_interval * 50, 500) # human-readable sample
    depth_eval_interval = max(eval_interval * 10, 5000)   # perplexity at 4/8/12/16 loops
    arithmetic_eval_interval = checkpoint_interval         # synthetic K in {2,4,8,12}

    def _do_checkpoint(step_val: int, n_loops_val: int,
                        name: Optional[str] = None) -> Optional[str]:
        """Disk-gated checkpoint save. Rotates aggressively when low on space."""
        if not admin.disk_ok_for_checkpoint(ckpt_dir):
            admin.emit_incident(
                "disk_low_ckpt_skipped", severity="CRIT", step=step_val,
                value=admin.disk_free_gb(ckpt_dir),
            )
            rotate_checkpoints(str(ckpt_dir), keep_last_n=2)
            return None
        model.assert_weight_tying()
        path = save_checkpoint(
            model, [muon, adamw], [sched_muon, sched_adamw], curriculum,
            step_val, n_loops_val, loss_history, tokens_seen, config,
            run.id if run is not None else None,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_name=name,
            admin_state=admin.snapshot_state(),
        )
        rotate_checkpoints(str(ckpt_dir), keep_last_n=5)
        return path

    model.train()
    step = start_step
    n_loops = 1  # placeholder; overwritten first iteration
    last_input_ids_sample: Optional[torch.Tensor] = None  # for periodic decoding
    while step < total_steps:
        # --- Admin: read config + dispatch pending commands at top of step ---
        cfg = admin.read_config(current_step=step)
        admin.process_pending_commands(executor={}, current_step=step)

        if admin.is_hard_stop():
            print(f"HARD STOP requested at step {step}; exiting without checkpoint.")
            finish_run(run)
            return model, {"step": step, "tokens_seen": tokens_seen,
                           "loss_history": loss_history}

        # Pause branch: heartbeat stays alive, commands still processed, no step.
        if cfg.pause_until_step is not None and step < cfg.pause_until_step:
            sys_now = system_metrics(str(ckpt_dir))
            cost_now = cost_metrics(run_start_wall, cfg.gpu_hourly_rate)
            admin.write_status(
                step=step, total_steps=total_steps, n_loops=n_loops,
                loss={"total": 0.0}, paused=True,
                pause_until_step=cfg.pause_until_step,
                gpu_mem_peak_gb=sys_now.get("gpu_mem_peak_gb", 0.0),
                disk_free_gb_ckpt=sys_now.get("disk_free_gb_ckpt", 0.0),
                wall_hours=cost_now["wall_hours"],
                estimated_cost_usd=cost_now["estimated_cost_usd"],
                max_loop_iters=config.max_loop_iters,
                config_applied_hash=admin.snapshot_state()["config_hash"],
            )
            time.sleep(1.0)
            if admin.should_stop():
                break
            continue

        n_loops = (cfg.n_loops_override
                   if cfg.n_loops_override is not None else curriculum.step(step))
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        # Toggle attention-logit sampling on the attention blocks only on
        # log-interval steps: SDPA fuses softmax with the matmul, so to see
        # pre-softmax magnitudes and compute entropy we redo QK. Zero cost
        # when the flag is off. Interval is admin-tunable.
        attn_diag_interval = cfg.diagnostic_intervals.get("attn_entropy", log_interval)
        sample_attn_this_step = ((step + 1) % attn_diag_interval == 0)
        _set_attn_diagnostics(model, sample_attn_this_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_step_start = time.perf_counter()
        loss_accum = 0.0
        last_loss_dict: Dict[str, Any] = {}
        last_ponder_cost: Optional[torch.Tensor] = None
        first_input_ids: Optional[torch.Tensor] = None
        first_shard_meta: Optional[dict] = None
        microbatch_losses: List[float] = []
        tokens_this_step = 0

        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = _fresh_train_iter()
                batch = next(train_iter)
            if len(batch) == 3:
                input_ids, targets, meta_list = batch
            else:
                input_ids, targets = batch
                meta_list = None
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tokens_seen += input_ids.numel()
            tokens_this_step += input_ids.numel()
            if first_input_ids is None:
                first_input_ids = input_ids
                first_shard_meta = meta_list[0] if meta_list else None

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ponder_cost = model(input_ids, n_loops=n_loops)
                loss, loss_dict = composite_loss_rdt(
                    model, logits, ponder_cost, targets, config.vocab_size,
                    coeffs=cfg.loss_coeffs,
                )
                loss = loss / grad_accum
            loss.backward()
            microbatch_losses.append(float(loss_dict["total"]))
            loss_accum += loss_dict["total"]
            last_loss_dict = loss_dict
            last_ponder_cost = ponder_cost.detach()

        loss_accum /= grad_accum  # report the average
        last_input_ids_sample = first_input_ids

        # --- NaN/Inf gate (pre-clip, pre-optimizer) ---
        nan_where = admin.check_nan_inf(last_loss_dict, model)
        if nan_where is not None:
            admin.emit_incident("nan_inf", severity="CRIT", step=step, value=nan_where)
            admin.enqueue_graceful_stop()
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            t_step_s = time.perf_counter() - t_step_start
            admin.update_step_time_rolling(t_step_s)
            admin.write_status(
                step=step, total_steps=total_steps, n_loops=n_loops,
                loss={"total": float("nan")}, nan_detected=True,
                max_loop_iters=config.max_loop_iters,
                config_applied_hash=admin.snapshot_state()["config_hash"],
            )
            break

        # Per-group grad norms (pre-clip) for diagnostic logging
        grad_norms = {name: _per_group_grad_norm(params) for name, params in groups.items()}
        per_layer_dict = per_layer_grad_norms(
            model.named_parameters(),
            prelude_layers=config.prelude_layers,
            coda_layers=config.coda_layers,
        )

        # Global clip (admin-tunable)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()

        muon.step()
        adamw.step()
        sched_muon.step()
        sched_adamw.step()

        # Apply admin LR multipliers post-scheduler, pre-next-step. Without
        # this, the multiplier would evaporate each iteration as the scheduler
        # overwrites param_groups[i]["lr"].
        muon.param_groups[0]["lr"] *= cfg.lr_mult.get("muon_default", 1.0)
        muon.param_groups[1]["lr"] *= cfg.lr_mult.get("muon_recurrent", 1.0)
        adamw.param_groups[0]["lr"] *= cfg.lr_mult.get("adamw_default", 1.0)
        adamw.param_groups[1]["lr"] *= cfg.lr_mult.get("adamw_recurrent", 1.0)

        # DeepSeek-V3 router-bias update: nudge per-expert router_bias toward
        # uniform load using counts accumulated across this step's forwards.
        # Update rate is admin-tunable (cfg.router_bias_update_rate).
        if cfg.router_bias_update:
            with torch.no_grad():
                ffn = model.recurrent.block.ffn
                counts = ffn._step_expert_counts
                total = counts.sum().clamp(min=1)
                target = total / ffn.n_experts
                imbalance = target - counts
                ffn.router_bias.add_(cfg.router_bias_update_rate * torch.sign(imbalance))
                ffn.reset_step_expert_counts()

        step += 1
        t_step_s = time.perf_counter() - t_step_start
        admin.note_step_advanced()
        admin.update_step_time_rolling(t_step_s)
        admin.update_loss_rolling(loss_accum)

        # --- Spectral-radius gates ---
        with torch.no_grad():
            spectral = model.recurrent.injection.get_A().abs().max().item()
        if spectral >= 1.0:
            admin.emit_incident("spectral_radius_ge_1", severity="CRIT",
                                 step=step, value=spectral)
            raise RuntimeError(
                f"LTI stability violated: ρ(A)={spectral:.6f} at step {step}. "
                f"This should be architecturally impossible."
            )
        if spectral >= cfg.spectral_warn_threshold:
            # Incident emission is unconditional on threshold crossing —
            # latency matters here, unlike the console print.
            admin.emit_incident("spectral_near_bound", severity="WARN",
                                 step=step, value=spectral,
                                 context={"threshold": cfg.spectral_warn_threshold})
            if step % log_interval == 0:
                print(f"  WARN: ρ(A)={spectral:.6f} ≥ "
                      f"{cfg.spectral_warn_threshold} at step {step}")

        # Structural invariant: unique-param count must not change.
        current_param_count = model.unique_param_count()
        if current_param_count != invariant_param_count:
            admin.emit_incident("invariant_param_count_changed", severity="CRIT",
                                 step=step, value=current_param_count,
                                 context={"expected": invariant_param_count})
            raise RuntimeError(
                f"Invariant violated: unique_param_count went from "
                f"{invariant_param_count:,} to {current_param_count:,} at step {step}. "
                f"Most likely cause: torch.compile broke weight tying. "
                f"Check head.weight vs embed.weight data_ptr."
            )

        loss_history.append(loss_accum)
        if len(loss_history) > 200:
            loss_history = loss_history[-200:]

        # --- Detectors (both use admin's rolling buffers) ---
        spike = admin.loss_spike(loss_accum)
        if spike is not None:
            admin.emit_incident("loss_spike", severity="WARN", step=step,
                                 value=spike[0], context={"baseline_median": spike[1]})
        regression = admin.step_time_regressed(t_step_s)
        if regression is not None:
            admin.emit_incident("step_time_regression", severity="WARN", step=step,
                                 value=regression[0],
                                 context={"baseline_median": regression[1]})

        # --- Per-step training metrics (wandb + local events.jsonl tee) ---
        if step % log_interval == 0:
            lrs = {
                "muon_default": muon.param_groups[0]["lr"],
                "muon_recurrent": muon.param_groups[1]["lr"],
                "adamw_default": adamw.param_groups[0]["lr"],
                "adamw_recurrent": adamw.param_groups[1]["lr"],
            }
            # Expected LR multiplied by admin LR multiplier so the desync
            # detector doesn't scream every time the Model tunes a group.
            exp_lrs = {
                "muon_default": (expected_lr(step, warmup_steps, total_steps, lr_muon)
                                  * cfg.lr_mult.get("muon_default", 1.0)),
                "muon_recurrent": (expected_lr(step, warmup_steps, total_steps, lr_muon * 0.5)
                                    * cfg.lr_mult.get("muon_recurrent", 1.0)),
                "adamw_default": (expected_lr(step, warmup_steps, total_steps, lr_adamw)
                                   * cfg.lr_mult.get("adamw_default", 1.0)),
                "adamw_recurrent": (expected_lr(step, warmup_steps, total_steps, lr_adamw * 0.5)
                                     * cfg.lr_mult.get("adamw_recurrent", 1.0)),
            }
            print(
                f"step {step:>6}/{total_steps} | "
                f"loss {last_loss_dict['total']:.4f} | "
                f"lm {last_loss_dict['lm']:.4f} | "
                f"n_loops {n_loops} | "
                f"grad {total_grad_norm:.2f} | "
                f"ρ(A) {spectral:.4f} | "
                f"t/step {t_step_s:.2f}s | "
                f"tokens {tokens_seen/1e9:.2f}B"
            )
            extra = {
                **timing_metrics(t_step_s, tokens_this_step),
                **system_metrics(str(ckpt_dir)),
                **cost_metrics(run_start_wall, cfg.gpu_hourly_rate),
                "microbatch_loss_cv": microbatch_loss_cv(microbatch_losses),
                "loss_rolling_mean_200": admin.loss_rolling_mean(),
            }

            metrics = log_training_step(
                run, step, last_loss_dict, grad_norms, total_grad_norm, lrs,
                n_loops, tokens_seen, spectral, model,
                ponder_cost=last_ponder_cost,
                expected_lrs_per_group=exp_lrs,
                invariant_param_count=invariant_param_count,
                loss_coeffs=cfg.loss_coeffs,
                lr_mults=cfg.lr_mult,
                per_layer_grad_norms_dict=per_layer_dict,
                extra_metrics=extra,
            )
            # Attach shard provenance to the admin stream only (wandb panels
            # don't like string metric values; admin JSONL is unconstrained).
            if first_shard_meta is not None:
                metrics = dict(metrics)
                metrics["first_shard"] = str(first_shard_meta.get("shard", ""))
                metrics["first_shard_row"] = int(first_shard_meta.get("row", 0))
            admin.log_metric(step, metrics)

        # Per-expert weight-norm & Muon-momentum diagnostics: O(n_experts)
        # tensor reductions, hence throttled.
        if step % expert_norm_interval == 0:
            log_expert_norms(run, step, model, muon_optim=muon)

        # Decoded-sample dump + token-id histogram: data-integrity catch-all.
        if step % decoded_sample_interval == 0 and last_input_ids_sample is not None:
            log_decoded_sample(run, step, last_input_ids_sample.cpu(), tokenizer)
            log_token_id_histogram(run, step, last_input_ids_sample.cpu(), config.vocab_size)

        # Fixed-prompt generation probe — latency excluded from step-time.
        gen_probe_interval = cfg.diagnostic_intervals.get("generation_probe", 500)
        if step % gen_probe_interval == 0 and fixed_prompts:
            try:
                probe_results = run_generation_probe(
                    model, tokenizer, fixed_prompts, device,
                    n_loops=n_loops, max_new_tokens=32,
                    temperature=1.0, top_k=50,
                )
                for r in probe_results:
                    admin.log_generation(
                        step, r["prompt"], r.get("output", ""),
                        r.get("n_loops", n_loops), r.get("temperature", 1.0),
                        r.get("top_k", 50), r.get("latency_s", 0.0),
                        source="panel",
                    )
            except Exception as e:
                admin.emit_incident("generation_probe_failed", severity="WARN",
                                     step=step, value=str(e))

        # --- Validation perplexity at current loop count (every eval_interval) ---
        if step % eval_interval == 0:
            val_iter = iter(create_dataloader(data_dir, micro_batch, max_seq_len,
                                               split="val"))
            val_ppl = _evaluate_perplexity(model, val_iter, config.vocab_size,
                                            n_loops, device)
            log_eval_metrics(run, step, {f"perplexity_loop{n_loops}": val_ppl})
            admin.log_metric(step, {f"eval/perplexity_loop{n_loops}": val_ppl,
                                     "kind": "eval"})
            admin.emit_incident("eval_perplexity", severity="INFO", step=step,
                                 value=val_ppl, context={"n_loops": n_loops})
            print(f"  [eval] step {step} perplexity@loop{n_loops} = {val_ppl:.2f}")

        # --- Depth-extrapolation probe ---
        if step % depth_eval_interval == 0:
            depth_ppls = {}
            for nl in (4, 8, 12, 16):
                vi = iter(create_dataloader(data_dir, micro_batch, max_seq_len,
                                             split="val"))
                depth_ppls[nl] = _evaluate_perplexity(model, vi, config.vocab_size,
                                                      nl, device, num_batches=8)
            log_depth_extrapolation(run, step, depth_ppls)
            admin.log_metric(step, {f"eval/perplexity_loop{k}": v
                                      for k, v in depth_ppls.items()})
            print(f"  [depth-eval] step {step} " +
                  "  ".join(f"loop{k}={v:.2f}" for k, v in depth_ppls.items()))

        # --- Synthetic multi-hop arithmetic probe ---
        if step % arithmetic_eval_interval == 0:
            try:
                acc = _evaluate_synthetic_arithmetic(model, tokenizer, device,
                                                      n_per_depth=20)
                log_synthetic_arithmetic(run, step, acc)
                admin.log_metric(step, {f"eval/arith/K{d}_loops{nl}": a
                                          for (d, nl), a in acc.items()})
                print(f"  [arith] step {step} " +
                      "  ".join(f"K{d}/L{nl}={a:.2f}" for (d, nl), a in acc.items()))
            except Exception as e:
                print(f"  [arith] eval skipped: {e}")

        # --- On-demand commands from admin queue (consumed post-step) ---
        if admin.pop_checkpoint_request():
            _do_checkpoint(step, n_loops)

        for req in admin.pop_eval_requests():
            nl = req["n_loops"]
            vi = iter(create_dataloader(data_dir, micro_batch, max_seq_len, split="val"))
            val_ppl = _evaluate_perplexity(model, vi, config.vocab_size, nl, device,
                                            num_batches=10)
            admin.log_metric(step, {f"eval/perplexity_loop{nl}": val_ppl,
                                     "source": "on_demand"})
            admin.emit_incident("eval_perplexity", severity="INFO", step=step,
                                 value=val_ppl,
                                 context={"n_loops": nl, "source": "on_demand"})

        if admin.pop_depth_eval_request():
            depth_ppls = {}
            for nl in (4, 8, 12, 16):
                vi = iter(create_dataloader(data_dir, micro_batch, max_seq_len,
                                             split="val"))
                depth_ppls[nl] = _evaluate_perplexity(model, vi, config.vocab_size,
                                                      nl, device, num_batches=8)
            log_depth_extrapolation(run, step, depth_ppls)
            admin.log_metric(step, {f"eval/perplexity_loop{k}": v
                                      for k, v in depth_ppls.items()})

        arith_req = admin.pop_arith_eval_request()
        if arith_req is not None:
            try:
                acc = _evaluate_synthetic_arithmetic(model, tokenizer, device,
                                                      n_per_depth=arith_req)
                log_synthetic_arithmetic(run, step, acc)
                admin.log_metric(step, {f"eval/arith/K{d}_loops{nl}": a
                                          for (d, nl), a in acc.items()})
            except Exception as e:
                admin.emit_incident("arith_eval_failed", severity="WARN", step=step,
                                     value=str(e))

        for gen_args in admin.pop_generate_requests():
            prompt = str(gen_args.get("prompt", ""))
            try:
                probe_results = run_generation_probe(
                    model, tokenizer, [prompt], device,
                    n_loops=int(gen_args.get("n_loops", n_loops)),
                    max_new_tokens=int(gen_args.get("max_new_tokens", 64)),
                    temperature=float(gen_args.get("temperature", 1.0)),
                    top_k=int(gen_args.get("top_k", 50)),
                )
                for r in probe_results:
                    admin.log_generation(
                        step, r["prompt"], r.get("output", ""),
                        r.get("n_loops", n_loops), r.get("temperature", 1.0),
                        r.get("top_k", 50), r.get("latency_s", 0.0),
                        source="command",
                    )
            except Exception as e:
                admin.emit_incident("on_demand_generate_failed", severity="WARN",
                                     step=step, value=str(e))

        for expert_id in admin.pop_reinit_expert_requests():
            try:
                ffn = model.recurrent.block.ffn
                if 0 <= expert_id < len(ffn.routed_experts):
                    expert = ffn.routed_experts[expert_id]
                    with torch.no_grad():
                        nn.init.normal_(expert.gate.weight, std=0.02)
                        nn.init.normal_(expert.up.weight, std=0.02)
                        nn.init.normal_(expert.down.weight, std=0.02)
                        # Depth-scale the residual output projection, matching
                        # the pattern from OpenMythos._init_weights.
                        training_max_loops = 8
                        effective_depth = (config.prelude_layers + training_max_loops
                                            + config.coda_layers)
                        depth_scale = 1.0 / math.sqrt(2.0 * effective_depth)
                        expert.down.weight.mul_(depth_scale)
                    admin.emit_incident("expert_reinit", severity="INFO", step=step,
                                         value=expert_id)
                else:
                    admin.emit_incident("expert_reinit_out_of_range", severity="WARN",
                                         step=step, value=expert_id)
            except Exception as e:
                admin.emit_incident("expert_reinit_failed", severity="WARN", step=step,
                                     value=str(e))

        if admin.pop_reset_router_bias_request():
            with torch.no_grad():
                model.recurrent.block.ffn.router_bias.zero_()
            admin.emit_incident("router_bias_reset", severity="INFO", step=step)

        # --- Periodic checkpoint ---
        if step % checkpoint_interval == 0:
            _do_checkpoint(step, n_loops)

        # --- Status snapshot + config_applied mirror (end of step) ---
        lrs_now = {
            "muon_default": muon.param_groups[0]["lr"],
            "muon_recurrent": muon.param_groups[1]["lr"],
            "adamw_default": adamw.param_groups[0]["lr"],
            "adamw_recurrent": adamw.param_groups[1]["lr"],
        }
        sys_now = system_metrics(str(ckpt_dir))
        cost_now = cost_metrics(run_start_wall, cfg.gpu_hourly_rate)
        admin.write_status(
            step=step, total_steps=total_steps, n_loops=n_loops,
            loss={
                "total": float(last_loss_dict.get("total", 0.0)),
                "lm": float(last_loss_dict.get("lm", 0.0)),
                "moe_aux": float(last_loss_dict.get("moe_aux", 0.0)),
                "moe_z": float(last_loss_dict.get("moe_z", 0.0)),
                "act_ponder": float(last_loss_dict.get("act_ponder", 0.0)),
            },
            loss_rolling_mean_200=admin.loss_rolling_mean(),
            grad_norm_total=total_grad_norm,
            spectral_radius=spectral,
            lrs=lrs_now,
            lr_mults_applied=dict(cfg.lr_mult),
            tokens_seen=tokens_seen,
            throughput_tokens_per_sec=(tokens_this_step / t_step_s if t_step_s > 0 else 0.0),
            step_time_s=t_step_s,
            gpu_mem_peak_gb=sys_now.get("gpu_mem_peak_gb", 0.0),
            disk_free_gb_ckpt=sys_now.get("disk_free_gb_ckpt", 0.0),
            wall_hours=cost_now["wall_hours"],
            estimated_cost_usd=cost_now["estimated_cost_usd"],
            pause_until_step=cfg.pause_until_step,
            max_loop_iters=config.max_loop_iters,
            config_applied_hash=admin.snapshot_state()["config_hash"],
        )
        admin.write_config_applied(cfg)

        if admin.should_stop():
            print(f"Graceful stop requested at step {step}.")
            break

    # Final checkpoint + cleanup
    _do_checkpoint(step, n_loops, name="final_checkpoint.pt")
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
    parser.add_argument("--config", choices=["rdt_1_5b", "tiny_rdt"], default="rdt_1_5b",
                        help="RDT config variant. Ignored when --model=dense.")
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
    parser.add_argument("--gpu_hourly_rate", type=float, default=3.50,
                         help="Cost tracker rate in $/hour; surfaced in admin status.")
    args = parser.parse_args()

    use_compile = not args.no_compile

    if args.model == "rdt":
        if args.config == "tiny_rdt":
            from configs.tiny_rdt import TINY_RDT_CONFIG as RDT_CONFIG
            rdt_subdir = "tiny_rdt"
        else:
            from configs.rdt_1_5b import RDT_1_5B_CONFIG as RDT_CONFIG
            rdt_subdir = "rdt_1.5b"
        train_rdt(
            RDT_CONFIG,
            output_dir=f"{args.output_dir}/{rdt_subdir}",
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
            gpu_hourly_rate=args.gpu_hourly_rate,
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
