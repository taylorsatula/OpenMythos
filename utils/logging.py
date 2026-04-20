#!/usr/bin/env python3
"""
Weights & Biases logging utilities for the OpenMythos training harness.

Exposes a deliberately large set of observability surfaces. Each one answers
one "is this failure mode active right now?" question that aggregate loss
cannot. The guiding principle is: a $200/55-hour training run is vastly
more expensive than 30 extra scalars per step. Log them.
"""

import math
import os
import shutil
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    or WANDB_API_KEY is unset (in which case all log_* calls are no-ops).
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


# ---------------------------------------------------------------------------
# Scalar helpers for per-step telemetry
# ---------------------------------------------------------------------------


def _safe_scalar(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def count_dead_experts(model, threshold_frac: float = 0.001) -> int:
    """Fraction of experts below `threshold_frac` of total routing slots in the
    most recent forward's `last_topk_idx` cache. Zero if no forward yet."""
    try:
        ffn = model.recurrent.block.ffn
        topk = ffn.last_topk_idx
    except AttributeError:
        return 0
    if topk is None:
        return 0
    counts = torch.bincount(topk.view(-1).to(torch.long), minlength=ffn.n_experts).float()
    total = counts.sum().clamp(min=1)
    return int(((counts / total) < threshold_frac).sum().item())


def router_logits_abs_max(model) -> float:
    try:
        return _safe_scalar(model.recurrent.block.ffn.last_router_logits_abs_max)
    except AttributeError:
        return 0.0


def attn_logits_abs_max(model) -> float:
    """Max over prelude + recurrent + coda attention blocks' last-sampled
    attention-logit abs-max. Only populated when the diagnostic flag is on
    (see training loop; default: every `log_interval` steps)."""
    vals = []
    try:
        for block in model.prelude:
            if hasattr(block.attn, "last_attn_logits_abs_max"):
                vals.append(_safe_scalar(block.attn.last_attn_logits_abs_max))
        if hasattr(model.recurrent.block.attn, "last_attn_logits_abs_max"):
            vals.append(_safe_scalar(model.recurrent.block.attn.last_attn_logits_abs_max))
        for block in model.coda:
            if hasattr(block.attn, "last_attn_logits_abs_max"):
                vals.append(_safe_scalar(block.attn.last_attn_logits_abs_max))
    except AttributeError:
        pass
    return max(vals) if vals else 0.0


def hidden_state_abs_max_final_loop(model) -> float:
    try:
        maxes = model.recurrent.last_hidden_abs_maxes
        return _safe_scalar(maxes[-1]) if maxes else 0.0
    except AttributeError:
        return 0.0


def hidden_state_abs_max_max(model) -> float:
    """Max across all loop iterations for the last forward."""
    try:
        maxes = model.recurrent.last_hidden_abs_maxes
        return max(_safe_scalar(m) for m in maxes) if maxes else 0.0
    except AttributeError:
        return 0.0


def halt_prob_means_per_loop(model) -> list:
    try:
        return [_safe_scalar(p) for p in model.recurrent.last_halt_prob_means]
    except AttributeError:
        return []


def mean_halt_step(ponder_cost: Optional[torch.Tensor]) -> float:
    if ponder_cost is None:
        return 0.0
    return float(ponder_cost.detach().float().mean().item())


def loss_term_fractions(loss_dict: dict, coeffs: Mapping[str, float]) -> dict:
    """
    Fraction of total loss contributed by each term. Catches coefficient drift:
    `lm/total` should stay >0.7 throughout; if it sinks toward 0.5, the
    regularizers have taken over and LM learning has stalled.

    `coeffs` must be the same coefficients used by `composite_loss_rdt` at
    this step (thread from admin/config.json). If `loss_dict` contains a
    `coeffs_applied` key (populated by composite_loss_rdt), that is used
    instead — the strongest guarantee that report matches reality.
    """
    c = dict(coeffs)
    if "coeffs_applied" in loss_dict and isinstance(loss_dict["coeffs_applied"], Mapping):
        c.update(loss_dict["coeffs_applied"])
    total = max(float(loss_dict.get("total", 0.0)), 1e-8)
    return {
        "lm_frac": float(loss_dict.get("lm", 0.0)) / total,
        "aux_frac": c.get("moe_aux", 0.0) * float(loss_dict.get("moe_aux", 0.0)) / total,
        "z_frac": c.get("moe_z", 0.0) * float(loss_dict.get("moe_z", 0.0)) / total,
        "ponder_frac": c.get("act_ponder", 0.0) * float(loss_dict.get("act_ponder", 0.0)) / total,
    }


def expected_lr(step: int, warmup: int, total: int, peak_lr: float, min_ratio: float = 0.1) -> float:
    """Closed-form LR at a given step. Compared to the actual optimizer LR
    every step; a nonzero diff means scheduler has desynced (typical cause:
    resume-from-checkpoint off-by-one). Caller multiplies by any admin
    LR-multiplier before comparing — otherwise the desync detector screams
    the moment a multiplier is set."""
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Administration-dashboard diagnostics (new, added for the admin harness)
# ---------------------------------------------------------------------------


def timing_metrics(step_time_s: float, tokens_this_step: int) -> Dict[str, float]:
    """Step wall-time + throughput. Caller brackets only forward+backward+optimizer,
    excluding diagnostics and eval, so this metric stays clean."""
    thr = tokens_this_step / step_time_s if step_time_s > 0 else 0.0
    return {
        "step_time_s": float(step_time_s),
        "throughput_tokens_per_sec": float(thr),
    }


def system_metrics(ckpt_dir: Optional[str] = None) -> Dict[str, float]:
    """GPU memory and checkpoint-filesystem free space. Both are free to collect."""
    out: Dict[str, float] = {}
    if torch.cuda.is_available():
        out["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    else:
        out["gpu_mem_peak_gb"] = 0.0
        out["gpu_mem_reserved_gb"] = 0.0
    if ckpt_dir:
        try:
            target = ckpt_dir if os.path.exists(ckpt_dir) else os.path.dirname(ckpt_dir) or "."
            du = shutil.disk_usage(target)
            out["disk_free_gb_ckpt"] = du.free / 1e9
        except (FileNotFoundError, OSError):
            out["disk_free_gb_ckpt"] = 0.0
    return out


def cost_metrics(run_start_wall: float, gpu_hourly_rate: float = 3.50) -> Dict[str, float]:
    wall_hours = (time.time() - run_start_wall) / 3600.0
    return {
        "wall_hours": float(wall_hours),
        "estimated_cost_usd": float(wall_hours * gpu_hourly_rate),
    }


def _match_any(name: str, patterns: Sequence[str]) -> bool:
    return any(p in name for p in patterns)


def per_layer_grad_norms(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    selectors: Optional[Mapping[str, Sequence[str]]] = None,
    prelude_layers: int = 4,
    coda_layers: int = 4,
) -> Dict[str, float]:
    """
    Per-layer L2 grad norm for the most informative buckets:
    first/last prelude block, recurrent-attn vs recurrent-ffn-router vs
    recurrent-ffn-shared, first/last coda block. Catches a single broken
    layer that the four-group global norm would hide.

    Pass an explicit `selectors` mapping to override the default bucket set.
    """
    if selectors is None:
        last_pre = prelude_layers - 1
        last_coda = coda_layers - 1
        selectors = {
            "prelude_first": [f"prelude.0."],
            "prelude_last": [f"prelude.{last_pre}."],
            "recurrent_attn": ["recurrent.block.attn."],
            "recurrent_ffn_router": ["recurrent.block.ffn.router"],
            "recurrent_ffn_shared": ["recurrent.block.ffn.shared_experts"],
            "recurrent_injection": ["recurrent.injection."],
            "recurrent_act": ["recurrent.act."],
            "coda_first": [f"coda.0."],
            "coda_last": [f"coda.{last_coda}."],
        }

    # One tensor list per bucket, then a single vector-norm per bucket.
    buckets: Dict[str, List[torch.Tensor]] = {k: [] for k in selectors}
    for name, p in named_params:
        if p.grad is None:
            continue
        for bucket, patterns in selectors.items():
            if _match_any(name, patterns):
                buckets[bucket].append(p.grad.detach())
                break  # one bucket per param

    out: Dict[str, float] = {}
    for bucket, tensors in buckets.items():
        if not tensors:
            out[bucket] = 0.0
            continue
        norms = torch.stack([torch.linalg.vector_norm(t) for t in tensors])
        out[bucket] = float(torch.linalg.vector_norm(norms).item())
    return out


def attn_entropy_mean(model) -> float:
    """Mean over prelude + recurrent + coda blocks of per-block cached
    attention-softmax entropy. Only populated when the `diagnostics` flag
    is on (training harness toggles it every log_interval steps).
    Low entropy (approaching 0) = attention collapsed onto a few tokens."""
    vals = []
    try:
        for block in model.prelude:
            if hasattr(block.attn, "last_attn_entropy"):
                vals.append(_safe_scalar(block.attn.last_attn_entropy))
        if hasattr(model.recurrent.block.attn, "last_attn_entropy"):
            vals.append(_safe_scalar(model.recurrent.block.attn.last_attn_entropy))
        for block in model.coda:
            if hasattr(block.attn, "last_attn_entropy"):
                vals.append(_safe_scalar(block.attn.last_attn_entropy))
    except AttributeError:
        pass
    return sum(vals) / len(vals) if vals else 0.0


def router_entropy(model) -> float:
    """Softmax entropy of the routing distribution on the last recurrent-loop
    forward. Cheap (already-materialized logits). Collapsing entropy → dead
    routes even when the dead-expert count still reads zero."""
    try:
        return _safe_scalar(model.recurrent.block.ffn.last_router_entropy)
    except AttributeError:
        return 0.0


def router_bias_l2(model) -> float:
    """L2 norm of the DeepSeek-V3 load-balancing bias buffer. Non-zero and
    drifting means the bias update is active; sudden jumps can indicate
    a dying expert the update is trying to rescue."""
    try:
        return float(model.recurrent.block.ffn.router_bias.detach().norm().item())
    except AttributeError:
        return 0.0


def microbatch_loss_cv(microbatch_losses: Sequence[float]) -> float:
    """Coefficient of variation (std/mean) across the `grad_accum` microbatches'
    per-microbatch total losses. A sudden jump reveals a stale or corrupt shard
    even when the mean across microbatches is unchanged."""
    clean = [x for x in microbatch_losses if x is not None and math.isfinite(x)]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    if abs(mean) < 1e-12:
        return 0.0
    std = statistics.pstdev(clean)
    return float(std / mean)


def run_generation_probe(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    n_loops: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Run a small fixed-prompt panel and return list of result dicts. Never
    raises — on per-prompt failure we return an `error` field. Caller is
    responsible for logging and for toggling model.train()/.eval()."""
    was_training = model.training
    model.eval()
    results: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for prompt in prompts:
                t0 = time.perf_counter()
                try:
                    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long,
                                       device=device).unsqueeze(0)
                    out = model.generate(ids, max_new_tokens=max_new_tokens,
                                          n_loops=n_loops, temperature=temperature, top_k=top_k)
                    text = tokenizer.decode(out[0][ids.shape[-1]:].tolist())
                    results.append({
                        "prompt": prompt, "output": text, "n_loops": n_loops,
                        "temperature": temperature, "top_k": top_k,
                        "latency_s": time.perf_counter() - t0,
                    })
                except Exception as e:  # noqa: BLE001 — never crash the run on an eval issue
                    results.append({
                        "prompt": prompt, "output": "", "error": f"{type(e).__name__}: {e}",
                        "n_loops": n_loops, "latency_s": time.perf_counter() - t0,
                    })
    finally:
        if was_training:
            model.train()
    return results


# ---------------------------------------------------------------------------
# Per-step log
# ---------------------------------------------------------------------------


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
    expected_lrs_per_group: Optional[Mapping[str, float]] = None,
    invariant_param_count: Optional[int] = None,
    loss_coeffs: Optional[Mapping[str, float]] = None,
    lr_mults: Optional[Mapping[str, float]] = None,
    per_layer_grad_norms_dict: Optional[Mapping[str, float]] = None,
    extra_metrics: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """
    Log per-optimizer-step training metrics to wandb. Returns the assembled
    metrics dict so the caller can also tee it to a local JSONL event stream
    (see `tools.mythos_admin_runtime.AdminRuntime.log_metric`).

    In addition to the basic loss/grad/LR trio, this emits a set of
    "broken-but-looks-working" detectors documented in HANDOFF §7.1, plus
    the administration-dashboard surface (timing, per-layer grad norms,
    microbatch variance, attention/routing entropy, router-bias drift):

      - Per-term loss fraction of total (coefficient-drift detector)
      - Expected-LR vs actual-LR diff (scheduler desync detector, with
        admin LR multipliers factored in)
      - Pre-softmax router logit abs-max (bf16 saturation detector)
      - Final-loop and max-across-loops hidden state abs-max
        (LTI drift / residual blow-up detector)
      - Per-loop ACT halting probability means (ACT collapse detector)
      - Structural invariant: unique parameter count (weight-tying detector)
      - Spectral radius max (LTI stability detector, threshold 0.999)
      - Per-layer grad norms, attn entropy, router entropy, router-bias L2
      - Throughput, step time, GPU memory, disk free (via `extra_metrics`)
    """
    coeffs = loss_coeffs if loss_coeffs is not None else {"moe_aux": 0.01, "moe_z": 1e-3, "act_ponder": 1e-3}
    dead_count = count_dead_experts(model)
    halt_mean = mean_halt_step(ponder_cost)
    fracs = loss_term_fractions(loss_dict, coeffs)

    metrics: Dict[str, Any] = {
        "train/loss": _safe_scalar(loss_dict.get("total", 0.0)),
        "train/lm_loss": _safe_scalar(loss_dict.get("lm", 0.0)),
        "train/moe_aux_loss": _safe_scalar(loss_dict.get("moe_aux", 0.0)),
        "train/moe_z_loss": _safe_scalar(loss_dict.get("moe_z", 0.0)),
        "train/act_ponder_cost": _safe_scalar(loss_dict.get("act_ponder", 0.0)),
        "train/loss_frac/lm": fracs["lm_frac"],
        "train/loss_frac/aux": fracs["aux_frac"],
        "train/loss_frac/z": fracs["z_frac"],
        "train/loss_frac/ponder": fracs["ponder_frac"],
        "train/grad_norm/total": total_grad_norm,
        "train/n_loops": n_loops,
        "train/tokens_seen": tokens_seen,
        "train/spectral_radius_max": spectral_radius_max,
        "train/moe_dead_expert_count": dead_count,
        "train/act_halt_mean": halt_mean,
        # Numerical health
        "train/router_logits_abs_max": router_logits_abs_max(model),
        "train/router_entropy": router_entropy(model),
        "train/router_bias_l2": router_bias_l2(model),
        "train/attn_logits_abs_max": attn_logits_abs_max(model),
        "train/attn_entropy_mean": attn_entropy_mean(model),
        "train/hidden_state_abs_max_final": hidden_state_abs_max_final_loop(model),
        "train/hidden_state_abs_max_any_loop": hidden_state_abs_max_max(model),
        "train/loss_coeff/moe_aux": coeffs.get("moe_aux", 0.0),
        "train/loss_coeff/moe_z": coeffs.get("moe_z", 0.0),
        "train/loss_coeff/act_ponder": coeffs.get("act_ponder", 0.0),
    }
    for g, v in grad_norms_per_group.items():
        metrics[f"train/grad_norm/{g}"] = v
    for g, v in lrs_per_group.items():
        metrics[f"train/lr/{g}"] = v
    if lr_mults is not None:
        for g, v in lr_mults.items():
            metrics[f"train/lr_mult/{g}"] = v

    # Expected vs actual LR (closed-form comparison; divergence flags scheduler desync).
    # Caller is responsible for factoring in lr_mults on the expected side — otherwise
    # the desync detector screams the moment any multiplier leaves 1.0.
    if expected_lrs_per_group is not None:
        for g, v in expected_lrs_per_group.items():
            metrics[f"train/lr_expected/{g}"] = v
            actual = lrs_per_group.get(g)
            if actual is not None:
                metrics[f"train/lr_diff/{g}"] = actual - v

    # Structural invariant: unique parameter count. Constant in a healthy run;
    # a delta means weight tying broke or a param was re-registered.
    if invariant_param_count is not None:
        metrics["train/invariant/unique_param_count"] = invariant_param_count

    # Per-loop ACT halt probability means. Flat across loops = collapsed ACT.
    halt_means = halt_prob_means_per_loop(model)
    for i, v in enumerate(halt_means):
        metrics[f"train/act/p_mean_loop{i}"] = v

    if per_layer_grad_norms_dict is not None:
        for layer, v in per_layer_grad_norms_dict.items():
            metrics[f"train/grad_norm_layer/{layer}"] = v

    if extra_metrics is not None:
        for k, v in extra_metrics.items():
            # Namespace: anything with a `/` is left untouched; otherwise prefix with train/.
            key = k if "/" in k else f"train/{k}"
            metrics[key] = v

    if run is not None:
        try:
            import wandb  # noqa: F401
            run.log(metrics, step=step)
        except ImportError:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Periodic deeper diagnostics (every log_interval × k steps)
# ---------------------------------------------------------------------------


def log_expert_norms(run, step: int, model, muon_optim=None):
    """
    Log per-expert weight-norm distribution and, if given, per-expert Muon
    momentum-buffer norm distribution. Rising dispersion here is an early
    warning for Muon × MoE drift before dead-expert count spikes.
    """
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    ffn = model.recurrent.block.ffn
    # Weight norms: routed experts × (gate + up + down)
    routed_norms = []
    for expert in ffn.routed_experts:
        n = (expert.gate.weight.detach().pow(2).sum()
             + expert.up.weight.detach().pow(2).sum()
             + expert.down.weight.detach().pow(2).sum()).sqrt().item()
        routed_norms.append(n)
    routed_norms_t = torch.tensor(routed_norms)
    metrics = {
        "diag/expert_norms/mean": routed_norms_t.mean().item(),
        "diag/expert_norms/std": routed_norms_t.std(unbiased=False).item(),
        "diag/expert_norms/max": routed_norms_t.max().item(),
        "diag/expert_norms/min": routed_norms_t.min().item(),
        "diag/expert_norms/max_min_ratio": (
            routed_norms_t.max() / routed_norms_t.min().clamp(min=1e-8)
        ).item(),
    }
    # Per-expert momentum norms from Muon (if provided)
    if muon_optim is not None:
        expert_ids = {id(expert.gate.weight) for expert in ffn.routed_experts}
        expert_ids |= {id(expert.up.weight) for expert in ffn.routed_experts}
        expert_ids |= {id(expert.down.weight) for expert in ffn.routed_experts}
        mom_norms = []
        for group in muon_optim.param_groups:
            for p in group["params"]:
                if id(p) in expert_ids:
                    state = muon_optim.state.get(p, {})
                    buf = state.get("momentum_buffer")
                    if buf is not None:
                        mom_norms.append(buf.detach().norm().item())
        if mom_norms:
            t = torch.tensor(mom_norms)
            metrics["diag/muon_mom/expert_mean"] = t.mean().item()
            metrics["diag/muon_mom/expert_max"] = t.max().item()
            metrics["diag/muon_mom/expert_std"] = t.std(unbiased=False).item()

    # Also log the expert-norm histogram
    metrics["diag/expert_norms/hist"] = wandb.Histogram(routed_norms)
    run.log(metrics, step=step)


def log_decoded_sample(run, step: int, input_ids: torch.Tensor, tokenizer,
                       max_rows: int = 2, max_chars: int = 400):
    """
    Decode a few rows of a training batch and log them as a wandb text table.
    Cheap human spot-check that catches tokenizer/data corruption that no
    aggregate metric will ever show.
    """
    if run is None or input_ids is None:
        return
    try:
        import wandb
    except ImportError:
        return
    rows = []
    n = min(max_rows, input_ids.size(0))
    for i in range(n):
        ids = input_ids[i].tolist()
        try:
            text = tokenizer.decode(ids)
        except Exception as e:
            text = f"[decode failed: {e}]"
        rows.append([step, i, text[:max_chars]])
    table = wandb.Table(columns=["step", "row", "decoded"], data=rows)
    run.log({"diag/decoded_sample": table}, step=step)


def log_token_id_histogram(run, step: int, input_ids: torch.Tensor, vocab_size: int,
                            n_buckets: int = 64):
    """
    Coarse histogram of token ids. Healthy LM corpora produce a long-tailed
    distribution biased toward low ids (common tokens); a uniform or degenerate
    distribution is a red flag for tokenizer/data corruption.
    """
    if run is None or input_ids is None:
        return
    try:
        import wandb
    except ImportError:
        return
    bucket_size = max(1, vocab_size // n_buckets)
    bucketed = (input_ids.view(-1) // bucket_size).clamp(max=n_buckets - 1).long()
    counts = torch.bincount(bucketed, minlength=n_buckets).float()
    run.log({"diag/token_id_hist": wandb.Histogram(np_histogram=(counts.tolist(),
                                                                  list(range(n_buckets + 1))))},
            step=step)


# ---------------------------------------------------------------------------
# Evaluation / capability logging
# ---------------------------------------------------------------------------


def log_eval_metrics(run, step: int, metrics: dict, prefix: str = "eval"):
    if run is None:
        return
    try:
        import wandb  # noqa: F401
    except ImportError:
        return
    run.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)


def log_depth_extrapolation(run, step: int, perplexities: dict):
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
    mean_halt: float,
    halt_by_token_type: Optional[Mapping[str, float]] = None,
    halt_histogram: Optional[List[int]] = None,
):
    """
    Log ACT halt-step statistics. `halt_by_token_type` is a mapping like
    {'punct': 2.1, 'stopword': 2.5, 'content': 6.7, 'digit': 4.3, 'code': 5.1}.
    Flat across buckets = ACT didn't learn differential computation allocation.
    """
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return
    metrics = {"eval/act_mean_halt_step": mean_halt}
    if halt_by_token_type is not None:
        for k, v in halt_by_token_type.items():
            metrics[f"eval/halt_step_by_type/{k}"] = v
    if halt_histogram is not None:
        metrics["eval/halt_histogram"] = wandb.Histogram(halt_histogram)
    run.log(metrics, step=step)


def log_synthetic_arithmetic(run, step: int, accuracy_by_depth_loops: dict):
    """
    `accuracy_by_depth_loops` is a mapping (depth, n_loops) -> exact-match.
    Log per depth, per n_loops, plus the extrapolation signal:
    accuracy at depth > training_max should be higher with more loops.
    """
    if run is None:
        return
    try:
        import wandb  # noqa: F401
    except ImportError:
        return
    metrics = {}
    for (depth, n_loops), acc in accuracy_by_depth_loops.items():
        metrics[f"eval/arith/K{depth}_loops{n_loops}"] = acc
    run.log(metrics, step=step)


def finish_run(run):
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except ImportError:
        pass
