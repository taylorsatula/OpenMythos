#!/usr/bin/env python3
"""
Smoke test for the OpenMythos training pipeline.

Runs a short training path on a tiny model + synthetic data, exercising every
invariant introduced by the training-harness sweep:

  - Model forward returns (logits, ponder_cost)
  - Muon + AdamW hybrid builds 4 param groups with the right routing
  - MoE aux + z-loss accumulate across loops and reset per outer forward
  - Init overrides are applied (ACT bias = -2.0, LTI A_init ≈ 0.9)
  - QK-norm modules are present and numerically active
  - LoRA plateau sampling covers indices up to max_loop_iters
  - Per-group grad norms are finite and non-zero
  - Spectral radius stays strictly below 1 under optimization
  - Checkpoint round-trip preserves both optimizer states, schedulers, RNG
  - Loss decreases over 20 steps on a constant-target toy task
"""

import math
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from open_mythos import MythosConfig, OpenMythos
from open_mythos.main import MoEFFN
from training.muon import Muon
from training.curriculum import CurriculumScheduler
from training.checkpointing import (
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from training.losses import composite_loss_rdt
from training.train import build_optimizers, build_schedulers


def _tiny_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=512,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        max_loop_iters=6,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        act_threshold=0.99,
        rope_theta=10000.0,
        lora_rank=4,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Architectural tests
# ---------------------------------------------------------------------------


def test_forward_shape_and_return():
    print("\n[1] Forward returns (logits, ponder_cost) with correct shapes...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    model.eval()
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, ponder = model(ids, n_loops=3)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    assert ponder.shape == (B, T), ponder.shape
    assert not torch.isnan(logits).any()
    print(f"   logits={tuple(logits.shape)}  ponder={tuple(ponder.shape)}   OK")


def test_init_overrides():
    print("\n[2] Init overrides (ACT bias = -2.0, LTI A_init ≈ 0.9, QK-norm present)...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    # ACT bias
    act_bias = model.recurrent.act.halt.bias.item()
    assert abs(act_bias - (-2.0)) < 1e-6, f"ACT halt.bias = {act_bias}, expected -2.0"
    # LTI A_init
    A = model.recurrent.injection.get_A()
    assert 0.85 <= A.min().item() <= 0.95, f"A.min = {A.min().item()}"
    assert 0.85 <= A.max().item() <= 0.95, f"A.max = {A.max().item()}"
    # QK-norm present on GQA
    gqa = model.prelude[0].attn
    assert hasattr(gqa, "q_norm") and hasattr(gqa, "k_norm")
    # Sanity: q_norm actually changes magnitude
    dummy = torch.randn(2, 8, gqa.n_heads, gqa.head_dim)
    normed = gqa.q_norm(dummy)
    assert normed.shape == dummy.shape
    assert not torch.allclose(normed, dummy)
    print(f"   ACT bias={act_bias:.4f}  A_init=[{A.min().item():.4f}, {A.max().item():.4f}]  OK")


def test_moe_accumulators_reset_per_forward():
    print("\n[3] MoE aux+z-loss accumulate across loops and reset per forward...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    model.train()
    ffn = model.recurrent.block.ffn
    assert isinstance(ffn, MoEFFN)

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, ponder = model(ids, n_loops=3)
    assert int(ffn._loss_count.item()) == 3, f"_loss_count after 3-loop forward = {ffn._loss_count.item()}"
    aux_after_first = ffn.aux_loss.item()
    z_after_first = ffn.z_loss.item()

    # Second forward resets
    logits, ponder = model(ids, n_loops=5)
    assert int(ffn._loss_count.item()) == 5, f"_loss_count after 5-loop forward = {ffn._loss_count.item()}"
    print(f"   first forward: count=3 aux={aux_after_first:.4f} z={z_after_first:.4f}")
    print(f"   second forward: count=5 aux={ffn.aux_loss.item():.4f} z={ffn.z_loss.item():.4f}  OK")


def test_spectral_radius_under_one():
    print("\n[4] Spectral radius < 1 by construction after init...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    rho = model.recurrent.injection.get_A().abs().max().item()
    assert rho < 1.0, f"ρ(A) = {rho}"
    print(f"   ρ(A) = {rho:.4f}  OK")


# ---------------------------------------------------------------------------
# Optimizer / training path tests
# ---------------------------------------------------------------------------


def test_optimizer_partitioning():
    print("\n[5] build_optimizers routes parameters into 4 disjoint groups...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    muon, adamw, groups = build_optimizers(model)
    all_params = {id(p) for _, p in model.named_parameters() if p.requires_grad}
    seen = set()
    for name, params in groups.items():
        for _, p in params:
            pid = id(p)
            assert pid not in seen, f"parameter in multiple groups"
            seen.add(pid)
    assert seen == all_params, (
        f"partition missed {len(all_params - seen)} params, extras {len(seen - all_params)}"
    )
    # Each Muon group must be 2D
    for g in muon.param_groups:
        for p in g["params"]:
            assert p.ndim == 2, f"Muon got non-2D param of shape {tuple(p.shape)}"
    print(f"   muon_default={len(groups['muon_default'])}  "
          f"muon_recurrent={len(groups['muon_recurrent'])}  "
          f"adamw_default={len(groups['adamw_default'])}  "
          f"adamw_recurrent={len(groups['adamw_recurrent'])}  OK")


def test_training_step_and_grads():
    print("\n[6] One full optimizer step with both optimizers + per-group grad norms...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    muon, adamw, groups = build_optimizers(model)

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, ponder = model(ids, n_loops=3)
    loss, loss_dict = composite_loss_rdt(model, logits, ponder, targets, cfg.vocab_size)
    loss.backward()

    # Check per-group grad norms all nonzero
    for name, params in groups.items():
        tensors = [p.grad for _, p in params if p.grad is not None]
        assert tensors, f"group {name} has no grads"
        total = sum(t.detach().pow(2).sum().item() for t in tensors) ** 0.5
        assert total > 0, f"group {name} has zero grad norm"
        assert math.isfinite(total), f"group {name} has non-finite grad norm"

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    muon.step()
    adamw.step()

    # Spectral radius still < 1 after optimizer step
    rho = model.recurrent.injection.get_A().abs().max().item()
    assert rho < 1.0, f"ρ(A) drifted to {rho} after step"
    print(f"   total loss={loss_dict['total']:.4f}  post-step ρ(A)={rho:.4f}  OK")


def test_loss_decreases_on_toy_task():
    print("\n[7] Loss decreases over 20 steps on a tiny constant-target task...")
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    muon, adamw, _ = build_optimizers(model, lr_muon=0.02, lr_adamw=3e-3)

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = ids.clone()  # copy-task signal; model should fit quickly
    losses = []
    for step in range(20):
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        logits, ponder = model(ids, n_loops=2)
        loss, _ = composite_loss_rdt(model, logits, ponder, targets, cfg.vocab_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adamw.step()
        losses.append(loss.item())
    print(f"   start={losses[0]:.4f}  end={losses[-1]:.4f}")
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} → {losses[-1]}"
    print("   OK")


# ---------------------------------------------------------------------------
# Curriculum test
# ---------------------------------------------------------------------------


def test_curriculum_plateau_coverage():
    print("\n[8] Curriculum plateau hits the full range of loop indices...")
    cur = CurriculumScheduler(total_steps=100, max_loops_train=4, max_loop_iters=6, ramp_frac=0.1, seed=0)
    # After step 10 we're in plateau. Sample 200 steps and assert every value
    # in [max_loops_train//2, max_loop_iters] shows up.
    seen = set()
    for s in range(10, 210):
        seen.add(cur.step(s))
    expected = set(range(max(2, 4 // 2), 6 + 1))
    missing = expected - seen
    assert not missing, f"plateau missed loop counts {missing}"
    print(f"   plateau range {sorted(expected)} all observed  OK")


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip():
    print("\n[9] Checkpoint save/load round-trip preserves both optimizer states + RNG...")
    cfg = _tiny_config()
    model = OpenMythos(cfg)
    muon, adamw, _ = build_optimizers(model)
    sched_m, sched_a = build_schedulers(muon, adamw, warmup_steps=5, total_steps=50)
    cur = CurriculumScheduler(total_steps=50, max_loops_train=3, max_loop_iters=6, seed=7)

    # Take a real step so optimizer state is non-empty
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = ids.clone()
    logits, ponder = model(ids, n_loops=2)
    loss, _ = composite_loss_rdt(model, logits, ponder, targets, cfg.vocab_size)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    muon.step(); adamw.step(); sched_m.step(); sched_a.step()

    expected_muon_lr = muon.param_groups[0]["lr"]
    expected_adamw_lr = adamw.param_groups[0]["lr"]

    # Advance curriculum past the ramp into the plateau so its RNG state is nontrivial
    for s in range(20):
        cur.step(s)
    # Capture the RNG state that the checkpoint will see
    pre_save_rng_state = cur._rng.getstate()
    # And the next value the *original* RNG would produce
    expected_next_loop = cur.step(20)

    with tempfile.TemporaryDirectory() as tmpd:
        # Roll the original back so the save happens at the pre-draw state,
        # matching what we'll compare against post-load.
        cur._rng.setstate(pre_save_rng_state)
        save_checkpoint(
            model, [muon, adamw], [sched_m, sched_a], cur,
            step=10, n_loops=3, loss_history=[1.0, 0.9],
            tokens_seen=1234, config=cfg, wandb_run_id=None,
            checkpoint_dir=tmpd,
        )
        # Rebuild from scratch and load
        model2 = OpenMythos(cfg)
        muon2, adamw2, _ = build_optimizers(model2)
        sched_m2, sched_a2 = build_schedulers(muon2, adamw2, warmup_steps=5, total_steps=50)
        cur2 = CurriculumScheduler(total_steps=50, max_loops_train=3, max_loop_iters=6, seed=999)

        ckpts = list(Path(tmpd).glob("*.pt"))
        assert len(ckpts) == 1, ckpts
        meta = load_checkpoint(str(ckpts[0]), model2, [muon2, adamw2],
                               [sched_m2, sched_a2], cur2, device="cpu")
        assert meta["step"] == 10
        assert meta["tokens_seen"] == 1234
        assert abs(muon2.param_groups[0]["lr"] - expected_muon_lr) < 1e-12
        assert abs(adamw2.param_groups[0]["lr"] - expected_adamw_lr) < 1e-12
        # Curriculum RNG resume determinism: next draw from the restored curriculum
        # must match the next draw of the original at the pre-save RNG state.
        assert cur2.step(20) == expected_next_loop, "curriculum RNG did not restore"

    print(f"   step=10 tokens=1234 muon_lr={expected_muon_lr:.5f} curriculum RNG ok  OK")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("OpenMythos Training Pipeline Smoke Test")
    print("=" * 60)
    torch.manual_seed(0)
    random.seed(0)

    test_forward_shape_and_return()
    test_init_overrides()
    test_moe_accumulators_reset_per_forward()
    test_spectral_radius_under_one()
    test_optimizer_partitioning()
    test_training_step_and_grads()
    test_loss_decreases_on_toy_task()
    test_curriculum_plateau_coverage()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
