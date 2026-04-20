#!/usr/bin/env python3
"""Composite loss for RDT training: LM + MoE aux + MoE z-loss + ACT ponder."""

from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


DEFAULT_COEFFS: Mapping[str, float] = {
    "moe_aux": 0.01,
    "moe_z": 1e-3,
    "act_ponder": 1e-3,
}


def compute_lm_loss(logits: torch.Tensor, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Standard language modeling cross-entropy with a one-position shift.

    logits:  (B, T, V)
    targets: (B, T)  — the token-level labels aligned with `logits` positions.
                      The first `logits` position predicts `targets[:, 1]`, etc.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_targets.view(-1),
        reduction="mean",
    )


def composite_loss_rdt(
    model,
    logits: torch.Tensor,
    ponder_cost: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    coeffs: Optional[Mapping[str, float]] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Four-term composite loss:

        total = lm + c_aux * moe_aux + c_z * moe_z + c_ponder * act_ponder

    Coefficients flow from the single source of truth (admin/config.json
    in the training harness). When `coeffs` is None, DEFAULT_COEFFS is used
    — matches pre-admin-dashboard behavior.

    `moe_aux` and `moe_z` are read from the recurrent MoE's running-mean
    accumulators, which are reset at the start of every `OpenMythos.forward`
    and written once per loop iteration by `MoEFFN.forward`. Both values are
    already means across loop iterations, so the coefficients carry the same
    semantics as the single-layer Switch-T / ST-MoE originals.

    Returns:
        (total_loss_tensor, loss_dict) where loss_dict contains scalar floats
        for each component plus the coefficients actually applied — so the
        caller and logger share one coefficient source.
    """
    c = dict(DEFAULT_COEFFS)
    if coeffs is not None:
        c.update(coeffs)

    lm_loss = compute_lm_loss(logits, targets, vocab_size)

    ffn = model.recurrent.block.ffn
    moe_aux = ffn.aux_loss if isinstance(ffn.aux_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)
    moe_z = ffn.z_loss if isinstance(ffn.z_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)

    act_ponder_loss = ponder_cost.mean()

    total = (
        lm_loss
        + c["moe_aux"] * moe_aux
        + c["moe_z"] * moe_z
        + c["act_ponder"] * act_ponder_loss
    )

    loss_dict = {
        "total": total.detach().item(),
        "lm": lm_loss.detach().item(),
        "moe_aux": moe_aux.detach().item() if isinstance(moe_aux, torch.Tensor) else float(moe_aux),
        "moe_z": moe_z.detach().item() if isinstance(moe_z, torch.Tensor) else float(moe_z),
        "act_ponder": act_ponder_loss.detach().item(),
        "coeffs_applied": dict(c),
    }
    return total, loss_dict
