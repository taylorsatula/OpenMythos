#!/usr/bin/env python3
"""Composite loss for RDT training: LM + MoE aux + MoE z-loss + ACT ponder."""

from typing import Tuple

import torch
import torch.nn.functional as F


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
    moe_aux_coeff: float = 0.01,
    moe_z_coeff: float = 1e-3,
    act_ponder_coeff: float = 1e-3,
) -> Tuple[torch.Tensor, dict]:
    """
    Four-term composite loss:

        total = lm + 0.01 * moe_aux + 1e-3 * moe_z + 1e-3 * act_ponder

    `moe_aux` and `moe_z` are read from the recurrent MoE's running-mean
    accumulators, which are reset at the start of every `OpenMythos.forward`
    and written once per loop iteration by `MoEFFN.forward`. Both values are
    already means across loop iterations, so the coefficients carry the same
    semantics as the single-layer Switch-T / ST-MoE originals.

    Returns:
        (total_loss_tensor, loss_dict) where loss_dict contains scalar floats
        for each component.
    """
    lm_loss = compute_lm_loss(logits, targets, vocab_size)

    ffn = model.recurrent.block.ffn
    moe_aux = ffn.aux_loss if isinstance(ffn.aux_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)
    moe_z = ffn.z_loss if isinstance(ffn.z_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)

    act_ponder_loss = ponder_cost.mean()

    total = (
        lm_loss
        + moe_aux_coeff * moe_aux
        + moe_z_coeff * moe_z
        + act_ponder_coeff * act_ponder_loss
    )

    loss_dict = {
        "total": total.detach().item(),
        "lm": lm_loss.detach().item(),
        "moe_aux": moe_aux.detach().item() if isinstance(moe_aux, torch.Tensor) else float(moe_aux),
        "moe_z": moe_z.detach().item() if isinstance(moe_z, torch.Tensor) else float(moe_z),
        "act_ponder": act_ponder_loss.detach().item(),
    }
    return total, loss_dict
