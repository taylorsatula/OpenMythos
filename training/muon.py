"""
Muon optimizer — Momentum Orthogonalized by Newton-Schulz.

Reference: Keller Jordan, "Muon: An optimizer for hidden layers in neural
networks" (2024). Also validated on large-scale MoE by Liu et al. (Moonshot),
"Muon is Scalable for LLM Training" / Moonlight 16B (Feb 2025).

Vendored single-file implementation used by OpenMythos for the 2D weight
matrices of the transformer body. 1D tensors (biases, norms), embedding
rows, and the weight-tied LM head go to AdamW instead.

Adapted from https://github.com/KellerJordan/modded-nanogpt under MIT.
"""

import torch
from torch import Tensor


@torch.no_grad()
def zeropower_via_newton_schulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Newton-Schulz iteration computing the zeroth power (orthogonal factor of
    the polar decomposition) of G. A 5-step quintic polynomial approximation
    to U @ Vᵀ where G = U Σ Vᵀ.

    Runs in fp32 regardless of input dtype — bf16 underflows the polynomial.
    Returns a tensor with G's original shape and dtype.
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype=torch.float32)
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonalized by Newton-Schulz.

    Intended for 2D matrix-shaped parameters in a transformer body. Applies
    heavy-ball or Nesterov momentum, orthogonalizes the update via Newton-
    Schulz, applies decoupled weight decay, then steps.

    Default LR is ~65× larger than AdamW's because NS-orthogonalized updates
    have approximate unit spectral norm — the update magnitude is bounded by
    the orthogonalization, not by the gradient.

    Args:
        params: iterable of parameters or group dicts. All parameters must be 2D.
        lr: learning rate (default 0.02)
        momentum: momentum coefficient β (default 0.95)
        nesterov: use Nesterov-style lookahead (default True)
        weight_decay: decoupled weight decay coefficient (default 0.0)
        ns_steps: number of Newton-Schulz iterations (default 5; raise to 6 if
                  the orthogonalization destabilizes)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise RuntimeError(
                        f"Muon only supports 2D parameters; got shape {tuple(p.shape)}. "
                        f"Route this tensor to AdamW instead."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newton_schulz5(update, steps=ns_steps)

                # Shape scaling: make wide/tall matrices step at comparable
                # effective per-element magnitudes (Muon paper, §Practical).
                scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(update, alpha=-lr * scale)

        return loss
