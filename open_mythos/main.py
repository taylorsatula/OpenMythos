import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0
    # DeepSeek-V3 router-bias update: after each optimizer step, nudge the
    # per-expert router_bias buffer toward uniform load. Enabled by default
    # for large expert counts (n_experts >= 128) where Switch-T aux alone
    # tends to leave tail experts dead. The update itself lives in the
    # training harness; MoEFFN accumulates per-step expert counts that the
    # harness reads.
    router_bias_update: bool = True
    # Per-step bias update rate γ. DeepSeek-V3 uses 1e-3; higher causes
    # routing to oscillate, lower fails to prevent dead-expert spiral.
    router_bias_update_rate: float = 1e-3


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed complex frequencies of shape (max_len, head_dim//2)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        # QK-norm (DeepSeek-V3 / Gemma-2): per-head RMSNorm on Q and K before RoPE.
        # Bounds attention logits across the looped recurrent block where the same
        # attention runs T times and errors compound.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(cfg.dropout)
        # Observability: if True, the forward computes an explicit QK matmul
        # under no_grad to report pre-softmax attention-logit abs-max. Kept off
        # during normal training (SDPA fuses the matmul and saves memory); the
        # training harness flips it on periodically for a single forward.
        self.diagnostics: bool = False
        self.last_attn_logits_abs_max: torch.Tensor = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        # expand KV to match Q heads
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Observability sample (opt-in). SDPA fuses softmax with the matmul,
        # so to see pre-softmax logits we redo the QK matmul under no_grad.
        # Flag is flipped off per-call by the training harness; zero cost
        # when disabled, single extra bmm when enabled.
        if self.diagnostics:
            with torch.no_grad():
                scale = 1.0 / math.sqrt(q.size(-1))
                attn_logits = torch.matmul(q.detach(), k.detach().transpose(-2, -1)) * scale
                self.last_attn_logits_abs_max = attn_logits.abs().amax()

        if mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif T > 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        # QK-norm on the per-head nope components only (rope components are already
        # magnitude-bounded by the complex phasor). Mirrors DeepSeek-V3's MLA stack.
        self.q_head_norm = RMSNorm(cfg.qk_nope_head_dim)
        self.k_head_norm = RMSNorm(cfg.qk_nope_head_dim)

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_nope = self.q_head_norm(q_nope)  # QK-norm on nope component
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k_nope = self.k_head_norm(k_nope)  # QK-norm on nope component
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        if mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif T > 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias; stays at zero unless the DeepSeek-V3 update loop is enabled.
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

        # Loss accumulators. RecurrentBlock calls this module n_loops times per
        # outer forward; we accumulate the Switch-T aux loss and ST-MoE z-loss
        # across those calls, then expose their mean via self.aux_loss / self.z_loss.
        # OpenMythos.forward calls reset_loss_accumulators() before the Prelude
        # so each outer forward starts fresh.
        self.register_buffer("_aux_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_z_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_loss_count", torch.tensor(0, dtype=torch.long), persistent=False)

        # Per-optimizer-step expert count accumulator for the DeepSeek-V3
        # router-bias update. Unlike the loss accumulators, this persists
        # across micro-batches within a single optimizer step and is reset
        # by the training harness AFTER applying the bias update. Saved in
        # checkpoints so resume lands in a consistent state.
        self.register_buffer("_step_expert_counts", torch.zeros(cfg.n_experts), persistent=True)

        # Last-call routing cache for eval/diagnostics (topk indices).
        # Not persisted; plain attribute so it can hold any shape.
        self.last_topk_idx: Optional[torch.Tensor] = None
        self.aux_loss: torch.Tensor = torch.tensor(0.0)
        self.z_loss: torch.Tensor = torch.tensor(0.0)
        # Observability: pre-softmax router-logit magnitude. High absolute
        # values mean bf16 saturation is imminent; z-loss should be keeping
        # this bounded. Healthy: stays under ~8. Warning: drifts above 12.
        self.last_router_logits_abs_max: torch.Tensor = torch.tensor(0.0)

    def reset_loss_accumulators(self) -> None:
        """
        Zero the running-mean buffers. Called once per outer forward pass.

        Must allocate fresh tensors rather than in-place zeroing, because the
        previous forward's `_aux_sum = _aux_sum + step_aux` reassignment left
        the attribute pointing at a tensor attached to a now-freed autograd
        graph; in-place zeroing that tensor triggers a "backward through freed
        graph" error on the next step.
        """
        self._aux_sum = torch.zeros_like(self._aux_sum)
        self._z_sum = torch.zeros_like(self._z_sum)
        self._loss_count = torch.zeros_like(self._loss_count)

    def reset_step_expert_counts(self) -> None:
        """
        Zero the per-optimizer-step expert-count accumulator. Called by the
        training harness AFTER applying the DeepSeek-V3 router-bias update.
        The counts themselves are populated in-place during forward, so a
        simple in-place zero is safe here — no autograd graph is attached.
        """
        self._step_expert_counts.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        logits = self.router(flat) + self.router_bias
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = scores.topk(self.topk, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # Switch-T aux: correlation between routing frequency and confidence.
        expert_mask = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=1)
        f = expert_mask.float().mean(dim=0)
        P = scores.mean(dim=0)
        step_aux = self.n_experts * (f * P).sum()
        # ST-MoE z-loss: keeps router logits bounded under bf16.
        step_z = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        self._aux_sum = self._aux_sum + step_aux
        self._z_sum = self._z_sum + step_z
        self._loss_count = self._loss_count + 1
        denom = self._loss_count.clamp(min=1).to(step_aux.dtype)
        self.aux_loss = self._aux_sum / denom
        self.z_loss = self._z_sum / denom
        # Cache most recent topk_idx for eval/utilization analysis.
        self.last_topk_idx = topk_idx.detach()
        # Capture pre-softmax router logit magnitude for observability.
        # Tracked outside autograd to avoid any graph retention concerns.
        with torch.no_grad():
            self.last_router_logits_abs_max = logits.detach().abs().amax()

        # Accumulate per-optimizer-step expert counts for the DeepSeek-V3
        # router-bias update. Counted every loop iteration and every
        # micro-batch; reset by the training harness after each step.
        # no_grad() keeps this off the autograd graph; topk_idx has no
        # grad anyway (it's integer), but bincount on a detached view
        # is the defensible form.
        with torch.no_grad():
            counts = torch.bincount(
                topk_idx.detach().view(-1).to(torch.long),
                minlength=self.n_experts,
            ).to(self._step_expert_counts.dtype)
            self._step_expert_counts.add_(counts)

        flat_expanded = flat.repeat_interleave(self.topk, dim=0)
        topk_idx_flat = topk_idx.view(B * T * self.topk)
        topk_scores_flat = topk_scores.view(B * T * self.topk)

        sort_idx = topk_idx_flat.argsort()
        flat_sorted = flat_expanded[sort_idx]
        topk_scores_sorted = topk_scores_flat[sort_idx]
        topk_idx_sorted = topk_idx_flat[sort_idx]

        out_flat = torch.zeros(B * T, D, device=flat.device, dtype=flat.dtype)
        start = 0
        for eid in range(self.n_experts):
            count = (topk_idx_sorted == eid).sum().item()
            if count == 0:
                continue
            end = start + count
            indices = sort_idx[start:end]
            orig_pos = indices // self.topk
            expert_tokens = flat_sorted[start:end]
            expert_scores = topk_scores_sorted[start:end]
            out_flat.scatter_add_(0, orig_pos.unsqueeze(-1).expand_as(expert_tokens),
                                  expert_scores.unsqueeze(-1) * self.routed_experts[eid](expert_tokens))
            start = end

        for shared in self.shared_experts:
            out_flat = out_flat + shared(flat)

        return out_flat.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        s = self.scale.weight[loop_t]
        down = self.down(x) * s
        return down @ self.B


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Clamp keeps the product finite in float32 for any gradient step size.
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of OpenMythos — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, lora_rank, max_loop_iters, act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding
        # Observability: hidden-state abs-max after each loop iteration.
        # Populated under no_grad during forward; cleared at the start of each
        # forward so it only reflects the most recent call. Rising magnitudes
        # across loops signal residual blow-up despite the LTI decay.
        self.last_hidden_abs_maxes: list = []
        # Per-loop halting probability means (pre-threshold), for ACT-collapse
        # diagnosis. Healthy: a spread across loops. Collapsed-early: all
        # concentrated at step 0 or 1. Collapsed-late: all ≈ 0 until max.
        self.last_halt_prob_means: list = []

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        Can be increased at inference for deeper reasoning (depth extrapolation).
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key

        Returns:
            Tuple of:
            - ACT-weighted sum of hidden states across iterations, shape (B, T, dim)
            - Cumulative ponder cost (sum of halting probabilities), shape (B, T)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        cumulative_ponder = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        # Reset observability accumulators for this forward
        self.last_hidden_abs_maxes = []
        self.last_halt_prob_means = []

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)
            still_running = ~halted

            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            cumulative_ponder = cumulative_ponder + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # Observability snapshots (no grad); one scalar per loop iteration.
            with torch.no_grad():
                self.last_hidden_abs_maxes.append(h.detach().abs().amax())
                self.last_halt_prob_means.append(p.detach().mean())

            if halted.all():
                break

        return h_out, cumulative_ponder


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled)
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights with N(0, 0.02), then apply the overrides that
        stabilize a looped/MoE model:

          1. Depth-scaled residual-stream output projections (attn.wo and ffn.down)
             scaled by 1/sqrt(2 * effective_depth). Prevents activation blowup
             through an architecture that behaves as ~16 layers deep.
          2. ACT halting bias biased to -2.0 so sigmoid ≈ 0.12 at init
             (avoids collapse-at-step-1).
          3. LTI injection parameterized so A_init ≈ 0.9 per channel at construction
             (near-identity injection preserves >90% of hidden state per loop early
             in training, vs the 37% that log_A = 0 would produce).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # (1) Depth-scaled output projections. Assumes training_max_loops = 8
        # (the curriculum's post-ramp target); mismatches only affect magnitudes
        # at initialization, which the optimizer absorbs.
        training_max_loops = 8
        effective_depth = self.cfg.prelude_layers + training_max_loops + self.cfg.coda_layers
        depth_scale = 1.0 / math.sqrt(2.0 * effective_depth)

        def _scale_residual_outputs(block: TransformerBlock) -> None:
            with torch.no_grad():
                block.attn.wo.weight.mul_(depth_scale)
                ffn = block.ffn
                if isinstance(ffn, Expert):
                    ffn.down.weight.mul_(depth_scale)
                elif isinstance(ffn, MoEFFN):
                    for expert in ffn.routed_experts:
                        expert.down.weight.mul_(depth_scale)
                    for expert in ffn.shared_experts:
                        expert.down.weight.mul_(depth_scale)

        for block in self.prelude:
            _scale_residual_outputs(block)
        _scale_residual_outputs(self.recurrent.block)
        for block in self.coda:
            _scale_residual_outputs(block)

        # (2) ACT halt bias.
        nn.init.constant_(self.recurrent.act.halt.bias, -2.0)

        # (3) LTI injection near-identity. A = exp(-exp(log_dt + log_A));
        # target A ≈ 0.9 → exp(log_dt + log_A) = -log(0.9) ≈ 0.1054.
        # Pick log_dt = 0, log_A = log(-log(0.9)) ≈ -2.25.
        with torch.no_grad():
            self.recurrent.injection.log_A.fill_(math.log(-math.log(0.9)))
            self.recurrent.injection.log_dt.zero_()

    def assert_weight_tying(self) -> None:
        """
        Structural invariant: `self.head.weight` and `self.embed.weight` share
        storage. Weight tying is set up in __init__ via attribute assignment,
        but `torch.compile` and some state-dict shenanigans can silently break
        this. Called at every checkpoint save; raises if violated.
        """
        if self.head.weight.data_ptr() != self.embed.weight.data_ptr():
            raise RuntimeError(
                "Weight tying broken: head.weight and embed.weight no longer "
                "share storage. Likely cause: torch.compile cloned the tensor, "
                "or a state_dict load replaced one without the other. "
                f"head.weight ptr={self.head.weight.data_ptr()}, "
                f"embed.weight ptr={self.embed.weight.data_ptr()}"
            )

    def unique_param_count(self) -> int:
        """
        Total parameter count with weight-tying deduplication.

        Standard `sum(p.numel() for p in self.parameters())` double-counts
        tied parameters. This method walks the parameter graph and counts
        each unique tensor only once. Logged every step as a structural
        invariant — if this number changes mid-training, something is very
        wrong (weight tying broke, or a parameter was re-registered).
        """
        seen = set()
        total = 0
        for p in self.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += p.numel()
        return total

    def muon_param_predicate(self, name: str, param: nn.Parameter) -> bool:
        """
        Return True if this parameter should be optimized by Muon (2D transformer
        weight matrix), False if it should go to AdamW. Used by the training
        harness to partition parameters into four groups (muon/adamw × default/recurrent).

        Exclusions: embedding (and its tied head), RMSNorm weights, 1D LoRA scale
        embedding, any non-2D tensor.
        """
        if param.ndim != 2:
            return False
        if "embed" in name or "head.weight" in name:
            return False
        if "lora.scale" in name:
            return False
        if ".norm" in name:  # RMSNorm weight tensors are 1D but guard anyway
            return False
        return True

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below the diagonal, -inf above.

        Args:
            seq_len -- sequence length
            device  -- target device

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids -- token indices of shape (B, T)
            n_loops   -- recurrent loop depth; defaults to cfg.max_loop_iters.
                         Increase at inference to extrapolate to harder problems.
            kv_cache  -- dict mutated in-place for autoregressive KV caching;
                         pass an empty dict {} and reuse across decode steps

        Returns:
            Tuple of:
            - Logits of shape (B, T, vocab_size)
            - Ponder cost from ACT halting, shape (B, T)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Reset the MoE aux/z-loss running-mean accumulators so each outer
        # forward starts from zero. RecurrentBlock will call MoEFFN.forward
        # n_loops times, and the final aux_loss/z_loss will be the mean across
        # those calls — preserving Switch-T coefficient semantics.
        self.recurrent.block.ffn.reset_loss_accumulators()

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[:T]
        mask = self._causal_mask(T, device) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x
        x, ponder_cost = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x)), ponder_cost

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops        -- recurrent loop depth for each decode step
            temperature    -- softmax temperature; lower = more greedy
            top_k          -- restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        for step in range(max_new_tokens):
            cur_ids = input_ids if step == 0 else input_ids[:, -1:]
            logits, _ = self.forward(cur_ids, n_loops=n_loops, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
