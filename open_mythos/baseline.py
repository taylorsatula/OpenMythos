#!/usr/bin/env python3
"""Dense transformer baseline for FLOP-matched comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from open_mythos.main import (
    MythosConfig,
    RMSNorm,
    GQAttention,
    MLAttention,
    Expert,
    precompute_rope_freqs,
    apply_rope,
)


class DenseBlock(nn.Module):
    """Standard transformer block without MoE."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


@dataclass
class DenseConfig:
    """Configuration for the dense baseline model."""
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    n_layers: int = 10
    vocab_size: int = 32000
    attn_type: str = "gqa"
    rope_theta: float = 500000.0
    dropout: float = 0.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128


class DenseTransformer(nn.Module):
    """FLOP-matched dense transformer baseline."""

    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.blocks = nn.ModuleList([DenseBlock(self._to_mythos_cfg()) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        self._init_weights()

    def _to_mythos_cfg(self) -> MythosConfig:
        return MythosConfig(
            dim=self.cfg.dim,
            n_heads=self.cfg.n_heads,
            n_kv_heads=self.cfg.n_kv_heads,
            max_seq_len=self.cfg.max_seq_len,
            attn_type=self.cfg.attn_type,
            kv_lora_rank=self.cfg.kv_lora_rank,
            q_lora_rank=self.cfg.q_lora_rank,
            qk_rope_head_dim=self.cfg.qk_rope_head_dim,
            qk_nope_head_dim=self.cfg.qk_nope_head_dim,
            v_head_dim=self.cfg.v_head_dim,
            rope_theta=self.cfg.rope_theta,
            dropout=self.cfg.dropout,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[:T]
        mask = self._causal_mask(T, device) if T > 1 else None

        for i, block in enumerate(self.blocks):
            x = block(x, freqs_cis, mask, kv_cache, cache_key=f"block_{i}")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        kv_cache: dict = {}
        for step in range(max_new_tokens):
            cur_ids = input_ids if step == 0 else input_ids[:, -1:]
            logits = self.forward(cur_ids, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


def dense_config_from_rdt(cfg: MythosConfig, n_layers: int) -> DenseConfig:
    """Create a DenseConfig FLOP-matched to an RDT config."""
    return DenseConfig(
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        max_seq_len=cfg.max_seq_len,
        n_layers=n_layers,
        vocab_size=cfg.vocab_size,
        attn_type=cfg.attn_type,
        rope_theta=cfg.rope_theta,
        dropout=cfg.dropout,
        kv_lora_rank=cfg.kv_lora_rank,
        q_lora_rank=cfg.q_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        v_head_dim=cfg.v_head_dim,
    )