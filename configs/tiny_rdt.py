#!/usr/bin/env python3
"""
Tiny RDT configuration for end-to-end pipeline sanity runs.

Not a research config. Sized so a ~50-step run completes in minutes on a
single GPU while still exercising every code path that the 1.5B run hits:
GQA, MoE routing + shared experts + router-bias update, recurrent LoRA,
ACT, prelude/coda stack, curriculum, checkpoint round-trip.

Embedding still dominates param count because vocab_size is overridden from
the tokenizer at runtime (~201K). That's unavoidable without swapping the
tokenizer and is fine — the tiny transformer body keeps step time low.
"""

from open_mythos.main import MythosConfig

TINY_RDT_CONFIG = MythosConfig(
    vocab_size=201088,
    dim=128,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=2048,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    act_threshold=0.99,
    rope_theta=500000.0,
    lora_rank=4,
    dropout=0.0,
)

MODEL_NAME = "tiny_rdt"
MAX_LOOPS_TRAIN = 2
MAX_LOOPS_INFERENCE = 4
