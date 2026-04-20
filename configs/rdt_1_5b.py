#!/usr/bin/env python3
"""
RDT 1.5B-class transformer-body configuration.

Total stored parameters target ~1.85B after embeddings (~410M at the
gpt-oss-20b ~201K vocab). Run `python -m utils.count_params` to confirm
the transformer-body count is within 5% of 1.5B before launching a full
training run — if not, bump `expert_dim` or `n_experts`.

`vocab_size` here is a placeholder. The training entry point overrides it
from `MythosTokenizer().vocab_size` before model construction.
"""

from open_mythos.main import MythosConfig

RDT_1_5B_CONFIG = MythosConfig(
    vocab_size=201088,            # placeholder; overridden at runtime from tokenizer
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
    max_loop_iters=16,
    prelude_layers=4,
    coda_layers=4,
    attn_type="gqa",
    # MoE sizing calibrated (via utils.count_params) so the transformer body
    # (non-embedding params) lands at ~1.5B. n_experts=128/expert_dim=512 only
    # gets to 656M body; n_experts=192/expert_dim=1024 → ~1.49B body.
    n_experts=192,
    n_shared_experts=2,
    n_experts_per_tok=4,
    expert_dim=1024,
    act_threshold=0.99,
    rope_theta=500000.0,
    lora_rank=16,
    dropout=0.0,
)

MODEL_NAME = "rdt_1.5b"
MAX_LOOPS_TRAIN = 8
MAX_LOOPS_INFERENCE = 16
