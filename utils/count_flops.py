#!/usr/bin/env python3
"""FLOP counting utility for RDT and dense baseline comparison."""

import math
from open_mythos import MythosConfig


def count_rdt_flops(
    cfg: MythosConfig,
    seq_len: int = 2048,
    n_loops: int = 8,
) -> dict:
    """
    Estimate FLOPs per forward pass for OpenMythos RDT.
    
    Returns breakdown by component.
    """
    d = cfg.dim
    h = cfg.n_heads
    kv = cfg.n_kv_heads
    head_dim = d // h
    ffn_hidden = cfg.expert_dim * 4 // 3  # SwiGLU inner dim
    
    n_prelude = cfg.prelude_layers
    n_coda = cfg.coda_layers
    
    embed_flops = 0
    
    attn_qkv_flops = seq_len * d * d * 3
    attn_scores_flops = seq_len * h * head_dim * seq_len * 2
    attn_out_flops = seq_len * h * seq_len * head_dim * 2
    attn_ffn_flops = seq_len * d * ffn_hidden * 3 + seq_len * ffn_hidden * d
    prelude_flops = n_prelude * (attn_qkv_flops + attn_scores_flops + attn_out_flops + attn_ffn_flops)
    
    expert_flops = seq_len * d * cfg.expert_dim * 3 + seq_len * cfg.expert_dim * d
    router_flops = seq_len * d * cfg.n_experts
    moe_flops = (
        router_flops +
        cfg.n_experts_per_tok * expert_flops +
        seq_len * d * cfg.n_shared_experts * cfg.expert_dim * cfg.n_experts_per_tok * 3
    )
    
    recurrent_flops = n_loops * (
        attn_qkv_flops +
        attn_scores_flops +
        attn_out_flops +
        moe_flops +
        seq_len * d * cfg.lora_rank * 2 +
        seq_len * d * 2
    )
    
    coda_flops = n_coda * (attn_qkv_flops + attn_scores_flops + attn_out_flops + attn_ffn_flops)
    
    head_flops = seq_len * d * cfg.vocab_size
    
    total = embed_flops + prelude_flops + recurrent_flops + coda_flops + head_flops
    
    return {
        'embed': embed_flops,
        'prelude': prelude_flops,
        'recurrent_per_loop': attn_qkv_flops + attn_scores_flops + attn_out_flops + moe_flops,
        'recurrent_total': recurrent_flops,
        'coda': coda_flops,
        'head': head_flops,
        'total': total,
        'total_tflops': total / 1e12,
    }


def count_dense_flops(
    dim: int = 2048,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    n_layers: int = 10,
    seq_len: int = 2048,
    vocab_size: int = 32000,
) -> dict:
    """
    Estimate FLOPs per forward pass for a dense transformer baseline.
    """
    d = dim
    h = n_heads
    head_dim = d // h
    ffn_hidden = d * 4 // 3
    
    attn_qkv_flops = seq_len * d * d * 3
    attn_scores_flops = seq_len * h * head_dim * seq_len * 2
    attn_out_flops = seq_len * h * seq_len * head_dim * 2
    attn_ffn_flops = seq_len * d * ffn_hidden * 3 + seq_len * ffn_hidden * d
    layer_flops = attn_qkv_flops + attn_scores_flops + attn_out_flops + attn_ffn_flops
    
    embed_flops = 0
    body_flops = n_layers * layer_flops
    head_flops = seq_len * d * vocab_size
    
    total = embed_flops + body_flops + head_flops
    
    return {
        'embed': embed_flops,
        'body': body_flops,
        'head': head_flops,
        'total': total,
        'total_tflops': total / 1e12,
    }


def find_matching_layers(
    cfg: MythosConfig,
    target_loops: int = 8,
    seq_len: int = 2048,
) -> int:
    """
    Find the number of dense transformer layers that match FLOPs
    of RDT at the given loop count.
    """
    rdt_info = count_rdt_flops(cfg, seq_len, target_loops)
    rdt_tflops = rdt_info['total_tflops']
    
    for n_layers in range(1, 100):
        dense_info = count_dense_flops(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            vocab_size=cfg.vocab_size,
        )
        if dense_info['total_tflops'] >= rdt_tflops:
            return n_layers
    
    return -1


def main():
    # Use the canonical config. Tokenizer vocab_size is applied if available
    # so the head FLOPs are counted accurately.
    from configs.rdt_1_5b import RDT_1_5B_CONFIG
    cfg = RDT_1_5B_CONFIG
    try:
        from open_mythos.tokenizer import MythosTokenizer
        cfg.vocab_size = MythosTokenizer().vocab_size
    except Exception:
        pass
    
    print("=" * 60)
    print("FLOP Counting Results")
    print("=" * 60)
    
    print("\nRDT FLOPs by loop count:")
    print("-" * 60)
    print(f"{'Loops':<8} {'Total TFLOPS':<15} {'Per-Loop TFLOPS':<18}")
    print("-" * 60)
    
    for loops in [1, 4, 8, 12, 16]:
        info = count_rdt_flops(cfg, seq_len=2048, n_loops=loops)
        print(f"{loops:<8} {info['total_tflops']:<15.2f} {info['recurrent_per_loop']/1e12:<18.2f}")
    
    print("\n" + "=" * 60)
    print("Dense Baseline FLOPs by layer count:")
    print("-" * 60)
    print(f"{'Layers':<8} {'Total TFLOPS':<15}")
    print("-" * 60)
    
    for layers in [4, 6, 8, 10, 12, 14, 16, 20, 24]:
        info = count_dense_flops(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            n_layers=layers,
            seq_len=2048,
            vocab_size=cfg.vocab_size,
        )
        print(f"{layers:<8} {info['total_tflops']:<15.2f}")
    
    print("\n" + "=" * 60)
    print("FLOP-matched Dense Layers for RDT @ 8 loops:")
    print("-" * 60)
    matched = find_matching_layers(cfg, target_loops=8)
    dense_info = count_dense_flops(
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        n_layers=matched,
        seq_len=2048,
        vocab_size=cfg.vocab_size,
    )
    rdt_info = count_rdt_flops(cfg, seq_len=2048, n_loops=8)
    print(f"Matched layers: {matched}")
    print(f"Dense TFLOPS: {dense_info['total_tflops']:.2f}")
    print(f"RDT TFLOPS @ 8 loops: {rdt_info['total_tflops']:.2f}")
    print(f"Ratio: {dense_info['total'] / rdt_info['total']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()