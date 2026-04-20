#!/usr/bin/env python3
"""Parameter counting utility for MythosConfig variants."""

import torch
from open_mythos import MythosConfig, OpenMythos


def count_params(model: torch.nn.Module) -> dict:
    """
    Count parameters by leaf module. Unique-tensor counting: a weight that
    appears in multiple modules (e.g. a weight-tied LM head) is counted
    once, not once per module.
    """
    counts = {}
    seen = set()
    total = 0

    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        n_unique = 0
        for p in module.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            n_unique += p.numel()
        if n_unique > 0:
            counts[name] = n_unique
            total += n_unique

    return counts, total


def main():
    # Use the canonical config. If MythosTokenizer is available, override
    # vocab_size to the tokenizer's actual size for an accurate total count.
    from configs.rdt_1_5b import RDT_1_5B_CONFIG
    cfg = RDT_1_5B_CONFIG
    try:
        from open_mythos.tokenizer import MythosTokenizer
        tokenizer = MythosTokenizer()
        cfg.vocab_size = tokenizer.vocab_size
        print(f"Using tokenizer vocab_size = {cfg.vocab_size:,}")
    except Exception as e:
        print(f"Could not load tokenizer ({e}); using placeholder vocab_size = {cfg.vocab_size:,}")

    model = OpenMythos(cfg)
    
    counts, total = count_params(model)
    
    print(f"\n{'='*60}")
    print(f"MythosConfig Parameter Count")
    print(f"{'='*60}")
    print(f"Total parameters: {total:,} ({total/1e9:.2f}B)")
    print(f"\nBreakdown by component:")
    print("-"*60)
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for name, n in sorted_counts[:30]:
        print(f"  {name:<45} {n:>12,}")
    
    if len(sorted_counts) > 30:
        print(f"  ... and {len(sorted_counts) - 30} more components")
    
    print(f"\n{'='*60}")
    
    active_per_token = (
        cfg.n_experts_per_tok * cfg.expert_dim * 3 +  # routed expert FFN
        cfg.n_shared_experts * cfg.expert_dim * cfg.n_experts_per_tok * 3 +  # shared expert FFN
        cfg.dim * 2  # attention
    )
    print(f"Approximate active parameters per token (forward pass):")
    print(f"  Attention: {cfg.dim * 2:,}")
    print(f"  Routed experts: {cfg.n_experts_per_tok * cfg.expert_dim * 3:,}")
    print(f"  Shared experts: {cfg.n_shared_experts * cfg.expert_dim * cfg.n_experts_per_tok * 3:,}")
    print(f"  Total: {active_per_token:,}")
    
    embed_params = cfg.vocab_size * cfg.dim
    print(f"\nEmbedding parameters: {embed_params:,} ({embed_params/1e6:.1f}M)")
    print(f"Non-embedding parameters: {total - embed_params:,} ({(total-embed_params)/1e9:.2f}B)")
    
    print(f"{'='*60}\n")
    
    param_dict = {
        'total': total,
        'embed': embed_params,
        'non_embed': total - embed_params,
        'by_component': counts,
    }
    
    return param_dict


if __name__ == "__main__":
    main()