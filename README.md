# OpenMythos — Training Harness Fork

A fork of [`kyegomez/OpenMythos`](https://github.com/kyegomez/OpenMythos) that adds a complete training harness on top of the upstream architecture code. The upstream repo provides the model definition (Recurrent-Depth Transformer with MoE FFN, MLA/GQA attention, LTI-stable injection, ACT halting); this fork adds everything else needed to actually train the thing end-to-end on a single H200.

> **Status: untested at scale.** The smoke test suite passes and the code is internally consistent with the design captured in [`HANDOFF.md`](HANDOFF.md). However, **no full-scale training run has been executed**. Expect bugs that only surface once the real data pipeline, H200 hardware, and 15B-token workload are exercised together. Treat this as a well-organized first-draft harness, not a battle-tested one.

---

## What's in the fork

### Upstream (preserved)
- `open_mythos/main.py` — RDT model (Prelude → looped Recurrent Block → Coda), MoE, GQA/MLA attention, LTI injection, ACT, LoRA, loop-index embedding.
- `open_mythos/baseline.py` — FLOP-matched dense transformer baseline.
- `open_mythos/tokenizer.py` — `MythosTokenizer` wrapping `openai/gpt-oss-20b`.
- `open_mythos/variants.py` — 1B–1T preset configs.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the upstream README with the full hypothesis write-up.

### Added in this fork

**Training pipeline**
- `training/train.py` — RDT + dense baseline training loops, Muon+AdamW hybrid optimizer, shared warmup-cosine schedule across both, step-unit-correct gradient accumulation, bf16 autocast (no GradScaler), global grad-clip, per-group grad-norm logging, spectral-radius safety gate, `torch.compile` on Prelude/Coda (not the recurrent loop).
- `training/muon.py` — vendored [Keller Jordan Muon](https://github.com/KellerJordan/modded-nanogpt) optimizer with Newton-Schulz orthogonalization in fp32, Nesterov momentum, shape-scaled updates, decoupled weight decay. Routed: 2D transformer matrices + MoE expert matrices. Everything else (embeddings, norms, 1D tensors) goes to AdamW. Moonshot's Moonlight 16B MoE paper validated this split.
- `training/losses.py` — composite loss (LM + MoE aux + MoE z-loss + ACT ponder), reading from the running-mean accumulators.
- `training/curriculum.py` — loop-depth curriculum: linear ramp 1→8 over first 30%, then plateau samples `n_loops ~ U[4, max_loop_iters]` so LoRA scale slots for loops 8..15 also receive gradient signal. RNG serialized in checkpoints for resume determinism.
- `training/checkpointing.py` — full state save/restore: both optimizer state dicts, both scheduler states, curriculum state (incl. RNG), torch/cuda/numpy/python RNG, step/tokens/loss-history, the `MythosConfig` used to build the model, and the wandb run id. Rank-0 guard. Checkpoint rotation.

**Architectural additions to `open_mythos/main.py`**
- **MoE aux + z-loss accumulators** with a `reset_loss_accumulators()` hook called at the top of every `OpenMythos.forward`. Previously `aux_loss` was overwritten every loop iteration, so T−1 loops worth of routing signal was discarded. Now the optimizer sees the mean across all loop iterations; Switch-T coefficient semantics preserved.
- **Observability hooks** on MoEFFN, GQAttention, and RecurrentBlock — `last_router_logits_abs_max`, opt-in `last_attn_logits_abs_max` (gated by a `diagnostics` flag; one extra QK matmul per attention block when on, zero cost when off), `last_hidden_abs_maxes` per loop, `last_halt_prob_means` per loop. These drive the "broken but looks working" detector set in `utils/logging.py`.
- **Structural invariant helpers** on `OpenMythos` — `assert_weight_tying()` (raises if `head.weight` and `embed.weight` ever stop sharing storage, which `torch.compile` can silently break) and `unique_param_count()` (tying-aware). Run at training start, logged every step, and re-asserted at every checkpoint.
- **DeepSeek-V3 router-bias update** (on by default at `n_experts=192`). `MoEFFN._step_expert_counts` accumulates per-expert routing counts across all micro-batches and loop iterations in an optimizer step; the training harness nudges `router_bias` by `γ · sign(target_load − observed_load)` after `optimizer.step()` and resets. Hard load-balance force complementary to the soft aux loss; avoids dead-expert spiral during early training. γ=1e-3 default; toggleable via `cfg.router_bias_update`.
- **ST-MoE z-loss** — `logsumexp(router_logits)² mean` with coefficient 1e-3. Keeps router logits bounded under bf16.
- **QK-norm** (DeepSeek-V3 / Gemma-2 style) on both GQA and MLA. Bounds attention logits across the looped recurrent block where the same attention runs T times.
- **Depth-scaled output projections** at init: `attn.wo` and `ffn.down` multiplied by `1/sqrt(2 · effective_depth) ≈ 0.177` (effective_depth = prelude + max_loops_train + coda = 16). Standard GPT-2 trick adapted for the effective depth of a looped model.
- **ACT halt bias `= −2.0`** so the halting head starts at sigmoid(−2) ≈ 0.12, preventing collapse-at-step-1.
- **LTI near-identity init**: `log_A = log(−log(0.9)) ≈ −2.25`, `log_dt = 0`, giving `A_init ≈ 0.9` per channel. Previously `log_A = 0` gave `A ≈ 0.37` — 63% hidden-state decay per loop before the transformer even contributes.
- `muon_param_predicate` helper for clean Muon-vs-AdamW parameter routing.

**Data pipeline**
- `data/dataloader.py` — real Parquet-backed `PackedSequenceDataset` (previously a random-tensor stub that silently fed the trainer with noise).
- `data/prepare_data.py` — real streaming download/filter/dedup/tokenize/pack pipeline for the 5-source mix (FineWeb-Edu, OpenWebMath, The Stack v2 dedup, Wikipedia, arXiv). Writes zstd-compressed Parquet shards.
- `data/synthetic.py` — multi-hop arithmetic generator with intermediate values shown, answer maskable at eval time, isolated `random.Random(seed)` per call.

**Utilities and eval**
- `utils/count_params.py` — unique-tensor counting (previously double-counted the tied LM head). Auto-pulls tokenizer vocab size.
- `utils/count_flops.py` — `find_matching_layers()` for the FLOP-matched dense baseline. Currently recommends **24 layers** for the configured RDT.
- `utils/logging.py` — full observability surface (HANDOFF §7.1): per-group grad norms + LRs, expected-vs-actual LR diff, pre-softmax router/attention/hidden-state abs-maxes, per-loop ACT halt means, per-term loss fraction of total, structural-invariant unique param count, dead-expert count, spectral radius, periodic per-expert weight-norm & Muon-momentum distributions, decoded-sample text tables, token-id histograms, depth-extrapolation probes, synthetic multi-hop arithmetic probes, wandb `resume='must'` support.
- `eval/expert_util.py` — hook-based expert counting via the `MoEFFN.last_topk_idx` cache (previously referenced a non-existent `router.expert_counts` attribute).
- `eval/perplexity.py`, `eval/act_profile.py` — dropped the `model.train()` side effect that clobbered caller mode.
- `eval/arithmetic.py` — masked-prompt evaluation; decode only generated tokens (avoids extracting answer from the echoed prompt).

**Scripts**
- `scripts/run_training.sh` — launch script. `CUDA_LAUNCH_BLOCKING=1` removed (it serializes kernel launches and ~3×'d the step time).
- `scripts/smoke_test.py` — 9 tests covering forward-return signature, init overrides (ACT bias, LTI A_init, QK-norm presence), MoE aux/z-loss accumulator reset per forward, spectral radius < 1, 4-group optimizer partition (muon/adamw × default/recurrent), gradient flow through all groups, loss-decrease on a copy-task toy, curriculum plateau coverage, and checkpoint round-trip (both optimizer states, both schedulers, curriculum RNG).

**Documentation**
- [`HANDOFF.md`](HANDOFF.md) — the full technical spec driving the training work. Covers optimizer, losses, curriculum, gradient accumulation, init, mixed precision, data pipeline, baseline FLOP matching, eval, logging, checkpointing, infrastructure, execution order, and a debugging guide.

---

## Target configuration

| | Value |
|---|---|
| Architecture | RDT with GQA attention + MoE FFN |
| Dim | 2048 |
| Prelude / Coda layers | 4 / 4 |
| Max loop iters (inference) | 16 |
| Max loops (training) | 8 (ramp) → sampled in plateau |
| Experts | 192 routed + 2 shared, `expert_dim=1024`, top-4 routing |
| Tokenizer | `openai/gpt-oss-20b`, ~201k vocab |
| Transformer body | **1.49B params** (measured) |
| Embedding + tied head | ~410M |
| Total | **1.90B params** |
| Context | 2048 |
| Target tokens | 15B |
| Global batch | 512k tokens (micro-batch 32 × grad_accum 8 × seq 2048) |
| Optimizer steps | ~29 000 |
| Peak LR | 0.02 (Muon), 3e-4 (AdamW); half for recurrent group in both |
| Schedule | Linear warmup 2 000 steps → cosine to 10% |
| Target hardware | H200-141GB single GPU on Vast.AI (~55 hr, ~$195) |
| Baseline dense layers | 24 (FLOP-matched to RDT @ 8 loops; ratio 1.004) |

---

## How to run

1. **Environment**: PyTorch ≥ 2.4, CUDA 12.x, `transformers`, `datasets`, `pyarrow`, `wandb`.
2. **Verify the config**:
   ```bash
   python -m utils.count_params   # should report transformer body ~1.49B
   python -m utils.count_flops    # should report 24 dense layers @ matched FLOPs
   ```
3. **Prepare data** (run once on a box with bandwidth; ~60 GB disk + HF auth):
   ```bash
   python -m data.prepare_data --target_tokens 15_000_000_000
   ```
4. **Smoke test** (CPU is fine):
   ```bash
   python -m scripts.smoke_test
   ```
5. **Train**:
   ```bash
   bash scripts/run_training.sh rdt
   # or for the baseline:
   bash scripts/run_training.sh baseline
   ```

Resume-from-checkpoint:
```bash
bash scripts/run_training.sh rdt --resume outputs/rdt_1.5b/checkpoints/checkpoint_step_5000.pt
```

---

## What is *not* yet done

- **No full training run has happened.** The code has not been exercised against the 15B-token FineWeb-Edu+code+math mix on H200 (or any other production setting). Expect some combination of: Parquet shard schema mismatches the first time `prepare_data.py` is run at scale, `torch.compile` recompile storms on sequence-length variations, MoE routing collapse under Muon that the smoke test can't surface at tiny scale, off-by-one in `next()` iteration handling when the data stream rolls over, numerical surprises in the Newton-Schulz iteration at bf16 boundaries, and/or activation-memory regressions from `torch.compile`'s chosen graph. The smoke test validates the *plumbing*; it cannot validate training dynamics.
- Document-boundary attention masking via FlexAttention — deferred.
- FP8 training via Transformer Engine — deferred.
- Distributed Muon — single-GPU only for now.

---

## Credits

- Upstream model, theory, and documentation: [Kye Gomez, OpenMythos](https://github.com/kyegomez/OpenMythos). See [`ARCHITECTURE.md`](ARCHITECTURE.md).
- Muon optimizer: [Keller Jordan, modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- Muon+MoE validation: [Liu et al., "Muon is Scalable for LLM Training" / Moonlight 16B](https://arxiv.org/abs/2502.16982) (Feb 2025).

## License

MIT — inherited from the upstream [OpenMythos](https://github.com/kyegomez/OpenMythos). See [`LICENSE`](LICENSE).
