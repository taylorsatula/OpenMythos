# OpenMythos Training Implementation — Technical Handoff

## Project Summary

Train a ~1.85B parameter Recurrent-Depth Transformer (RDT) language model based on the OpenMythos architecture (`https://github.com/kyegomezb/OpenMythos`). The goal is a proof-of-concept that validates three core RDT properties: stable training under recurrence, depth extrapolation at inference, and adaptive computation allocation via ACT halting. A FLOP-matched dense transformer baseline must be trained on identical data for controlled comparison.

The OpenMythos codebase provides a complete architecture definition in PyTorch (`open_mythos/main.py`) but no training loop, dataset pipeline, evaluation harness, or auxiliary losses. This document specifies everything needed to build those components.

---

## 1. Architecture Configuration

Use the existing `MythosConfig` dataclass. The target config for a ~1.85B total parameter model, of which ~1.5B is the transformer body (prelude + recurrent + coda). The remaining ~350M sits in the shared embedding + tied head, driven by the ~200K-vocab tokenizer choice (see §4.3). The RDT hypothesis is tested on the transformer body; the embedding is held identical between RDT and the dense baseline so it cancels from FLOP/param comparisons.

```python
MythosConfig(
    vocab_size=201088,        # overridden at runtime from tokenizer.vocab_size
    dim=2048,
    n_heads=16,
    n_kv_heads=4,             # GQA: 4 KV groups
    max_seq_len=2048,
    max_loop_iters=16,        # max loops at inference; training ramps up to 8
    prelude_layers=4,
    coda_layers=4,
    attn_type="gqa",          # GQA for first run; MLA adds complexity without aiding the RDT hypothesis test
    n_experts=128,
    n_shared_experts=2,
    n_experts_per_tok=4,
    expert_dim=512,
    act_threshold=0.99,
    rope_theta=500000.0,
    lora_rank=16,
)
```

### Why GQA over MLA for this run

MLA (Multi-Latent Attention) is a KV cache optimization for serving efficiency. It adds implementation complexity and an additional failure surface (the latent compression path). For a proof-of-concept focused on validating recurrence properties, GQA is simpler, better understood, and sufficient. MLA can be tested in a follow-up run if the RDT properties validate.

### Parameter budget target

Approximate breakdown at `dim=2048, n_experts=128, expert_dim=512, vocab≈200K`:

- Embeddings (tied with head): ~410M
- Prelude (4 dense blocks): ~110M
- Coda (4 dense blocks): ~110M
- Recurrent MoE block: ~436M (128 × 2048 × 512 × 3 routed + shared + attention)
- Overhead (LTI, ACT, LoRA, norms): ~1M
- **Total stored: ~1.07B transformer body + ~410M embedding ≈ 1.48B**

If total lands under 1.85B after `utils/count_params.py` verification, increase `expert_dim` to 640 or `n_experts` to 160 to reach the target without touching `dim`. **Do not finalize the config until `python -m utils.count_params` confirms transformer-body parameters within 5% of 1.5B.**

---

## 2. Critical Code Modifications to OpenMythos

The following changes must be made to `open_mythos/main.py` before training. Do not restructure the file unnecessarily — make surgical modifications.

### 2.1 Vectorized MoE Dispatch

The current `MoEFFN.forward()` uses nested Python for-loops over experts and top-k slots. This is the primary performance bottleneck.

Replace with a gather/scatter pattern:

```
1. Flatten input: (B*T, D)
2. Compute router logits, softmax, topk → topk_scores (B*T, K), topk_idx (B*T, K)
3. Renormalize topk_scores
4. Reshape for batched dispatch:
   - Repeat input K times: (B*T*K, D)
   - Flatten topk_idx: (B*T*K,)
   - Sort by expert_id to group tokens per expert
   - For each expert, slice the sorted tokens, run the expert, scatter results back
5. Weight by topk_scores and sum
6. Add shared expert outputs
```

The inner loop over experts is acceptable (it's O(n_experts) Python calls but each call processes a batch of tokens through a single linear layer, which saturates the GPU). The critical fix is eliminating the per-token inner loop.

### 2.2 Expose Per-Step Halting Probabilities

The `RecurrentBlock.forward()` currently accumulates `h_out` internally but does not return the halting probabilities needed for the ACT ponder cost. Modify the return signature:

```python
def forward(self, h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None):
    # ... existing loop logic ...
    return h_out, cumulative_ponder_cost
```

Where `cumulative_ponder_cost` is the sum of halting probabilities across all loop steps for all positions: `sum over t of (p_t * still_running_t)`, reduced to a scalar. This is the "remainder" formulation from Graves (2016).

### 2.3 LoRA Tensor Allocation Fix

In `LoRAAdapter.forward()`, replace `s = self.scale(torch.tensor(loop_t, device=x.device))` with:

```python
s = self.scale.weight[loop_t]  # direct index into embedding weight
```

### 2.4 Propagate Ponder Cost Through OpenMythos.forward()

Modify `OpenMythos.forward()` to return both logits and ponder cost:

```python
def forward(self, input_ids, n_loops=None, kv_cache=None):
    # ... existing logic ...
    x, ponder_cost = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
    # ... coda, norm, head ...
    return logits, ponder_cost
```

The `generate()` method does not need the ponder cost and can discard it.

---

## 3. Training Harness

### 3.1 Optimizer — Muon + AdamW hybrid

Use **Muon** (Jordan 2024; Moonlight / Liu et al. 2025 validated on 16B MoE) for the 2D weight matrices of the transformer body, and **AdamW** for everything else. Muon's Newton-Schulz-orthogonalized updates give roughly 35% better token efficiency than AdamW on dense LM workloads and accelerate grokking-style phase transitions — directly relevant to the RDT systematic-generalization test (Section 10, criterion #6).

Vendor the canonical implementation at `training/muon.py` (Keller Jordan's reference or the Moonlight variant — the two are interchangeable at this scale).

**Parameter routing**:

- **Muon**: 2D matrices in transformer blocks — attention `wq/wk/wv/wo`, FFN `gate/up/down`, MoE `router`, routed expert and shared expert `gate/up/down`, LoRA `down` and `B`. (Moonlight confirmed Muon is stable on MoE expert matrices despite their sparse gradient updates.)
- **AdamW**: everything else — embedding + tied head (sparse row-wise updates don't orthogonalize sensibly), RMSNorm weights, LTI `log_A / log_dt / B`, LoRA `scale` embedding, ACT `halt` bias, any other 1D or embedding-style parameter.

**Four groups total** — two per optimizer, with the 0.5× LR split preserved on the recurrent group in both:

```python
from training.muon import Muon

recurrent_keys = ('recurrent.', 'injection.', 'act.', 'lora.')
muon_default, muon_recurrent = [], []
adamw_default, adamw_recurrent = [], []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    is_recurrent = any(k in name for k in recurrent_keys)
    is_muon_eligible = (
        param.ndim == 2
        and 'embed' not in name
        and 'lora.scale' not in name
        and '.norm' not in name
    )
    bucket = (
        (muon_recurrent if is_recurrent else muon_default)
        if is_muon_eligible
        else (adamw_recurrent if is_recurrent else adamw_default)
    )
    bucket.append(param)

muon = Muon([
    {'params': muon_default,   'lr': 0.02, 'weight_decay': 0.1},
    {'params': muon_recurrent, 'lr': 0.01, 'weight_decay': 0.05},
], momentum=0.95, nesterov=True)

adamw = torch.optim.AdamW([
    {'params': adamw_default,   'lr': 3e-4,  'weight_decay': 0.1},
    {'params': adamw_recurrent, 'lr': 1.5e-4, 'weight_decay': 0.05},
], betas=(0.9, 0.95), eps=1e-8, fused=True)

optimizers = [muon, adamw]
```

Per training step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
for optim in optimizers:
    optim.step()
    optim.zero_grad(set_to_none=True)
for sched in schedulers:
    sched.step()
```

The recurrent group still gets half the base LR in both optimizers. Under Muon the "gradient-magnitude-scales-with-loop-count" argument dissolves (Newton-Schulz bounds step size), but the gradient-*variance*-scales-with-loop-count argument remains — halving the recurrent LR is conservative insurance worth keeping until the smoke test shows otherwise. Re-evaluate after the first 2,000 steps using per-group grad-norm logging (§7.1).

Gradient clipping at `max_norm=1.0` is applied globally (before either optimizer steps). Muon's orthogonalization operates on the clipped gradients — clipping does not interfere with its geometry.

### 3.1a Step Unit Convention

**Throughout this document, "step" means one optimizer update**, i.e. one `optimizer.step()` call after `gradient_accumulation_steps` micro-batches have been accumulated. All of the following are in optimizer-update units:

- `total_steps` (~29,000 for 15B tokens at 512K tokens/step)
- `warmup_steps` (2,000)
- `eval_interval` (500)
- `depth_extrapolation_interval` (5,000)
- `checkpoint_interval` (2,500)

The outer training loop must only increment its step counter inside the `if (micro_step + 1) % grad_accum == 0` branch. The LR scheduler, curriculum, wandb logging, eval, and checkpoint triggers all consume this single counter.

### 3.2 Learning Rate Schedule

Cosine decay with linear warmup, applied as a shared `lr_lambda` across both optimizers:

- Warmup: 2,000 optimizer updates (linear from 0 to peak LR)
- Decay: cosine from peak to 10% of peak over remaining ~27,000 updates
- All four parameter groups follow the same schedule shape; their base LRs (0.02 / 0.01 Muon, 3e-4 / 1.5e-4 AdamW) decay to 2e-3 / 1e-3 Muon, 3e-5 / 1.5e-5 AdamW.

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

schedulers = [
    torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda),
    torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda),
]
```

`scheduler.step()` is called once per optimizer update (after both `muon.step()` and `adamw.step()` have run).

### 3.3 Loop Depth Curriculum

The number of recurrent loops per training step ramps linearly from 1 to `max_loops_train=8` over the first 30% of training, then plateaus.

**Plateau behavior (revised)**: to keep the `LoRAAdapter.scale` embedding slots for loop indices 8–15 from going untrained, during the plateau phase sample `n_loops ~ Uniform{4, 5, ..., max_loop_iters}` per step instead of fixing at 8. This preserves average training cost near the 8-loop target while producing gradient signal for every LoRA depth slot, so inference at `n_loops > max_loops_train` no longer hits uninitialized LoRA deltas.

Training max loops = 16. Mean loops during plateau ≈ 10. Inference can extrapolate to `max_loop_iters` or beyond.

### 3.4 Composite Loss Function

Four terms, summed:

```python
# 1. Language modeling loss
lm_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

# 2. MoE load-balancing auxiliary loss — accumulated mean over loop iterations (see §3.5)
moe_aux_loss = model.recurrent.block.ffn.aux_loss

# 3. Router z-loss — penalizes unbounded router logit growth (ST-MoE, Zoph 2022)
moe_z_loss = model.recurrent.block.ffn.z_loss

# 4. ACT ponder cost (Graves 2016)
act_ponder_loss = ponder_cost.mean()

total_loss = lm_loss + 0.01 * moe_aux_loss + 0.001 * moe_z_loss + 0.001 * act_ponder_loss
```

Coefficient guidance:
- `moe_aux_coeff = 0.01`: Switch Transformer default.
- `moe_z_coeff = 0.001`: ST-MoE default. Keeps router logits bounded under bf16.
- `act_ponder_coeff = 0.001`: Deliberately low. Adjust per pitfalls table (§11).

### 3.5 MoE Auxiliary + Z-Loss Implementation

Inside `MoEFFN.__init__`:

```python
self.register_buffer('_aux_sum', torch.tensor(0.0), persistent=False)
self.register_buffer('_z_sum',   torch.tensor(0.0), persistent=False)
self.register_buffer('_loss_count', torch.tensor(0), persistent=False)
```

Inside `MoEFFN.forward`, after computing router logits and scores:

```python
# Switch-T aux: penalizes correlation between routing frequency and confidence
expert_mask = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=1)
f = expert_mask.float().mean(dim=0)
P = scores.mean(dim=0)
step_aux = self.n_experts * (f * P).sum()

# ST-MoE z-loss: penalizes unbounded logit magnitude
step_z = (torch.logsumexp(logits, dim=-1) ** 2).mean()

self._aux_sum = self._aux_sum + step_aux
self._z_sum   = self._z_sum + step_z
self._loss_count = self._loss_count + 1
self.aux_loss = self._aux_sum / self._loss_count.clamp(min=1)
self.z_loss   = self._z_sum   / self._loss_count.clamp(min=1)
```

**Reset hook**: `OpenMythos.forward()` must zero `_aux_sum`, `_z_sum`, and `_loss_count` on the recurrent MoE before the Prelude runs, so each outer forward pass sees a clean accumulator. Averaging over loop iterations preserves the Switch-T coefficient semantics (which were tuned assuming one aux value per forward).

### 3.6 Gradient Accumulation

Target batch size: 512K tokens = 256 sequences × 2048 tokens.

On H200-141GB with a ~1.85B param model (BF16 weights + FP32 optimizer states):

- Model weights (BF16): ~3.7 GB
- Optimizer states (FP32, AdamW): ~15 GB
- Gradients (BF16 during accum, FP32 at step): ~3.7–15 GB
- Logits buffer at 201K vocab, batch=32, seq=2048, bf16: ~52 GB (dominant)
- Activation memory at n_loops=8 through recurrent block: ~10–15 GB with default; ~3–5 GB with activation checkpointing
- Comfortable micro_batch: **32**
- Gradient accumulation: 256 / 32 = **8**

```python
accumulation_steps = 8
optimizer.zero_grad(set_to_none=True)
for micro_step in range(accumulation_steps):
    batch = next(data_iter)
    logits, ponder_cost = model(batch['input_ids'], n_loops=current_n_loops)
    loss = compute_composite_loss(logits, batch['targets'], ponder_cost, model)
    (loss / accumulation_steps).backward()
# clip + step happens once per N micro-steps
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
step += 1
```

### 3.6a Activation Checkpointing (optional on H200)

On H200-141GB the default activation memory fits, so checkpointing is **not required** for the specified config. Leave it disabled by default for the ~15% throughput gain. Re-enable by wrapping the inner recurrent-loop TransformerBlock call in `torch.utils.checkpoint.checkpoint(use_reentrant=False)` if:

- `n_loops` is pushed beyond 8 during training plateau sampling
- A follow-up run increases `dim` or sequence length
- Migration to smaller-VRAM hardware

### 3.7 Mixed Precision

Use `torch.autocast('cuda', dtype=torch.bfloat16)` for the forward pass. Keep optimizer states in FP32.

**Gradient scaling (`GradScaler`) must not be used with BF16.** BF16 has the same exponent range as FP32 and does not underflow; `GradScaler` is for FP16 only and will spuriously skip optimizer steps under BF16. Call `loss.backward(); clip; optimizer.step()` directly.

On H200, FP8 training via Transformer Engine is available and offers ~1.5–2× throughput on compute-bound ops. Deferred to follow-up runs — the FP8 path adds a calibration step and dependency surface that is not justified for the PoC.

### 3.8 Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` applied after accumulation, before `optimizer.step()`. Monitor grad norm per parameter group (§7.1); if the recurrent group's norm consistently runs >5× the default group's, the 0.5× LR split is under-correcting and should be revisited.

### 3.9 Initialization Overrides

After `OpenMythos._init_weights()` runs the default N(0, 0.02) init, apply these overrides:

1. **Depth-scaled output projections** (GPT-2 style): scale the residual-stream-output matrices (`attn.wo` and `ffn.down`) by `1/sqrt(2 * effective_depth)`, where `effective_depth = prelude_layers + max_loops_train + coda_layers = 16`. In code: after default init, walk the modules and for each such Linear, multiply weight by `1 / sqrt(32) ≈ 0.177`. Prevents residual magnitude growth through the looped depth.

2. **ACT halting bias**: `nn.init.constant_(model.recurrent.act.halt.bias, -2.0)`. Biases initial halting probability to ~0.12, preventing collapse-at-step-1 during early training.

3. **LTI injection near-identity**: initialize `LTIInjection.log_A` such that `A_init ≈ 0.9` per channel (i.e., `log_A = log(-log(0.9)) ≈ -2.25`); `log_dt = 0`. Keeps >90% of hidden state across early loops before the transformer contribution is meaningful, rather than the 37% the zero-init produces.

4. **Router bias**: initialized at zero (`register_buffer`). The DeepSeek-V3 bias update from §3.11 is **enabled by default** (`cfg.router_bias_update=True`); after each optimizer step it nudges the buffer toward uniform load.

### 3.10 Training Stability & Throughput Defaults

- **QK-norm**: in `GQAttention.forward`, apply `RMSNorm(head_dim)` to Q and K per-head before RoPE. Prevents attention-logit explosion, a standard mid-scale failure mode amplified by running the same attention T times in the recurrent loop. Adopted by DeepSeek-V3, Gemma-2/3. Cost: `2 * n_heads * head_dim` parameters per attention layer.
- **`torch.compile` on Prelude and Coda**: wrap each of `self.prelude` and `self.coda` (as `nn.Sequential` or individually) with `torch.compile(mode="reduce-overhead")`. Do NOT compile the recurrent block — dynamic `n_loops` and the Python-level MoE dispatch trigger recompiles or fallbacks. Expected: 20–30% throughput gain on those portions.
- **`set_to_none=True` on zero_grad**: skips the zeroing kernel.
- **Pinned memory + `non_blocking=True`** on host→device transfers in the dataloader. Already standard for the DataLoader's `pin_memory=True`.

### 3.11 DeepSeek-V3 Router Bias Update (on by default)

At `n_experts=192` the Switch-T aux loss alone tends to leave tail experts dead during early training. Augment it with the DeepSeek-V3 bias-update scheme:

```python
# After muon.step() + adamw.step() but before step += 1:
if cfg.router_bias_update:
    with torch.no_grad():
        ffn = model.recurrent.block.ffn
        counts = ffn._step_expert_counts          # accumulated across this step's forwards
        target = counts.sum() / ffn.n_experts
        imbalance = target - counts
        ffn.router_bias.add_(cfg.router_bias_update_rate * torch.sign(imbalance))
        ffn.reset_step_expert_counts()
```

Notes:

- **Non-learned.** `router_bias` is a buffer; no gradient flows through it and it does not interact with Muon's Newton-Schulz geometry. It rides on top of the gradient-driven `router.weight` (which Muon does optimize).
- **Per-expert count accumulator.** `MoEFFN._step_expert_counts` is a persistent buffer; it is populated inside `MoEFFN.forward` and spans both micro-batches and loop iterations within a single optimizer step. The training harness calls `reset_step_expert_counts()` after applying the update. Saved in checkpoints so resume is deterministic.
- **γ choice.** `cfg.router_bias_update_rate=1e-3` is the DeepSeek-V3 paper default. Higher values cause routing oscillation; lower fails to prevent dead-expert spiral under adversarial routing. Monitor `train/moe_dead_expert_count`.
- **Toggle.** Setting `cfg.router_bias_update=False` disables the update; the buffer still exists and is still added to router logits (at zeros), but nothing modifies it. Use as a debugging knob if routing appears to oscillate.

---

## 4. Dataset Pipeline

### 4.1 Composition

| Dataset | Proportion | Token Count (~15B total) | Source |
|---------|-----------|-------------------------|--------|
| FineWeb-Edu (score ≥ 3) | 60% | ~9B | `HuggingFaceFW/fineweb-edu` |
| OpenWebMath | 10% | ~1.5B | `open-web-math/open-web-math` |
| The Stack v2 (deduped) | 15% | ~2.25B | `bigcode/the-stack-v2-dedup` |
| Wikipedia + WikiBooks | 10% | ~1.5B | `wikimedia/wikipedia`, `wikimedia/wikibooks` |
| arXiv (abstracts + intros) | 5% | ~0.75B | `togethercomputer/RedPajama-Data-V2` arxiv subset |

### 4.2 Filtering

- **FineWeb-Edu**: Filter to `score >= 3` using the metadata field.
- **The Stack v2**: Filter to Python, Rust, TypeScript, C. Exclude files under 50 lines and files that are primarily comments/config/generated.
- **arXiv**: Extract only `abstract` and first section (introduction). Drop papers with >80% LaTeX notation density in the intro.
- **All sources**: Dedup at document level using SHA-256 hash of first 1000 characters. Remove documents under 100 tokens after tokenization.

### 4.3 Tokenization

Use the **`openai/gpt-oss-20b` tokenizer** via the existing `MythosTokenizer` wrapper. Rationale: strong code and math compression for the 25%+ code+math slice of the corpus, no gated-repo auth required, already wired. ~201K vocabulary; embedding + tied head cost ~410M params.

```python
from open_mythos.tokenizer import MythosTokenizer
tokenizer = MythosTokenizer()
vocab_size = tokenizer.vocab_size
```

**The tokenizer's actual `vocab_size` must be assigned to `cfg.vocab_size` at training start** before the model is constructed. Configs hold a placeholder only; the runtime source of truth is the tokenizer.

### 4.4 Sequence Packing

Pack tokenized documents into fixed-length sequences of 2048 tokens. Documents are concatenated with an EOS token separator. If a document is longer than 2048, split at 2048-token boundaries (no overlap).

Do not pad — always pack to fill. Padding wastes compute proportional to the padding ratio × loop count, which is significant in an RDT.

**Known limitation**: the causal mask allows attention across document boundaries within a packed sequence. A cleaner solution uses document-level attention masking via `FlexAttention`, but that integration is deferred to a follow-up run. Empirically the loss impact is 1–3% perplexity; acceptable for the PoC.

### 4.5 Data Loading

Use `datasets` library with interleaved streaming. Pre-tokenize and pack into arrow/parquet shards stored on local NVMe before training starts. Shuffle at the shard level.

---

## 5. Dense Baseline Model

The baseline must be FLOP-matched to the RDT at its training loop count.

### FLOP Matching Procedure

Do not hardcode `n_layers`. After the final RDT config is frozen (and after §3.5's aux accumulation is implemented — the MoE active ratio depends on correct aux balancing), run:

```bash
python -m utils.count_flops
```

`utils.count_flops.find_matching_layers()` computes the smallest `n_layers` such that `dense_total_FLOPs >= rdt_total_FLOPs` at `n_loops=max_loops_train`. The match is on **total per-token forward FLOPs including embedding and head** (which are identical across models). Record the computed layer count in `configs/baseline.py` with a comment referencing the RDT config it was matched against.

### Baseline Config

```python
BaselineConfig = dict(
    vocab_size=201088,        # overridden at runtime from tokenizer
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    n_layers=<from count_flops>,
    max_seq_len=2048,
    ffn_hidden_dim=2730,      # dim * 4 // 3 for SwiGLU
)
```

The baseline is a simplified OpenMythos: stack `TransformerBlock(use_moe=False)` N times with no recurrence, no ACT, no LoRA, no loop-index embedding, no LTI injection. Same tokenizer, same data, same optimizer (single group at lr=3e-4, wd=0.1), same schedule, same 15B-token budget.

---

## 6. Evaluation Suite

### 6.1 Validation Perplexity (every 500 steps)

Hold out 0.1% of the training data mix as a validation set. Compute perplexity on this set at the current training loop count.

### 6.2 Depth Extrapolation Test (every 5,000 steps)

Run the validation set at loop counts [4, 8, 12, 16]. Log perplexity at each depth. If perplexity improves at 12 or 16 vs 8, depth extrapolation is present.

For the dense baseline, log its single perplexity value for comparison.

### 6.3 ACT Halting Profile (every 5,000 steps)

On the validation set, record the loop step at which each position halts. Produce:
- Histogram of halt steps across all positions
- Mean halt step broken out by token type (punctuation, function words, content words, code tokens)

If halt step is uniform across token types, ACT is not learning useful allocation.

### 6.4 MoE Expert Utilization (every 5,000 steps)

On the validation set, record expert selections per token across all loop iterations. Compute:
- Per-expert token count (should be roughly uniform)
- Shannon entropy of expert selection distribution
- Number of "dead" experts receiving <0.1% of tokens
- Per-loop expert selection patterns

### 6.5 Synthetic Multi-Hop Arithmetic (post-training eval)

Generate synthetic data:
- Format: `Compute: 3 + 7 = 10, 10 * 2 = 20, 20 - 4 = 16. Answer: 16`  *(with intermediate values; answer masked during eval)*
- Training-range depths: K = 2, 3, 4, 5
- Extrapolation depths: K = 6, 8, 10, 12, 15
- 500 examples per depth, operations: +, -, × (single-digit operands, results capped at 3 digits)

Fine-tune both the trained RDT and baseline on the K=2–5 training set for 1,000 steps (lr=1e-5). Evaluate exact-match accuracy at each depth. Run the RDT at loop counts [8, 12, 16].

### 6.6 Standard Benchmarks (post-training)

Run ARC-Easy and HellaSwag via `lm-evaluation-harness`. Sanity check that the model has learned language.

---

## 7. Logging and Checkpointing

### 7.1 Logging

Use Weights & Biases (`wandb`). Log per optimizer step:
- `train/loss` (composite)
- `train/lm_loss`
- `train/moe_aux_loss`
- `train/moe_z_loss`
- `train/act_ponder_cost`
- `train/grad_norm/total`, `train/grad_norm/muon_default`, `train/grad_norm/muon_recurrent`, `train/grad_norm/adamw_default`, `train/grad_norm/adamw_recurrent` (per-group, pre-clip)
- `train/lr/muon_default`, `train/lr/muon_recurrent`, `train/lr/adamw_default`, `train/lr/adamw_recurrent`
- `train/n_loops` (current loop depth from curriculum)
- `train/tokens_seen`
- `train/spectral_radius_max` (max of LTI injection's A matrix — should always be < 1)
- `train/moe_dead_expert_count` (experts receiving <0.1% of tokens in current batch)
- `train/act_halt_mean` (average halt step across batch)

Log per eval (every 500 steps for perplexity, every 5000 for full suite):
- `eval/perplexity_loopN` for each evaluated loop count
- `eval/act_mean_halt_step`
- `eval/expert_entropy`
- `eval/dead_expert_count`

### 7.2 Checkpointing

Save full checkpoint every 2,500 optimizer steps:
- Model state dict (rank-0 only under DDP/FSDP)
- **Both** optimizer state dicts (`muon.state_dict()` and `adamw.state_dict()`, keyed separately)
- **Both** scheduler state dicts
- RNG states (torch, cuda, numpy, python random)
- Current step number
- Current loop depth (from curriculum)
- Wandb run ID
- The `MythosConfig` used to construct the model (for resume shape-safety)

Store checkpoints on persistent storage. Sync to cloud (S3, GCS, or similar) after each save. Rotate: keep the 5 most recent plus the step-0, halfway, and final checkpoints. Budget ~100GB locally.

Implement resume-from-checkpoint: the training script must accept a `--resume` flag pointing to a checkpoint file and restore all state exactly. This is non-negotiable for Vast.AI training where instances can be preempted.

Use `torch.load(path, weights_only=False, map_location='cpu')` and move tensors to device post-load. The `weights_only=False` is explicit because the checkpoint holds the pickled `MythosConfig`.

### 7.3 Early Termination Signals

Stop training early and investigate if:
- `spectral_radius_max >= 1.0` (LTI stability violated — should be architecturally impossible but check anyway)
- Loss spikes by >2× and does not recover within 500 steps
- `grad_norm` consistently clips at 1.0 for >1000 consecutive steps (recurrent LR too high)
- `dead_expert_count > n_experts * 0.25` (routing collapse — consider enabling the DeepSeek-V3 bias update from §11)
- ACT halts at step 1 for >95% of positions after ramp completes (ponder cost too high or halting head not learning)

---

## 8. Infrastructure

### 8.1 Hardware Target

**H200-141GB on Vast.AI. Single GPU.** Estimated cost per run: ~$195 at ~$3.50/hr for ~55 hours.

Rationale: H200 provides 141GB HBM3e, enough to run the ~201K-vocab tokenizer path without fused-CE gymnastics (the per-step logits buffer is ~52GB at `batch=32, seq=2048, bf16`). BF16 compute is ~3× A100 per TFLOP, bringing wall-clock under the A100 baseline at comparable total cost. FP8 training is available but deferred to follow-ups.

**Preemption risk**: H200 availability on Vast.AI is tighter than A100. The checkpoint-on-step + cloud-sync + resume path (§7.2) is therefore load-bearing, not optional.

### 8.2 Software Stack

- PyTorch >= 2.4 with CUDA 12.x
- `transformers` (tokenizer only)
- `datasets` (data loading)
- `wandb`
- `lm-eval-harness`

### 8.3 Disk Requirements

- Dataset (pre-tokenized shards): ~60GB
- Checkpoints (rotated): ~100GB
- Code and logs: ~1GB
- Total: 170GB minimum on local NVMe

### 8.4 Data Preparation (run before training)

`prepare_data.py` must:
1. Download and filter each dataset component
2. Tokenize with the `MythosTokenizer` (gpt-oss-20b)
3. Pack into 2048-token sequences (EOS-separated)
4. Save as parquet/arrow shards (~1GB each) to local NVMe
5. Create a validation split (0.1%)
6. Print token counts per source and total

Run once before training. I/O bound.

---

## 9. File Structure

```
openmythos-train/
├── open_mythos/
│   ├── __init__.py
│   ├── main.py            # modified per §2 (MoE vectorization, ponder cost, QK-norm, aux/z-loss accumulation)
│   ├── baseline.py        # dense transformer baseline model
│   └── tokenizer.py       # MythosTokenizer wrapping gpt-oss-20b
├── data/
│   ├── prepare_data.py    # dataset download, filter, tokenize, pack
│   ├── dataloader.py      # packed sequence DataLoader
│   └── synthetic.py       # multi-hop arithmetic generation (answer masked at eval time)
├── training/
│   ├── train.py           # main training loop
│   ├── losses.py          # composite loss (LM + MoE aux + z-loss + ACT ponder)
│   ├── curriculum.py      # loop depth schedule with plateau sampling
│   ├── muon.py            # vendored Muon optimizer (Jordan 2024 / Moonlight variant)
│   └── checkpointing.py   # save/resume logic (rank-0 aware, config + both optim states + RNG)
├── eval/
│   ├── evaluate.py        # orchestrates all evals
│   ├── perplexity.py      # validation perplexity at variable loop counts
│   ├── act_profile.py     # halting histogram analysis
│   ├── expert_util.py     # MoE routing analysis
│   └── arithmetic.py      # synthetic multi-hop eval
├── utils/
│   ├── count_params.py    # parameter counting utility
│   ├── count_flops.py     # FLOP estimation + find_matching_layers()
│   └── logging.py         # wandb setup
├── configs/
│   ├── rdt_1.5b.py        # RDT model config
│   └── baseline.py        # dense baseline config (n_layers set from count_flops)
├── scripts/
│   ├── run_training.sh    # launch script
│   └── run_eval.sh        # post-training eval
└── HANDOFF.md             # this document
```

---

## 10. Success Criteria

1. **Training loss decreases stably** as loop depth ramps from 1 to 8. No divergence, no persistent loss spikes. Spectral radius remains < 1 throughout.
2. **Depth extrapolation**: perplexity at loop=12 or 16 is lower than at loop=8 on the validation set. Single most important result.
3. **ACT learns non-uniform halting**: halt-step distribution correlates with token complexity.
4. **MoE routing differentiates**: high expert entropy, low dead expert count, different loops show different expert patterns.
5. **Competitive perplexity per FLOP**: the RDT matches or beats the dense baseline at equal transformer-body FLOP budget. Within 5% on perplexity while winning on depth extrapolation counts as positive.
6. **Compositional generalization**: on synthetic multi-hop arithmetic, RDT maintains accuracy at K>5 where the baseline degrades.

---

## 11. Known Pitfalls and Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss spikes at loop ramp transitions | Gradient magnitude jumps when loop count increases | Smooth the ramp (fractional loop counts via probabilistic sampling), or reduce recurrent group LR further |
| ACT halts everything at step 1 | Ponder cost coefficient too high, or halting head bias not at -2.0 (§3.9) | Reduce `act_ponder_coeff` to 0.0005; verify `ACTHalting.halt.bias == -2.0` |
| ACT never halts (always runs max loops) | Ponder cost coefficient too low, or halting head not receiving meaningful gradients | Increase `act_ponder_coeff` to 0.005 |
| Expert routing collapse (all tokens → same experts) | Aux or z-loss coefficient too low, or routing dynamics stuck | Increase `moe_aux_coeff` to 0.02; if still collapsed, enable DeepSeek-V3 bias update: after each optimizer step, compute per-expert load from last batch's `expert_mask`, then `router_bias[i] += 1e-3 * sign(target_load - observed_load[i])` |
| Spectral radius approaching 1.0 | LTI parameterization is correct by construction, so this shouldn't happen | Check that `log_A` is not being overridden or bypassed; verify `get_A()` output in a test |
| Gradient norm consistently at clip threshold | Recurrent LR too high, or loop count too high for current training stage | Reduce recurrent group LR; slow the loop depth ramp |
| Training much slower than expected | MoE dispatch still using Python loops, or torch.compile fell back | Verify vectorized dispatch; confirm `torch.compile` didn't silently fall back on the Prelude/Coda (check `TORCH_LOGS=+dynamo`) |
| VRAM OOM | Microbatch too large at high loop counts | Reduce microbatch; increase gradient accumulation; enable activation checkpointing (§3.6a) |
| Perplexity degrades at higher loop counts (inference) | Overthinking — hidden state drifts past solution | Expected beyond some depth. Degradation onset should be beyond training depth; if it occurs *at* training depth, the model isn't learning stable fixed points |
| Inference at n_loops > max_loops_train degrades sharply | LoRA scale slots for loop indices past training depth are still uninitialized | Confirm plateau sampling from §3.3 is active; if disabled, fall back to zeroing `LoRAAdapter.delta` for t > max_loops_train at inference |
| LTI injection decays signal too aggressively early in training | `log_A` init was left at 0 (gives A ≈ 0.37) | Apply §3.9 init: `log_A = log(-log(0.9)) ≈ -2.25` → A_init ≈ 0.9 |
| Muon updates produce NaN | Newton-Schulz iteration instability, typically from mixed-precision or extremely small gradient norms | Ensure the orthogonalization step runs in fp32 regardless of autocast dtype; raise the Newton-Schulz iteration count from 5 to 6 if instability persists |
| Loss explodes in first few hundred steps with Muon | Muon LR too high for unwarmed model | Confirm warmup is active on the Muon optimizer (not just AdamW); lower Muon peak LR to 0.01 / 0.005 as a first bisection |
| Grokking phase transition never happens on synthetic eval | Muon weight decay too high, collapsing expressive capacity before compositional structure emerges | Reduce `weight_decay` on Muon groups from 0.1 / 0.05 to 0.01. Moonlight noted Muon is more sensitive to WD than AdamW |
| Muon-routed MoE expert matrices drift apart in magnitude | Sparse per-expert gradient frequency interacts with momentum buffer stale-ness | Monitor `train/grad_norm/muon_recurrent` vs `muon_default`; if the ratio exceeds ~3× after step 2000, consider moving expert matrices to AdamW as a fallback |

---

## 12. Execution Order

1. Clone repo, set up environment, install dependencies
2. Vendor Muon into `training/muon.py` (Keller Jordan's reference implementation or the Moonlight variant)
3. Apply §2 modifications to `open_mythos/main.py`, §3.9 init overrides, §3.10 QK-norm
4. Run `python -m utils.count_params` against `configs/rdt_1.5b.py`; adjust `expert_dim` or `n_experts` to hit transformer-body target
5. Run `python -m utils.count_flops` to determine dense baseline `n_layers`; write into `configs/baseline.py`
6. Run `python -m data.prepare_data` to build tokenized shards on local NVMe
7. Smoke test: `python -m scripts.smoke_test` — 10 training steps on synthetic data verifying loss decrease, spectral radius < 1, MoE aux + z-loss + ACT ponder compute, gradients flow to all four optimizer groups, per-group grad norms logged, both Muon and AdamW states saved and restored cleanly
8. Train RDT for 15B tokens (~29,000 optimizer steps)
9. Train dense baseline for 15B tokens (same data order if possible)
10. Run full eval suite on both models
11. Generate comparison report
