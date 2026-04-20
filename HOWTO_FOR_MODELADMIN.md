# HOWTO for the Model Administrator

You (a Claude instance) are responsible for monitoring and administering a
~55-hour, ~$200 OpenMythos RDT training run from this terminal. There is
no human in the loop. This document is your operations manual.

## Orientation

**What's running.** A 1.5B-parameter recurrent-depth transformer training
on `configs/rdt_1_5b.py` for ~29K optimizer steps. Hard gate: spectral
radius `ρ(A) < 1.0` — the loop will raise if violated. Everything else
you own.

**What you control.** A JSON config file (runtime-polled) and a one-shot
command queue. All intervention is file-based. No sockets, no TUI.

**What you read.** A single status snapshot (`status.json`), plus three
append-only JSONL streams: metrics, incidents, generations.

**Economics.** One hour of drift costs ~$3.50. A bad intervention that
corrupts a checkpoint is more expensive than pausing to think. **Measure
twice, cut once.**

## The file map

All paths live under `outputs/rdt_1.5b/` (or whatever `--output` was
passed to `train.py`). The dashboard lays out:

```
outputs/rdt_1.5b/
├── checkpoints/                     — model snapshots, disk-gated
├── admin/
│   ├── config.json                  — persistent knobs YOU edit
│   ├── config_applied.json          — last-known-good (rollback source)
│   ├── prompts.json                 — fixed-prompt panel for gen probe
│   ├── status.json                  — current training state (read this first)
│   ├── audit.jsonl                  — every accepted change, with diff
│   └── commands/
│       ├── pending/                 — YOU drop command JSON here
│       ├── applied/                 — loop moves here + populates result
│       └── rejected/                — bounds violation / unknown command
└── events/
    ├── metrics/metrics.NNN.jsonl    — per-step training metrics (chunked 5K lines)
    ├── incidents.jsonl              — WARN/CRIT events (append-only)
    └── generations.jsonl            — fixed-prompt panel + on-demand gens
```

Everything is UTF-8 JSON or newline-delimited JSON. Read with `Read` /
`Grep`, write through the CLI.

## Daily workflow

### 1. First touch on any session

```bash
python -m tools.mythos_admin status
```

This gives you: current step, loss trajectory, spectral radius, throughput,
disk, cost-to-date, last 5 incidents. Do this before anything else.

If `heartbeat_at` is older than ~5 minutes, the loop is dead. If
`last_step_advance_at` is much older than `heartbeat_at`, the loop is
wedged (dataloader stall, NaN hang, etc.) — do not issue knob changes;
diagnose first.

### 2. Recent metrics

```bash
python -m tools.mythos_admin recent -n 30
```

Summarizes the last 30 logged steps. For the raw JSONL:

```bash
python -m tools.mythos_admin recent -n 30 --raw
```

Or `Grep` the chunk directly:

```bash
Grep pattern="train/loss" path="outputs/rdt_1.5b/events/metrics"
```

### 3. Incident triage

```bash
python -m tools.mythos_admin incidents -n 50
python -m tools.mythos_admin incidents --severity CRIT
```

See the **Incident catalog** below for what each `kind` means.

### 4. Tuning a knob

```bash
python -m tools.mythos_admin config get lr_mult.muon_default
python -m tools.mythos_admin config set lr_mult.muon_default 0.5
```

The CLI validates bounds, writes atomically, appends an audit entry. The
loop picks up the change at the next step (mtime-gated poll).

### 5. One-shot command

```bash
python -m tools.mythos_admin cmd checkpoint_now --wait 60
python -m tools.mythos_admin cmd generate --arg 'prompt="2+2="' --arg max_new_tokens=16 --wait 30
```

`--wait N` polls `applied/` for N seconds and prints the result. Without
`--wait` the command fires and forgets — check `applied/<filename>.json`
later.

### 6. Rollback a bad config change

```bash
python -m tools.mythos_admin config rollback
```

Copies `config_applied.json` → `config.json`. `config_applied.json`
mirrors the last config that completed a successful step. If the next
step after your edit NaN'd, rollback is safe.

## Config reference

`admin/config.json` — edit via `config set <dotted_key> <json_value>`. All
writes are bounds-checked (CLI) and re-validated (loop). Out-of-bounds =
file not written. Invalid JSON = loop uses last-good and emits
`config_parse_failed`.

| Key | Type | Bounds | Default | Notes |
|---|---|---|---|---|
| `lr_mult.muon_default` | float | [0.0, 2.0] | 1.0 | Multiplier on Muon default-group LR, applied every step post-scheduler. |
| `lr_mult.muon_recurrent` | float | [0.0, 2.0] | 1.0 | Recurrent-group Muon LR multiplier (base is already 0.5× of default). |
| `lr_mult.adamw_default` | float | [0.0, 2.0] | 1.0 | AdamW default-group. |
| `lr_mult.adamw_recurrent` | float | [0.0, 2.0] | 1.0 | AdamW recurrent-group. |
| `loss_coeffs.moe_aux` | float | [0.0, 0.2] | 0.01 | Switch-T aux load-balance coefficient. |
| `loss_coeffs.moe_z` | float | [0.0, 0.1] | 1e-3 | ST-MoE z-loss (router-logit magnitude bound). |
| `loss_coeffs.act_ponder` | float | [0.0, 0.1] | 1e-3 | ACT ponder cost (halting regularizer). |
| `grad_clip` | float | (0, 10.0] | 1.0 | Global max-norm; change cautiously, 1.0 is load-bearing. |
| `spectral_warn_threshold` | float | [0.5, 0.9999] | 0.999 | ρ(A) soft-warn threshold — WARN incident on crossing. |
| `router_bias_update` | bool | — | true | DeepSeek-V3 per-step router bias nudge. |
| `router_bias_update_rate` | float | [0.0, 0.01] | 1e-3 | Bias-update γ. 1e-3 is paper default. |
| `n_loops_override` | int ∪ null | [1, max_loop_iters] | null | Bypass curriculum. Null = follow curriculum. |
| `pause_until_step` | int ∪ null | [current, current+500] | null | Suspend the loop until the given absolute step. **Auto-expires at 500 steps past current** — re-issue for longer pauses. |
| `diagnostic_intervals.attn_entropy` | int | [1, 10⁵] | 10 | Cadence of attn-entropy sampling (costs one extra QK bmm when on). |
| `diagnostic_intervals.expert_norms` | int | [1, 10⁵] | 100 | Per-expert weight-norm diagnostics cadence. |
| `diagnostic_intervals.generation_probe` | int | [1, 10⁵] | 500 | Fixed-prompt generation cadence. |
| `diagnostic_intervals.decoded_sample` | int | [1, 10⁵] | 500 | Decoded-batch sample cadence. |
| `gpu_hourly_rate` | float | [0.0, 100.0] | 3.50 | Cost-tracker $/hr for status.json. |

**LR multiplier semantics.** Applied fresh each step after the
warmup/cosine scheduler. Serialized into the checkpoint — resume restores
it. The expected-LR desync detector factors multipliers in, so setting a
multiplier does **not** false-alarm the detector.

## Command reference

One-shot commands are dropped as `commands/pending/NNN-uuid.json`. The
loop scans `pending/` in sorted order at the top of each step.

| Command | Args | What it does |
|---|---|---|
| `checkpoint_now` | — | Save a checkpoint this step (disk-gated: skipped with CRIT incident if <2× largest-ckpt free). |
| `eval_now` | `n_loops=int` | Run perplexity at given loop count (10 val batches). |
| `depth_eval_now` | — | Perplexity sweep at 4/8/12/16 loops. |
| `arith_eval_now` | `n_per_depth=int` | Synthetic arithmetic probe, K∈{2,4,8,12}. |
| `generate` | `prompt=str`, `max_new_tokens=int`, `n_loops=int`, `temperature=float`, `top_k=int` | One-shot sample from the model under training. Appends to `events/generations.jsonl`. |
| `reinit_expert` | `expert_id=int` | Reinitialize a routed expert's gate/up/down weights to N(0, 0.02), with depth-scale on `down`. Optimizer state (Muon momentum) is **not** cleared. |
| `reset_router_bias` | — | Zero the DeepSeek-V3 router-bias buffer. |
| `graceful_stop` | — | Save a final checkpoint, close wandb, exit cleanly. SIGTERM triggers this automatically. |
| `hard_stop` | `confirm_step=int` | Exit immediately without checkpoint. **Rejected** unless `confirm_step == current_step`. |

CLI ergonomics: `--arg key=value` is repeatable; values are JSON-parsed
(quotes needed for strings). `--wait SECONDS` polls `applied/` until the
loop picks up the command.

### Examples

```bash
# Force a checkpoint before trying something risky
python -m tools.mythos_admin cmd checkpoint_now --wait 120

# Sample the model's current generation quality
python -m tools.mythos_admin cmd generate \
    --arg 'prompt="The capital of France is"' \
    --arg max_new_tokens=16 \
    --arg n_loops=8 \
    --arg temperature=1.0 \
    --wait 30

# Rescue a dying expert (id 17)
python -m tools.mythos_admin cmd reinit_expert --arg expert_id=17

# Stop cleanly
python -m tools.mythos_admin cmd graceful_stop --wait 300

# Stop right now (only if genuinely broken)
python -m tools.mythos_admin cmd hard_stop --confirm-current
# or equivalently:
python -m tools.mythos_admin cmd hard_stop --arg confirm_step=$(python -c "import json; print(json.load(open('outputs/rdt_1.5b/admin/status.json'))['step'])")
```

`--confirm-current` auto-fills `confirm_step` from the current status —
convenient but still safety-checked at the loop.

## Incident catalog

`events/incidents.jsonl` — one JSON object per line. Fields: `at`, `step`,
`severity` (INFO/WARN/CRIT), `kind`, `value`, `context`.

| Kind | Severity | Meaning | What to do |
|---|---|---|---|
| `spectral_near_bound` | WARN | ρ(A) ≥ `spectral_warn_threshold`. | Not urgent unless rising. If it crosses 0.9999, consider lowering recurrent-group LR multipliers (0.5× each). Loop HARD-raises at 1.0. |
| `spectral_radius_ge_1` | CRIT | ρ(A) ≥ 1.0. | Loop already raised — unrecoverable in-run. Resume from the last good checkpoint. |
| `loss_spike` | WARN | Total loss > 5× rolling-median over last 200 steps. | Check `microbatch_loss_cv` in metrics — high CV means a bad shard. Non-bad-shard spikes often self-heal within 20–50 steps; don't reflex-tune LR. |
| `step_time_regression` | WARN | Step time > 2× rolling-median over last 50 steps. | Almost always system (disk/IO/thermals). Check `disk_free_gb_ckpt` in status, check for swap via `rss_gb` growth. |
| `nan_inf` | CRIT | NaN/Inf detected in loss or any gradient (pre-clip). | Loop enqueues `graceful_stop` and exits. Resume from last checkpoint, consider halving LR multipliers. |
| `invariant_param_count_changed` | CRIT | `unique_param_count` changed (weight-tying broke, a param was re-registered). | Loop raised. Resume; file an issue — torch.compile likely cloned the tied tensor. |
| `config_parse_failed` | CRIT | `config.json` unreadable or invalid JSON. | Loop is using last-known-good. Run `config rollback`, then fix. |
| `config_bounds_violated` | WARN | A config read produced out-of-range values. | Loop ignored the file; fix via CLI (which pre-validates). |
| `command_rejected` | WARN | A command from `pending/` was malformed, unknown, or safety-blocked. | Check `rejected/` for the file and reason. |
| `expert_reinit` | INFO | A `reinit_expert` command ran. | Informational. Watch `moe_dead_expert_count` and the expert's weight norm to confirm it's learning again. |
| `expert_reinit_out_of_range` | WARN | `reinit_expert` expert_id outside `[0, n_experts)`. | Fix your command. |
| `router_bias_reset` | INFO | `reset_router_bias` ran. | Informational. Router bias regrows over ~100 steps. |
| `eval_perplexity` | INFO | Perplexity result (scheduled or on-demand). | Informational. Rising perplexity at fixed n_loops = training regression. |
| `generation_probe_failed` | WARN | Fixed-prompt probe raised. | Usually tokenizer or sampling issue. Check `value` for the exception. |
| `disk_low_ckpt_skipped` | CRIT | Disk too full to safely checkpoint. | Free space (rotate aggressively is done automatically to keep_last_n=2). If persistent, stop the run. |
| `sigterm_received` | WARN | External SIGTERM (kill, systemd stop). | Loop will finish current step, save final checkpoint, exit. |

## Intervention recipes

### Loss is climbing (slowly)

1. `status` — check rolling mean. Climbing by >10% over 200 steps is concerning; <5% over 1000 is often noise.
2. `incidents` — any `loss_spike` or `step_time_regression`?
3. `recent 50 --raw | grep loss_frac` — is `lm_frac` collapsing below 0.7? If so, the regularizers have taken over. Lower `loss_coeffs.moe_aux` by 2× or `loss_coeffs.act_ponder` by 2×.
4. If none of the above, consider halving all `lr_mult.*` and watching 200 steps.

### Loss is climbing (fast, >5× spike)

1. `cmd checkpoint_now --wait 120` — save before making changes.
2. `incidents --severity CRIT` — any NaN? If yes, this is a recovery scenario — the loop enqueued graceful_stop.
3. Check `recent --raw | grep microbatch_loss_cv` — if >0.5 this is a bad shard, it will self-heal.
4. If it's not a bad shard, rollback: `config rollback` + watch 20 steps.
5. Last resort: halve all LR multipliers.

### Dead experts climbing

1. `recent --raw | grep moe_dead_expert_count` — confirm it's rising.
2. `recent --raw | grep router_entropy` — if entropy is collapsing, routing is degenerating.
3. If a single expert is clearly dying, `cmd reinit_expert --arg expert_id=<id>`. You need to identify it manually from wandb's per-expert histograms.
4. If many experts are dying, increase `loss_coeffs.moe_aux` by 2× (caps at 0.2).
5. Consider raising `router_bias_update_rate` to 2e-3 (caps at 0.01). Higher causes oscillation.

### Gradient explosion (grad norm >10× baseline)

1. `cmd checkpoint_now` — save.
2. Halve `grad_clip` (it's 1.0 by default; try 0.5).
3. Halve all `lr_mult.*`.
4. Watch 100 steps.

### Spectral radius approaching 1.0

1. `recent --raw | grep spectral_radius_max` — confirm the trajectory.
2. Halve `lr_mult.muon_recurrent` and `lr_mult.adamw_recurrent` (the recurrent-group multipliers). The LTI injection's `log_A` is in the recurrent group.
3. If it still climbs, tighten `spectral_warn_threshold` to 0.995 to get earlier WARN incidents.

### Throughput regression (steps taking 2× longer)

1. `status` — check `gpu_mem_peak_gb` and `disk_free_gb_ckpt`.
2. Check for OOM near-misses via memory growth in `recent`.
3. If disk is low: `cmd checkpoint_now` may skip (CRIT incident); rotate more aggressively or stop the run.
4. Most step-time regressions are filesystem/thermal and out of your hands.

### The loop looks dead

1. `status` — if `heartbeat_at` is >5 min stale, process is actually dead (check wandb).
2. If `heartbeat_at` is fresh but `last_step_advance_at` is stale, it's wedged — usually dataloader.
3. Don't send commands to a wedged loop; they'll queue forever.

### Resume from checkpoint

The run script handles `--resume <path>`. Your job:

1. `ls outputs/rdt_1.5b/checkpoints/` — find the latest.
2. Start the run with `--resume <path>`.
3. First thing on session restart: `status` + `incidents -n 20` to get context. Admin state (LR multipliers, rolling buffers, last-applied command id) is restored from checkpoint. **Rolling wall-time / cost resets** — the resumed process is a new wall-start.

### Pause / resume within-run

```bash
# Pause for 200 steps
current=$(python -c "import json; print(json.load(open('outputs/rdt_1.5b/admin/status.json'))['step'])")
python -m tools.mythos_admin config set pause_until_step $((current + 200))
```

The loop writes status + heartbeats while paused but does not step. Pause
auto-expires when `step >= pause_until_step`. **Max 500 steps in the
future** per issue; re-issue for longer.

To resume early:

```bash
python -m tools.mythos_admin config set pause_until_step null
```

## Safety rules

**Do not** directly edit `config.json` with Write or Edit unless you understand
atomicity. Use `config set` — it tmp+renames atomically and validates
bounds. The loop retries once on parse error, but "partial" writes land
you in `config_parse_failed` CRIT incidents.

**Do not** touch `config_applied.json` — the loop owns it. Editing it breaks
rollback.

**Do not** hand-edit files in `commands/applied/` or `commands/rejected/`.
They are immutable history.

**Do not** delete `metrics/*.jsonl` mid-run. Chunks grow unbounded but
rotate at 5K lines; disk is checked each checkpoint.

**Do not** issue `hard_stop` unless the run is genuinely broken —
checkpoints exist for a reason. If unsure, `graceful_stop`.

**Do not** set LR multipliers above 2.0 — the CLI rejects this but the
underlying rationale is: warmup+scheduler already set the nominal LR.
Doubling it once might be defensible; tripling it is always wrong.

**Before any non-trivial change** — LR shift, coefficient change, expert
reinit — `cmd checkpoint_now --wait 120` first.

## Direct file access (as an alternative to the CLI)

You can Read any dashboard file with the normal Read/Grep tools:

```
Read /Users/.../outputs/rdt_1.5b/admin/status.json
Grep pattern="CRIT" path="/Users/.../outputs/rdt_1.5b/events/incidents.jsonl"
```

For writes: prefer the CLI. It does atomic writes and bounds-checking
that raw Edit/Write cannot. If you must write directly (e.g., batching
multiple config changes into one commit), write to a tempfile first and
then move into place via Bash — `mv` is atomic on POSIX.

## FAQ

**Q: Which metrics chunk is current?** — `status.json.metrics_chunk`.
Newer chunks have higher numbers.

**Q: How do I see the last 200 steps across chunk boundaries?** —
`recent` only reads the newest chunk. For history crossing chunks, glob
the `metrics/*.jsonl` files and concatenate. They are in order.

**Q: Is the coefficient-mismatch bug in logging.py fixed?** — Yes.
`composite_loss_rdt` now returns `coeffs_applied` in its `loss_dict`, and
`log_training_step` prefers that over its own argument. The `lm_frac`
detector is correct regardless of which source supplies the coefficients.

**Q: Can I change `max_loop_iters`?** — No. That's a model-shape knob,
frozen at `__init__`. Only `n_loops_override` (runtime per-forward) is
tunable, and it's bounded by `max_loop_iters`.

**Q: Do my config changes survive a resume?** — Yes, `config.json` lives
on disk and is read by the resumed loop. The LR multipliers specifically
also live in the checkpoint (`admin_state.config_dict`).

**Q: How do I know the loop has applied my config change?** — Compare
`config_hash` vs `config_applied_hash` in `status.json`. Equal = applied.
They differ only briefly between your edit and the next step completion.
`config diff` prints what's pending.

**Q: Why did my command vanish from `pending/`?** — Look in `applied/`
(success) or `rejected/` (malformed / safety block). They are never
deleted; the presence of a file there is your ack.

**Q: Is there a health check I should run every N minutes?** — Yes.
Check `status.json`: (1) `heartbeat_at` is fresh, (2) `last_step_advance_at`
is within one step-time of heartbeat, (3) `disk_free_gb_ckpt > 30`, (4)
no CRIT in `incidents_tail`. This is a ~2-second poll.

## Where to file issues you can't fix

You can't fix code mid-run. If the incident suggests a bug (not a
numerical issue), save context and continue:

1. `cmd checkpoint_now` to preserve state.
2. Copy relevant files: `admin/audit.jsonl`, recent `metrics/*.jsonl`,
   `events/incidents.jsonl`, the checkpoint path.
3. Note the incident `kind` + `context`.
4. On next human handoff, present the facts. Do not attempt code changes
   in-flight; the training process imports code at start and will not
   pick up edits mid-run.

## Verify the dashboard itself

```bash
python -m scripts.smoke_admin
```

64 assertions covering config seeding, atomic writes, bounds, command
dispatch, safety guards, NaN detection, rolling detectors, chunk
rotation, snapshot round-trip. Runs in <10s on CPU. A clean pass means
the admin surface is functional; it does **not** test the training loop
itself.
