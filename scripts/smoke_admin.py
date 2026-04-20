#!/usr/bin/env python3
"""
Smoke test for the OpenMythos administration dashboard.

Exercises the admin control plane in isolation — no model, no data, no GPU.
Tests:
    - Config seeding + atomic write
    - CLI config set (bounds-check: accept, reject)
    - Command queue: atomic drop, dispatch, applied/ + audit
    - Command safety: hard_stop requires confirm_step; pause bounds
    - Incident emission
    - Status write + read
    - Snapshot / load round-trip
    - Rolling-buffer detectors (loss spike, step-time regression)
    - Metrics chunk rotation

Runs in <10s on CPU. No external deps beyond stdlib + torch.

Usage:
    python -m scripts.smoke_admin
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from tools.mythos_admin import main as cli_main
from tools.mythos_admin_runtime import (
    AdminRuntime,
    DEFAULT_CONFIG,
    METRIC_CHUNK_LINES,
    ROLLING_LOSS,
    ROLLING_STEP_TIME,
    validate_config,
)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class _Assertions:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def ok(self, cond: bool, label: str, detail: str = ""):
        if cond:
            self.passes += 1
            print(f"  {PASS}  {label}")
        else:
            self.failures.append(label + (f" — {detail}" if detail else ""))
            print(f"  {FAIL}  {label}" + (f"\n        {detail}" if detail else ""))

    def summary(self) -> int:
        total = self.passes + len(self.failures)
        print()
        if self.failures:
            print(f"{FAIL}  {len(self.failures)}/{total} failed")
            for f in self.failures:
                print(f"    - {f}")
            return 1
        print(f"{PASS}  {self.passes}/{total} passed")
        return 0


def _run_cli(argv: list) -> int:
    """Run the CLI with given argv (without leading 'python -m tools.mythos_admin')."""
    return cli_main(argv)


def _section(name: str):
    print(f"\n== {name} ==")


def test_config_seeding(root: Path, a: _Assertions) -> None:
    _section("config seeding")
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.seed_config_if_missing()
    a.ok(rt.config_path.exists(), "config.json created")
    a.ok(rt.prompts_path.exists(), "prompts.json created")
    cfg = json.loads(rt.config_path.read_text())
    a.ok(cfg.get("schema_version") == DEFAULT_CONFIG["schema_version"],
         "schema_version set")
    a.ok(cfg.get("lr_mult", {}).get("muon_default") == 1.0,
         "default lr_mult = 1.0")


def test_config_read_defensive(root: Path, a: _Assertions) -> None:
    _section("config read (mtime-gated + defensive parse)")
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.seed_config_if_missing()

    cfg1 = rt.read_config(current_step=0)
    a.ok(cfg1.grad_clip == 1.0, "initial grad_clip")

    # mtime unchanged → cached config returned
    cfg2 = rt.read_config(current_step=1)
    a.ok(cfg1 is cfg2, "cached when mtime unchanged")

    # Corrupt the file; runtime should emit incident + return last-good
    rt.config_path.write_text("{ not valid json")
    # Bump mtime forward so cache invalidates
    import os as _os
    _os.utime(rt.config_path, (time.time() + 1, time.time() + 1))
    cfg3 = rt.read_config(current_step=2)
    a.ok(cfg3 is cfg1 or cfg3.grad_clip == 1.0, "fallback on parse error")
    incidents = rt.incidents_path.read_text().splitlines() if rt.incidents_path.exists() else []
    a.ok(any("config_parse_failed" in l for l in incidents),
         "incident emitted on parse failure")


def test_cli_config_set(root: Path, a: _Assertions) -> None:
    _section("CLI config set (accept + reject)")
    out_str = str(root)
    # Reset config
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.seed_config_if_missing()

    # Accept: in-range multiplier
    rc = _run_cli(["--output", out_str, "config", "set", "lr_mult.muon_default", "0.5"])
    a.ok(rc == 0, "in-range lr_mult accepted")
    cfg = json.loads(rt.config_path.read_text())
    a.ok(cfg["lr_mult"]["muon_default"] == 0.5, "value persisted")

    # Reject: grad_clip out of range
    rc = _run_cli(["--output", out_str, "config", "set", "grad_clip", "50.0"])
    a.ok(rc == 2, "out-of-range grad_clip rejected (exit 2)")
    cfg2 = json.loads(rt.config_path.read_text())
    a.ok(cfg2.get("grad_clip", 1.0) == 1.0, "original grad_clip unchanged")

    # Reject: unknown coefficient out-of-range
    rc = _run_cli(["--output", out_str, "config", "set", "loss_coeffs.moe_aux", "5.0"])
    a.ok(rc == 2, "out-of-range moe_aux rejected")


def test_command_queue(root: Path, a: _Assertions) -> None:
    _section("command queue (pending → applied)")
    out_str = str(root)
    rt = AdminRuntime(root, run_start_wall=time.time())

    # Enqueue checkpoint_now via CLI
    rc = _run_cli(["--output", out_str, "cmd", "checkpoint_now"])
    a.ok(rc == 0, "checkpoint_now enqueued")
    pending_before = list(rt.pending_dir.iterdir())
    a.ok(len(pending_before) == 1, "exactly one file in pending/")

    # Process
    rt.process_pending_commands(executor={}, current_step=100)
    a.ok(rt.pop_checkpoint_request() is True, "checkpoint flag set")
    a.ok(len(list(rt.pending_dir.iterdir())) == 0, "pending/ drained")
    a.ok(len(list(rt.applied_dir.iterdir())) == 1, "command moved to applied/")
    applied_file = next(rt.applied_dir.iterdir())
    applied_payload = json.loads(applied_file.read_text())
    a.ok(applied_payload.get("applied_at_step") == 100,
         "applied_at_step recorded")
    a.ok("result" in applied_payload, "result populated")


def test_command_with_args(root: Path, a: _Assertions) -> None:
    _section("command with args (generate + reinit_expert)")
    out_str = str(root)
    rt = AdminRuntime(root, run_start_wall=time.time())

    rc = _run_cli(["--output", out_str, "cmd", "generate",
                    "--arg", 'prompt="hello world"',
                    "--arg", "max_new_tokens=8",
                    "--arg", "n_loops=4"])
    a.ok(rc == 0, "generate command enqueued")

    rc = _run_cli(["--output", out_str, "cmd", "reinit_expert",
                    "--arg", "expert_id=3"])
    a.ok(rc == 0, "reinit_expert enqueued")

    rt.process_pending_commands(executor={}, current_step=200)

    gen_requests = rt.pop_generate_requests()
    a.ok(len(gen_requests) == 1, "generate request queued")
    a.ok(gen_requests[0].get("prompt") == "hello world",
         "prompt parsed correctly")
    a.ok(gen_requests[0].get("max_new_tokens") == 8,
         "max_new_tokens parsed correctly")

    reinit_requests = rt.pop_reinit_expert_requests()
    a.ok(reinit_requests == [3], "expert_id parsed correctly")


def test_hard_stop_confirm(root: Path, a: _Assertions) -> None:
    _section("hard_stop safety (confirm_step required)")
    out_str = str(root)
    rt = AdminRuntime(root, run_start_wall=time.time())

    # Without confirm
    rc = _run_cli(["--output", out_str, "cmd", "hard_stop"])
    a.ok(rc == 0, "hard_stop enqueued (accept pre-validation)")
    rt.process_pending_commands(executor={}, current_step=500)
    a.ok(rt.is_hard_stop() is False, "loop-side rejected without confirm_step")
    a.ok(any(f.name.startswith("0000000") for f in rt.rejected_dir.iterdir()),
         "rejected/ has the bad command")

    # With wrong confirm
    _run_cli(["--output", out_str, "cmd", "hard_stop", "--arg", "confirm_step=999"])
    rt.process_pending_commands(executor={}, current_step=500)
    a.ok(rt.is_hard_stop() is False, "wrong confirm_step rejected")

    # With correct confirm
    _run_cli(["--output", out_str, "cmd", "hard_stop", "--arg", "confirm_step=500"])
    rt.process_pending_commands(executor={}, current_step=500)
    a.ok(rt.is_hard_stop() is True, "correct confirm_step accepted")


def test_pause_bounds(root: Path, a: _Assertions) -> None:
    _section("pause_until_step bounds")
    ok, errors = validate_config(
        {**DEFAULT_CONFIG, "pause_until_step": 50}, current_step=100
    )
    a.ok(not ok, "past-step pause rejected")
    a.ok(any("pause_until_step" in e for e in errors),
         "pause_until_step error surfaced")

    ok, errors = validate_config(
        {**DEFAULT_CONFIG, "pause_until_step": 100 + 600}, current_step=100
    )
    a.ok(not ok, "pause > current+500 rejected")

    ok, errors = validate_config(
        {**DEFAULT_CONFIG, "pause_until_step": 100 + 300}, current_step=100
    )
    a.ok(ok, "pause within window accepted")


def test_unknown_n_loops(root: Path, a: _Assertions) -> None:
    _section("n_loops_override bounds")
    ok, _ = validate_config(
        {**DEFAULT_CONFIG, "n_loops_override": 0}, current_step=0,
        max_loop_iters=16,
    )
    a.ok(not ok, "n_loops_override=0 rejected")
    ok, _ = validate_config(
        {**DEFAULT_CONFIG, "n_loops_override": 100}, current_step=0,
        max_loop_iters=16,
    )
    a.ok(not ok, "n_loops_override above max rejected")
    ok, _ = validate_config(
        {**DEFAULT_CONFIG, "n_loops_override": 8}, current_step=0,
        max_loop_iters=16,
    )
    a.ok(ok, "in-range n_loops_override accepted")


def test_nan_detection(root: Path, a: _Assertions) -> None:
    _section("NaN detection")
    rt = AdminRuntime(root, run_start_wall=time.time())

    class _Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(4))

    m = _Dummy()
    m.p.grad = torch.zeros(4)
    out = rt.check_nan_inf({"total": 1.0, "lm": 1.0}, m)
    a.ok(out is None, "healthy loss + grads → no NaN")

    out = rt.check_nan_inf({"total": float("nan"), "lm": 1.0}, m)
    a.ok(out is not None and "total" in out, "NaN in loss detected")

    m.p.grad = torch.tensor([1.0, float("inf"), 2.0, 3.0])
    out = rt.check_nan_inf({"total": 1.0, "lm": 1.0}, m)
    a.ok(out is not None and "grad.p" in out, "Inf in grad detected")


def test_rolling_detectors(root: Path, a: _Assertions) -> None:
    _section("rolling-buffer detectors")
    rt = AdminRuntime(root, run_start_wall=time.time())
    # Fill rolling loss with ~3.0, then spike
    for _ in range(ROLLING_LOSS):
        rt.update_loss_rolling(3.0)
    spike = rt.loss_spike(5.0)  # 5/3 = 1.67, below 5x threshold
    a.ok(spike is None, "small bump not a spike")
    spike = rt.loss_spike(20.0)  # 20/3 > 5
    a.ok(spike is not None, "5× rolling median flagged")
    a.ok(spike[1] == 3.0, "baseline correctly reported")

    for _ in range(ROLLING_STEP_TIME):
        rt.update_step_time_rolling(1.0)
    reg = rt.step_time_regressed(1.5)
    a.ok(reg is None, "1.5× not a regression")
    reg = rt.step_time_regressed(3.0)
    a.ok(reg is not None, "3× step-time flagged")


def test_status_write(root: Path, a: _Assertions) -> None:
    _section("status write + heartbeat distinction")
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.seed_config_if_missing()
    _ = rt.read_config(current_step=0)  # prime cached_config_dict
    rt.write_status(step=1, total_steps=100, n_loops=4,
                    loss={"total": 3.2, "lm": 3.1},
                    grad_norm_total=0.5, spectral_radius=0.9)
    st = json.loads(rt.status_path.read_text())
    a.ok(st["step"] == 1, "step recorded")
    a.ok(st["loss"]["total"] == 3.2, "loss recorded")
    a.ok("heartbeat_at" in st, "heartbeat_at present")
    a.ok("last_step_advance_at" in st, "last_step_advance_at present")
    # Simulate: heartbeat advances without step (wedged dataloader)
    t0 = st["last_step_advance_at"]
    time.sleep(1.1)
    rt.write_status(step=1, total_steps=100, n_loops=4,
                    loss={"total": 3.2}, grad_norm_total=0.5,
                    spectral_radius=0.9)
    st2 = json.loads(rt.status_path.read_text())
    a.ok(st2["heartbeat_at"] != st["heartbeat_at"],
         "heartbeat_at advances on every write")
    a.ok(st2["last_step_advance_at"] == t0,
         "last_step_advance_at unchanged when step didn't advance")


def test_incident_emission(root: Path, a: _Assertions) -> None:
    _section("incident emission + tail")
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.emit_incident("test_warn", severity="WARN", step=10, value=0.99)
    rt.emit_incident("test_crit", severity="CRIT", step=11, value="bad")
    lines = rt.incidents_path.read_text().splitlines()
    a.ok(len(lines) >= 2, "incidents appended")
    last = json.loads(lines[-1])
    a.ok(last["severity"] == "CRIT", "last severity correct")
    a.ok(last["kind"] == "test_crit", "kind round-tripped")


def test_snapshot_load_roundtrip(root: Path, a: _Assertions) -> None:
    _section("snapshot + load_state round-trip")
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.seed_config_if_missing()
    rt.read_config(current_step=0)
    rt.update_loss_rolling(3.0)
    rt.update_loss_rolling(2.9)
    rt._last_applied_command_id = "00000042-abcd1234"  # simulate

    snap = rt.snapshot_state()
    a.ok("config_hash" in snap and snap["config_hash"].startswith("sha256:"),
         "config_hash in snapshot")
    a.ok(snap["last_applied_command_id"] == "00000042-abcd1234",
         "last_applied_command_id serialized")

    rt2 = AdminRuntime(root, run_start_wall=time.time())
    rt2.load_state(snap)
    a.ok(rt2._last_applied_command_id == "00000042-abcd1234",
         "last_applied_command_id restored")
    a.ok(len(rt2._loss_rolling) == 2, "loss_rolling restored")


def test_metric_chunk_rotation(root: Path, a: _Assertions) -> None:
    _section("metric chunk rotation")
    rt = AdminRuntime(root, run_start_wall=time.time())
    for i in range(METRIC_CHUNK_LINES + 5):
        rt.log_metric(i, {"train/loss": 3.0, "step_n": i})
    chunks = sorted(rt.metrics_dir.glob("metrics.*.jsonl"))
    a.ok(len(chunks) >= 2, f"≥2 chunks after {METRIC_CHUNK_LINES+5} writes")
    a.ok(rt._metric_chunk >= 1, "chunk counter advanced")


def test_cli_status_output(root: Path, a: _Assertions) -> None:
    _section("CLI status + audit surface readable")
    out_str = str(root)
    # Ensure status exists
    rt = AdminRuntime(root, run_start_wall=time.time())
    rt.seed_config_if_missing()
    rt.read_config(current_step=0)
    rt.write_status(step=42, total_steps=29000, n_loops=6,
                    loss={"total": 2.8, "lm": 2.75}, grad_norm_total=0.4,
                    spectral_radius=0.88,
                    lrs={"muon_default": 0.01})
    rc = _run_cli(["--output", out_str, "status"])
    a.ok(rc == 0, "status command succeeds")
    rc = _run_cli(["--output", out_str, "audit", "-n", "10"])
    a.ok(rc == 0, "audit command succeeds")
    rc = _run_cli(["--output", out_str, "incidents", "-n", "10"])
    a.ok(rc == 0, "incidents command succeeds")


def main() -> int:
    print("=" * 60)
    print("OpenMythos administration dashboard smoke test")
    print("=" * 60)

    a = _Assertions()
    tmp_root = Path(tempfile.mkdtemp(prefix="mythos_admin_smoke_"))
    try:
        test_config_seeding(tmp_root, a)
        test_config_read_defensive(tmp_root, a)
        # Fresh dir for CLI-heavy tests
        shutil.rmtree(tmp_root)
        tmp_root = Path(tempfile.mkdtemp(prefix="mythos_admin_smoke_"))
        test_cli_config_set(tmp_root, a)
        test_command_queue(tmp_root, a)
        test_command_with_args(tmp_root, a)

        shutil.rmtree(tmp_root)
        tmp_root = Path(tempfile.mkdtemp(prefix="mythos_admin_smoke_"))
        test_hard_stop_confirm(tmp_root, a)

        test_pause_bounds(tmp_root, a)
        test_unknown_n_loops(tmp_root, a)
        test_nan_detection(tmp_root, a)

        shutil.rmtree(tmp_root)
        tmp_root = Path(tempfile.mkdtemp(prefix="mythos_admin_smoke_"))
        test_rolling_detectors(tmp_root, a)
        test_status_write(tmp_root, a)
        test_incident_emission(tmp_root, a)
        test_snapshot_load_roundtrip(tmp_root, a)

        shutil.rmtree(tmp_root)
        tmp_root = Path(tempfile.mkdtemp(prefix="mythos_admin_smoke_"))
        test_metric_chunk_rotation(tmp_root, a)
        test_cli_status_output(tmp_root, a)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return a.summary()


if __name__ == "__main__":
    sys.exit(main())
