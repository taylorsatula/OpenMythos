#!/usr/bin/env python3
"""
Administration CLI for the OpenMythos training run.

Usage:
    python -m tools.mythos_admin <subcommand> [options]

Subcommands:
    status [--full]
    watch [--kind metrics|incidents|generations]
    recent [-n 100]
    config get [<dotted_key>]
    config set <dotted_key> <json_value>
    config rollback
    cmd <name> [--arg key=value ...] [--wait SECONDS]
    audit [-n 50]
    diff
    incidents [-n 50] [--severity WARN|CRIT]

All commands take `--output <path>` (default: outputs/rdt_1.5b). Atomic
writes throughout; rejects out-of-range values before writing.

The CLI shares schema / bounds with `tools.mythos_admin_runtime`, so
validation happens consistently on both sides (CLI pre-write, loop
post-read).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.mythos_admin_runtime import (
    BOUNDS,
    COEFF_BOUNDS,
    DEFAULT_CONFIG,
    INTERVAL_BOUNDS,
    KNOWN_COMMANDS,
    LR_MULT_BOUNDS,
    next_command_filename,
    validate_config,
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _paths(output_dir: str) -> Dict[str, Path]:
    out = Path(output_dir)
    admin = out / "admin"
    events = out / "events"
    return {
        "output": out,
        "admin": admin,
        "events": events,
        "config": admin / "config.json",
        "config_applied": admin / "config_applied.json",
        "prompts": admin / "prompts.json",
        "status": admin / "status.json",
        "audit": admin / "audit.jsonl",
        "commands": admin / "commands",
        "pending": admin / "commands" / "pending",
        "applied": admin / "commands" / "applied",
        "rejected": admin / "commands" / "rejected",
        "metrics_dir": events / "metrics",
        "incidents": events / "incidents.jsonl",
        "generations": events / "generations.jsonl",
    }


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    return json.loads(config_path.read_text())


def _save_config(config_path: Path, cfg: Dict[str, Any]) -> None:
    _atomic_write_json(config_path, cfg)


# ---------------------------------------------------------------------------
# Dotted-key helpers
# ---------------------------------------------------------------------------


def _dotted_get(d: Dict[str, Any], key: str) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def _dotted_set(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur: Any = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# Parse free-form values from the CLI
# ---------------------------------------------------------------------------


def _parse_value(raw: str) -> Any:
    """Parse a CLI value as JSON if possible; otherwise pass through as a string."""
    if raw.lower() == "none":
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if not p["status"].exists():
        print(f"status.json not found at {p['status']}. Has the run started?")
        return 1
    status = json.loads(p["status"].read_text())
    if args.full:
        print(json.dumps(status, indent=2))
        return 0

    # Triage view
    step = status.get("step", "?")
    total = status.get("total_steps", "?")
    loss = status.get("loss", {}) or {}
    lrs = status.get("lrs", {}) or {}
    lr_mults = status.get("lr_mults_applied", {}) or {}
    spectral = status.get("spectral_radius", 0.0)
    grad = status.get("grad_norm_total", 0.0)
    thr = status.get("throughput_tokens_per_sec", 0.0)
    step_time = status.get("step_time_s", 0.0)
    mem = status.get("gpu_mem_peak_gb", 0.0)
    disk = status.get("disk_free_gb_ckpt", 0.0)
    tokens = status.get("tokens_seen", 0)
    wall = status.get("wall_hours", 0.0)
    cost = status.get("estimated_cost_usd", 0.0)
    heartbeat = status.get("heartbeat_at", "?")
    last_step_at = status.get("last_step_advance_at", "?")
    pause = status.get("pause_until_step")
    chash = status.get("config_hash", "?")
    cahash = status.get("config_applied_hash", chash)

    print(f"step {step}/{total}  n_loops={status.get('n_loops', '?')}  "
          f"paused={pause is not None}")
    print(f"  loss.total {loss.get('total', 0.0):.4f}  "
          f"lm {loss.get('lm', 0.0):.4f}  "
          f"rolling200 {status.get('loss_rolling_mean_200', 0.0):.4f}")
    print(f"  grad {grad:.3f}  ρ(A) {spectral:.4f}  "
          f"throughput {thr:,.0f} tok/s  step_time {step_time:.2f}s")
    if lr_mults:
        mult_str = ", ".join(f"{k}={v:.2g}" for k, v in lr_mults.items() if v != 1.0)
        if mult_str:
            print(f"  lr_mults: {mult_str}")
    if lrs:
        print(f"  lrs: " + "  ".join(f"{k}={v:.2e}" for k, v in lrs.items()))
    print(f"  gpu_mem_peak {mem:.1f} GB   disk_free {disk:.1f} GB")
    print(f"  tokens {tokens/1e9:.2f}B   wall {wall:.2f} h   cost ${cost:.2f}")
    print(f"  heartbeat {heartbeat}   last_step_advance {last_step_at}")
    if chash != cahash:
        print(f"  ⚠ config drift: config={chash} != applied={cahash}")

    tail = status.get("incidents_tail") or []
    if tail:
        print()
        print("Recent incidents:")
        for rec in tail[-5:]:
            print(f"  [{rec.get('severity', '?')}] step={rec.get('step')}  "
                  f"{rec.get('kind')}  {rec.get('value', '')}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if args.kind == "metrics":
        # Watch newest metrics chunk; if it rotates, follow the new one.
        target = _newest_metrics_chunk(p["metrics_dir"])
        if target is None:
            print(f"no metrics yet at {p['metrics_dir']}")
            return 1
    elif args.kind == "incidents":
        target = p["incidents"]
    elif args.kind == "generations":
        target = p["generations"]
    else:
        print(f"unknown --kind: {args.kind}")
        return 2

    _tail_follow(target, args.kind, p["metrics_dir"] if args.kind == "metrics" else None)
    return 0


def _newest_metrics_chunk(metrics_dir: Path) -> Optional[Path]:
    if not metrics_dir.exists():
        return None
    chunks = sorted(metrics_dir.glob("metrics.*.jsonl"))
    return chunks[-1] if chunks else None


def _tail_follow(path: Path, kind: str, metrics_dir: Optional[Path]) -> None:
    """Simple tail -f. Polls every 500 ms. For metrics, auto-switches to the
    next chunk when the current one stops growing and a later one exists."""
    current = path
    while not current.exists():
        time.sleep(0.5)
        if kind == "metrics":
            current = _newest_metrics_chunk(metrics_dir) or current
    f = current.open("r")
    f.seek(0, os.SEEK_END)
    try:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                if kind == "metrics":
                    newest = _newest_metrics_chunk(metrics_dir)
                    if newest is not None and newest != current:
                        f.close()
                        current = newest
                        f = current.open("r")
                        # new chunk; start from its beginning
                continue
            print(line.rstrip())
    except KeyboardInterrupt:
        pass
    finally:
        f.close()


def cmd_recent(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    newest = _newest_metrics_chunk(p["metrics_dir"])
    if newest is None:
        print("no metrics yet")
        return 1
    lines = newest.read_text().splitlines()[-args.n :]
    if args.raw:
        print("\n".join(lines))
        return 0
    # Summarized view
    print(f"{'step':>7}  {'loss':>8}  {'lm':>8}  {'grad':>6}  {'ρ(A)':>7}  {'lr_mu':>9}  {'tok/s':>9}  {'step_s':>7}")
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = rec.get("step", 0)
        tl = rec.get("train/loss", rec.get("loss", {}).get("total", 0.0) if isinstance(rec.get("loss"), dict) else 0.0)
        lm = rec.get("train/lm_loss", 0.0)
        gn = rec.get("train/grad_norm/total", 0.0)
        sp = rec.get("train/spectral_radius_max", rec.get("spectral_radius", 0.0))
        lr_mu = rec.get("train/lr/muon_default", 0.0)
        tps = rec.get("train/throughput_tokens_per_sec", 0.0)
        st = rec.get("train/step_time_s", 0.0)
        print(f"{step:>7}  {tl:>8.4f}  {lm:>8.4f}  {gn:>6.2f}  {sp:>7.4f}  {lr_mu:>9.2e}  {tps:>9,.0f}  {st:>7.2f}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if args.config_action == "get":
        cfg = _load_config(p["config"])
        if args.key:
            try:
                print(json.dumps(_dotted_get(cfg, args.key), indent=2))
            except KeyError:
                print(f"unknown key: {args.key}", file=sys.stderr)
                return 1
        else:
            print(json.dumps(cfg, indent=2))
        return 0

    if args.config_action == "set":
        cfg = _load_config(p["config"])
        before = json.loads(json.dumps(cfg))
        value = _parse_value(args.value)
        _dotted_set(cfg, args.key, value)
        current_step = _current_step(p)
        max_loops = _max_loop_iters_from_status(p)
        ok, errors = validate_config(cfg, current_step=current_step, max_loop_iters=max_loops)
        if not ok:
            print("REJECTED (bounds violation):", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            return 2
        _save_config(p["config"], cfg)
        _append_audit(p["audit"], {
            "at": _now_iso(), "step": current_step, "kind": "cli_config_set",
            "key": args.key, "before": _dotted_try(before, args.key),
            "after": value, "source": "cli",
        })
        print(f"OK: {args.key} = {json.dumps(value)}")
        return 0

    if args.config_action == "rollback":
        if not p["config_applied"].exists():
            print("no config_applied.json to roll back to", file=sys.stderr)
            return 1
        applied = json.loads(p["config_applied"].read_text())
        _save_config(p["config"], applied)
        _append_audit(p["audit"], {
            "at": _now_iso(), "step": _current_step(p), "kind": "cli_config_rollback",
            "source": "cli",
        })
        print("rolled back to config_applied.json")
        return 0

    print(f"unknown config action: {args.config_action}", file=sys.stderr)
    return 2


def cmd_cmd(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if args.name not in KNOWN_COMMANDS:
        print(f"unknown command: {args.name}", file=sys.stderr)
        print(f"known: {sorted(KNOWN_COMMANDS)}", file=sys.stderr)
        return 2

    # Parse --arg key=value pairs.
    cmd_args: Dict[str, Any] = {}
    for pair in args.arg or []:
        if "=" not in pair:
            print(f"malformed --arg: {pair}; expected key=value", file=sys.stderr)
            return 2
        k, v = pair.split("=", 1)
        cmd_args[k] = _parse_value(v)

    # Special: hard_stop autofills confirm_step if the Model provides --confirm-current.
    if args.name == "hard_stop" and args.confirm_current:
        cmd_args["confirm_step"] = _current_step(p)

    filename = next_command_filename(p["pending"], args.name)
    payload = {
        "name": args.name,
        "args": cmd_args,
        "queued_at": _now_iso(),
    }
    target = p["pending"] / filename
    # Atomic: write to unique name; no existing reader can see a partial file.
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, target)
    print(f"queued: {filename}")

    if args.wait:
        return _wait_for_ack(p, filename, timeout_s=args.wait)
    return 0


def _wait_for_ack(p: Dict[str, Path], filename: str, timeout_s: float) -> int:
    applied = p["applied"] / filename
    rejected = p["rejected"] / filename
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if applied.exists():
            payload = json.loads(applied.read_text())
            print(f"APPLIED at step {payload.get('applied_at_step')}:")
            print(json.dumps(payload.get("result"), indent=2))
            return 0
        if rejected.exists():
            payload = json.loads(rejected.read_text())
            print(f"REJECTED: {payload.get('reason', 'unknown')}")
            return 3
        time.sleep(0.25)
    print(f"timeout after {timeout_s}s (file still in pending/)", file=sys.stderr)
    return 4


def cmd_audit(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if not p["audit"].exists():
        print("no audit log")
        return 0
    lines = p["audit"].read_text().splitlines()[-args.n :]
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = rec.get("kind", "?")
        step = rec.get("step", "?")
        at = rec.get("at", "?")
        if kind == "command":
            print(f"[{at}] step={step}  cmd={rec.get('name')}  args={rec.get('args')}  result={rec.get('result')}")
        elif kind == "config_change":
            print(f"[{at}] step={step}  config_change  {rec.get('before_hash')} → {rec.get('after_hash')}")
            diff = rec.get("diff") or {}
            for k, v in diff.items():
                print(f"    {k}: {v}")
        elif kind.startswith("cli_"):
            print(f"[{at}] step={step}  {kind}  key={rec.get('key')}  "
                  f"before={rec.get('before')}  after={rec.get('after')}")
        else:
            print(f"[{at}] step={step}  {kind}  {rec}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if not p["config"].exists():
        print("no config.json")
        return 1
    if not p["config_applied"].exists():
        print("no config_applied.json yet (loop has not completed a step)")
        return 1
    cur = json.loads(p["config"].read_text())
    app = json.loads(p["config_applied"].read_text())
    if cur == app:
        print("clean: config.json == config_applied.json")
        return 0
    from tools.mythos_admin_runtime import _shallow_diff
    diff = _shallow_diff(app, cur)
    print("config.json (pending) vs config_applied.json (applied):")
    print(json.dumps(diff, indent=2))
    return 0


def cmd_incidents(args: argparse.Namespace) -> int:
    p = _paths(args.output)
    if not p["incidents"].exists():
        print("no incidents yet")
        return 0
    lines = p["incidents"].read_text().splitlines()
    if args.severity:
        lines = [l for l in lines if f'"severity": "{args.severity}"' in l]
    for line in lines[-args.n :]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sev = rec.get("severity", "?")
        step = rec.get("step", "?")
        at = rec.get("at", "?")
        kind = rec.get("kind", "?")
        val = rec.get("value", "")
        ctx = rec.get("context") or {}
        ctx_str = "  " + json.dumps(ctx) if ctx else ""
        print(f"[{at}] [{sev}] step={step}  {kind}  value={val}{ctx_str}")
    return 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _current_step(p: Dict[str, Path]) -> int:
    try:
        return int(json.loads(p["status"].read_text()).get("step", 0))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0


def _max_loop_iters_from_status(p: Dict[str, Path]) -> int:
    try:
        return int(json.loads(p["status"].read_text()).get("max_loop_iters", 16))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 16


def _dotted_try(d: Dict[str, Any], key: str) -> Any:
    try:
        return _dotted_get(d, key)
    except KeyError:
        return None


def _append_audit(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec))
        f.write("\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mythos_admin",
        description="Administer the OpenMythos training run.",
    )
    ap.add_argument("--output", default="outputs/rdt_1.5b",
                    help="Path to the training output dir (default: outputs/rdt_1.5b)")
    sub = ap.add_subparsers(dest="subcommand", required=True)

    p_status = sub.add_parser("status", help="Show current training status")
    p_status.add_argument("--full", action="store_true", help="Dump full status.json")
    p_status.set_defaults(func=cmd_status)

    p_watch = sub.add_parser("watch", help="Tail a stream")
    p_watch.add_argument("--kind", choices=["metrics", "incidents", "generations"],
                          default="metrics")
    p_watch.set_defaults(func=cmd_watch)

    p_recent = sub.add_parser("recent", help="Last N metric lines summarized")
    p_recent.add_argument("-n", type=int, default=30)
    p_recent.add_argument("--raw", action="store_true", help="Print raw JSONL")
    p_recent.set_defaults(func=cmd_recent)

    p_cfg = sub.add_parser("config", help="Read or modify admin/config.json")
    cfg_sub = p_cfg.add_subparsers(dest="config_action", required=True)
    p_cg = cfg_sub.add_parser("get")
    p_cg.add_argument("key", nargs="?", default=None,
                       help="Dotted key, e.g. lr_mult.muon_default")
    p_cs = cfg_sub.add_parser("set")
    p_cs.add_argument("key")
    p_cs.add_argument("value")
    cfg_sub.add_parser("rollback")
    p_cfg.set_defaults(func=cmd_config)

    p_cmd = sub.add_parser("cmd", help="Enqueue a one-shot command")
    p_cmd.add_argument("name", choices=sorted(KNOWN_COMMANDS))
    p_cmd.add_argument("--arg", action="append", default=[],
                       help="Command argument as key=value (JSON-parsed); repeatable")
    p_cmd.add_argument("--wait", type=float, default=0.0,
                       help="Poll for ack up to N seconds; 0 = fire-and-forget")
    p_cmd.add_argument("--confirm-current", action="store_true",
                        help="For hard_stop: auto-fill confirm_step from current status")
    p_cmd.set_defaults(func=cmd_cmd)

    p_audit = sub.add_parser("audit", help="Tail the audit log")
    p_audit.add_argument("-n", type=int, default=50)
    p_audit.set_defaults(func=cmd_audit)

    p_diff = sub.add_parser("diff", help="Show config.json vs config_applied.json")
    p_diff.set_defaults(func=cmd_diff)

    p_inc = sub.add_parser("incidents", help="Tail the incidents log")
    p_inc.add_argument("-n", type=int, default=50)
    p_inc.add_argument("--severity", choices=["INFO", "WARN", "CRIT"])
    p_inc.set_defaults(func=cmd_incidents)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
