#!/usr/bin/env python3
"""
Loop-side glue for the OpenMythos administration dashboard.

An `AdminRuntime` instance is created once at training start. The loop calls
its methods at specific hooks: top-of-step to read config and consume
commands, end-of-step to write metrics/status, on gate trips to emit
incidents. It owns no thread — everything runs inline on the main thread.

File layout (all paths relative to output_path):

    admin/
        config.json              — persistent knobs (atomic rewrite by CLI)
        config_applied.json      — mirror after successful step (rollback)
        prompts.json             — fixed-prompt panel
        commands/pending/        — NNN-uuid.json files, model writes here
        commands/applied/        — loop moves here with result populated
        commands/rejected/       — bounds violation / unknown command
        status.json              — atomic rewrite every step
        audit.jsonl              — one line per accepted intervention
    events/
        metrics/metrics.NNN.jsonl — chunked 5K lines
        incidents.jsonl          — WARN/CRIT events
        generations.jsonl        — generation-probe outputs
"""

import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Schema / defaults
# ---------------------------------------------------------------------------


SCHEMA_VERSION = 1

DEFAULT_CONFIG: Dict[str, Any] = {
    "lr_mult": {
        "muon_default": 1.0,
        "muon_recurrent": 1.0,
        "adamw_default": 1.0,
        "adamw_recurrent": 1.0,
    },
    "loss_coeffs": {"moe_aux": 0.01, "moe_z": 1e-3, "act_ponder": 1e-3},
    "grad_clip": 1.0,
    "spectral_warn_threshold": 0.999,
    "router_bias_update": True,
    "router_bias_update_rate": 1e-3,
    "n_loops_override": None,
    "pause_until_step": None,
    "diagnostic_intervals": {
        "attn_entropy": 10,
        "expert_norms": 100,
        "generation_probe": 500,
        "decoded_sample": 500,
    },
    "gpu_hourly_rate": 3.50,
    "schema_version": SCHEMA_VERSION,
}

DEFAULT_PROMPTS: List[str] = [
    "The answer to 37 * 41 is",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small village perched on a cliff overlooking the sea,",
    "User: What's the difference between a list and a tuple in Python?\nAssistant:",
    "if x > 0 and y < 0: print('opposite signs')  # explanation:",
]

BOUNDS: Dict[str, Tuple[float, float]] = {
    "grad_clip": (1e-6, 10.0),
    "spectral_warn_threshold": (0.5, 0.9999),
    "router_bias_update_rate": (0.0, 0.01),
    "gpu_hourly_rate": (0.0, 100.0),
}

LR_MULT_BOUNDS = (0.0, 2.0)
COEFF_BOUNDS = {
    "moe_aux": (0.0, 0.2),
    "moe_z": (0.0, 0.1),
    "act_ponder": (0.0, 0.1),
}
INTERVAL_BOUNDS = (1, 100000)

KNOWN_COMMANDS = {
    "checkpoint_now",
    "eval_now",
    "depth_eval_now",
    "arith_eval_now",
    "generate",
    "reinit_expert",
    "reset_router_bias",
    "graceful_stop",
    "hard_stop",
}


# ---------------------------------------------------------------------------
# Config wrapper — attribute access on top of the validated dict
# ---------------------------------------------------------------------------


@dataclass
class Config:
    lr_mult: Dict[str, float]
    loss_coeffs: Dict[str, float]
    grad_clip: float
    spectral_warn_threshold: float
    router_bias_update: bool
    router_bias_update_rate: float
    n_loops_override: Optional[int]
    pause_until_step: Optional[int]
    diagnostic_intervals: Dict[str, int]
    gpu_hourly_rate: float
    schema_version: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        merged = _deep_merge(DEFAULT_CONFIG, d)
        return cls(**{k: merged[k] for k in DEFAULT_CONFIG})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Bounds validation (used by CLI pre-write and runtime post-read)
# ---------------------------------------------------------------------------


def validate_config(cfg_dict: Dict[str, Any], current_step: int = 0,
                    max_loop_iters: int = 16) -> Tuple[bool, List[str]]:
    """Return (is_valid, errors)."""
    errors: List[str] = []

    def _check_range(name: str, val: Any, lo: float, hi: float,
                     inclusive_lo: bool = True, inclusive_hi: bool = True):
        if not isinstance(val, (int, float)):
            errors.append(f"{name}: must be numeric, got {type(val).__name__}")
            return
        if inclusive_lo and val < lo or (not inclusive_lo) and val <= lo:
            errors.append(f"{name}: {val} < {lo}")
        if inclusive_hi and val > hi or (not inclusive_hi) and val >= hi:
            errors.append(f"{name}: {val} > {hi}")

    lr_mult = cfg_dict.get("lr_mult", {})
    for k in ("muon_default", "muon_recurrent", "adamw_default", "adamw_recurrent"):
        _check_range(f"lr_mult.{k}", lr_mult.get(k, 1.0), *LR_MULT_BOUNDS)

    coeffs = cfg_dict.get("loss_coeffs", {})
    for k, (lo, hi) in COEFF_BOUNDS.items():
        _check_range(f"loss_coeffs.{k}", coeffs.get(k, 0.0), lo, hi)

    gc = cfg_dict.get("grad_clip", 1.0)
    _check_range("grad_clip", gc, *BOUNDS["grad_clip"], inclusive_lo=False)

    _check_range("spectral_warn_threshold",
                 cfg_dict.get("spectral_warn_threshold", 0.999),
                 *BOUNDS["spectral_warn_threshold"])
    _check_range("router_bias_update_rate",
                 cfg_dict.get("router_bias_update_rate", 1e-3),
                 *BOUNDS["router_bias_update_rate"])
    _check_range("gpu_hourly_rate",
                 cfg_dict.get("gpu_hourly_rate", 3.50),
                 *BOUNDS["gpu_hourly_rate"])

    n_loops = cfg_dict.get("n_loops_override")
    if n_loops is not None:
        if not isinstance(n_loops, int) or n_loops < 1 or n_loops > max_loop_iters:
            errors.append(f"n_loops_override: must be int in [1, {max_loop_iters}] or null, got {n_loops!r}")

    pause = cfg_dict.get("pause_until_step")
    if pause is not None:
        if not isinstance(pause, int):
            errors.append(f"pause_until_step: must be int or null, got {type(pause).__name__}")
        elif pause < current_step:
            errors.append(f"pause_until_step ({pause}) < current_step ({current_step})")
        elif pause > current_step + 500:
            errors.append(
                f"pause_until_step ({pause}) > current_step + 500 ({current_step + 500})"
            )

    intervals = cfg_dict.get("diagnostic_intervals", {})
    for k, v in intervals.items():
        if not isinstance(v, int):
            errors.append(f"diagnostic_intervals.{k}: must be int, got {type(v).__name__}")
        else:
            _check_range(f"diagnostic_intervals.{k}", v, *INTERVAL_BOUNDS)

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return float(o.detach().float().item())
        return o.detach().float().tolist()
    if hasattr(o, "__float__"):
        return float(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _config_hash(cfg_dict: Dict[str, Any]) -> str:
    blob = json.dumps(cfg_dict, sort_keys=True, default=_json_default).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=_json_default))
        f.write("\n")


# ---------------------------------------------------------------------------
# AdminRuntime
# ---------------------------------------------------------------------------


METRIC_CHUNK_LINES = 5000
ROLLING_LOSS = 200
ROLLING_STEP_TIME = 50


class AdminRuntime:
    """Single-process loop-side runtime. Main thread only (signal handling)."""

    def __init__(
        self,
        output_path: Path,
        run_start_wall: float,
        gpu_hourly_rate: float = 3.50,
        max_loop_iters: int = 16,
    ):
        self.output_path = Path(output_path)
        self.admin_dir = self.output_path / "admin"
        self.events_dir = self.output_path / "events"
        self.commands_dir = self.admin_dir / "commands"
        self.pending_dir = self.commands_dir / "pending"
        self.applied_dir = self.commands_dir / "applied"
        self.rejected_dir = self.commands_dir / "rejected"
        self.metrics_dir = self.events_dir / "metrics"

        for d in (self.admin_dir, self.events_dir, self.commands_dir,
                  self.pending_dir, self.applied_dir, self.rejected_dir,
                  self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.config_path = self.admin_dir / "config.json"
        self.config_applied_path = self.admin_dir / "config_applied.json"
        self.prompts_path = self.admin_dir / "prompts.json"
        self.status_path = self.admin_dir / "status.json"
        self.audit_path = self.admin_dir / "audit.jsonl"
        self.incidents_path = self.events_dir / "incidents.jsonl"
        self.generations_path = self.events_dir / "generations.jsonl"

        self.run_start_wall = run_start_wall
        self.gpu_hourly_rate = gpu_hourly_rate
        self.max_loop_iters = max_loop_iters

        self._last_config_mtime: Optional[float] = None
        self._cached_config: Optional[Config] = None
        self._cached_config_dict: Optional[Dict[str, Any]] = None

        self._metric_chunk: int = self._determine_current_chunk()
        self._metric_lines_in_chunk: int = self._count_lines_in_current_chunk()

        # Rolling buffers — used by loss-spike and step-time regression detectors.
        self._loss_rolling: List[float] = []
        self._step_time_rolling: List[float] = []

        # Runtime state set by commands.
        self._checkpoint_requested: bool = False
        self._eval_requested: List[Dict[str, Any]] = []  # queued eval commands
        self._depth_eval_requested: bool = False
        self._arith_eval_requested: Optional[int] = None
        self._generate_requests: List[Dict[str, Any]] = []
        self._reinit_expert_requests: List[int] = []
        self._reset_router_bias_requested: bool = False
        self._graceful_stop_requested: bool = False
        self._hard_stop_requested: bool = False
        self._last_applied_command_id: Optional[str] = None
        self._last_step_advance_at: str = _now_iso()

        # Incidents tail — kept in memory for fast status.json triage surface.
        self._incidents_tail: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Seeding / bootstrap
    # -----------------------------------------------------------------------

    def seed_config_if_missing(self, argparse_defaults: Optional[Dict[str, Any]] = None) -> None:
        """Write DEFAULT_CONFIG on first launch; pick up argparse lr/clip overrides."""
        if self.config_path.exists():
            return
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
        if argparse_defaults:
            if "grad_clip" in argparse_defaults:
                cfg["grad_clip"] = argparse_defaults["grad_clip"]
            if "gpu_hourly_rate" in argparse_defaults:
                cfg["gpu_hourly_rate"] = argparse_defaults["gpu_hourly_rate"]
        _atomic_write_json(self.config_path, cfg)
        if not self.prompts_path.exists():
            _atomic_write_json(self.prompts_path, DEFAULT_PROMPTS)

    def load_prompts(self) -> List[str]:
        if not self.prompts_path.exists():
            return list(DEFAULT_PROMPTS)
        try:
            return list(json.loads(self.prompts_path.read_text()))
        except (json.JSONDecodeError, OSError):
            return list(DEFAULT_PROMPTS)

    def register_sigterm(self) -> None:
        """SIGTERM → graceful_stop. Safe to call from main thread only."""
        try:
            signal.signal(signal.SIGTERM, self._sigterm_handler)
        except ValueError:
            # Not main thread, or signals unsupported — fine.
            pass

    def _sigterm_handler(self, signum: int, frame) -> None:  # noqa: ARG002
        self._graceful_stop_requested = True
        self.emit_incident("sigterm_received", severity="WARN", step=None,
                           context={"signum": signum})

    # -----------------------------------------------------------------------
    # Config I/O
    # -----------------------------------------------------------------------

    def read_config(self, current_step: int = 0) -> Config:
        """Mtime-gated, defensive read. Returns last-known-good on parse failure."""
        try:
            mtime = self.config_path.stat().st_mtime
        except FileNotFoundError:
            if self._cached_config is not None:
                return self._cached_config
            self.seed_config_if_missing()
            mtime = self.config_path.stat().st_mtime

        if self._cached_config is not None and mtime == self._last_config_mtime:
            return self._cached_config

        raw: Optional[str] = None
        try:
            raw = self.config_path.read_text()
            cfg_dict = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            # Partial write or transient read error. One retry after a short pause.
            time.sleep(0.05)
            try:
                raw = self.config_path.read_text()
                cfg_dict = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                self.emit_incident("config_parse_failed", severity="CRIT",
                                   step=current_step, context={"error": str(e),
                                                                "raw_head": (raw or "")[:200]})
                if self._cached_config is not None:
                    return self._cached_config
                cfg_dict = json.loads(json.dumps(DEFAULT_CONFIG))

        ok, errors = validate_config(cfg_dict, current_step=current_step,
                                     max_loop_iters=self.max_loop_iters)
        if not ok:
            self.emit_incident("config_bounds_violated", severity="WARN",
                               step=current_step, context={"errors": errors})
            if self._cached_config is not None:
                # Keep last-good in memory; don't overwrite disk (CLI's job).
                return self._cached_config
            cfg_dict = json.loads(json.dumps(DEFAULT_CONFIG))

        cfg = Config.from_dict(cfg_dict)
        if self._cached_config is not None and self._config_diff(self._cached_config_dict, cfg_dict):
            self._write_audit_config_change(current_step, self._cached_config_dict, cfg_dict)

        self._cached_config = cfg
        self._cached_config_dict = cfg_dict
        self._last_config_mtime = mtime
        return cfg

    def _config_diff(self, before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        return json.dumps(before, sort_keys=True, default=_json_default) != \
               json.dumps(after, sort_keys=True, default=_json_default)

    def _write_audit_config_change(self, step: int, before: Dict[str, Any],
                                    after: Dict[str, Any]) -> None:
        _append_jsonl(self.audit_path, {
            "at": _now_iso(), "step": step, "kind": "config_change",
            "before_hash": _config_hash(before), "after_hash": _config_hash(after),
            "diff": _shallow_diff(before, after), "source": "config.json",
        })

    def write_config_applied(self, config: Config) -> None:
        _atomic_write_json(self.config_applied_path, config.to_dict())

    # -----------------------------------------------------------------------
    # Command queue
    # -----------------------------------------------------------------------

    def process_pending_commands(
        self,
        executor: Dict[str, Callable[..., Dict[str, Any]]],
        current_step: int,
    ) -> None:
        """Scan pending/ in sorted order, dispatch to executor callables, move to applied/."""
        try:
            pending = sorted(self.pending_dir.iterdir())
        except FileNotFoundError:
            return

        for cmd_path in pending:
            if not cmd_path.is_file() or not cmd_path.name.endswith(".json"):
                continue
            try:
                cmd_data = json.loads(cmd_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                self._move_rejected(cmd_path, {"reason": f"parse_error: {e}"})
                continue

            name = cmd_data.get("name")
            args = cmd_data.get("args", {}) or {}
            if name not in KNOWN_COMMANDS:
                self._move_rejected(cmd_path, {"reason": f"unknown_command: {name}"})
                continue

            # Safety: hard_stop requires confirm_step == current_step.
            if name == "hard_stop":
                confirm = args.get("confirm_step")
                if confirm != current_step:
                    self._move_rejected(cmd_path, {
                        "reason": f"hard_stop confirm_step mismatch: got {confirm}, current {current_step}"
                    })
                    continue

            # Local flag-only commands bypass the executor map.
            try:
                if name == "checkpoint_now":
                    self._checkpoint_requested = True
                    result = {"queued": True, "will_apply_at_step": current_step}
                elif name == "eval_now":
                    nl = int(args.get("n_loops", 8))
                    self._eval_requested.append({"n_loops": nl})
                    result = {"queued": True, "n_loops": nl}
                elif name == "depth_eval_now":
                    self._depth_eval_requested = True
                    result = {"queued": True}
                elif name == "arith_eval_now":
                    self._arith_eval_requested = int(args.get("n_per_depth", 20))
                    result = {"queued": True, "n_per_depth": self._arith_eval_requested}
                elif name == "generate":
                    self._generate_requests.append(args)
                    result = {"queued": True, "prompt_head": str(args.get("prompt", ""))[:80]}
                elif name == "reinit_expert":
                    eid = int(args["expert_id"])
                    self._reinit_expert_requests.append(eid)
                    result = {"queued": True, "expert_id": eid}
                elif name == "reset_router_bias":
                    self._reset_router_bias_requested = True
                    result = {"queued": True}
                elif name == "graceful_stop":
                    self._graceful_stop_requested = True
                    result = {"queued": True}
                elif name == "hard_stop":
                    self._hard_stop_requested = True
                    result = {"queued": True, "exit_immediately": True}
                else:
                    # Unreachable given KNOWN_COMMANDS filter.
                    result = {"error": "unhandled"}
            except (KeyError, ValueError, TypeError) as e:
                self._move_rejected(cmd_path, {"reason": f"arg_error: {e}"})
                continue

            self._move_applied(cmd_path, current_step, result)
            self._last_applied_command_id = cmd_path.stem
            _append_jsonl(self.audit_path, {
                "at": _now_iso(), "step": current_step, "kind": "command",
                "name": name, "args": args, "result": result,
                "source": cmd_path.name,
            })

    def _move_applied(self, cmd_path: Path, step: int, result: Dict[str, Any]) -> None:
        payload = json.loads(cmd_path.read_text())
        payload["applied_at_step"] = step
        payload["applied_at"] = _now_iso()
        payload["result"] = result
        dest = self.applied_dir / cmd_path.name
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
        os.replace(tmp, dest)
        cmd_path.unlink(missing_ok=True)

    def _move_rejected(self, cmd_path: Path, extra: Dict[str, Any]) -> None:
        try:
            payload = json.loads(cmd_path.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {"raw": cmd_path.read_text(errors="replace")[:500]}
        payload.update(extra)
        payload["rejected_at"] = _now_iso()
        dest = self.rejected_dir / cmd_path.name
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
        os.replace(tmp, dest)
        cmd_path.unlink(missing_ok=True)
        self.emit_incident("command_rejected", severity="WARN", step=None,
                           context={"file": cmd_path.name, **extra})

    # Pops & getters for the loop to drain queued requests.
    def pop_checkpoint_request(self) -> bool:
        req = self._checkpoint_requested
        self._checkpoint_requested = False
        return req

    def pop_eval_requests(self) -> List[Dict[str, Any]]:
        out = list(self._eval_requested)
        self._eval_requested.clear()
        return out

    def pop_depth_eval_request(self) -> bool:
        req = self._depth_eval_requested
        self._depth_eval_requested = False
        return req

    def pop_arith_eval_request(self) -> Optional[int]:
        req = self._arith_eval_requested
        self._arith_eval_requested = None
        return req

    def pop_generate_requests(self) -> List[Dict[str, Any]]:
        out = list(self._generate_requests)
        self._generate_requests.clear()
        return out

    def pop_reinit_expert_requests(self) -> List[int]:
        out = list(self._reinit_expert_requests)
        self._reinit_expert_requests.clear()
        return out

    def pop_reset_router_bias_request(self) -> bool:
        req = self._reset_router_bias_requested
        self._reset_router_bias_requested = False
        return req

    def enqueue_graceful_stop(self) -> None:
        self._graceful_stop_requested = True

    def should_stop(self) -> bool:
        return self._graceful_stop_requested or self._hard_stop_requested

    def is_hard_stop(self) -> bool:
        return self._hard_stop_requested

    # -----------------------------------------------------------------------
    # Status / metrics / incidents / generations
    # -----------------------------------------------------------------------

    def write_status(self, **fields: Any) -> None:
        """Atomic rewrite of status.json. Caller passes current snapshot fields."""
        now = _now_iso()
        wall_hours = (time.time() - self.run_start_wall) / 3600.0
        status = {
            "heartbeat_at": now,
            "last_step_advance_at": self._last_step_advance_at,
            "wall_hours": round(wall_hours, 3),
            "estimated_cost_usd": round(wall_hours * self.gpu_hourly_rate, 2),
            "config_hash": _config_hash(self._cached_config_dict or DEFAULT_CONFIG),
            "last_applied_command_id": self._last_applied_command_id,
            "metrics_chunk": self._metric_chunk,
            "incidents_tail": list(self._incidents_tail[-5:]),
        }
        status.update(fields)
        _atomic_write_json(self.status_path, status)

    def note_step_advanced(self) -> None:
        """Call once per actual step increment so `last_step_advance_at` distinguishes
        liveness (status refresh) from progress (step counter moving)."""
        self._last_step_advance_at = _now_iso()

    def log_metric(self, step: int, metrics: Dict[str, Any]) -> None:
        """Append one line to the current metrics chunk; rotate if needed."""
        chunk_path = self.metrics_dir / f"metrics.{self._metric_chunk:03d}.jsonl"
        payload = {"at": _now_iso(), "step": step, **metrics}
        with chunk_path.open("a") as f:
            f.write(json.dumps(payload, default=_json_default))
            f.write("\n")
        self._metric_lines_in_chunk += 1
        if self._metric_lines_in_chunk >= METRIC_CHUNK_LINES:
            self._metric_chunk += 1
            self._metric_lines_in_chunk = 0

    def emit_incident(
        self,
        kind: str,
        severity: str = "WARN",
        step: Optional[int] = None,
        value: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "at": _now_iso(), "step": step, "severity": severity,
            "kind": kind, "value": value, "context": context or {},
        }
        _append_jsonl(self.incidents_path, rec)
        self._incidents_tail.append(rec)
        if len(self._incidents_tail) > 20:
            self._incidents_tail = self._incidents_tail[-20:]

    def log_generation(self, step: int, prompt: str, output: str, n_loops: int,
                       temperature: float, top_k: int, latency_s: float,
                       source: str = "panel") -> None:
        _append_jsonl(self.generations_path, {
            "at": _now_iso(), "step": step, "prompt": prompt, "output": output,
            "n_loops": n_loops, "temperature": temperature, "top_k": top_k,
            "latency_s": round(latency_s, 3), "source": source,
        })

    # -----------------------------------------------------------------------
    # Detectors
    # -----------------------------------------------------------------------

    def check_nan_inf(self, loss_dict: Dict[str, float], model: torch.nn.Module
                      ) -> Optional[str]:
        """Return a short description if NaN/Inf detected in loss_dict values or any
        parameter gradient; None otherwise."""
        for k, v in loss_dict.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return f"loss.{k}={v}"
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all():
                return f"grad.{name}"
        return None

    def update_loss_rolling(self, total_loss: float) -> None:
        if total_loss is None or math.isnan(total_loss) or math.isinf(total_loss):
            return
        self._loss_rolling.append(total_loss)
        if len(self._loss_rolling) > ROLLING_LOSS:
            self._loss_rolling = self._loss_rolling[-ROLLING_LOSS:]

    def update_step_time_rolling(self, step_time_s: float) -> None:
        if step_time_s is None or step_time_s <= 0:
            return
        self._step_time_rolling.append(step_time_s)
        if len(self._step_time_rolling) > ROLLING_STEP_TIME:
            self._step_time_rolling = self._step_time_rolling[-ROLLING_STEP_TIME:]

    def loss_spike(self, total_loss: float, threshold: float = 5.0,
                   min_history: int = 200) -> Optional[Tuple[float, float]]:
        if len(self._loss_rolling) < min_history or total_loss is None:
            return None
        baseline = statistics.median(self._loss_rolling)
        if baseline <= 0:
            return None
        if total_loss > threshold * baseline:
            return (total_loss, baseline)
        return None

    def step_time_regressed(self, step_time_s: float, threshold: float = 2.0,
                             min_history: int = 50) -> Optional[Tuple[float, float]]:
        if len(self._step_time_rolling) < min_history or step_time_s is None:
            return None
        baseline = statistics.median(self._step_time_rolling)
        if baseline <= 0:
            return None
        if step_time_s > threshold * baseline:
            return (step_time_s, baseline)
        return None

    def loss_rolling_mean(self) -> float:
        if not self._loss_rolling:
            return 0.0
        return sum(self._loss_rolling) / len(self._loss_rolling)

    # -----------------------------------------------------------------------
    # Checkpointing helpers
    # -----------------------------------------------------------------------

    def disk_ok_for_checkpoint(self, ckpt_dir: Path, required_free_gb: float = 30.0,
                                min_multiple: float = 2.0) -> bool:
        """Return True if at least max(required_free_gb, min_multiple × largest existing ckpt)
        is free on the checkpoint filesystem."""
        try:
            du = shutil.disk_usage(ckpt_dir if Path(ckpt_dir).exists() else ckpt_dir.parent)
        except FileNotFoundError:
            return True
        free_gb = du.free / 1e9
        largest_gb = 0.0
        try:
            for p in Path(ckpt_dir).glob("checkpoint_step_*.pt"):
                largest_gb = max(largest_gb, p.stat().st_size / 1e9)
        except (FileNotFoundError, OSError):
            pass
        threshold = max(required_free_gb, min_multiple * largest_gb)
        return free_gb >= threshold

    def disk_free_gb(self, path: Path) -> float:
        try:
            du = shutil.disk_usage(path if path.exists() else path.parent)
            return round(du.free / 1e9, 2)
        except (FileNotFoundError, OSError):
            return 0.0

    def snapshot_state(self) -> Dict[str, Any]:
        """State blob to serialize into checkpoint payload."""
        return {
            "config_hash": _config_hash(self._cached_config_dict or DEFAULT_CONFIG),
            "config_dict": self._cached_config_dict,
            "last_applied_command_id": self._last_applied_command_id,
            "metric_chunk": self._metric_chunk,
            "loss_rolling": list(self._loss_rolling),
            "step_time_rolling": list(self._step_time_rolling),
            "run_start_wall": self.run_start_wall,
        }

    def load_state(self, admin_state: Optional[Dict[str, Any]]) -> None:
        if not admin_state:
            return
        self._last_applied_command_id = admin_state.get("last_applied_command_id")
        self._loss_rolling = list(admin_state.get("loss_rolling", []))
        self._step_time_rolling = list(admin_state.get("step_time_rolling", []))
        # Don't restore run_start_wall — the resumed process has a new wall-start.
        # metric_chunk restored from disk scan (more authoritative than checkpoint).

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _determine_current_chunk(self) -> int:
        existing = sorted(self.metrics_dir.glob("metrics.*.jsonl"))
        if not existing:
            return 0
        m = re.match(r"metrics\.(\d+)\.jsonl", existing[-1].name)
        return int(m.group(1)) if m else 0

    def _count_lines_in_current_chunk(self) -> int:
        path = self.metrics_dir / f"metrics.{self._metric_chunk:03d}.jsonl"
        if not path.exists():
            return 0
        try:
            with path.open("rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0


def _shallow_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Report keys that changed at the top or one level deep."""
    diff: Dict[str, Any] = {}
    all_keys = set(before) | set(after)
    for k in all_keys:
        b, a = before.get(k), after.get(k)
        if isinstance(b, dict) and isinstance(a, dict):
            sub = _shallow_diff(b, a)
            if sub:
                diff[k] = sub
        elif b != a:
            diff[k] = {"before": b, "after": a}
    return diff


# ---------------------------------------------------------------------------
# Command-file helpers shared with the CLI
# ---------------------------------------------------------------------------


_COUNTER_PATH_SUFFIX = ".counter"


def next_command_filename(pending_dir: Path, cmd_name: str) -> str:
    """Return a new command filename with a monotonic counter. Uses a tiny
    counter file so two processes can't collide (though we don't expect
    concurrent CLI callers)."""
    pending_dir.mkdir(parents=True, exist_ok=True)
    counter_path = pending_dir.parent / (cmd_name + _COUNTER_PATH_SUFFIX)
    try:
        counter = int(counter_path.read_text())
    except (FileNotFoundError, ValueError):
        counter = _scan_existing_counter(pending_dir)
    counter += 1
    counter_path.write_text(str(counter))
    return f"{counter:08d}-{uuid.uuid4().hex[:8]}.json"


def _scan_existing_counter(pending_dir: Path) -> int:
    """Seed counter from existing filenames across pending/applied/rejected."""
    base = pending_dir.parent
    high = 0
    for sub in ("pending", "applied", "rejected"):
        d = base / sub
        if not d.exists():
            continue
        for p in d.iterdir():
            m = re.match(r"(\d+)-", p.name)
            if m:
                high = max(high, int(m.group(1)))
    return high
