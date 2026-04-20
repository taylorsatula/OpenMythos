#!/usr/bin/env python3
"""
Loop-depth curriculum for RDT training.

Ramp: linear 1 → max_loops_train over the first `ramp_frac` of training.
Plateau: sample n_loops uniformly from [max_loops_train//2, max_loop_iters]
         so LoRA scale-embedding slots past max_loops_train still receive
         gradient signal; without this, slots 8..max_loop_iters-1 stay at
         their N(0, 0.02) init and inference at n_loops>max_loops_train
         hits uninitialized LoRA deltas.
"""

import random
from typing import Optional


class CurriculumScheduler:
    def __init__(
        self,
        total_steps: int,
        max_loops_train: int = 8,
        max_loop_iters: int = 16,
        ramp_frac: float = 0.3,
        seed: int = 42,
    ):
        self.total_steps = total_steps
        self.max_loops_train = max_loops_train
        self.max_loop_iters = max_loop_iters
        self.ramp_end = max(1, int(total_steps * ramp_frac))
        self._rng = random.Random(seed)
        self._plateau_lo = max(2, max_loops_train // 2)
        self._plateau_hi = max_loop_iters

    def step(self, current_step: int) -> int:
        """Return the n_loops to use for the current optimizer step."""
        if current_step < self.ramp_end:
            # Linear ramp from 1 → max_loops_train
            return max(1, int(1 + (self.max_loops_train - 1) * current_step / self.ramp_end))
        return self._rng.randint(self._plateau_lo, self._plateau_hi)

    def state_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "max_loops_train": self.max_loops_train,
            "max_loop_iters": self.max_loop_iters,
            "ramp_end": self.ramp_end,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.total_steps = state["total_steps"]
        self.max_loops_train = state["max_loops_train"]
        self.max_loop_iters = state["max_loop_iters"]
        self.ramp_end = state["ramp_end"]
        self._rng.setstate(state["rng_state"])
        self._plateau_lo = max(2, self.max_loops_train // 2)
        self._plateau_hi = self.max_loop_iters
