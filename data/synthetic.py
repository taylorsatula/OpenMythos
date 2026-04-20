#!/usr/bin/env python3
"""
Synthetic multi-hop arithmetic data generation.

Used for the depth-extrapolation eval (§6.5 of HANDOFF.md): generate
K-step arithmetic chains in the documented format

    Compute: 3 + 7 = 10, 10 * 2 = 20, 20 - 4 = 16. Answer: 16

with intermediate values shown. When `mask_answer=True` the Answer is
blanked so the model must produce it.

Training depths: K ∈ {2, 3, 4, 5}
Extrapolation depths: K ∈ {6, 8, 10, 12, 15}
"""

import random
from pathlib import Path
from typing import Iterator, List, Tuple


OPERATIONS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
]


def generate_arithmetic_chain(
    depth: int,
    seed: int = None,
    mask_answer: bool = False,
) -> Tuple[str, int]:
    """
    Generate a multi-hop arithmetic chain of the given depth.

    Format (intermediate values shown):
        Compute: {a} {op} {b} = {r1}, {r1} {op} {c} = {r2}, ... . Answer: {final}

    Args:
        depth: number of operations in the chain (K)
        seed: optional seed for a local RNG (does not touch global random state)
        mask_answer: if True, the returned prompt omits the final value after
                     "Answer:" so a model must produce it

    Returns:
        (prompt_string, final_value)
    """
    rng = random.Random(seed)
    value = rng.randint(1, 9)
    steps: List[str] = []
    for _ in range(depth):
        op_name, op_fn = rng.choice(OPERATIONS)
        # Keep multiplication operands small so results stay in 3-digit range
        operand = rng.randint(1, 5) if op_name == "*" else rng.randint(1, 9)
        prev = value
        value = op_fn(value, operand)
        steps.append(f"{prev} {op_name} {operand} = {value}")

    question = "Compute: " + ", ".join(steps) + "."
    if mask_answer:
        prompt = question + " Answer:"
    else:
        prompt = question + f" Answer: {value}"
    return prompt, value


def generate_dataset(
    depths: List[int] = None,
    examples_per_depth: int = 500,
    output_dir: str = "data/synthetic",
) -> Path:
    """
    Write one file per depth. Each line is a full (answer-visible) example
    for human inspection; the evaluation code re-generates with `mask_answer=True`.
    """
    if depths is None:
        depths = [2, 3, 4, 5, 6, 8, 10, 12, 15]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for depth in depths:
        filepath = output_path / f"arithmetic_K{depth}.txt"
        with open(filepath, "w") as f:
            for i in range(examples_per_depth):
                prompt, _ = generate_arithmetic_chain(depth, seed=depth * 100003 + i)
                f.write(f"{prompt}\n")
    return output_path


def load_synthetic_data(
    depth: int,
    data_dir: str = "data/synthetic",
    mask_answer: bool = True,
) -> List[Tuple[str, int]]:
    """
    Load generated synthetic data for a depth. If `mask_answer=True`, the
    returned prompts have the trailing answer stripped.

    Returns a list of (prompt, answer) tuples.
    """
    filepath = Path(data_dir) / f"arithmetic_K{depth}.txt"
    if not filepath.exists():
        print(f"Synthetic data not found at {filepath}; generating…")
        generate_dataset(output_dir=data_dir)

    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("Answer:", 1)
            if len(parts) != 2:
                continue
            question_prefix, answer_str = parts[0].strip(), parts[1].strip()
            try:
                answer = int(answer_str)
            except ValueError:
                continue
            prompt = question_prefix + " Answer:" if mask_answer else line
            data.append((prompt, answer))
    return data


class SyntheticArithmeticIterator:
    """
    Infinite iterator over training-depth arithmetic chains.

    Uses an isolated Random instance; seeding is deterministic per call.
    """

    def __init__(self, depths: List[int] = None, seed: int = 42, mask_answer: bool = False):
        self.depths = depths if depths is not None else [2, 3, 4, 5]
        self.seed = seed
        self.mask_answer = mask_answer
        self.counter = 0
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        return self

    def __next__(self) -> Tuple[str, int]:
        depth = self._rng.choice(self.depths)
        self.counter += 1
        return generate_arithmetic_chain(
            depth, seed=self.seed + self.counter, mask_answer=self.mask_answer,
        )


if __name__ == "__main__":
    out = generate_dataset()
    print(f"Generated synthetic data in {out}")
    for depth in [2, 5, 10]:
        p, a = generate_arithmetic_chain(depth, seed=0)
        print(f"K={depth}: {p[:120]}{'…' if len(p) > 120 else ''} (answer={a})")
