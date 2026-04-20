#!/usr/bin/env python3
"""
Dataset preparation: stream → filter → dedup → tokenize → pack → parquet shards.

Produces shards under `data/tokenized_shards/` with the schema:
    column 'tokens': list[int64] of length seq_len + 1 (2049 by default)
consumed by `data.dataloader.PackedSequenceDataset`.

Composition per HANDOFF §4.1:
    60%  FineWeb-Edu (score >= 3)
    10%  OpenWebMath
    15%  The Stack v2 dedup (Python/Rust/TypeScript/C)
    10%  Wikipedia
    5%   arXiv abstracts (via RedPajama-Data-V2 arxiv subset)

Usage:
    python -m data.prepare_data --output_dir data/tokenized_shards --target_tokens 15_000_000_000

Disk: ~60 GB at 15B tokens. I/O bound; not GPU bound.
"""

import argparse
import hashlib
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_CONFIGS: Dict[str, dict] = {
    "fineweb_edu": {
        "hf": "HuggingFaceFW/fineweb-edu",
        "hf_config": "sample-10BT",
        "split": "train",
        "proportion": 0.60,
        "filter": lambda s: s.get("score", 0) >= 3 and s.get("text"),
        "text_key": "text",
    },
    "open_web_math": {
        "hf": "open-web-math/open-web-math",
        "hf_config": None,
        "split": "train",
        "proportion": 0.10,
        "filter": lambda s: bool(s.get("text")),
        "text_key": "text",
    },
    "the_stack": {
        "hf": "bigcode/the-stack-v2-dedup",
        "hf_config": None,
        "split": "train",
        "proportion": 0.15,
        "filter": lambda s: (
            s.get("lang") in ("Python", "Rust", "TypeScript", "C")
            and s.get("content")
            and s["content"].count("\n") >= 50
        ),
        "text_key": "content",
    },
    "wikipedia": {
        "hf": "wikimedia/wikipedia",
        "hf_config": "20231101.en",
        "split": "train",
        "proportion": 0.10,
        "filter": lambda s: bool(s.get("text")),
        "text_key": "text",
    },
    "arxiv": {
        "hf": "togethercomputer/RedPajama-Data-V2",
        "hf_config": "default",
        "split": "train",
        "proportion": 0.05,
        "filter": lambda s: s.get("meta", {}).get("source") == "arxiv" and s.get("text"),
        "text_key": "text",
    },
}


def _doc_hash(text: str) -> str:
    """SHA-256 of the first 1000 characters — cheap document-level dedup."""
    return hashlib.sha256(text[:1000].encode("utf-8", errors="ignore")).hexdigest()


def stream_source(name: str, cfg: dict) -> Iterator[str]:
    """Stream filtered text strings from one source."""
    from datasets import load_dataset
    kwargs = dict(streaming=True, split=cfg["split"])
    if cfg.get("hf_config"):
        ds = load_dataset(cfg["hf"], cfg["hf_config"], **kwargs)
    else:
        ds = load_dataset(cfg["hf"], **kwargs)
    text_key = cfg["text_key"]
    filt = cfg["filter"]
    for sample in ds:
        try:
            if not filt(sample):
                continue
        except Exception:
            continue
        text = sample.get(text_key)
        if text:
            yield text


def mixed_stream(
    sources: Dict[str, dict],
    seed: int = 42,
    max_dedup_cache: int = 10_000_000,
) -> Iterator[Tuple[str, str]]:
    """
    Interleave source streams in proportion to their configured shares,
    yielding deduplicated (source_name, text) pairs.

    Dedup cache is LRU-bounded at `max_dedup_cache` hashes to avoid
    unbounded memory growth during a full run.
    """
    rng = random.Random(seed)
    iters = {name: stream_source(name, cfg) for name, cfg in sources.items()}
    names = list(sources.keys())
    weights = [sources[n]["proportion"] for n in names]

    seen_hashes: "dict[str, None]" = {}  # dict preserves insertion order
    while iters:
        name = rng.choices(names, weights=weights, k=1)[0]
        try:
            text = next(iters[name])
        except StopIteration:
            # Retire this source; renormalize weights.
            idx = names.index(name)
            iters.pop(name)
            names.pop(idx)
            weights.pop(idx)
            if not iters:
                return
            continue

        h = _doc_hash(text)
        if h in seen_hashes:
            continue
        seen_hashes[h] = None
        if len(seen_hashes) > max_dedup_cache:
            # Evict the oldest ~10% to keep memory bounded
            evict_n = max_dedup_cache // 10
            for k in list(seen_hashes.keys())[:evict_n]:
                seen_hashes.pop(k, None)

        yield name, text


class Packer:
    """Pack tokenized documents into fixed-length sequences with EOS separators."""

    def __init__(self, seq_len: int, eos_id: int):
        self.row_len = seq_len + 1  # +1 for the CE shift
        self.eos_id = eos_id
        self.buffer: List[int] = []

    def add(self, tokens: List[int]) -> Iterator[List[int]]:
        """Extend the buffer with `tokens` + EOS; yield every full row."""
        self.buffer.extend(tokens)
        self.buffer.append(self.eos_id)
        while len(self.buffer) >= self.row_len:
            row = self.buffer[: self.row_len]
            self.buffer = self.buffer[self.row_len:]
            yield row

    def flush_partial(self) -> None:
        """Discard any incomplete trailing buffer — cleaner than padding."""
        self.buffer = []


class ShardWriter:
    """Write rows into Parquet shards of ~`shard_bytes_target` bytes each."""

    def __init__(self, output_dir: Path, split: str, shard_bytes_target: int = 1_000_000_000):
        self.output_dir = output_dir
        self.split = split
        self.shard_bytes_target = shard_bytes_target
        self.rows: List[List[int]] = []
        self.shard_idx = 0
        self.total_rows = 0
        self.total_tokens = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _approx_bytes(self) -> int:
        if not self.rows:
            return 0
        return len(self.rows) * len(self.rows[0]) * 8  # int64

    def add(self, row: List[int]) -> None:
        self.rows.append(row)
        self.total_tokens += len(row) - 1  # don't count the shift slot
        if self._approx_bytes() >= self.shard_bytes_target:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pydict({"tokens": self.rows})
        path = self.output_dir / f"{self.split}_{self.shard_idx:04d}.parquet"
        pq.write_table(table, path, compression="zstd")
        print(f"  wrote {path} ({len(self.rows):,} rows, {table.nbytes / 1e6:.1f} MB)")
        self.total_rows += len(self.rows)
        self.rows = []
        self.shard_idx += 1


def prepare_shards(
    output_dir: str = "data/tokenized_shards",
    target_tokens: int = 15_000_000_000,
    seq_len: int = 2048,
    val_fraction: float = 0.001,
    seed: int = 42,
    shard_bytes_target: int = 1_000_000_000,
) -> None:
    """
    Download → filter → tokenize → pack → shard. Writes `{split}_{NNNN}.parquet`
    under `output_dir`. Stops once `target_tokens` of training-split tokens
    have been packed.
    """
    from open_mythos.tokenizer import MythosTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = MythosTokenizer()
    # Resolve the EOS token id from the underlying HF tokenizer.
    eos_id = tokenizer.tokenizer.eos_token_id
    if eos_id is None:
        # Fall back to <|endoftext|>-style marker; gpt-oss-20b exposes this.
        eos_id = tokenizer.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if eos_id is None or eos_id < 0:
        raise RuntimeError("Tokenizer has no usable EOS id; set one explicitly.")

    print(f"Tokenizer: {tokenizer.tokenizer.name_or_path}")
    print(f"EOS id:    {eos_id}")
    print(f"Seq len:   {seq_len} (+1 for shift)")
    print(f"Target:    {target_tokens:,} tokens")
    print()

    rng = random.Random(seed)
    packer = Packer(seq_len=seq_len, eos_id=eos_id)
    train_writer = ShardWriter(output_path, split="train", shard_bytes_target=shard_bytes_target)
    val_writer = ShardWriter(output_path, split="val", shard_bytes_target=shard_bytes_target // 4)

    start = time.time()
    for _, text in mixed_stream(DATASET_CONFIGS, seed=seed):
        tokens = tokenizer.encode(text)
        if len(tokens) < 100:
            continue
        for row in packer.add(tokens):
            writer = val_writer if rng.random() < val_fraction else train_writer
            writer.add(row)

        if train_writer.total_tokens + train_writer._approx_bytes() // 8 >= target_tokens:
            break
        if (train_writer.total_rows + len(train_writer.rows)) % 1000 == 0 and train_writer.rows:
            elapsed = time.time() - start
            rate = (train_writer.total_tokens + 1) / max(elapsed, 1)
            print(f"  [{elapsed/60:.1f} min] "
                  f"train_tokens={train_writer.total_tokens:,} "
                  f"rate={rate/1e6:.2f}M tok/s")

    train_writer.flush()
    val_writer.flush()

    print("\nDone.")
    print(f"  train: {train_writer.total_rows:,} rows, {train_writer.total_tokens:,} tokens")
    print(f"  val:   {val_writer.total_rows:,} rows, {val_writer.total_tokens:,} tokens")

    # Sanity: train/val 16-gram overlap. A healthy dedup yields <0.1% overlap
    # on a random sample; higher suggests the dedup cache dropped near-dups
    # into opposite splits and your val perplexity will be optimistic.
    overlap = _train_val_ngram_overlap(output_path, n=16, n_samples=512)
    print(f"\n  train/val 16-gram overlap (sample n=512): {overlap*100:.3f}%")
    if overlap > 0.01:
        print(f"  WARN: overlap >1% — inspect dedup logic; val metrics will lie.")
    elif overlap > 0.001:
        print(f"  NOTE: overlap >0.1% — review dedup; boundary leakage possible.")


def _train_val_ngram_overlap(output_path: "Path", n: int = 16, n_samples: int = 512) -> float:
    """
    Rough estimate of n-gram leakage between train and val splits.

    Samples up to `n_samples` rows from each split, extracts all n-grams,
    and reports the fraction of val n-grams that also appear in the train
    sample. This is a leak-detection heuristic, not a guarantee of non-leak;
    large n (16) with small sample sizes (hundreds) gives a high-specificity
    signal: false positives are unlikely at natural language with vocab~200k.
    """
    import random

    train_paths = sorted(output_path.glob("train_*.parquet"))
    val_paths = sorted(output_path.glob("val_*.parquet"))
    if not train_paths or not val_paths:
        return 0.0

    rng = random.Random(0)

    def _collect_ngrams(paths, n_rows):
        rows_read = 0
        ngrams: set = set()
        for p in paths:
            if rows_read >= n_rows:
                break
            tbl = pq.read_table(p, columns=["tokens"])
            for row in tbl.column("tokens"):
                tokens = row.as_py()
                for i in range(0, len(tokens) - n + 1, max(1, (len(tokens) // 64))):
                    ngrams.add(tuple(tokens[i : i + n]))
                rows_read += 1
                if rows_read >= n_rows:
                    break
        return ngrams

    train_ngrams = _collect_ngrams(train_paths, n_samples)
    val_ngrams = _collect_ngrams(val_paths, n_samples)
    if not val_ngrams:
        return 0.0
    shared = train_ngrams & val_ngrams
    return len(shared) / len(val_ngrams)


def main():
    ap = argparse.ArgumentParser(description="Prepare tokenized training shards")
    ap.add_argument("--output_dir", default="data/tokenized_shards")
    ap.add_argument("--target_tokens", type=int, default=15_000_000_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--val_fraction", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_bytes", type=int, default=1_000_000_000)
    args = ap.parse_args()
    prepare_shards(
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        seed=args.seed,
        shard_bytes_target=args.shard_bytes,
    )


if __name__ == "__main__":
    main()
