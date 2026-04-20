#!/usr/bin/env python3
"""
Packed-sequence DataLoader for OpenMythos training.

Reads Parquet shards produced by `data/prepare_data.py`. Each shard contains
a `tokens` column where every row is a list of `seq_len + 1` token ids,
already EOS-separated and packed to length. The "+ 1" is for the
cross-entropy shift; training splits each row into (input_ids, targets).

Shard naming:
    train_0000.parquet, train_0001.parquet, ...
    val_0000.parquet,   val_0001.parquet,   ...
"""

import random
from pathlib import Path
from typing import Tuple

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset


class PackedSequenceDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str = "data/tokenized_shards",
        seq_len: int = 2048,
        split: str = "train",
        shuffle_shards: bool = True,
        seed: int = 42,
        return_meta: bool = False,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.split = split
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.return_meta = return_meta

        self.shard_paths = sorted(Path(data_dir).glob(f"{split}_*.parquet"))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shards matching '{split}_*.parquet' in {data_dir}. "
                f"Run `python -m data.prepare_data` first."
            )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        num_workers = worker.num_workers if worker else 1
        wid = worker.id if worker else 0

        my_shards = self.shard_paths[wid::num_workers]
        if self.shuffle_shards:
            random.Random(self.seed + wid).shuffle(my_shards)

        expected_len = self.seq_len + 1
        for shard in my_shards:
            table = pq.read_table(shard, columns=["tokens"])
            shard_name = shard.name
            for row_idx, row in enumerate(table.column("tokens")):
                tokens = row.as_py()
                if len(tokens) != expected_len:
                    # Shards written for a different seq_len; skip gracefully.
                    continue
                t = torch.tensor(tokens, dtype=torch.long)
                if self.return_meta:
                    yield t[:-1], t[1:], {"shard": shard_name, "row": row_idx}
                else:
                    yield t[:-1], t[1:]


def _collate(rows):
    input_ids = torch.stack([r[0] for r in rows])
    targets = torch.stack([r[1] for r in rows])
    return input_ids, targets


def _collate_with_meta(rows):
    input_ids = torch.stack([r[0] for r in rows])
    targets = torch.stack([r[1] for r in rows])
    meta = [r[2] for r in rows]
    return input_ids, targets, meta


def create_dataloader(
    data_dir: str = "data/tokenized_shards",
    batch_size: int = 32,
    seq_len: int = 2048,
    split: str = "train",
    num_workers: int = 4,
    seed: int = 42,
    return_meta: bool = False,
) -> DataLoader:
    dataset = PackedSequenceDataset(
        data_dir=data_dir, seq_len=seq_len, split=split, seed=seed,
        return_meta=return_meta,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_with_meta if return_meta else _collate,
    )
