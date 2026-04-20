#!/usr/bin/env python3
"""
Dense baseline configuration — FLOP-matched to the RDT at its training loop count.

IMPORTANT: `N_LAYERS` below is a placeholder. Before running the baseline, execute
  python -m utils.count_flops
to determine the matching layer count against the current RDT config at
`max_loops_train=8`, then update `N_LAYERS` here. The FLOP comparison is on
total per-token forward FLOPs including embedding and head (both cancel).
"""

from configs.rdt_1_5b import RDT_1_5B_CONFIG
from open_mythos.baseline import dense_config_from_rdt

# Set via utils.count_flops.find_matching_layers(RDT_1_5B_CONFIG, target_loops=8).
# Last computed: 24 layers (RDT 3.37 TFLOPS vs dense 3.38 TFLOPS; ratio 1.004).
# Re-run `python -m utils.count_flops` if the RDT config changes.
N_LAYERS = 24

DENSE_BASELINE_CONFIG = dense_config_from_rdt(RDT_1_5B_CONFIG, n_layers=N_LAYERS)

MODEL_NAME = "dense_baseline"
