#!/usr/bin/env python
"""CLI for the GeneVector spatial benchmark.

Examples
--------
    # Full FOV1, all 5 target variants, default 3000 epochs.
    python scripts/run_spatial_benchmark.py --seed 42

    # Smoke run: tiny FOV, few epochs, single variant.
    python scripts/run_spatial_benchmark.py --quick

    # Subset of target variants.
    python scripts/run_spatial_benchmark.py \\
        --target-variants mi graph_xcorr graph_xcorr_shuffled
"""

from __future__ import annotations

import argparse
import json
import sys

from genevector.benchmarks.spatial import (
    BenchmarkConfig,
    DEFAULT_TARGET_VARIANTS,
    run_benchmark,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--threshold", type=float, default=1e-6)
    p.add_argument("--emb-dimension", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-neighs", type=int, default=6)
    p.add_argument(
        "--target-variants",
        nargs="+",
        default=list(DEFAULT_TARGET_VARIANTS),
        choices=list(DEFAULT_TARGET_VARIANTS),
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="benchmarks_artifacts/spatial",
    )
    p.add_argument(
        "--num-cells",
        type=int,
        default=None,
        help="Override layout num_cells (default: full FOV1).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke run: num_cells=300, epochs=20, threshold=1e-3.",
    )
    return p.parse_args(argv)


_DEFAULTS = {"epochs": 3000, "threshold": 1e-6, "num_cells": None}


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    layout_kwargs: dict | None = None
    epochs = args.epochs
    threshold = args.threshold
    num_cells = args.num_cells

    if args.quick:
        if num_cells == _DEFAULTS["num_cells"]:
            num_cells = 300
        if args.epochs == _DEFAULTS["epochs"]:
            epochs = 20
        if args.threshold == _DEFAULTS["threshold"]:
            threshold = 1e-3

    if num_cells is not None:
        layout_kwargs = {"num_cells": num_cells}

    return BenchmarkConfig(
        target_variants=args.target_variants,
        seed=args.seed,
        epochs=epochs,
        threshold=threshold,
        emb_dimension=args.emb_dimension,
        device=args.device,
        n_neighs=args.n_neighs,
        layout_kwargs=layout_kwargs,
        output_root=args.output_root,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _build_config(args)
    manifest = run_benchmark(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
