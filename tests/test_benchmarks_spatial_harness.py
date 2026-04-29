"""Tests for genevector.benchmarks.spatial.harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from genevector.benchmarks.spatial import (
    BenchmarkConfig,
    DEFAULT_TARGET_VARIANTS,
    run_benchmark,
)
from genevector.benchmarks.spatial.harness import (
    _build_spatial_graph,
    _resolve_variant,
    _shuffle_graph_preserve_degree,
)
from genevector.benchmarks.synthetic import (
    create_anndata_from_synthetic,
    generate_synthetic_data,
)

SEED = 42


def test_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert list(cfg.target_variants) == list(DEFAULT_TARGET_VARIANTS)
    assert cfg.seed == 42
    assert cfg.epochs == 3000
    assert cfg.emb_dimension == 64
    assert cfg.device == "cpu"
    assert cfg.output_root == "benchmarks_artifacts/spatial"


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("mi", ("mi", "none")),
        ("pearson", ("pearson", "none")),
        ("spearman", ("spearman", "none")),
        ("graph_xcorr", ("graph_xcorr", "real")),
        ("graph_xcorr_shuffled", ("graph_xcorr", "shuffled")),
    ],
)
def test_resolve_variant(variant, expected):
    assert _resolve_variant(variant) == expected


def test_resolve_variant_unknown_raises():
    with pytest.raises(ValueError, match="Unknown target variant"):
        _resolve_variant("not_a_variant")


def test_shuffle_graph_preserves_degree_sequence():
    df, _ = generate_synthetic_data(num_cells=400, seed=SEED)
    adata = create_anndata_from_synthetic(df)
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    real = _build_spatial_graph(adata, n_neighs=6)

    shuffled = _shuffle_graph_preserve_degree(real, seed=SEED)

    real_deg = np.asarray(real.sum(axis=1)).ravel()
    shuf_deg = np.asarray(shuffled.sum(axis=1)).ravel()
    assert np.array_equal(real_deg, shuf_deg)
    # Same number of nonzeros.
    assert real.nnz == shuffled.nnz
    # And at least some edges differ — degree-preserving shuffle should
    # reroute connectivity, not return the same matrix.
    assert (real != shuffled).nnz > 0


def test_build_spatial_graph_is_symmetric_csr():
    df, _ = generate_synthetic_data(num_cells=300, seed=SEED)
    adata = create_anndata_from_synthetic(df)
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    G = _build_spatial_graph(adata, n_neighs=6)
    assert sparse.issparse(G)
    assert G.shape == (adata.n_obs, adata.n_obs)
    # Spatial connectivities are symmetric.
    diff = G - G.T
    assert diff.nnz == 0 or np.allclose(diff.toarray(), 0.0)


def test_run_benchmark_quick(tmp_path: Path):
    cfg = BenchmarkConfig(
        target_variants=list(DEFAULT_TARGET_VARIANTS),
        seed=SEED,
        epochs=5,
        threshold=1e-2,
        emb_dimension=8,
        device="cpu",
        n_neighs=6,
        layout_kwargs={"num_cells": 300},
        output_root=str(tmp_path / "spatial"),
        log_interval=100,
    )
    manifest = run_benchmark(cfg)

    seed_dir = tmp_path / "spatial" / f"seed_{SEED}"
    for fname in (
        "adata.h5ad",
        "ground_truth.json",
        "graph_real.npz",
        "graph_shuffled.npz",
        "all_runs.json",
    ):
        assert (seed_dir / fname).exists(), f"missing {fname}"

    for variant in DEFAULT_TARGET_VARIANTS:
        assert (seed_dir / f"embedding_{variant}.vec").exists()
        assert (seed_dir / f"embedding_{variant}2.vec").exists()
        meta_path = seed_dir / f"meta_{variant}.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        for key in (
            "target",
            "target_variant",
            "seed",
            "graph_variant",
            "epochs_run",
            "final_loss",
            "wall_time_seconds",
            "n_genes",
            "n_cells",
        ):
            assert key in meta, f"{variant} missing meta key {key}"
        assert meta["target_variant"] == variant
        assert meta["seed"] == SEED

    # Manifest sanity.
    assert {r["target_variant"] for r in manifest["runs"]} == set(DEFAULT_TARGET_VARIANTS)
    assert manifest["config"]["seed"] == SEED


@pytest.mark.slow
def test_run_benchmark_full(tmp_path: Path):
    cfg = BenchmarkConfig(
        target_variants=["mi", "graph_xcorr", "graph_xcorr_shuffled"],
        seed=SEED,
        epochs=50,
        threshold=1e-3,
        emb_dimension=32,
        device="cpu",
        n_neighs=6,
        layout_kwargs={"num_cells": 1500},
        output_root=str(tmp_path / "spatial"),
        log_interval=200,
    )
    manifest = run_benchmark(cfg)

    seed_dir = tmp_path / "spatial" / f"seed_{SEED}"
    assert (seed_dir / "all_runs.json").exists()

    real = sparse.load_npz(seed_dir / "graph_real.npz")
    shuf = sparse.load_npz(seed_dir / "graph_shuffled.npz")
    assert real.shape == shuf.shape
    assert np.array_equal(
        np.asarray(real.sum(axis=1)).ravel(),
        np.asarray(shuf.sum(axis=1)).ravel(),
    )

    targets_in_manifest = {r["target_variant"] for r in manifest["runs"]}
    assert targets_in_manifest == {"mi", "graph_xcorr", "graph_xcorr_shuffled"}
