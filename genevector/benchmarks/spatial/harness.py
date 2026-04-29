"""Spatial benchmark harness.

Trains GeneVector on the synthetic pathology FOV across a configurable set of
target variants (``mi``, ``pearson``, ``spearman``, ``graph_xcorr``,
``graph_xcorr_shuffled``) and writes per-run artifacts under
``output_root/seed_{seed}/``.

Artifact layout::

    {output_root}/seed_{seed}/
        adata.h5ad
        ground_truth.json
        graph_real.npz
        graph_shuffled.npz
        embedding_{variant}.vec        (+ embedding_{variant}2.vec)
        meta_{variant}.json
        all_runs.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import torch
from scipy import sparse

from genevector.benchmarks.synthetic import build_pathology
from genevector.data import GeneVectorDataset
from genevector.model import GeneVector

DEFAULT_TARGET_VARIANTS: tuple[str, ...] = (
    "mi",
    "pearson",
    "spearman",
    "graph_xcorr",
    "graph_xcorr_shuffled",
)

_VARIANT_TO_TARGET: dict[str, tuple[str, str]] = {
    "mi": ("mi", "none"),
    "pearson": ("pearson", "none"),
    "spearman": ("spearman", "none"),
    "graph_xcorr": ("graph_xcorr", "real"),
    "graph_xcorr_shuffled": ("graph_xcorr", "shuffled"),
}


@dataclass
class BenchmarkConfig:
    """Configuration for ``run_benchmark``.

    Parameters
    ----------
    target_variants : sequence of str
        Subset of :data:`DEFAULT_TARGET_VARIANTS` to run.
    seed : int
        Seed shared by the synthetic dataset and the embedding model.
    epochs : int
        Maximum training epochs per variant.
    threshold : float
        Loss-delta convergence threshold for ``GeneVector.train``.
    emb_dimension : int
        Embedding dimension.
    device : str
        Torch device for training.
    n_neighs : int
        Spatial graph k for the squidpy k-NN graph.
    layout_kwargs : dict, optional
        Overrides for :func:`build_pathology`'s layout step.
    overlay_kwargs : dict, optional
        Overrides for :func:`build_pathology`'s overlay step.
    output_root : str
        Root directory under which ``seed_{seed}/`` artifacts are written.
    log_interval : int
        Epoch interval for printing training loss.
    """

    target_variants: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_TARGET_VARIANTS)
    )
    seed: int = 42
    epochs: int = 3000
    threshold: float = 1e-6
    emb_dimension: int = 64
    device: str = "cpu"
    n_neighs: int = 6
    layout_kwargs: dict | None = None
    overlay_kwargs: dict | None = None
    output_root: str = "benchmarks_artifacts/spatial"
    log_interval: int = 100


def _resolve_variant(variant: str) -> tuple[str, str]:
    """Return ``(registered_target, graph_variant)`` for a benchmark variant."""
    try:
        return _VARIANT_TO_TARGET[variant]
    except KeyError as exc:
        valid = ", ".join(DEFAULT_TARGET_VARIANTS)
        raise ValueError(
            f"Unknown target variant: {variant!r}. Valid options: {valid}"
        ) from exc


def _build_spatial_graph(adata, n_neighs: int = 6) -> sparse.csr_matrix:
    """Symmetric k-NN spatial connectivity graph from ``adata.obsm['spatial']``."""
    import squidpy as sq

    sq.gr.spatial_neighbors(
        adata,
        spatial_key="spatial",
        coord_type="generic",
        n_neighs=n_neighs,
    )
    conn = adata.obsp["spatial_connectivities"]
    A = sparse.csr_matrix(conn).astype(np.float64)
    # squidpy's k-NN connectivities are not symmetric; take the union to
    # produce an undirected adjacency.
    A_sym = A.maximum(A.T)
    return sparse.csr_matrix(A_sym).astype(np.float64)


def _shuffle_graph_preserve_degree(
    graph: sparse.spmatrix,
    seed: int = 0,
    n_swap_factor: int = 10,
) -> sparse.csr_matrix:
    """Degree-preserving random shuffle via ``networkx.double_edge_swap``.

    Parameters
    ----------
    graph : scipy.sparse matrix
        Symmetric adjacency matrix (assumed undirected).
    seed : int
        RNG seed for reproducibility.
    n_swap_factor : int
        Approximate number of swap attempts per edge.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shuffled adjacency with the same per-node degree sequence.
    """
    G_sparse = sparse.csr_matrix(graph).astype(np.float64)
    n = G_sparse.shape[0]
    G = nx.from_scipy_sparse_array(G_sparse)
    G.remove_edges_from(nx.selfloop_edges(G))
    n_edges = G.number_of_edges()
    if n_edges < 2:
        return G_sparse
    n_swap = max(1, n_edges * n_swap_factor)
    try:
        nx.double_edge_swap(G, nswap=n_swap, max_tries=n_swap * 10, seed=seed)
    except (nx.NetworkXAlgorithmError, nx.NetworkXError):
        # Graph may be too constrained for swaps; return what we have.
        pass
    A = nx.to_scipy_sparse_array(
        G, nodelist=range(n), format="csr", weight=None, dtype=np.float64
    )
    return sparse.csr_matrix(A)


def _config_to_jsonable(config: BenchmarkConfig) -> dict:
    d = asdict(config)
    d["target_variants"] = list(d["target_variants"])
    return d


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run the spatial benchmark across ``config.target_variants``.

    Builds the synthetic pathology FOV, constructs real and shuffled spatial
    graphs, trains a GeneVector model per variant, and writes embeddings and
    metadata to disk.

    Parameters
    ----------
    config : BenchmarkConfig
        Run configuration.

    Returns
    -------
    dict
        Manifest with ``config`` and per-variant ``runs`` metadata. Also
        written to ``{output_root}/seed_{seed}/all_runs.json``.
    """
    output_root = Path(config.output_root)
    seed_dir = output_root / f"seed_{config.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    adata, ground_truth = build_pathology(
        layout_kwargs=config.layout_kwargs,
        overlay_kwargs=config.overlay_kwargs,
        seed=config.seed,
    )
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    adata.write_h5ad(seed_dir / "adata.h5ad")
    with open(seed_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2, sort_keys=True)

    graph_real = _build_spatial_graph(adata, n_neighs=config.n_neighs)
    sparse.save_npz(seed_dir / "graph_real.npz", graph_real)

    graph_shuffled = _shuffle_graph_preserve_degree(graph_real, seed=config.seed)
    sparse.save_npz(seed_dir / "graph_shuffled.npz", graph_shuffled)

    runs = []
    for variant in config.target_variants:
        registered_target, graph_variant = _resolve_variant(variant)
        target_kwargs: dict = {}
        if graph_variant == "real":
            target_kwargs["graph"] = graph_real
        elif graph_variant == "shuffled":
            target_kwargs["graph"] = graph_shuffled

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        run_adata = adata.copy()
        dataset = GeneVectorDataset(
            run_adata,
            target=registered_target,
            target_kwargs=target_kwargs,
            use_cache=False,
        )

        emb_path = seed_dir / f"embedding_{variant}.vec"
        model = GeneVector(
            dataset,
            output_file=str(emb_path),
            emb_dimension=config.emb_dimension,
            device=config.device,
        )

        t0 = time.time()
        model.train(
            config.epochs,
            threshold=config.threshold,
            update_interval=config.log_interval,
        )
        wall = time.time() - t0
        final_loss = (
            float(model.mean_loss_values[-1])
            if model.mean_loss_values
            else float("nan")
        )

        meta = {
            "target": registered_target,
            "target_variant": variant,
            "seed": config.seed,
            "graph_variant": graph_variant,
            "epochs_run": int(model.epoch),
            "final_loss": final_loss,
            "wall_time_seconds": wall,
            "n_genes": int(adata.n_vars),
            "n_cells": int(adata.n_obs),
        }
        with open(seed_dir / f"meta_{variant}.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        runs.append(meta)

    manifest = {"config": _config_to_jsonable(config), "runs": runs}
    with open(seed_dir / "all_runs.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest
