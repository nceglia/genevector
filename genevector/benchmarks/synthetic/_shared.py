"""Shared dataclass + helpers for synthetic dataset templates (v2.0 schema)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


GROUND_TRUTH_VERSION = "2.0"


@dataclass
class GroundTruth:
    """Unified ground-truth record returned by every template builder."""

    template: str
    seed: int
    params: dict
    phenotypes: dict[str, dict] = field(default_factory=dict)
    genes: dict[str, dict] = field(default_factory=dict)
    pairs: list[dict] = field(default_factory=list)
    version: str = GROUND_TRUTH_VERSION

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=_jsonable))

    def add_phenotype(self, name: str, n_cells: int, markers: list[str]):
        self.phenotypes[name] = {"n_cells": int(n_cells), "markers": list(markers)}

    def add_gene(
        self,
        name: str,
        role: str,
        phenotype: str | None = None,
        **metadata,
    ):
        self.genes[name.upper()] = {
            "role": role,
            "phenotype": phenotype,
            "metadata": metadata,
        }

    def add_pair(self, gene_a: str, gene_b: str, kind: str, **metadata):
        self.pairs.append(
            {
                "gene_a": gene_a.upper(),
                "gene_b": gene_b.upper(),
                "kind": kind,
                "metadata": metadata,
            }
        )


def _jsonable(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# ─── Geometry helpers ────────────────────────────────────────


def place_cells_uniform(
    n_cells: int,
    bounds: tuple[float, float] = (0.0, 100.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random placement in a 2D box. Returns ``(n_cells, 2)``."""
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(bounds[0], bounds[1], size=(n_cells, 2))


def place_cells_blobs(
    counts_per_phenotype: dict[str, int],
    centers: dict[str, tuple[float, float]],
    std: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs centered at given coordinates.

    Returns
    -------
    coords : (n, 2) array
    labels : (n,) array of phenotype names, in the order rows appear in ``coords``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    chunks_coords = []
    chunks_labels = []
    for name, n in counts_per_phenotype.items():
        if n <= 0:
            continue
        cx, cy = centers[name]
        chunks_coords.append(rng.normal(loc=(cx, cy), scale=std, size=(n, 2)))
        chunks_labels.append(np.full(n, name, dtype=object))
    if not chunks_coords:
        return np.zeros((0, 2), dtype=np.float64), np.array([], dtype=object)
    coords = np.vstack(chunks_coords).astype(np.float64)
    labels = np.concatenate(chunks_labels)
    return coords, labels


def place_cells_intermixed(
    counts_per_phenotype: dict[str, int],
    mixing: float,
    bounds: tuple[float, float] = (0.0, 100.0),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-phenotype intermixing controlled by ``mixing`` ∈ [0, 1].

    ``mixing=0`` → fully segregated (left half / right half along x).
    ``mixing=1`` → uniformly mixed.
    Linearly interpolates by mixing each cell's segregated x with a fresh
    uniform-x draw, weighted by ``mixing``.
    """
    if not 0.0 <= mixing <= 1.0:
        raise ValueError(f"mixing must be in [0, 1], got {mixing!r}")
    if len(counts_per_phenotype) != 2:
        raise ValueError(
            f"place_cells_intermixed expects exactly 2 phenotypes, "
            f"got {len(counts_per_phenotype)}"
        )

    rng = rng if rng is not None else np.random.default_rng()
    lo, hi = bounds
    mid = 0.5 * (lo + hi)

    pieces_coords = []
    pieces_labels = []
    for idx, (name, n) in enumerate(counts_per_phenotype.items()):
        if n <= 0:
            continue
        # Segregated x range: phenotype 0 → [lo, mid]; phenotype 1 → [mid, hi].
        seg_lo = lo if idx == 0 else mid
        seg_hi = mid if idx == 0 else hi
        x_seg = rng.uniform(seg_lo, seg_hi, size=n)
        x_uniform = rng.uniform(lo, hi, size=n)
        x = (1.0 - mixing) * x_seg + mixing * x_uniform
        y = rng.uniform(lo, hi, size=n)
        pieces_coords.append(np.column_stack([x, y]))
        pieces_labels.append(np.full(n, name, dtype=object))

    coords = np.vstack(pieces_coords).astype(np.float64)
    labels = np.concatenate(pieces_labels)
    return coords, labels


# ─── Expression helpers ──────────────────────────────────────


def marker_expression(
    n_cells: int,
    mask: np.ndarray,
    scale_mean: float = 0.85,
    scale_std: float = 0.09,
    background_noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """High Gaussian draws where ``mask=True``, low background elsewhere.

    Output is clipped to [0, 1].
    """
    rng = rng if rng is not None else np.random.default_rng()
    bg_sigma = background_noise * scale_mean
    out = rng.normal(0.0, bg_sigma, size=n_cells)
    n_high = int(mask.sum())
    if n_high:
        out[mask] = rng.normal(scale_mean, scale_std, size=n_high)
    return np.clip(out, 0.0, 1.0)


def stochastic_marker_expression(
    n_cells: int,
    mask: np.ndarray,
    expression_density: float = 0.5,
    scale_mean: float = 0.85,
    scale_std: float = 0.09,
    background_noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Like ``marker_expression`` but only ``expression_density`` of mask=True
    cells actually express the marker (Bernoulli draw)."""
    if not 0.0 <= expression_density <= 1.0:
        raise ValueError(
            f"expression_density must be in [0, 1], got {expression_density!r}"
        )
    rng = rng if rng is not None else np.random.default_rng()
    drawn = mask & (rng.random(n_cells) < expression_density)
    return marker_expression(
        n_cells, drawn, scale_mean, scale_std, background_noise, rng
    )


# ─── Construction helper ─────────────────────────────────────


def build_anndata(
    X_dense: np.ndarray,
    gene_names: list[str],
    phenotype_labels: np.ndarray,
    coords: np.ndarray,
) -> ad.AnnData:
    """Build an AnnData satisfying the template contract:

    - ``adata.X`` is ``scipy.sparse.csr_matrix`` of float counts
    - ``adata.obs['phenotype']`` is a categorical
    - ``adata.obsm['spatial']`` is float64 ``(n_cells, 2)``
    - ``adata.var_names`` are uppercase, unique
    """
    upper = [g.upper() for g in gene_names]
    if len(set(upper)) != len(upper):
        raise ValueError("gene_names must be unique (case-insensitively)")

    n_cells = X_dense.shape[0]
    obs = pd.DataFrame(
        {"phenotype": pd.Categorical(np.asarray(phenotype_labels))},
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(index=pd.Index(upper, name=None))

    adata = ad.AnnData(
        X=sparse.csr_matrix(X_dense.astype(float)),
        obs=obs,
        var=var,
    )
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    return adata
