"""Axial-gradient template — 1D pathology with monotone + peaked genes."""

from __future__ import annotations

import numpy as np

from ._shared import GroundTruth, build_anndata, marker_expression


def build_gradient_dataset(
    n_cells: int = 3000,
    n_monotone_genes: int = 5,
    n_peaked_genes: int = 3,
    axis_length: float = 100.0,
    axis_jitter: float = 2.0,
    expression_steepness: float = 0.05,
    n_identity_markers: int = 5,
    scale_mean: float = 0.85,
    scale_std: float = 0.09,
    background_noise: float = 0.1,
    seed: int = 42,
):
    """Axial-gradient (1D pathology) dataset.

    Cells arranged along a 1D axis (with small lateral jitter for realism).
    Genes vary smoothly along the axis: some monotonically (sigmoid centered
    at the midpoint, increasing) and some with a peak in the middle
    (Gaussian centered at midpoint). Models tissue structures with a
    dominant axis (crypt-villus, neural-tube dorsoventral patterning).

    The 1D structure is the regime where ``graph_xcorr`` most clearly
    differs from ``mi``: each cell has 2 graph neighbors, axial position
    fully determines expression, and neighborhood smoothing recovers the
    gradient cleanly.

    Parameters
    ----------
    n_cells : int
    n_monotone_genes : int
        Genes that increase from axis position 0 to ``axis_length``
        (MONO_0..N-1). Sigmoid centered at ``axis_length / 2``.
    n_peaked_genes : int
        Genes with a Gaussian peak at the axis midpoint (PEAK_0..N-1).
    axis_length : float
        Length of the 1D axis.
    axis_jitter : float
        Lateral noise (std of normal) around the axis line.
    expression_steepness : float
        Sigmoid steepness for monotone genes; Gaussian std (relative to
        ``axis_length``) for peaked genes is ``axis_length / 8``.
    n_identity_markers : int
        Bulk identity markers expressed on every cell at marker scale
        (IDENT_0..N-1). Set to 0 for a "pure" gradient signal.
    scale_mean, scale_std, background_noise : float
        Expression-scale parameters.
    seed : int

    Returns
    -------
    adata : AnnData (single phenotype "Tissue")
    ground_truth : dict (v2.0 schema)
    """
    params = {
        "n_cells": int(n_cells),
        "n_monotone_genes": int(n_monotone_genes),
        "n_peaked_genes": int(n_peaked_genes),
        "axis_length": float(axis_length),
        "axis_jitter": float(axis_jitter),
        "expression_steepness": float(expression_steepness),
        "n_identity_markers": int(n_identity_markers),
        "scale_mean": float(scale_mean),
        "scale_std": float(scale_std),
        "background_noise": float(background_noise),
        "seed": int(seed),
    }
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.0, axis_length, size=n_cells)
    y = rng.normal(0.0, axis_jitter, size=n_cells)
    coords = np.column_stack([x, y]).astype(np.float64)

    midpoint = 0.5 * axis_length
    peak_sigma = axis_length / 8.0
    bg_sigma = background_noise * scale_mean

    mono_names = [f"MONO_{i}" for i in range(n_monotone_genes)]
    peak_names = [f"PEAK_{i}" for i in range(n_peaked_genes)]
    ident_names = [f"IDENT_{i}" for i in range(n_identity_markers)]
    gene_names = mono_names + peak_names + ident_names

    columns: list[np.ndarray] = []
    # Monotone genes: sigmoid in (x - midpoint).
    for _ in range(n_monotone_genes):
        intensity = 1.0 / (1.0 + np.exp(-expression_steepness * (x - midpoint)))
        col = scale_mean * intensity + rng.normal(0.0, bg_sigma, size=n_cells) \
              + rng.normal(0.0, scale_std, size=n_cells) * intensity
        columns.append(np.clip(col, 0.0, 1.0))
    # Peaked genes: Gaussian centered at midpoint.
    for _ in range(n_peaked_genes):
        intensity = np.exp(-0.5 * ((x - midpoint) / peak_sigma) ** 2)
        col = scale_mean * intensity + rng.normal(0.0, bg_sigma, size=n_cells) \
              + rng.normal(0.0, scale_std, size=n_cells) * intensity
        columns.append(np.clip(col, 0.0, 1.0))
    # Identity markers: uniform expression.
    all_mask = np.ones(n_cells, dtype=bool)
    for _ in range(n_identity_markers):
        columns.append(
            marker_expression(
                n_cells, all_mask,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    X = np.column_stack(columns) if columns else np.zeros((n_cells, 0))

    labels = np.full(n_cells, "Tissue", dtype=object)
    adata = build_anndata(X, gene_names, labels, coords)

    gt = GroundTruth(template="gradient", seed=int(seed), params=params)
    gt.add_phenotype("Tissue", int(n_cells), ident_names)

    for name in mono_names:
        gt.add_gene(
            name, "axial_gradient", phenotype="Tissue",
            shape="monotone", direction="+x",
            steepness=float(expression_steepness),
            midpoint=float(midpoint),
        )
    for name in peak_names:
        gt.add_gene(
            name, "axial_gradient", phenotype="Tissue",
            shape="peaked", peak_x=float(midpoint),
            peak_sigma=float(peak_sigma),
        )
    for name in ident_names:
        gt.add_gene(name, "identity_marker", phenotype="Tissue")

    # Pairs:
    # - Monotone-monotone: same direction → +1.
    for i in range(len(mono_names)):
        for j in range(i + 1, len(mono_names)):
            gt.add_pair(mono_names[i], mono_names[j], "axial_gradient",
                        correlation_sign=1)
    # - Peaked-peaked: same peak region → +1.
    for i in range(len(peak_names)):
        for j in range(i + 1, len(peak_names)):
            gt.add_pair(peak_names[i], peak_names[j], "axial_gradient",
                        correlation_sign=1)
    # - Monotone vs peaked: monotone integrates the half-axis past the peak
    #   → mean correlation around 0; mark as 0 sign for the metadata.
    for m in mono_names:
        for p in peak_names:
            gt.add_pair(m, p, "axial_gradient", correlation_sign=0)

    return adata, gt.to_json()
