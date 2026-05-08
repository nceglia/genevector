"""Niche-induction template — tumor blob + density-induced niche genes on T cells."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ._shared import (
    GroundTruth,
    build_anndata,
    marker_expression,
    place_cells_blobs,
    place_cells_uniform,
)


def build_niche_dataset(
    n_niche_genes: int = 5,
    n_tumor_cells: int = 2000,
    n_t_cells: int = 1500,
    tumor_blob_std: float = 15.0,
    niche_radius: float = 8.0,
    niche_threshold: float = 0.3,
    niche_k: float = 10.0,
    n_identity_markers: int = 10,
    bounds: tuple[float, float] = (0.0, 100.0),
    scale_mean: float = 0.85,
    scale_std: float = 0.09,
    background_noise: float = 0.1,
    seed: int = 42,
):
    """Niche-induction dataset.

    Three phenotypes: Tumor cells in a central blob, T cells distributed
    around it. T cells whose neighborhood (within ``niche_radius``) has tumor
    fraction above ``niche_threshold`` express niche genes; T cells far from
    the tumor do not. The split produces ``T_near`` and ``T_far`` phenotypes
    that share T-cell identity markers — the only distinguishing signal is
    niche-gene activation.

    Models the niche subset of ``build_pathology`` cleanly, without LIG/REC,
    TRARE, or housekeeping clutter. ``graph_xcorr`` should recover NICHE_i
    correlation with tumor markers; ``mi`` / ``pearson`` cannot.

    Parameters
    ----------
    n_niche_genes : int
        Number of niche genes (NICHE_0..N-1).
    n_tumor_cells, n_t_cells : int
        Cell counts.
    tumor_blob_std : float
        Std of Gaussian blob for tumor placement.
    niche_radius, niche_threshold, niche_k : float
        Niche induction parameters: T cell expresses niche genes with
        ``prob = sigmoid(k * (tumor_frac_in_radius - threshold))``.
    n_identity_markers : int
        Identity markers per cell type (TUMOR_M0..., TCELL_M0...).
    bounds : tuple
        2D coordinate bounds.
    scale_mean, scale_std, background_noise : float
        Expression-scale parameters.
    seed : int

    Returns
    -------
    adata : AnnData
    ground_truth : dict (v2.0 schema)
    """
    params = {
        "n_niche_genes": int(n_niche_genes),
        "n_tumor_cells": int(n_tumor_cells),
        "n_t_cells": int(n_t_cells),
        "tumor_blob_std": float(tumor_blob_std),
        "niche_radius": float(niche_radius),
        "niche_threshold": float(niche_threshold),
        "niche_k": float(niche_k),
        "n_identity_markers": int(n_identity_markers),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "scale_mean": float(scale_mean),
        "scale_std": float(scale_std),
        "background_noise": float(background_noise),
        "seed": int(seed),
    }
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    center = (0.5 * (lo + hi), 0.5 * (lo + hi))

    # Tumor cells: central Gaussian blob.
    tumor_coords, _ = place_cells_blobs(
        {"Tumor": int(n_tumor_cells)},
        {"Tumor": center},
        std=tumor_blob_std,
        rng=rng,
    )
    # T cells: uniformly distributed across the FOV.
    t_coords = place_cells_uniform(int(n_t_cells), bounds, rng)

    coords = np.vstack([tumor_coords, t_coords]).astype(np.float64)
    n_cells = coords.shape[0]
    tumor_mask = np.zeros(n_cells, dtype=bool)
    tumor_mask[: int(n_tumor_cells)] = True
    t_mask = ~tumor_mask

    # Local tumor density per cell (fraction of within-radius neighbors that
    # are Tumor). Used both for the T_near/T_far split and for niche-gene
    # induction probability.
    tree = cKDTree(coords)
    nbr_lists = tree.query_ball_point(coords, r=float(niche_radius))
    tumor_frac = np.zeros(n_cells, dtype=np.float64)
    for i, nbrs in enumerate(nbr_lists):
        nbrs_no_self = [j for j in nbrs if j != i]
        if nbrs_no_self:
            tumor_frac[i] = float(np.sum(tumor_mask[nbrs_no_self])) / len(nbrs_no_self)

    near_mask = t_mask & (tumor_frac >= niche_threshold)
    far_mask = t_mask & (tumor_frac < niche_threshold)

    labels = np.empty(n_cells, dtype=object)
    labels[tumor_mask] = "Tumor"
    labels[near_mask] = "T_near"
    labels[far_mask] = "T_far"

    # Genes
    tumor_marker_names = [f"TUMOR_M{i}" for i in range(n_identity_markers)]
    tcell_marker_names = [f"TCELL_M{i}" for i in range(n_identity_markers)]
    niche_names = [f"NICHE_{i}" for i in range(n_niche_genes)]
    gene_names = tumor_marker_names + tcell_marker_names + niche_names

    columns: list[np.ndarray] = []
    for _ in range(n_identity_markers):
        columns.append(
            marker_expression(
                n_cells, tumor_mask,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    for _ in range(n_identity_markers):
        columns.append(
            marker_expression(
                n_cells, t_mask,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    # Niche-gene induction: sigmoid in (tumor_frac - threshold) on T cells.
    bg_sigma = background_noise * scale_mean
    p_trigger = 1.0 / (1.0 + np.exp(-niche_k * (tumor_frac - niche_threshold)))
    trigger_probs = np.where(t_mask, p_trigger, 0.0)
    for _ in range(n_niche_genes):
        col = rng.normal(0.0, bg_sigma, size=n_cells)
        triggered = rng.random(n_cells) < trigger_probs
        n_trig = int(triggered.sum())
        if n_trig:
            col[triggered] = rng.normal(scale_mean, scale_std, size=n_trig)
        columns.append(np.clip(col, 0.0, 1.0))
    X = np.column_stack(columns) if columns else np.zeros((n_cells, 0))

    adata = build_anndata(X, gene_names, labels.astype(str), coords)

    gt = GroundTruth(template="niche", seed=int(seed), params=params)
    gt.add_phenotype("Tumor", int(tumor_mask.sum()), tumor_marker_names)
    gt.add_phenotype("T_near", int(near_mask.sum()), tcell_marker_names + niche_names)
    gt.add_phenotype("T_far", int(far_mask.sum()), tcell_marker_names)

    for name in tumor_marker_names:
        gt.add_gene(name, "identity_marker", phenotype="Tumor")
    for name in tcell_marker_names:
        gt.add_gene(name, "identity_marker", phenotype="T_cell")
    for name in niche_names:
        gt.add_gene(
            name, "niche_gene", phenotype="T_near",
            niche_radius=float(niche_radius),
            niche_threshold=float(niche_threshold),
            niche_k=float(niche_k),
            modulator_phenotype="Tumor",
        )

    for niche in niche_names:
        for tm in tumor_marker_names:
            gt.add_pair(
                niche, tm, "niche_induction",
                modulator_phenotype="Tumor",
                expressed_phenotype="T_near",
                niche_radius=float(niche_radius),
            )
    # Within-cell shared programs.
    for i in range(len(tumor_marker_names)):
        for j in range(i + 1, len(tumor_marker_names)):
            gt.add_pair(tumor_marker_names[i], tumor_marker_names[j],
                        "shared_program", phenotype="Tumor")
    for i in range(len(tcell_marker_names)):
        for j in range(i + 1, len(tcell_marker_names)):
            gt.add_pair(tcell_marker_names[i], tcell_marker_names[j],
                        "shared_program", phenotype="T_cell")

    return adata, gt.to_json()
