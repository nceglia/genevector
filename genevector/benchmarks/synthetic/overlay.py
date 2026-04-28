"""
Diagnostic overlay for synthetic spatial AnnData.

Extends ``var_names`` with four classes of diagnostic genes:

1. **Paracrine L/R pairs** (``LIG_*`` / ``REC_*``) — ligands expressed only on
   ``Tumor`` cells, receptors only on ``T cell`` cells. Drawn independently so
   within-cell mutual information between any (LIG_i, REC_i) pair is small.
2. **Niche genes** (``NICHE_*``) — Bernoulli-triggered on T cells with
   probability ``sigmoid(k * (tumor_neighbor_frac - threshold))`` inside a
   neighborhood radius. Triggered draws are at marker scale; otherwise
   background noise.
3. **T-rare distinguishing genes** (``TRARE_*``) — high on the ``T_rare``
   subtype only (a *small* within-cell signal so ``mi`` can partially recover
   the subtype).
4. **Housekeeping** (``HK_*``) — uniform marker-scale expression across all
   phenotypes; negative controls.

T cells are split into ``T_stromal`` / ``T_intratumoral`` / ``T_rare`` by local
tumor density. The original ``phenotype`` is preserved in
``obs["phenotype_coarse"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata
from scipy.spatial import cKDTree


_GROUND_TRUTH_VERSION = "1.0"


def apply_overlay(
    adata,
    n_paracrine: int = 5,
    n_niche: int = 5,
    n_trare: int = 3,
    n_housekeeping: int = 5,
    niche_k: float = 10.0,
    niche_threshold: float = 0.3,
    neighbor_radius: float = 1.5,
    t_intratumoral_threshold: float = 0.3,
    t_rare_threshold: float = 0.7,
    marker_scale_mean: float | None = None,
    marker_scale_std: float | None = None,
    background_noise: float = 0.1,
    seed: int = 42,
):
    """
    Apply diagnostic overlay to a synthetic spatial AnnData.

    Parameters
    ----------
    adata : AnnData
        Synthetic AnnData with ``obs["phenotype"]``, ``obs["cell_type"]``,
        and ``obsm["spatial"]`` (as produced by
        ``create_anndata_from_synthetic``).
    n_paracrine, n_niche, n_trare, n_housekeeping : int
        Number of genes to add in each diagnostic class.
    niche_k, niche_threshold : float
        Sigmoid steepness and threshold for niche-gene trigger probability.
    neighbor_radius : float
        Radius for the local-tumor-density query.
    t_intratumoral_threshold, t_rare_threshold : float
        Tumor-neighbor-fraction cut-points splitting T cells into
        ``T_stromal`` / ``T_intratumoral`` / ``T_rare``.
    marker_scale_mean, marker_scale_std : float, optional
        Mean / std of "expressed" marker draws. If ``None``, inferred from
        existing marker columns (values above 0.5).
    background_noise : float
        Background expression std as a fraction of ``marker_scale_mean``.
    seed : int
        Seed for the overlay RNG.

    Returns
    -------
    adata : AnnData
        New AnnData with extended ``var_names`` and updated ``phenotype``.
    ground_truth : dict
        JSON-serializable record of all overlay decisions.
    """
    rng = np.random.default_rng(seed)

    if marker_scale_mean is None or marker_scale_std is None:
        inferred_mean, inferred_std = _infer_marker_scale(adata)
        if marker_scale_mean is None:
            marker_scale_mean = inferred_mean
        if marker_scale_std is None:
            marker_scale_std = inferred_std

    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    tumor_mask = cell_type == "Tumor"
    t_mask = cell_type == "T cell"

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    tumor_frac = _compute_neighbor_phenotype_fraction(
        coords, tumor_mask, neighbor_radius
    )

    phenotype_coarse = adata.obs["phenotype"].astype(str).copy()
    new_phenotype, t_subtype_counts = _split_t_subtypes(
        adata.obs["phenotype"].astype(str).to_numpy(),
        t_mask,
        tumor_frac,
        t_intratumoral_threshold,
        t_rare_threshold,
    )

    trare_mask = new_phenotype == "T_rare"

    LIG, REC, paracrine_meta = _add_paracrine_genes(
        n_paracrine,
        tumor_mask,
        t_mask,
        marker_scale_mean,
        marker_scale_std,
        background_noise,
        rng,
    )
    NICHE, niche_meta = _add_niche_genes(
        n_niche,
        t_mask,
        tumor_frac,
        niche_k,
        niche_threshold,
        neighbor_radius,
        marker_scale_mean,
        marker_scale_std,
        background_noise,
        rng,
    )
    TRARE, trare_names = _add_trare_genes(
        n_trare,
        trare_mask,
        t_mask,
        marker_scale_mean,
        marker_scale_std,
        background_noise,
        rng,
    )
    HK, hk_names = _add_housekeeping_genes(
        n_housekeeping,
        adata.n_obs,
        marker_scale_mean,
        marker_scale_std,
        rng,
    )

    lig_names = [pair["ligand"] for pair in paracrine_meta]
    rec_names = [pair["receptor"] for pair in paracrine_meta]
    niche_names = [item["gene"] for item in niche_meta]
    added_gene_names = lig_names + rec_names + niche_names + trare_names + hk_names

    X_existing = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_added = np.hstack([LIG, REC, NICHE, TRARE, HK])
    X_new = np.hstack([X_existing, X_added])

    new_obs = adata.obs.copy()
    new_obs["phenotype"] = new_phenotype
    new_obs["phenotype_coarse"] = phenotype_coarse.to_numpy()

    new_var_index = list(adata.var_names) + added_gene_names
    new_var = pd.DataFrame(index=pd.Index(new_var_index, name=adata.var.index.name))
    for col in adata.var.columns:
        existing = adata.var[col].tolist()
        new_var[col] = existing + [None] * len(added_gene_names)

    new_adata = anndata.AnnData(
        X=X_new,
        obs=new_obs,
        var=new_var,
        obsm={k: np.asarray(v).copy() for k, v in adata.obsm.items()},
        uns={k: v for k, v in adata.uns.items()},
    )

    ground_truth = {
        "version": _GROUND_TRUTH_VERSION,
        "seed": int(seed),
        "paracrine_pairs": paracrine_meta,
        "niche_genes": niche_meta,
        "trare_genes": trare_names,
        "housekeeping_genes": hk_names,
        "t_subtypes": {
            "T_stromal": {
                "n_cells": int(t_subtype_counts["T_stromal"]),
                "tumor_frac_range": [0.0, float(t_intratumoral_threshold)],
            },
            "T_intratumoral": {
                "n_cells": int(t_subtype_counts["T_intratumoral"]),
                "tumor_frac_range": [
                    float(t_intratumoral_threshold),
                    float(t_rare_threshold),
                ],
            },
            "T_rare": {
                "n_cells": int(t_subtype_counts["T_rare"]),
                "tumor_frac_range": [float(t_rare_threshold), 1.0],
            },
        },
        "added_gene_names": added_gene_names,
    }

    return new_adata, ground_truth


def _infer_marker_scale(adata, threshold: float = 0.5) -> tuple[float, float]:
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    high = X[X > threshold]
    if high.size == 0:
        return 0.85, 0.1
    return float(high.mean()), float(high.std())


def _compute_neighbor_phenotype_fraction(
    coords: np.ndarray, target_mask: np.ndarray, radius: float
) -> np.ndarray:
    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=radius)
    n = coords.shape[0]
    frac = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbrs = [j for j in neighbor_lists[i] if j != i]
        if not nbrs:
            continue
        frac[i] = float(np.sum(target_mask[nbrs])) / len(nbrs)
    return frac


def _split_t_subtypes(
    phenotype: np.ndarray,
    t_mask: np.ndarray,
    tumor_frac: np.ndarray,
    intratumoral_threshold: float,
    rare_threshold: float,
):
    new_phenotype = phenotype.copy().astype(object)
    counts = {"T_stromal": 0, "T_intratumoral": 0, "T_rare": 0}
    if not t_mask.any():
        return new_phenotype, counts

    t_idx = np.where(t_mask)[0]
    for i in t_idx:
        f = tumor_frac[i]
        if f < intratumoral_threshold:
            label = "T_stromal"
        elif f < rare_threshold:
            label = "T_intratumoral"
        else:
            label = "T_rare"
        new_phenotype[i] = label
        counts[label] += 1
    return new_phenotype, counts


def _add_paracrine_genes(
    n: int,
    ligand_mask: np.ndarray,
    receptor_mask: np.ndarray,
    scale_mean: float,
    scale_std: float,
    bg_noise: float,
    rng: np.random.Generator,
):
    n_cells = ligand_mask.shape[0]
    bg_sigma = bg_noise * scale_mean

    LIG = rng.normal(0.0, bg_sigma, size=(n_cells, n))
    REC = rng.normal(0.0, bg_sigma, size=(n_cells, n))
    n_lig_high = int(ligand_mask.sum())
    n_rec_high = int(receptor_mask.sum())
    if n_lig_high:
        LIG[ligand_mask] = rng.normal(scale_mean, scale_std, size=(n_lig_high, n))
    if n_rec_high:
        REC[receptor_mask] = rng.normal(scale_mean, scale_std, size=(n_rec_high, n))
    LIG = np.clip(LIG, 0.0, 1.0)
    REC = np.clip(REC, 0.0, 1.0)

    meta = [
        {
            "ligand": f"LIG_{i}",
            "receptor": f"REC_{i}",
            "ligand_phenotype": "Tumor",
            "receptor_phenotype": "T_*",
        }
        for i in range(n)
    ]
    return LIG, REC, meta


def _add_niche_genes(
    n: int,
    target_mask: np.ndarray,
    tumor_frac: np.ndarray,
    k: float,
    threshold: float,
    radius: float,
    scale_mean: float,
    scale_std: float,
    bg_noise: float,
    rng: np.random.Generator,
):
    n_cells = target_mask.shape[0]
    bg_sigma = bg_noise * scale_mean

    NICHE = rng.normal(0.0, bg_sigma, size=(n_cells, n))
    p = 1.0 / (1.0 + np.exp(-k * (tumor_frac - threshold)))
    trigger_probs = np.where(target_mask[:, None], p[:, None], 0.0)
    triggered = rng.random(size=(n_cells, n)) < trigger_probs
    n_triggered = int(triggered.sum())
    if n_triggered:
        NICHE[triggered] = rng.normal(scale_mean, scale_std, size=n_triggered)
    NICHE = np.clip(NICHE, 0.0, 1.0)

    meta = [
        {
            "gene": f"NICHE_{i}",
            "expressed_in": "T_*",
            "modulator_phenotype": "Tumor",
            "k": float(k),
            "threshold": float(threshold),
            "radius": float(radius),
        }
        for i in range(n)
    ]
    return NICHE, meta


def _add_trare_genes(
    n: int,
    trare_mask: np.ndarray,
    t_mask: np.ndarray,
    scale_mean: float,
    scale_std: float,
    bg_noise: float,
    rng: np.random.Generator,
):
    n_cells = trare_mask.shape[0]
    bg_sigma = bg_noise * scale_mean
    other_t_sigma = 0.2 * scale_mean

    TRARE = rng.normal(0.0, bg_sigma, size=(n_cells, n))
    other_t_mask = t_mask & (~trare_mask)
    n_other_t = int(other_t_mask.sum())
    if n_other_t:
        TRARE[other_t_mask] = rng.normal(0.0, other_t_sigma, size=(n_other_t, n))
    n_rare = int(trare_mask.sum())
    if n_rare:
        TRARE[trare_mask] = rng.normal(scale_mean, scale_std, size=(n_rare, n))
    TRARE = np.clip(TRARE, 0.0, 1.0)

    names = [f"TRARE_{i}" for i in range(n)]
    return TRARE, names


def _add_housekeeping_genes(
    n: int,
    n_cells: int,
    scale_mean: float,
    scale_std: float,
    rng: np.random.Generator,
):
    HK = rng.normal(scale_mean, scale_std, size=(n_cells, n))
    HK = np.clip(HK, 0.0, 1.0)
    names = [f"HK_{i}" for i in range(n)]
    return HK, names
