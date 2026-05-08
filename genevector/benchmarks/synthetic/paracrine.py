"""Paracrine signaling template — two intermixed cell types with L/R pairs."""

from __future__ import annotations

import numpy as np

from ._shared import (
    GroundTruth,
    build_anndata,
    marker_expression,
    place_cells_intermixed,
    stochastic_marker_expression,
)


def build_paracrine_dataset(
    n_pairs: int = 5,
    n_cells_per_type: int = 2000,
    mixing: float = 0.7,
    expression_density: float = 0.3,
    n_identity_markers: int = 10,
    bounds: tuple[float, float] = (0.0, 100.0),
    scale_mean: float = 0.85,
    scale_std: float = 0.09,
    background_noise: float = 0.1,
    seed: int = 42,
):
    """Two-cell-type paracrine signaling dataset.

    Generates two intermixed cell populations ("Source" and "Target"). Each
    pair consists of a ligand expressed on Source cells and a receptor
    expressed on Target cells, never co-expressed within the same cell.
    Spatial intermixing means Source and Target cells are frequently
    adjacent, so neighborhood-aware methods (graph_xcorr) can detect the
    coupling that within-cell methods miss.

    Identity markers per cell type create a competing within-cell signal —
    the realistic case where paracrine must be recovered against a strong
    cell-type identity backdrop.

    Parameters
    ----------
    n_pairs : int
        Number of independent ligand-receptor pairs.
    n_cells_per_type : int
        Cell count per phenotype.
    mixing : float in [0, 1]
        0 = fully segregated populations, 1 = uniformly mixed.
    expression_density : float in [0, 1]
        Fraction of cells of the relevant type that express each LIG/REC.
    n_identity_markers : int
        Identity markers per cell type. Set to 0 for a "pure" paracrine
        signal with no competing within-cell structure.
    bounds : tuple
        2D coordinate bounds.
    scale_mean, scale_std : float
        Mean and std of expressed-marker draws.
    background_noise : float
        Background-noise std as a fraction of ``scale_mean``.
    seed : int
        RNG seed.

    Returns
    -------
    adata : AnnData
    ground_truth : dict (v2.0 schema; LIG/REC pairs have kind="paracrine")
    """
    params = {
        "n_pairs": int(n_pairs),
        "n_cells_per_type": int(n_cells_per_type),
        "mixing": float(mixing),
        "expression_density": float(expression_density),
        "n_identity_markers": int(n_identity_markers),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "scale_mean": float(scale_mean),
        "scale_std": float(scale_std),
        "background_noise": float(background_noise),
        "seed": int(seed),
    }
    rng = np.random.default_rng(seed)

    # Layout: two intermixed phenotypes.
    counts = {"Source": int(n_cells_per_type), "Target": int(n_cells_per_type)}
    coords, labels = place_cells_intermixed(counts, mixing, bounds, rng)
    source_mask = labels == "Source"
    target_mask = labels == "Target"
    n_cells = coords.shape[0]

    lig_names = [f"LIG_{i}" for i in range(n_pairs)]
    rec_names = [f"REC_{i}" for i in range(n_pairs)]
    src_marker_names = [f"MARKER_S{i}" for i in range(n_identity_markers)]
    tgt_marker_names = [f"MARKER_T{i}" for i in range(n_identity_markers)]
    gene_names = lig_names + rec_names + src_marker_names + tgt_marker_names

    columns: list[np.ndarray] = []
    for _ in range(n_pairs):
        columns.append(
            stochastic_marker_expression(
                n_cells, source_mask, expression_density,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    for _ in range(n_pairs):
        columns.append(
            stochastic_marker_expression(
                n_cells, target_mask, expression_density,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    for _ in range(n_identity_markers):
        columns.append(
            marker_expression(
                n_cells, source_mask,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    for _ in range(n_identity_markers):
        columns.append(
            marker_expression(
                n_cells, target_mask,
                scale_mean, scale_std, background_noise, rng,
            )
        )
    X = np.column_stack(columns) if columns else np.zeros((n_cells, 0))

    adata = build_anndata(X, gene_names, labels.astype(str), coords)

    gt = GroundTruth(template="paracrine", seed=int(seed), params=params)
    gt.add_phenotype("Source", int(source_mask.sum()), src_marker_names + lig_names)
    gt.add_phenotype("Target", int(target_mask.sum()), tgt_marker_names + rec_names)

    for name in lig_names:
        gt.add_gene(name, "ligand", phenotype="Source",
                    expression_density=float(expression_density))
    for name in rec_names:
        gt.add_gene(name, "receptor", phenotype="Target",
                    expression_density=float(expression_density))
    for name in src_marker_names:
        gt.add_gene(name, "identity_marker", phenotype="Source")
    for name in tgt_marker_names:
        gt.add_gene(name, "identity_marker", phenotype="Target")

    for lig, rec in zip(lig_names, rec_names):
        gt.add_pair(
            lig, rec, "paracrine",
            ligand_phenotype="Source",
            receptor_phenotype="Target",
            mixing=float(mixing),
        )
    for i in range(len(src_marker_names)):
        for j in range(i + 1, len(src_marker_names)):
            gt.add_pair(src_marker_names[i], src_marker_names[j],
                        "shared_program", phenotype="Source")
    for i in range(len(tgt_marker_names)):
        for j in range(i + 1, len(tgt_marker_names)):
            gt.add_pair(tgt_marker_names[i], tgt_marker_names[j],
                        "shared_program", phenotype="Target")

    return adata, gt.to_json()
