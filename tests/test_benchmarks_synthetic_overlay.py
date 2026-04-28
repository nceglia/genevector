"""Tests for genevector.benchmarks.synthetic.overlay and pathology."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from sklearn.metrics import mutual_info_score

from genevector.benchmarks.synthetic import (
    generate_synthetic_data,
    create_anndata_from_synthetic,
    apply_overlay,
    build_pathology,
)

SEED = 42


@pytest.fixture(scope="module")
def base_adata():
    df, _ = generate_synthetic_data(num_cells=2000, seed=SEED)
    return create_anndata_from_synthetic(df)


@pytest.fixture(scope="module")
def overlaid(base_adata):
    return apply_overlay(base_adata.copy(), seed=SEED)


def _quantile_bin(x, q=5):
    return pd.qcut(x, q=q, labels=False, duplicates="drop")


def test_genes_appended(base_adata, overlaid):
    adata, _ = overlaid
    expected = (
        [f"LIG_{i}" for i in range(5)]
        + [f"REC_{i}" for i in range(5)]
        + [f"NICHE_{i}" for i in range(5)]
        + [f"TRARE_{i}" for i in range(3)]
        + [f"HK_{i}" for i in range(5)]
    )
    for gene in expected:
        assert gene in adata.var_names
    assert adata.n_vars == base_adata.n_vars + 23


def test_phenotype_split(base_adata, overlaid):
    adata, _ = overlaid
    phenos = set(adata.obs["phenotype"].astype(str))
    for sub in ("T_stromal", "T_intratumoral", "T_rare"):
        assert sub in phenos
        assert (adata.obs["phenotype"].astype(str) == sub).sum() > 0
    assert "phenotype_coarse" in adata.obs.columns
    coarse = adata.obs["phenotype_coarse"].astype(str)
    t_mask = adata.obs["cell_type"].astype(str) == "T cell"
    assert coarse[t_mask].str.startswith("T_").all()


def test_paracrine_within_cell_mi_low(overlaid):
    adata, _ = overlaid
    for i in (0, 2):
        lig = adata[:, f"LIG_{i}"].X.flatten()
        rec = adata[:, f"REC_{i}"].X.flatten()
        mi = mutual_info_score(_quantile_bin(lig), _quantile_bin(rec))
        assert mi < 0.05, f"LIG_{i} ↔ REC_{i} MI = {mi:.4f} (>= 0.05)"


def test_niche_correlates_with_tumor_density(overlaid):
    adata, _ = overlaid
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    tumor_mask = (adata.obs["cell_type"].astype(str) == "Tumor").to_numpy()
    tree = cKDTree(coords)
    nbrs = tree.query_ball_point(coords, r=1.5)
    tumor_frac = np.array(
        [
            float(np.sum(tumor_mask[[j for j in nb if j != i]]))
            / max(1, len(nb) - (i in nb))
            for i, nb in enumerate(nbrs)
        ]
    )
    t_idx = np.where(adata.obs["cell_type"].astype(str).to_numpy() == "T cell")[0]
    for i in (0, 1):
        expr = adata[:, f"NICHE_{i}"].X.flatten()[t_idx]
        r = np.corrcoef(expr, tumor_frac[t_idx])[0, 1]
        assert r > 0.4, f"NICHE_{i} Pearson r = {r:.3f} (<= 0.4)"


def test_trare_distinct_in_rare(overlaid):
    adata, _ = overlaid
    pheno = adata.obs["phenotype"].astype(str).to_numpy()
    expr = adata[:, "TRARE_0"].X.flatten()
    rare_mean = expr[pheno == "T_rare"].mean()
    other_mean = max(
        expr[pheno == "T_stromal"].mean(),
        expr[pheno == "T_intratumoral"].mean(),
    )
    assert rare_mean > 3 * other_mean, (
        f"TRARE_0 rare={rare_mean:.3f} vs other_T={other_mean:.3f}"
    )


def test_housekeeping_uniform(overlaid):
    adata, _ = overlaid
    pheno = adata.obs["phenotype"].astype(str).to_numpy()
    expr = adata[:, "HK_0"].X.flatten()
    means = pd.Series(expr).groupby(pheno).mean()
    ratio = means.max() / means.min()
    assert ratio < 1.5, f"HK_0 max/min phenotype mean ratio = {ratio:.3f}"


def test_ground_truth_schema(overlaid):
    _, gt = overlaid
    for key in (
        "version",
        "paracrine_pairs",
        "niche_genes",
        "trare_genes",
        "housekeeping_genes",
        "t_subtypes",
        "added_gene_names",
    ):
        assert key in gt
    assert len(gt["paracrine_pairs"]) == 5
    round_trip = json.loads(json.dumps(gt))
    assert round_trip == gt


def test_determinism(base_adata):
    a1, gt1 = apply_overlay(base_adata.copy(), seed=SEED)
    a2, gt2 = apply_overlay(base_adata.copy(), seed=SEED)
    assert np.allclose(a1.X, a2.X)
    assert gt1 == gt2


def test_build_pathology_e2e():
    adata, gt = build_pathology(layout_kwargs={"num_cells": 1500}, seed=SEED)
    assert adata.n_vars >= 23
    assert "T_stromal" in set(adata.obs["phenotype"].astype(str))
    assert gt["version"] == "1.0"
    json.dumps(gt)


def test_no_t_cells_no_split():
    df, _ = generate_synthetic_data(
        num_cells=1000,
        tumor_fraction=0.7,
        stromal_fraction=0.1,
        t_fraction=0.0,
        b_fraction=0.1,
        myeloid_fraction=0.1,
        rare_t_cell=False,
        seed=SEED,
    )
    adata = create_anndata_from_synthetic(df)
    new_adata, gt = apply_overlay(adata, seed=SEED)
    for sub in ("T_stromal", "T_intratumoral", "T_rare"):
        assert gt["t_subtypes"][sub]["n_cells"] == 0
    assert "phenotype_coarse" in new_adata.obs.columns
