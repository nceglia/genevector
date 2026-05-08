"""Tests for the v2.0 synthetic dataset template catalog."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial import cKDTree

from genevector.benchmarks.synthetic import (
    GROUND_TRUTH_VERSION,
    build_gradient_dataset,
    build_niche_dataset,
    build_paracrine_dataset,
    build_pathology,
    list_templates,
)


SEED = 42


@pytest.fixture(scope="module")
def paracrine_data():
    return build_paracrine_dataset(n_cells_per_type=300, n_pairs=3, seed=SEED)


@pytest.fixture(scope="module")
def niche_data():
    return build_niche_dataset(n_tumor_cells=300, n_t_cells=300, seed=SEED)


@pytest.fixture(scope="module")
def gradient_data():
    return build_gradient_dataset(n_cells=400, seed=SEED)


def _assert_contract(adata):
    assert sparse.issparse(adata.X) and isinstance(adata.X, sparse.csr_matrix)
    assert "phenotype" in adata.obs.columns
    assert hasattr(adata.obs["phenotype"], "cat")
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
    assert adata.obsm["spatial"].dtype == np.float64
    var_names = list(adata.var_names)
    assert all(v == v.upper() for v in var_names)
    assert len(set(var_names)) == len(var_names)


def test_list_templates_complete():
    keys = set(list_templates().keys())
    assert keys == {"pathology", "paracrine", "niche", "gradient"}


def test_paracrine_contract(paracrine_data):
    adata, _ = paracrine_data
    _assert_contract(adata)


def test_niche_contract(niche_data):
    adata, _ = niche_data
    _assert_contract(adata)


def test_gradient_contract(gradient_data):
    adata, _ = gradient_data
    _assert_contract(adata)
    assert set(adata.obs["phenotype"].astype(str)) == {"Tissue"}


def test_v2_schema(paracrine_data, niche_data, gradient_data):
    for _, gt in (paracrine_data, niche_data, gradient_data):
        for key in ("template", "version", "seed", "params",
                    "phenotypes", "genes", "pairs"):
            assert key in gt
        assert gt["version"] == GROUND_TRUTH_VERSION == "2.0"
        assert json.loads(json.dumps(gt)) == gt


def test_paracrine_pair_kinds(paracrine_data):
    _, gt = paracrine_data
    paracrine_pairs = [p for p in gt["pairs"] if p["kind"] == "paracrine"]
    assert len(paracrine_pairs) == 3  # n_pairs from fixture
    assert any(p["kind"] == "paracrine" for p in gt["pairs"])


def test_niche_induction_signal(niche_data):
    adata, _ = niche_data
    pheno = adata.obs["phenotype"].astype(str).to_numpy()
    near_expr = adata[pheno == "T_near", "NICHE_0"].X.toarray().mean()
    far_expr = adata[pheno == "T_far", "NICHE_0"].X.toarray().mean()
    assert near_expr > 3 * far_expr, f"near={near_expr:.3f} far={far_expr:.3f}"


def test_gradient_monotone(gradient_data):
    adata, _ = gradient_data
    expr = adata[:, "MONO_0"].X.toarray().ravel()
    x = adata.obsm["spatial"][:, 0]
    r = np.corrcoef(expr, x)[0, 1]
    assert r > 0.5, f"MONO_0 vs x correlation = {r:.3f}"


def test_paracrine_mixing_changes_adjacency():
    a_low, _ = build_paracrine_dataset(
        n_cells_per_type=400, n_pairs=2, mixing=0.0,
        n_identity_markers=0, seed=SEED,
    )
    a_high, _ = build_paracrine_dataset(
        n_cells_per_type=400, n_pairs=2, mixing=1.0,
        n_identity_markers=0, seed=SEED,
    )

    def cross_frac(adata):
        coords = adata.obsm["spatial"]
        labels = adata.obs["phenotype"].astype(str).to_numpy()
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=6)  # self + 5 neighbors
        cross = 0
        total = 0
        for i in range(coords.shape[0]):
            nbrs = idx[i, 1:]
            cross += int((labels[nbrs] != labels[i]).sum())
            total += len(nbrs)
        return cross / total

    assert cross_frac(a_high) > cross_frac(a_low)


def test_determinism(paracrine_data):
    a1, gt1 = build_paracrine_dataset(n_cells_per_type=300, n_pairs=3, seed=SEED)
    a2, gt2 = build_paracrine_dataset(n_cells_per_type=300, n_pairs=3, seed=SEED)
    assert np.array_equal(a1.X.toarray(), a2.X.toarray())
    assert gt1 == gt2


def test_pathology_v2_migration():
    _, gt = build_pathology(layout_kwargs={"num_cells": 800}, seed=SEED)
    assert gt["template"] == "pathology"
    assert gt["version"] == "2.0"
    for key in ("phenotypes", "genes", "pairs"):
        assert key in gt
    roles = {g["role"] for g in gt["genes"].values()}
    assert {"ligand", "receptor", "niche_gene", "housekeeping"} <= roles
    kinds = {p["kind"] for p in gt["pairs"]}
    assert "paracrine" in kinds
