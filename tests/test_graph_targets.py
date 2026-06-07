"""Tests for genevector._graph_targets — graph-aware co-expression targets."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.sparse import csr_matrix

from genevector.metrics import TARGETS
from genevector._graph_targets import (
    target_graph_xcorr,
    target_graph_mi,
    target_graph_cross_mi,
)


# ─── Shared fixtures ──────────────────────────────────────────

def _make_graph_and_X():
    adj = csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float64))
    X = np.array([
        [1, 0],
        [0, 2],
        [3, 1],
    ], dtype=np.float64)
    return adj, X


# ─── Tests ────────────────────────────────────────────────────

def test_graph_xcorr_registered():
    assert "graph_xcorr" in TARGETS


def test_graph_xcorr_requires_graph():
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    with pytest.raises(ValueError, match="graph required"):
        target_graph_xcorr(X, ["a", "b"])


def test_graph_xcorr_output_structure():
    adj, X = _make_graph_and_X()
    genes = ["g0", "g1"]
    scores = target_graph_xcorr(X, genes, graph=adj)
    assert isinstance(scores, dict)
    for g in genes:
        assert g in scores
        assert g not in scores[g]
    assert scores["g0"]["g1"] == scores["g1"]["g0"]


def test_graph_xcorr_range():
    rng = np.random.default_rng(42)
    n, d = 50, 10
    X = rng.standard_normal((n, d))
    adj = csr_matrix(rng.integers(0, 2, size=(n, n)).astype(np.float64))
    genes = [f"g{i}" for i in range(d)]
    scores = target_graph_xcorr(X, genes, graph=adj)
    for g1 in genes:
        for g2 in scores[g1]:
            assert -1.0 - 1e-6 <= scores[g1][g2] <= 1.0 + 1e-6


def test_graph_xcorr_symmetric():
    rng = np.random.default_rng(7)
    n, d = 30, 5
    X = rng.standard_normal((n, d))
    adj = csr_matrix(rng.integers(0, 2, size=(n, n)).astype(np.float64))
    genes = [f"g{i}" for i in range(d)]
    scores = target_graph_xcorr(X, genes, graph=adj)
    for g1 in genes:
        for g2 in scores[g1]:
            assert scores[g1][g2] == pytest.approx(scores[g2][g1])


def test_graph_xcorr_known_direction():
    n = 50
    # Chain graph: 0-1-2-...-49
    row = list(range(n - 1)) + list(range(1, n))
    col = list(range(1, n)) + list(range(n - 1))
    data = [1.0] * len(row)
    adj = csr_matrix((data, (row, col)), shape=(n, n))

    # Gene A: high in even cells
    # Gene B: high in odd cells (neighbors of even cells in chain)
    # Gene C: random
    rng = np.random.default_rng(0)
    gene_a = np.array([5.0 if i % 2 == 0 else 0.0 for i in range(n)])
    gene_b = np.array([5.0 if i % 2 == 1 else 0.0 for i in range(n)])
    gene_c = rng.standard_normal(n)
    X = np.column_stack([gene_a, gene_b, gene_c])

    genes = ["A", "B", "C"]
    scores = target_graph_xcorr(X, genes, graph=adj)
    assert abs(scores["A"]["B"]) > abs(scores["A"]["C"])


def test_graph_xcorr_custom_aggr():
    adj, X = _make_graph_and_X()
    genes = ["g0", "g1"]
    scores = target_graph_xcorr(
        X, genes, graph=adj,
        aggr=lambda X, G, **kw: np.zeros_like(X),
    )
    for g1 in genes:
        for g2 in scores[g1]:
            assert scores[g1][g2] == pytest.approx(0.0, abs=1e-5)


def test_graph_xcorr_no_self_pairs():
    adj, X = _make_graph_and_X()
    genes = ["g0", "g1"]
    scores = target_graph_xcorr(X, genes, graph=adj)
    for g in genes:
        assert g not in scores[g]


def test_graph_xcorr_with_sparse_X():
    adj, X_dense = _make_graph_and_X()
    X_sparse = csr_matrix(X_dense)
    genes = ["g0", "g1"]
    scores_dense = target_graph_xcorr(X_dense, genes, graph=adj)
    scores_sparse = target_graph_xcorr(X_sparse, genes, graph=adj)
    for g1 in genes:
        for g2 in scores_dense[g1]:
            assert scores_dense[g1][g2] == pytest.approx(scores_sparse[g1][g2])


# ─── graph_mi (symmetric) and graph_cross_mi (asymmetric) ─────

def _make_chain_panel(n=60):
    # chain graph; a spatial gradient shared by A and B (co-located), C is the same
    # marginal with the spatial structure destroyed. Continuous + mostly nonzero so the
    # MI mask keeps cells.
    row = list(range(n - 1)) + list(range(1, n))
    col = list(range(1, n)) + list(range(n - 1))
    adj = csr_matrix(([1.0] * len(row), (row, col)), shape=(n, n))
    rng = np.random.default_rng(0)
    grad = np.linspace(1.0, 6.0, n)
    gene_a = np.clip(grad + rng.normal(0, 0.2, n), 0, None)
    gene_b = np.clip(grad + rng.normal(0, 0.2, n), 0, None)  # co-located with A
    gene_c = rng.permutation(grad)                            # no spatial structure
    X = np.column_stack([gene_a, gene_b, gene_c])
    return X, adj, ["A", "B", "C"]


def test_graph_mi_registered():
    assert "graph_mi" in TARGETS
    assert "graph_cross_mi" in TARGETS


def test_graph_mi_requires_graph():
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    with pytest.raises(ValueError, match="graph required"):
        target_graph_mi(X, ["a", "b"])
    with pytest.raises(ValueError, match="graph required"):
        target_graph_cross_mi(X, ["a", "b"])


def test_graph_mi_symmetric_and_no_self():
    X, adj, genes = _make_chain_panel()
    scores = target_graph_mi(X, genes, graph=adj)
    for g in genes:
        assert g not in scores[g]
    for g1 in genes:
        for g2 in scores[g1]:
            assert scores[g1][g2] == pytest.approx(scores[g2][g1], abs=1e-6)


def test_graph_mi_detects_neighbor_coexpression():
    # A high in even cells, B high in their chain neighbours → strong graph MI(A,B)
    X, adj, genes = _make_chain_panel()
    scores = target_graph_mi(X, genes, graph=adj)
    assert abs(scores["A"]["B"]) > abs(scores["A"]["C"])


def test_graph_mi_sparse_equals_dense():
    X, adj, genes = _make_chain_panel(n=40)
    sd = target_graph_mi(X, genes, graph=adj)
    ss = target_graph_mi(csr_matrix(X), genes, graph=adj)
    for g1 in genes:
        for g2 in sd[g1]:
            assert sd[g1][g2] == pytest.approx(ss[g1][g2], abs=1e-6)


def test_graph_cross_mi_no_self_pairs():
    X, adj, genes = _make_chain_panel()
    scores = target_graph_cross_mi(X, genes, graph=adj)
    for g in genes:
        assert g not in scores[g]


# ─── GPU (torch) graph_mi == numpy graph_mi (validated on CPU torch) ──

def test_graph_mi_torch_matches_numpy():
    pytest.importorskip("torch")
    X, adj, genes = _make_chain_panel(n=50)
    cpu = target_graph_mi(X, genes, graph=adj, backend="numpy")
    gpu = target_graph_mi(X, genes, graph=adj, backend="gpu", device="cpu")
    for g1 in genes:
        for g2 in cpu[g1]:
            assert cpu[g1][g2] == pytest.approx(gpu[g1][g2], abs=1e-6)


def test_graph_cross_mi_torch_matches_numpy():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    n, d = 80, 6
    X = rng.poisson(1.5, size=(n, d)).astype(float)
    adj = csr_matrix((rng.random((n, n)) < 0.2).astype(np.float64))
    genes = [f"g{i}" for i in range(d)]
    cpu = target_graph_cross_mi(X, genes, graph=adj, backend="numpy")
    gpu = target_graph_cross_mi(X, genes, graph=adj, backend="gpu", device="cpu")
    for g1 in genes:
        for g2 in cpu[g1]:
            assert cpu[g1][g2] == pytest.approx(gpu[g1][g2], abs=1e-6)


def test_cross_mi_torch_chunking_consistent():
    pytest.importorskip("torch")
    from genevector._graph_targets import _cross_mi_matrix, _cross_mi_matrix_torch
    from genevector.metrics import discretize_genes
    rng = np.random.default_rng(2)
    X = rng.poisson(1.0, size=(120, 8)).astype(float)
    Ad, na = discretize_genes(X)
    Bd, nb = discretize_genes(X + rng.poisson(0.5, X.shape))
    ref = _cross_mi_matrix(Ad, na, Bd, nb)
    full = _cross_mi_matrix_torch(Ad, na, Bd, nb, device="cpu", max_elems=10**9)
    chunked = _cross_mi_matrix_torch(Ad, na, Bd, nb, device="cpu", max_elems=120 * 2)
    np.testing.assert_allclose(ref, full, atol=1e-6)
    np.testing.assert_allclose(ref, chunked, atol=1e-6)


def test_graph_mi_rust_matches_numpy():
    from genevector.metrics import HAS_RUST
    if not HAS_RUST:
        pytest.skip("rust extension (_rust) not built")
    X, adj, genes = _make_chain_panel(n=60)
    cpu = target_graph_mi(X, genes, graph=adj, backend="numpy")
    rust = target_graph_mi(X, genes, graph=adj, backend="rust")
    for g1 in genes:
        for g2 in cpu[g1]:
            assert cpu[g1][g2] == pytest.approx(rust[g1][g2], abs=1e-6)
