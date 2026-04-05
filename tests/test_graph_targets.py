"""Tests for genevector._graph_targets — graph-aware co-expression targets."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.sparse import csr_matrix

from genevector.metrics import TARGETS
from genevector._graph_targets import target_graph_xcorr


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
