"""Tests for genevector._aggregation — aggregation registry and mean aggregation."""

import numpy as np
import numpy.testing as npt
from scipy.sparse import csr_matrix

from genevector._aggregation import (
    AGGREGATIONS,
    get_aggregation,
    _row_normalize,
    _to_dense,
)


# ─── Shared fixtures ──────────────────────────────────────────

def _make_graph_and_X():
    """3-node graph: 0↔1, 1↔2.  X is (3, 2)."""
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

def test_registry_has_mean():
    assert "mean" in AGGREGATIONS


def test_mean_output_shape():
    rng = np.random.default_rng(42)
    n, d = 20, 5
    X = rng.standard_normal((n, d))
    graph = csr_matrix(rng.integers(0, 2, size=(n, n)).astype(np.float64))
    mean_fn = get_aggregation("mean")
    out = mean_fn(X, graph)
    assert out.shape == X.shape


def test_mean_known_value():
    adj, X = _make_graph_and_X()
    mean_fn = get_aggregation("mean")
    out = mean_fn(X, adj)
    expected = np.array([
        [0, 2],
        [2, 0.5],
        [0, 2],
    ], dtype=np.float64)
    npt.assert_array_almost_equal(out, expected)


def test_mean_include_self():
    adj, X = _make_graph_and_X()
    mean_fn = get_aggregation("mean")
    neighbor_mean = np.array([
        [0, 2],
        [2, 0.5],
        [0, 2],
    ], dtype=np.float64)
    expected = 0.5 * X + 0.5 * neighbor_mean
    out = mean_fn(X, adj, include_self=True)
    npt.assert_array_almost_equal(out, expected)


def test_mean_zero_degree_node():
    adj = csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],  # isolated
    ], dtype=np.float64))
    X = np.array([
        [1, 0],
        [0, 2],
        [3, 1],
        [5, 5],
    ], dtype=np.float64)
    mean_fn = get_aggregation("mean")
    out = mean_fn(X, adj)
    npt.assert_array_almost_equal(out[3], [0.0, 0.0])
    assert not np.any(np.isnan(out))


def test_row_normalize_sums():
    adj = csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64))
    W = _row_normalize(adj)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    for i in range(4):
        if adj[i].sum() == 0:
            assert row_sums[i] == 0.0
        else:
            npt.assert_almost_equal(row_sums[i], 1.0)


def test_to_dense_sparse_input():
    sp = csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float64))
    out = _to_dense(sp)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    npt.assert_array_equal(out, [[1, 2], [3, 4]])


def test_to_dense_dense_input():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    out = _to_dense(arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    npt.assert_array_equal(out, [[1, 2], [3, 4]])


def test_get_aggregation_callable():
    fn = lambda X, G, **kw: X * 2
    result = get_aggregation(fn)
    assert result is fn
    out = result(np.array([1, 2, 3]), None)
    npt.assert_array_equal(out, [2, 4, 6])


def test_get_aggregation_unknown():
    try:
        get_aggregation("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
