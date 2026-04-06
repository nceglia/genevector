"""genevector/_aggregation.py — graph aggregation registry and implementations."""

import numpy as np
from scipy.sparse import issparse, diags

# ─── Aggregation registry ─────────────────────────────────────

AGGREGATIONS = {}


def register_aggregation(name):
    """Decorator to register an aggregation function."""
    def wrapper(fn):
        AGGREGATIONS[name] = fn
        return fn
    return wrapper


def get_aggregation(name):
    """Look up a registered aggregation function by name, or return a callable directly.

    Parameters
    ----------
    name : str or callable
        Name of registered aggregation, or a callable with signature
        ``f(X_dense, graph, **params) -> np.ndarray``.

    Returns
    -------
    callable
        The aggregation function.

    Raises
    ------
    ValueError
        If name is a string and not registered.
    """
    if callable(name):
        return name
    if name not in AGGREGATIONS:
        available = ", ".join(sorted(AGGREGATIONS.keys()))
        raise ValueError(f"Unknown aggregation '{name}'. Available: {available}")
    return AGGREGATIONS[name]


# ─── Helpers ──────────────────────────────────────────────────

def _row_normalize(graph):
    """Row-normalize a sparse adjacency matrix so each row sums to 1.

    Parameters
    ----------
    graph : scipy.sparse matrix
        Adjacency matrix. Zero-degree rows are left as zeros.

    Returns
    -------
    scipy.sparse matrix
        Row-normalized adjacency.
    """
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = diags(1.0 / row_sums)
    return D_inv @ graph


def _to_dense(X):
    """Convert sparse or dense matrix to dense float64 numpy array.

    Parameters
    ----------
    X : scipy.sparse matrix or np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Dense array with dtype float64.
    """
    if issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


# ─── Built-in aggregations ────────────────────────────────────

@register_aggregation("mean")
def aggr_mean(X_dense, graph, include_self=False, **kwargs):
    """Mean aggregation over graph neighbors.

    Parameters
    ----------
    X_dense : np.ndarray
        Expression matrix (cells x genes).
    graph : scipy.sparse matrix
        Adjacency matrix (any graph topology).
    include_self : bool
        If True, blend 50/50 between self and neighbor mean.

    Returns
    -------
    np.ndarray
        Aggregated expression, same shape as X_dense.
    """
    W = _row_normalize(graph)
    neighbor_mean = W @ X_dense
    if include_self:
        return 0.5 * X_dense + 0.5 * neighbor_mean
    return neighbor_mean
