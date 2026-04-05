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
    if callable(name):
        return name
    if name not in AGGREGATIONS:
        available = ", ".join(sorted(AGGREGATIONS.keys()))
        raise ValueError(f"Unknown aggregation '{name}'. Available: {available}")
    return AGGREGATIONS[name]


# ─── Helpers ──────────────────────────────────────────────────

def _row_normalize(graph):
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = diags(1.0 / row_sums)
    return D_inv @ graph


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


# ─── Built-in aggregations ────────────────────────────────────

@register_aggregation("mean")
def aggr_mean(X_dense, graph, include_self=False, **kwargs):
    W = _row_normalize(graph)
    neighbor_mean = W @ X_dense
    if include_self:
        return 0.5 * X_dense + 0.5 * neighbor_mean
    return neighbor_mean
