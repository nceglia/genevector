"""genevector/_graph_targets.py — graph-aware co-expression targets."""

import numpy as np
from scipy.sparse import issparse

from ._aggregation import get_aggregation, _to_dense
from .metrics import register_target, _matrix_to_score_dict


@register_target("graph_xcorr")
def target_graph_xcorr(X, gene_names, graph=None, aggr="mean",
                       aggr_params=None, **kwargs):
    """Cross-correlation between self-expression and graph-neighbor-aggregated expression.

    Computes bivariate cross-correlation for all gene pairs: for each gene A and gene B,
    measures the correlation between A's expression in each cell and B's aggregated
    expression in that cell's graph neighbors. The result is symmetrized.

    Parameters
    ----------
    X : sparse or dense matrix
        Expression matrix (cells x genes).
    gene_names : list of str
        Gene symbols.
    graph : scipy.sparse matrix
        Adjacency matrix (spatial, TCR, or any graph topology).
    aggr : str or callable
        Aggregation method name or function. Default "mean".
    aggr_params : dict, optional
        Extra keyword arguments passed to the aggregation function.

    Returns
    -------
    dict of dict
        scores[gene_a][gene_b] = float, cross-correlation in [-1, 1].
    """
    if graph is None:
        raise ValueError(
            "graph required. Pass any scipy sparse adjacency matrix "
            "via target_kwargs={'graph': G}"
        )
    aggr_fn = get_aggregation(aggr)
    X_dense = _to_dense(X)
    X_agg = aggr_fn(X_dense, graph, **(aggr_params or {}))

    n_cells = X_dense.shape[0]

    X_std = (X_dense - X_dense.mean(axis=0)) / (X_dense.std(axis=0) + 1e-8)
    X_agg_std = (X_agg - X_agg.mean(axis=0)) / (X_agg.std(axis=0) + 1e-8)

    xcorr = (X_std.T @ X_agg_std) / n_cells
    xcorr_sym = (xcorr + xcorr.T) / 2
    np.fill_diagonal(xcorr_sym, 0)

    return _matrix_to_score_dict(xcorr_sym, gene_names)
