"""genevector/_graph_targets.py — graph-aware co-expression targets."""

import numpy as np
from scipy.sparse import issparse

from ._aggregation import get_aggregation, _to_dense
from .metrics import register_target, _matrix_to_score_dict


@register_target("graph_xcorr")
def target_graph_xcorr(X, gene_names, graph=None, aggr="mean",
                       aggr_params=None, **kwargs):
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
