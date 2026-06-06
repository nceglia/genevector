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


def _cross_mi_matrix(A_disc, na, B_disc, nb):
    """Pairwise MI between every self-gene column of A and neighbor-gene column of B.

    Returns M where M[i, j] = MI(self gene i, neighbor-aggregated gene j). Asymmetric.
    """
    from .metrics import _mi_from_joint
    P = A_disc.shape[1]
    M = np.zeros((P, P), dtype=np.float64)
    for i in range(P):
        if na[i] <= 1:
            continue
        a = A_disc[:, i]
        for j in range(P):
            if nb[j] <= 1:
                continue
            b = B_disc[:, j]
            mask = (a > 0) | (b > 0)
            if mask.sum() == 0:
                continue
            joint = np.zeros((na[i], nb[j]), dtype=np.float64)
            np.add.at(joint, (a[mask], b[mask]), 1)
            M[i, j] = _mi_from_joint(joint)
    return M


def _graph_mi_core(X, gene_names, graph, aggr, aggr_params, n_bins, signed):
    """Shared computation for graph_mi / graph_cross_mi: returns signed cross-MI matrix.

    M[i, j] = (sign of self_i vs neighbor_j cross-correlation) * MI(self_i, neighbor_j).
    Aggregating expression over the graph BEFORE estimating the gene-gene relationship
    denoises the per-cell counts, making it robust to dropout on sparse spatial panels.
    """
    if graph is None:
        raise ValueError(
            "graph required. Pass any scipy sparse adjacency matrix "
            "via target_kwargs={'graph': G}"
        )
    from .metrics import discretize_genes
    aggr_fn = get_aggregation(aggr)
    X_dense = _to_dense(X)
    X_agg = aggr_fn(X_dense, graph, **(aggr_params or {}))

    A_disc, na = discretize_genes(X_dense, n_bins=n_bins)
    B_disc, nb = discretize_genes(X_agg, n_bins=n_bins)
    M = _cross_mi_matrix(A_disc, na, B_disc, nb)

    if signed:
        X_std = (X_dense - X_dense.mean(0)) / (X_dense.std(0) + 1e-8)
        A_std = (X_agg - X_agg.mean(0)) / (X_agg.std(0) + 1e-8)
        sign = np.sign((X_std.T @ A_std) / X_dense.shape[0])
        M = sign * M
    return M


@register_target("graph_mi")
def target_graph_mi(X, gene_names, graph=None, aggr="mean", aggr_params=None,
                    n_bins=10, signed=True, **kwargs):
    """Symmetric graph mutual information between self and neighbor-aggregated expression.

    The MI analogue of ``graph_xcorr``: captures non-linear spatial co-expression while
    the neighbor aggregation denoises sparse counts. Symmetrized over (i, j).

    Returns
    -------
    dict of dict
        scores[gene_a][gene_b] = signed MI in (roughly) [-log2(n_bins), log2(n_bins)].
    """
    M = _graph_mi_core(X, gene_names, graph, aggr, aggr_params, n_bins, signed)
    M_sym = (M + M.T) / 2
    np.fill_diagonal(M_sym, 0)
    return _matrix_to_score_dict(M_sym, gene_names)


@register_target("graph_cross_mi")
def target_graph_cross_mi(X, gene_names, graph=None, aggr="mean", aggr_params=None,
                          n_bins=10, signed=True, **kwargs):
    """Asymmetric cross-neighbor MI: MI(gene_a in cell, gene_b in neighbors).

    Directional spatial signal (e.g. ligand in a cell predicting receptor in its
    neighbours). Not symmetrized — the model's separate input/output weights can encode
    the asymmetry. Encodes niche/communication directionality in the gene embedding.
    """
    M = _graph_mi_core(X, gene_names, graph, aggr, aggr_params, n_bins, signed)
    np.fill_diagonal(M, 0)
    return _matrix_to_score_dict(M, gene_names)
