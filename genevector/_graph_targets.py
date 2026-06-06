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


def _cross_mi_matrix_torch(A_disc, na, B_disc, nb, device="cuda", max_elems=20_000_000):
    """GPU/torch cross-MI: M[i, j] = MI(self gene i, neighbor gene j). Mirrors _cross_mi_matrix.

    For each self gene one ``scatter_add`` builds every neighbour gene's joint histogram at
    once (chunked for memory), so this is O(P) Python iterations rather than O(P^2). Runs on
    any torch device — pass ``device='cpu'`` to validate against the numpy path.

    Padding to the max bin count is harmless: extra bins stay empty and contribute 0 to MI.
    Zeroing the (0, 0) cell of each joint reproduces the ``(a>0)|(b>0)`` mask used elsewhere
    (cells where both genes are zero are dropped).
    """
    import torch
    A_disc = np.asarray(A_disc)
    B_disc = np.asarray(B_disc)
    n_cells, P = A_disc.shape
    MA = int(na.max()) if na.size else 1
    MB = int(nb.max()) if nb.size else 1
    cellsize = MA * MB
    chunk = max(1, min(P, int(max_elems // max(n_cells, 1))))

    A_t = torch.as_tensor(A_disc, device=device, dtype=torch.long)
    B_t = torch.as_tensor(B_disc, device=device, dtype=torch.long)
    nb_t = torch.as_tensor(np.asarray(nb), device=device, dtype=torch.long)
    M = torch.zeros((P, P), device=device, dtype=torch.float64)

    for i in range(P):
        if int(na[i]) <= 1:
            continue
        a = A_t[:, i]  # (n_cells,)
        for start in range(0, P, chunk):
            end = min(start + chunk, P)
            c = end - start
            Bc = B_t[:, start:end]  # (n_cells, c)
            joff = (torch.arange(c, device=device, dtype=torch.long) * cellsize).unsqueeze(0)
            flat = (joff + a.unsqueeze(1) * MB + Bc).reshape(-1)  # (n_cells*c,)
            hist = torch.zeros(c * cellsize, device=device, dtype=torch.float64)
            hist.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
            hist = hist.reshape(c, MA, MB)
            hist[:, 0, 0] = 0.0  # drop both-zero cells == (a>0)|(b>0) mask
            total = hist.sum(dim=(1, 2), keepdim=True)
            p = hist / total.clamp_min(1.0)
            px = p.sum(dim=2, keepdim=True)
            py = p.sum(dim=1, keepdim=True)
            ratio = torch.where(p > 0, p / (px * py).clamp_min(1e-12), torch.ones_like(p))
            mi = (p * torch.log2(ratio)).sum(dim=(1, 2))  # (c,)
            valid = (total.reshape(-1) > 0) & (nb_t[start:end] > 1)
            M[i, start:end] = torch.where(valid, mi, torch.zeros_like(mi))
    return M.cpu().numpy()


def _cross_mi_matrix_rust(A_disc, na, B_disc, nb):
    """rayon-parallel Rust cross-MI: M[i, j] = MI(self gene i, neighbor gene j).

    Mirrors _cross_mi_matrix but parallel across the P*P pairs. ~80-110x faster than numpy
    and a few x faster than the torch CPU kernel on many-core machines (the diagonal is left
    zero — it is zeroed downstream anyway).
    """
    from ._rust import compute_cross_mi_pairs
    P = A_disc.shape[1]
    triples = compute_cross_mi_pairs(
        np.ascontiguousarray(A_disc, dtype=np.int32), np.asarray(na, dtype=np.int32),
        np.ascontiguousarray(B_disc, dtype=np.int32), np.asarray(nb, dtype=np.int32), None)
    M = np.zeros((P, P), dtype=np.float64)
    for i, j, v in triples:
        M[i, j] = v
    return M


def _graph_mi_core(X, gene_names, graph, aggr, aggr_params, n_bins, signed,
                   backend="auto", device="cpu"):
    """Shared computation for graph_mi / graph_cross_mi: returns signed cross-MI matrix.

    M[i, j] = (sign of self_i vs neighbor_j cross-correlation) * MI(self_i, neighbor_j).
    Aggregating expression over the graph BEFORE estimating the gene-gene relationship
    denoises the per-cell counts, making it robust to dropout on sparse spatial panels.

    ``backend`` selects the cross-MI kernel:
      - "auto" (default): GPU torch when ``device=="cuda"``; else the rayon Rust kernel if the
        ``_rust`` extension is built (fastest on multi-core CPU); else the torch CPU kernel
        (~20-35x numpy); else numpy.
      - "rust" / "gpu" / "numpy": force that kernel (each degrades gracefully if unavailable).
    All kernels are numerically identical (diff ~1e-15); only speed differs.
    """
    if graph is None:
        raise ValueError(
            "graph required. Pass any scipy sparse adjacency matrix "
            "via target_kwargs={'graph': G}"
        )
    from .metrics import discretize_genes, HAS_RUST
    from ._logging import get_logger
    aggr_fn = get_aggregation(aggr)
    X_dense = _to_dense(X)
    X_agg = aggr_fn(X_dense, graph, **(aggr_params or {}))

    A_disc, na = discretize_genes(X_dense, n_bins=n_bins)
    B_disc, nb = discretize_genes(X_agg, n_bins=n_bins)

    chosen = backend
    if chosen == "auto":
        chosen = "gpu" if device == "cuda" else ("rust" if HAS_RUST else "gpu")

    M = None
    if chosen == "rust":
        try:
            M = _cross_mi_matrix_rust(A_disc, na, B_disc, nb)
        except Exception as e:
            get_logger(__name__).warning(f"graph_mi rust backend failed ({e}); trying torch.")
            chosen = "gpu"
    if M is None and chosen == "gpu":
        try:
            import torch
            dev = device
            if dev == "cuda" and not torch.cuda.is_available():
                get_logger(__name__).warning("device='cuda' requested but CUDA unavailable; "
                                             "using torch CPU.")
                dev = "cpu"
            M = _cross_mi_matrix_torch(A_disc, na, B_disc, nb, device=dev)
        except Exception as e:  # graceful fall back to numpy
            get_logger(__name__).warning(f"graph_mi torch backend failed ({e}); using numpy.")
            M = None
    if M is None:
        M = _cross_mi_matrix(A_disc, na, B_disc, nb)

    if signed:
        X_std = (X_dense - X_dense.mean(0)) / (X_dense.std(0) + 1e-8)
        A_std = (X_agg - X_agg.mean(0)) / (X_agg.std(0) + 1e-8)
        sign = np.sign((X_std.T @ A_std) / X_dense.shape[0])
        M = sign * M
    return M


@register_target("graph_mi")
def target_graph_mi(X, gene_names, graph=None, aggr="mean", aggr_params=None,
                    n_bins=10, signed=True, backend="auto", device="cpu", **kwargs):
    """Symmetric graph mutual information between self and neighbor-aggregated expression.

    The MI analogue of ``graph_xcorr``: captures non-linear spatial co-expression while
    the neighbor aggregation denoises sparse counts. Symmetrized over (i, j). The cross-MI
    kernel is auto-selected (GPU torch on ``device="cuda"``, else the rayon Rust extension if
    built, else torch CPU, else numpy); force one with ``backend`` in {"rust","gpu","numpy"}.

    Returns
    -------
    dict of dict
        scores[gene_a][gene_b] = signed MI in (roughly) [-log2(n_bins), log2(n_bins)].
    """
    M = _graph_mi_core(X, gene_names, graph, aggr, aggr_params, n_bins, signed,
                       backend=backend, device=device)
    M_sym = (M + M.T) / 2
    np.fill_diagonal(M_sym, 0)
    return _matrix_to_score_dict(M_sym, gene_names)


@register_target("graph_cross_mi")
def target_graph_cross_mi(X, gene_names, graph=None, aggr="mean", aggr_params=None,
                          n_bins=10, signed=True, backend="auto", device="cpu", **kwargs):
    """Asymmetric cross-neighbor MI: MI(gene_a in cell, gene_b in neighbors).

    Directional spatial signal (e.g. ligand in a cell predicting receptor in its
    neighbours). Not symmetrized — the model's separate input/output weights can encode
    the asymmetry. Encodes niche/communication directionality in the gene embedding. The cross-MI
    kernel is auto-selected (GPU torch on ``device="cuda"``, else the rayon Rust extension if
    built, else torch CPU, else numpy); force one with ``backend`` in {"rust","gpu","numpy"}.
    """
    M = _graph_mi_core(X, gene_names, graph, aggr, aggr_params, n_bins, signed,
                       backend=backend, device=device)
    np.fill_diagonal(M, 0)
    return _matrix_to_score_dict(M, gene_names)
