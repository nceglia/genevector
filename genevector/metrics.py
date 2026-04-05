"""genevector/metrics.py — co-expression target functions."""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.stats import spearmanr
import collections
import itertools
import tqdm

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from ._rust import compute_mi_pairs as _rust_mi_pairs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# ─── Discretization ────────────────────────────────────────────

def discretize_genes(X, n_bins=10):
    """
    Discretize each gene's expression into integer bin indices.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix or np.ndarray
        Cells x genes expression matrix.
    n_bins : int
        Number of bins per gene (excluding the zero bin).

    Returns
    -------
    X_disc : np.ndarray, shape (n_cells, n_genes), dtype=np.int32
        Discretized expression. 0 = zero expression,
        1..n_bins = quantile bins of nonzero expression.
    n_bins_per_gene : np.ndarray, shape (n_genes,), dtype=np.int32
        Actual number of bins used per gene (may be < n_bins
        if a gene has fewer unique nonzero values).
    """
    if issparse(X):
        X = np.asarray(X.todense())
    n_cells, n_genes = X.shape
    X_disc = np.zeros((n_cells, n_genes), dtype=np.int32)
    n_bins_per_gene = np.zeros(n_genes, dtype=np.int32)

    for g in range(n_genes):
        col = X[:, g].ravel()
        nonzero_mask = col > 0
        nonzero_vals = col[nonzero_mask]
        if len(nonzero_vals) == 0:
            n_bins_per_gene[g] = 1  # just the zero bin
            continue
        unique_vals = np.unique(nonzero_vals)
        actual_bins = min(n_bins, len(unique_vals))
        if actual_bins <= 1:
            X_disc[nonzero_mask, g] = 1
            n_bins_per_gene[g] = 2  # 0 and 1
        else:
            quantiles = np.linspace(0, 100, actual_bins + 1)[1:-1]
            edges = np.percentile(nonzero_vals, quantiles)
            X_disc[nonzero_mask, g] = np.searchsorted(edges, nonzero_vals, side='right') + 1
            n_bins_per_gene[g] = actual_bins + 1  # including zero bin

    return X_disc, n_bins_per_gene


# ─── MI helper ─────────────────────────────────────────────────

def _mi_from_joint(pxy):
    """Compute MI from a joint probability distribution."""
    pxy = pxy / pxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = np.outer(px, py)
    nz = (pxy > 0) & (px_py > 0)
    return np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz]))


# ─── Tier A: Vectorized NumPy MI ───────────────────────────────

def compute_mi_vectorized(X, gene_names, n_bins=10, signed=False):
    """
    Compute MI for all gene pairs using vectorized discretization.

    Parameters
    ----------
    X : sparse or dense matrix, shape (n_cells, n_genes)
    gene_names : list of str
    n_bins : int
    signed : bool
        If True, multiply MI by sign of Pearson correlation.

    Returns
    -------
    mi_scores : dict of dict
        mi_scores[gene_a][gene_b] = float
    """
    X_disc, n_bins_per_gene = discretize_genes(X, n_bins=n_bins)
    n_genes = len(gene_names)

    if signed:
        if issparse(X):
            X_dense = np.asarray(X.todense())
        else:
            X_dense = X
        corr_matrix = np.nan_to_num(np.corrcoef(X_dense.T), nan=0.0)

    mi_scores = collections.defaultdict(lambda: collections.defaultdict(float))
    pairs = list(itertools.combinations(range(n_genes), 2))

    for idx_a, idx_b in tqdm.tqdm(pairs, desc="Computing MI"):
        na = n_bins_per_gene[idx_a]
        nb = n_bins_per_gene[idx_b]
        if na <= 1 or nb <= 1:
            continue

        col_a = X_disc[:, idx_a]
        col_b = X_disc[:, idx_b]
        mask = (col_a > 0) | (col_b > 0)
        if mask.sum() == 0:
            continue

        joint = np.zeros((na, nb), dtype=np.float64)
        np.add.at(joint, (col_a[mask], col_b[mask]), 1)

        mi = _mi_from_joint(joint)

        if signed:
            sign = np.sign(corr_matrix[idx_a, idx_b])
            mi = sign * mi

        gene_a = gene_names[idx_a]
        gene_b = gene_names[idx_b]
        mi_scores[gene_a][gene_b] = round(mi, 5)
        mi_scores[gene_b][gene_a] = round(mi, 5)

    return mi_scores


# ─── Tier B: Numba JIT MI ──────────────────────────────────────

if HAS_NUMBA:
    @njit(parallel=True)
    def _compute_all_mi_numba(X_disc, n_bins_per_gene, n_genes):
        """
        Returns a flat array of MI values for upper-triangle pairs.
        """
        n_pairs = n_genes * (n_genes - 1) // 2
        mi_values = np.zeros(n_pairs, dtype=np.float64)

        for flat_idx in prange(n_pairs):
            # recover (i, j) from flat index
            i = int(n_genes - 1 - int(np.sqrt(-8 * flat_idx + 4 * n_genes * (n_genes - 1) - 7) / 2 - 0.5))
            j = flat_idx + i * (i + 1) // 2 - n_genes * i + n_genes

            na = n_bins_per_gene[i]
            nb = n_bins_per_gene[j]
            if na <= 1 or nb <= 1:
                continue

            n_cells = X_disc.shape[0]
            joint = np.zeros((na, nb), dtype=np.float64)
            count = 0
            for c in range(n_cells):
                a = X_disc[c, i]
                b = X_disc[c, j]
                if a > 0 or b > 0:
                    joint[a, b] += 1.0
                    count += 1

            if count == 0:
                continue

            total = 0.0
            for ai in range(na):
                for bi in range(nb):
                    total += joint[ai, bi]

            px = np.zeros(na, dtype=np.float64)
            py = np.zeros(nb, dtype=np.float64)
            for ai in range(na):
                for bi in range(nb):
                    joint[ai, bi] /= total
                    px[ai] += joint[ai, bi]
                    py[bi] += joint[ai, bi]

            mi = 0.0
            for ai in range(na):
                for bi in range(nb):
                    if joint[ai, bi] > 0 and px[ai] > 0 and py[bi] > 0:
                        mi += joint[ai, bi] * np.log2(joint[ai, bi] / (px[ai] * py[bi]))

            mi_values[flat_idx] = mi

        return mi_values


def compute_mi_numba(X, gene_names, n_bins=10, signed=False):
    """Numba-accelerated MI computation."""
    if not HAS_NUMBA:
        raise ImportError("numba is required for compute_mi_numba. "
                          "Install with: pip install numba")

    X_disc, n_bins_per_gene = discretize_genes(X, n_bins=n_bins)
    n_genes = len(gene_names)

    mi_flat = _compute_all_mi_numba(X_disc, n_bins_per_gene, n_genes)

    if signed:
        if issparse(X):
            X_dense = np.asarray(X.todense())
        else:
            X_dense = X
        corr_matrix = np.nan_to_num(np.corrcoef(X_dense.T), nan=0.0)

    mi_scores = collections.defaultdict(lambda: collections.defaultdict(float))
    pair_idx = 0
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            mi = mi_flat[pair_idx]
            if signed:
                mi = np.sign(corr_matrix[i, j]) * mi
            gene_a = gene_names[i]
            gene_b = gene_names[j]
            mi_scores[gene_a][gene_b] = round(float(mi), 5)
            mi_scores[gene_b][gene_a] = round(float(mi), 5)
            pair_idx += 1

    return mi_scores


# ─── Tier C: GPU MI ────────────────────────────────────────────

def compute_mi_gpu(X, gene_names, n_bins=10, signed=False, device="cuda"):
    """
    GPU-accelerated MI using PyTorch scatter_add for joint histograms.
    """
    import torch

    X_disc, n_bins_per_gene = discretize_genes(X, n_bins=n_bins)
    n_cells, n_genes = X_disc.shape
    X_disc_t = torch.tensor(X_disc, dtype=torch.long, device=device)

    if signed:
        if issparse(X):
            X_dense = np.asarray(X.todense())
        else:
            X_dense = X
        corr_matrix = np.nan_to_num(np.corrcoef(X_dense.T), nan=0.0)

    mi_scores = collections.defaultdict(lambda: collections.defaultdict(float))

    for i in tqdm.tqdm(range(n_genes), desc="GPU MI"):
        na = int(n_bins_per_gene[i])
        if na <= 1:
            continue

        col_a = X_disc_t[:, i]

        j_indices = list(range(i + 1, n_genes))
        if not j_indices:
            continue

        cols_b = X_disc_t[:, j_indices]
        mask = (col_a.unsqueeze(1) > 0) | (cols_b > 0)

        for local_j, j in enumerate(j_indices):
            nb = int(n_bins_per_gene[j])
            if nb <= 1:
                continue

            m = mask[:, local_j]
            a_vals = col_a[m]
            b_vals = cols_b[m, local_j]

            if a_vals.numel() == 0:
                continue

            flat_idx = a_vals * nb + b_vals
            joint = torch.zeros(na * nb, dtype=torch.float32, device=device)
            joint.scatter_add_(0, flat_idx.long(), torch.ones_like(flat_idx, dtype=torch.float32))

            joint = joint.reshape(na, nb)

            total = joint.sum()
            if total == 0:
                continue
            joint /= total
            px = joint.sum(dim=1)
            py = joint.sum(dim=0)
            px_py = torch.outer(px, py)

            nz = (joint > 0) & (px_py > 0)
            mi = (joint[nz] * torch.log2(joint[nz] / px_py[nz])).sum().item()

            if signed:
                mi = float(np.sign(corr_matrix[i, j])) * mi

            gene_a = gene_names[i]
            gene_b = gene_names[j]
            mi_scores[gene_a][gene_b] = round(mi, 5)
            mi_scores[gene_b][gene_a] = round(mi, 5)

    return mi_scores


# ─── Tier D: Rust MI ───────────────────────────────────────────

def compute_mi_rust(X, gene_names, n_bins=10, signed=False):
    """Rust-accelerated MI computation via PyO3."""
    if not HAS_RUST:
        raise ImportError("Rust backend not available. "
                          "Build with: maturin develop --release")

    X_disc, n_bins_per_gene = discretize_genes(X, n_bins=n_bins)

    corr_signs = None
    if signed:
        if issparse(X):
            X_dense = np.asarray(X.todense())
        else:
            X_dense = X
        corr_signs = np.nan_to_num(np.corrcoef(X_dense.T), nan=0.0).astype(np.float32)

    triples = _rust_mi_pairs(X_disc, n_bins_per_gene, corr_signs)

    mi_scores = collections.defaultdict(lambda: collections.defaultdict(float))
    for i, j, mi in triples:
        gene_a = gene_names[i]
        gene_b = gene_names[j]
        mi_scores[gene_a][gene_b] = round(mi, 5)
        mi_scores[gene_b][gene_a] = round(mi, 5)

    return mi_scores


# ─── Target function registry ──────────────────────────────────

TARGETS = {}


def register_target(name):
    """Decorator to register a target function."""
    def wrapper(fn):
        TARGETS[name] = fn
        return fn
    return wrapper


def get_target_function(name):
    if name not in TARGETS:
        available = ", ".join(sorted(TARGETS.keys()))
        raise ValueError(f"Unknown target '{name}'. Available: {available}")
    return TARGETS[name]


# ─── Built-in targets ──────────────────────────────────────────

@register_target("mi")
def target_mi(X, gene_names, signed=False, backend="auto",
              device="cpu", n_bins=10, **kwargs):
    """Mutual information (optionally signed by Pearson correlation)."""
    if backend == "auto":
        if device == "cuda":
            return compute_mi_gpu(X, gene_names, n_bins=n_bins,
                                  signed=signed, device=device)
        elif HAS_RUST:
            return compute_mi_rust(X, gene_names, n_bins=n_bins,
                                   signed=signed)
        elif HAS_NUMBA:
            return compute_mi_numba(X, gene_names, n_bins=n_bins,
                                    signed=signed)
        else:
            return compute_mi_vectorized(X, gene_names, n_bins=n_bins,
                                         signed=signed)
    elif backend == "rust":
        return compute_mi_rust(X, gene_names, n_bins=n_bins,
                               signed=signed)
    elif backend == "numpy":
        return compute_mi_vectorized(X, gene_names, n_bins=n_bins,
                                     signed=signed)
    elif backend == "numba":
        return compute_mi_numba(X, gene_names, n_bins=n_bins,
                                signed=signed)
    elif backend == "gpu":
        return compute_mi_gpu(X, gene_names, n_bins=n_bins,
                              signed=signed, device=device)
    else:
        raise ValueError(f"Unknown MI backend: {backend}")


@register_target("pearson")
def target_pearson(X, gene_names, **kwargs):
    """Pearson correlation between all gene pairs."""
    if issparse(X):
        X = np.asarray(X.todense())
    corr = np.corrcoef(X.T)
    return _matrix_to_score_dict(corr, gene_names)


@register_target("spearman")
def target_spearman(X, gene_names, **kwargs):
    """Spearman rank correlation between all gene pairs."""
    if issparse(X):
        X = np.asarray(X.todense())
    corr, _ = spearmanr(X)
    if corr.ndim == 0:
        corr = np.array([[1.0, corr], [corr, 1.0]])
    return _matrix_to_score_dict(corr, gene_names)


@register_target("jaccard")
def target_jaccard(X, gene_names, **kwargs):
    """Jaccard index on binarized expression (gene detected / not detected)."""
    if issparse(X):
        binary = (X > 0).astype(np.float32)
        binary_dense = np.asarray(binary.todense())
    else:
        binary_dense = (X > 0).astype(np.float32)

    intersection = binary_dense.T @ binary_dense
    sums = binary_dense.sum(axis=0)
    union = sums[:, None] + sums[None, :] - intersection
    union[union == 0] = 1
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)
    return _matrix_to_score_dict(np.array(jaccard), gene_names)


@register_target("cosine")
def target_cosine(X, gene_names, **kwargs):
    """Cosine similarity between gene expression vectors (each gene is a vector across cells)."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    if issparse(X):
        sim = cos_sim(X.T)
    else:
        sim = cos_sim(X.T)
    np.fill_diagonal(sim, 0)
    return _matrix_to_score_dict(sim, gene_names)


def _matrix_to_score_dict(matrix, gene_names):
    """Convert a symmetric score matrix to nested dict."""
    scores = collections.defaultdict(lambda: collections.defaultdict(float))
    n = len(gene_names)
    for i in range(n):
        for j in range(n):
            if i != j:
                scores[gene_names[i]][gene_names[j]] = round(float(matrix[i, j]), 5)
    return scores


# ─── Import graph targets to trigger registration ──────────────
from . import _graph_targets  # noqa: F401, E402
