"""genevector/cache.py — target score caching with hash-based keys."""

import hashlib
import json
import os
import numpy as np
from scipy.sparse import issparse
import collections

CACHE_DIR = os.path.expanduser("~/.genevector/cache")


def _hash_array(X):
    """Fast hash of sparse or dense matrix."""
    if issparse(X):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(X.data).tobytes())
        h.update(np.ascontiguousarray(X.indices).tobytes())
        h.update(np.ascontiguousarray(X.indptr).tobytes())
        h.update(str(X.shape).encode())
        return h.hexdigest()[:16]
    else:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(X).tobytes())
        h.update(str(X.shape).encode())
        return h.hexdigest()[:16]


def compute_cache_key(X, gene_names, target_name, target_kwargs, signed_mi):
    """
    Deterministic hash for a given dataset + target configuration.

    Returns
    -------
    key : str
        A 32-char hex string uniquely identifying this computation.
    """
    h = hashlib.sha256()
    h.update(_hash_array(X).encode())
    h.update(json.dumps(sorted(gene_names)).encode())
    h.update(target_name.encode() if isinstance(target_name, str) else b"custom")
    h.update(json.dumps(target_kwargs, sort_keys=True, default=str).encode())
    h.update(str(signed_mi).encode())
    return h.hexdigest()[:32]


def get_cache_path(cache_key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"scores_{cache_key}.npz")


def save_scores(cache_key, mi_scores, gene_names):
    """
    Serialize score dict to a compressed npz file.

    Stores scores as a dense matrix + gene name list for fast load.
    """
    n = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    matrix = np.zeros((n, n), dtype=np.float32)

    for g1, inner in mi_scores.items():
        i = gene_to_idx.get(g1)
        if i is None:
            continue
        for g2, val in inner.items():
            j = gene_to_idx.get(g2)
            if j is None:
                continue
            matrix[i, j] = val

    path = get_cache_path(cache_key)
    np.savez_compressed(path, scores=matrix, genes=np.array(gene_names))
    print(f"Cached scores to {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return path


def load_scores(cache_key):
    """
    Load cached scores. Returns (mi_scores dict, gene_names) or (None, None).
    """
    path = get_cache_path(cache_key)
    if not os.path.exists(path):
        return None, None

    data = np.load(path, allow_pickle=False)
    matrix = data["scores"]
    gene_names = list(data["genes"])
    n = len(gene_names)

    scores = collections.defaultdict(lambda: collections.defaultdict(float))
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] != 0:
                scores[gene_names[i]][gene_names[j]] = round(float(matrix[i, j]), 5)

    print(f"Loaded cached scores from {path}")
    return scores, gene_names


def clear_cache():
    """Remove all cached score files."""
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")
