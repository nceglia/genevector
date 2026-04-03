import numpy as np
import os
from scipy.sparse import csr_matrix
from genevector.cache import compute_cache_key, save_scores, load_scores, clear_cache


def test_cache_roundtrip():
    genes = ["A", "B", "C"]
    scores = {
        "A": {"B": 0.5, "C": 0.3},
        "B": {"A": 0.5, "C": 0.1},
        "C": {"A": 0.3, "B": 0.1},
    }
    key = "test_roundtrip_key_00000000000000"
    save_scores(key, scores, genes)
    loaded, loaded_genes = load_scores(key)
    assert loaded_genes == genes
    for g1 in scores:
        for g2 in scores[g1]:
            assert abs(loaded[g1][g2] - scores[g1][g2]) < 1e-4


def test_cache_key_deterministic():
    X = csr_matrix(np.random.RandomState(0).rand(100, 10).astype(np.float32))
    genes = [f"G{i}" for i in range(10)]
    k1 = compute_cache_key(X, genes, "mi", {}, True)
    k2 = compute_cache_key(X, genes, "mi", {}, True)
    assert k1 == k2


def test_cache_key_changes():
    X = csr_matrix(np.random.RandomState(0).rand(100, 10).astype(np.float32))
    genes = [f"G{i}" for i in range(10)]
    k1 = compute_cache_key(X, genes, "mi", {}, True)
    k2 = compute_cache_key(X, genes, "pearson", {}, True)
    assert k1 != k2


def test_cache_miss():
    loaded, genes = load_scores("nonexistent_key_12345678901234")
    assert loaded is None
    assert genes is None
