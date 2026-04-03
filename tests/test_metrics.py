import numpy as np
from scipy.sparse import csr_matrix
from genevector.metrics import (
    discretize_genes,
    compute_mi_vectorized,
    target_pearson,
    target_spearman,
    target_jaccard,
    target_cosine,
    get_target_function,
    TARGETS,
)


def _make_test_data(n_cells=500, n_genes=50, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    X[X < 1] = 0
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return csr_matrix(X), gene_names


def test_discretize_output_shape():
    X, genes = _make_test_data()
    X_disc, n_bins = discretize_genes(X, n_bins=10)
    assert X_disc.shape == X.shape
    assert len(n_bins) == X.shape[1]
    assert X_disc.dtype == np.int32
    # zero expression should map to bin 0
    X_dense = np.asarray(X.todense())
    assert np.all(X_disc[X_dense == 0] == 0)


def test_mi_symmetric():
    X, genes = _make_test_data(n_genes=20)
    scores = compute_mi_vectorized(X, genes, n_bins=5)
    for g1 in scores:
        for g2 in scores[g1]:
            assert abs(scores[g1][g2] - scores[g2][g1]) < 1e-5


def test_mi_nonnegative_unsigned():
    X, genes = _make_test_data(n_genes=20)
    scores = compute_mi_vectorized(X, genes, n_bins=5, signed=False)
    for g1 in scores:
        for g2 in scores[g1]:
            assert scores[g1][g2] >= 0


def test_pearson_range():
    X, genes = _make_test_data(n_genes=10)
    scores = target_pearson(X, genes)
    for g1 in scores:
        for g2 in scores[g1]:
            assert -1 <= scores[g1][g2] <= 1


def test_spearman_range():
    X, genes = _make_test_data(n_genes=10)
    scores = target_spearman(X, genes)
    for g1 in scores:
        for g2 in scores[g1]:
            assert -1 <= scores[g1][g2] <= 1


def test_jaccard_range():
    X, genes = _make_test_data(n_genes=10)
    scores = target_jaccard(X, genes)
    for g1 in scores:
        for g2 in scores[g1]:
            assert 0 <= scores[g1][g2] <= 1


def test_cosine_range():
    X, genes = _make_test_data(n_genes=10)
    scores = target_cosine(X, genes)
    for g1 in scores:
        for g2 in scores[g1]:
            assert -1 <= scores[g1][g2] <= 1


def test_registry():
    assert "mi" in TARGETS
    assert "pearson" in TARGETS
    assert "spearman" in TARGETS
    assert "jaccard" in TARGETS
    assert "cosine" in TARGETS


def test_custom_target():
    from genevector.metrics import register_target
    import collections

    @register_target("constant")
    def target_constant(X, gene_names, **kwargs):
        scores = collections.defaultdict(lambda: collections.defaultdict(float))
        for g1 in gene_names:
            for g2 in gene_names:
                if g1 != g2:
                    scores[g1][g2] = 1.0
        return scores

    fn = get_target_function("constant")
    assert fn is not None
    X, genes = _make_test_data(n_genes=5)
    scores = fn(X, genes)
    assert scores[genes[0]][genes[1]] == 1.0
