"""Shared pytest fixtures for aggregation and graph-target tests."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def tiny_graph():
    return csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float64))


@pytest.fixture
def star_graph():
    adj = np.zeros((5, 5), dtype=np.float64)
    for i in range(1, 5):
        adj[0, i] = 1.0
        adj[i, 0] = 1.0
    return csr_matrix(adj)


@pytest.fixture
def ring_with_isolate():
    adj = np.zeros((6, 6), dtype=np.float64)
    for i in range(5):
        j = (i + 1) % 5
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return csr_matrix(adj)


@pytest.fixture
def tiny_X(tiny_graph):
    X = np.array([
        [1., 0.],
        [0., 2.],
        [3., 1.],
        [2., 4.],
    ], dtype=np.float64)
    return X, tiny_graph


@pytest.fixture
def uniform_X_star(star_graph):
    X = np.ones((5, 3), dtype=np.float64)
    return X, star_graph


@pytest.fixture
def random_X_ring(ring_with_isolate):
    X = np.random.RandomState(42).rand(6, 10)
    return X, ring_with_isolate


@pytest.fixture
def star_distances(star_graph):
    row, col = star_graph.nonzero()
    data = np.zeros(len(row), dtype=np.float64)
    for k in range(len(row)):
        leaf = max(row[k], col[k])
        data[k] = float(leaf)
    distances = csr_matrix((data, (row, col)), shape=(5, 5))
    return star_graph, distances


@pytest.fixture
def tiny_adata():
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.RandomState(42)
    X = csr_matrix(rng.poisson(2, size=(20, 6)).astype(np.float64))
    gene_names = [f"GENE{i}" for i in range(6)]
    cell_names = [f"CELL{i}" for i in range(20)]

    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
        obs=pd.DataFrame(index=cell_names),
    )

    conn = csr_matrix(rng.random((20, 20)) < 0.15, dtype=np.float64)
    conn = conn + conn.T
    conn.data[:] = 1.0
    conn.setdiag(0)
    conn.eliminate_zeros()
    adata.obsp["spatial_connectivities"] = conn

    dist_vals = rng.rand(conn.nnz) * 4.5 + 0.5
    distances = conn.copy()
    distances.data[:] = dist_vals
    adata.obsp["spatial_distances"] = distances

    return adata
