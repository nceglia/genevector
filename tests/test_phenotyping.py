"""Tests for the phenotyping improvements in CellEmbedding.

Covers the dataset_vector fix, opt-in debias/contrastive scoring, spatial label
propagation, count-adaptive cell-vector denoising, and the marker-dict QC.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

anndata = pytest.importorskip("anndata")
import pandas as pd

from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Tiny 3-type dataset (2 markers each) trained end-to-end."""
    rng = np.random.RandomState(0)
    n_per = 40
    types = ["A", "B", "C"]
    blocks, labels, coords = [], [], []
    for t, ct in enumerate(types):
        base = np.full((n_per, 6), 0.5)
        base[:, 2 * t:2 * t + 2] += 6.0  # this type's 2 markers high
        blocks.append(rng.poisson(base))
        labels += [ct] * n_per
        coords.append(rng.rand(n_per, 2) + np.array([t * 5.0, 0.0]))  # spatially separated
    X = csr_matrix(np.vstack(blocks).astype(np.float64))
    genes = [f"G{i}" for i in range(6)]
    adata = anndata.AnnData(X=X,
                            var=pd.DataFrame(index=genes),
                            obs=pd.DataFrame({"ct": labels},
                                             index=[f"C{i}" for i in range(len(labels))]))
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = np.vstack(coords)

    vec = str(tmp_path_factory.mktemp("vec") / "emb.vec")
    ds = GeneVectorDataset(adata.copy(), load_expression=True, signed_mi=True,
                           device="cpu", use_cache=False, mi_backend="numpy")
    gv = GeneVector(ds, output_file=vec, emb_dimension=20, c=100, gain=10,
                    init_ortho=True, device="cpu")
    gv.train(150, update_interval=75)
    embed = GeneEmbedding(vec, ds, vector="average")
    cembed = CellEmbedding(ds, embed)
    agv = cembed.get_adata()
    markers = {"A": ["G0", "G1"], "B": ["G2", "G3"], "C": ["G4", "G5"]}
    return adata, embed, cembed, agv, markers


def test_dataset_vector_is_nonzero(trained):
    _, _, cembed, _, _ = trained
    # previously initialised to zeros (bug); now the mean cell vector.
    assert np.linalg.norm(cembed.dataset_vector) > 0


def test_phenotype_probability_default(trained):
    _, _, cembed, agv, markers = trained
    out = cembed.phenotype_probability(agv, markers, temperature=0.05, target_col="gv")
    assert "gv" in out.obs
    # probability columns sum to ~1 per cell
    pcols = [c for c in out.obs.columns if "Pseudo-probability" in c]
    assert len(pcols) == 3
    s = out.obs[pcols].to_numpy().sum(1)
    npt = np.testing
    npt.assert_allclose(s, 1.0, atol=1e-5)
    # recovers the planted structure reasonably
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(out.obs["ct"], out.obs["gv"]) > 0.5


def test_debias_and_contrastive_run(trained):
    _, _, cembed, agv, markers = trained
    o1 = cembed.phenotype_probability(agv, markers, temperature=0.05, target_col="d", debias=0.5)
    o2 = cembed.phenotype_probability(agv, markers, temperature=0.05, target_col="c", contrastive=True)
    assert set(o1.obs["d"]) <= set(markers)
    assert set(o2.obs["c"]) <= set(markers)


def test_label_propagation_runs_and_preserves_simplex(trained):
    _, _, cembed, agv, markers = trained
    W = cembed._row_normalize_graph  # ensure helper exists
    from sklearn.neighbors import kneighbors_graph
    G = kneighbors_graph(agv.obsm["spatial"], 5, mode="connectivity")
    G = ((G + G.T) > 0).astype(float).tocsr()
    out = cembed.phenotype_probability(agv, markers, temperature=0.05, target_col="lp",
                                       lp_graph=G, lp_alpha=0.4, lp_iter=3)
    pcols = [c for c in out.obs.columns if "Pseudo-probability" in c]
    np.testing.assert_allclose(out.obs[pcols].to_numpy().sum(1), 1.0, atol=1e-5)


def test_denoise_cell_vectors(trained):
    adata, embed, cembed, agv, markers = trained
    # fresh cembed so we don't mutate the shared fixture's matrix
    ds = GeneVectorDataset(adata.copy(), load_expression=True, device="cpu",
                           use_cache=False, mi_backend="numpy")
    ds.mi_scores = {g: {h: 0.0 for h in ds.data.genes if h != g} for g in ds.data.genes}
    ce = CellEmbedding(ds, embed)
    from sklearn.neighbors import kneighbors_graph
    coords = adata.obsm["spatial"]
    G = kneighbors_graph(coords, 5, mode="connectivity")
    G = ((G + G.T) > 0).astype(float).tocsr()
    before = np.array(ce.matrix).copy()
    ce.denoise_cell_vectors(G, adaptive=True, counts=np.asarray(adata.layers["counts"].sum(1)).ravel())
    after = np.array(ce.matrix)
    assert after.shape == before.shape
    assert not np.allclose(after, before)  # vectors actually changed
    assert hasattr(ce, "uncorrected_matrix")


def test_qc_marker_dict_flags(trained):
    adata, _, cembed, _, markers = trained
    bad = dict(markers)
    bad["A"] = ["G0", "NOTAGENE"]      # absent marker
    bad["B"] = ["G2", "G4"]            # G4 is really a C marker -> off_target
    qc = cembed.qc_marker_dict(adata, bad, layer="counts", verbose=False)
    assert {"phenotype", "marker", "in_panel", "specificity", "flag"} <= set(qc.columns)
    assert (qc[qc.marker == "NOTAGENE"]["flag"] == "absent").all()
    g4 = qc[(qc.phenotype == "B") & (qc.marker == "G4")]["flag"].iloc[0]
    assert g4.startswith("off_target")
