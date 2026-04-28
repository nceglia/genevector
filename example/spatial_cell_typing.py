# %% Imports and dataset build
"""
Minimal spatial GeneVector + cell typing scratchpad.

- Builds a synthetic FOV via ``genevector.benchmarks.synthetic`` (seed=42).
- Builds a Delaunay spatial graph with squidpy (distance-based triangulation).
- Trains with target ``graph_xcorr`` using that graph.
- Runs ``CellEmbedding.phenotype_probability`` (softmax / sparsemax / normalized_exponential).

Open in VS Code / Cursor and run cells via ``#%%``, or: ``ipython -i example/spatial_cell_typing.py``.

Requires: scanpy, squidpy, anndata, and this package.
"""
from pathlib import Path

import numpy as np
import scanpy as sc
import squidpy as sq

from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding
from genevector.benchmarks.synthetic import (
    generate_synthetic_data,
    create_anndata_from_synthetic,
)

OUT_DIR = Path(__file__).resolve().parent
EMB_VEC = OUT_DIR / "spatial_cell_typing.vec"

SPATIAL_KEY = "spatial"

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=120, facecolor="white")

df, marker_assignments = generate_synthetic_data(
    num_cells=10000,
    tumor_fraction=0.6,
    stromal_fraction=0.1,
    t_fraction=0.1,
    b_fraction=0.1,
    myeloid_fraction=0.1,
    aggregates_fraction=0.5,
    infiltrated_fraction=0.1,
    scattered_tumor_fraction=0.05,
    tumor_clustered_fraction=0,
    border_width=5,
    num_markers=15,
    num_tumor_gradient=5,
    rare_t_cell=True,
    seed=42,
)
adata = create_anndata_from_synthetic(df)

# %% Spatial coordinates (squidpy expects float64 ndarray)
adata.obsm[SPATIAL_KEY] = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64)

# %% Delaunay spatial graph (squidpy)
sq.gr.spatial_neighbors(
    adata,
    spatial_key=SPATIAL_KEY,
    coord_type="generic",
    delaunay=True,
)

graph = adata.obsp["spatial_connectivities"]

# %% GeneVector dataset (spatial cross-correlation target)
dataset = GeneVectorDataset(
    adata,
    target="graph_xcorr",
    target_kwargs={"graph": graph, "aggr": "mean"},
    use_cache=True,
)

# %% Train gene embeddings (raise epochs / tighten threshold for production runs)
model = GeneVector(dataset, output_file=str(EMB_VEC), emb_dimension=64, device="cpu")
model.train(3000, threshold=1e-6)
model.plot()

# %% Cell embeddings + UMAP on X_genevector
gene_embed = GeneEmbedding(str(EMB_VEC), dataset, vector="average")
cell_embed = CellEmbedding(dataset, gene_embed)
adata_gv = cell_embed.get_adata(n_neighbors=30, min_dist=0.25)

# %% Marker-based cell typing — pull cell-type → marker map from the synthetic build
PHENOTYPE_MARKERS = marker_assignments

missing = sorted(
    {g for genes in PHENOTYPE_MARKERS.values() for g in genes} - set(adata_gv.var_names)
)
if missing:
    print("Markers not in var_names (update PHENOTYPE_MARKERS):", missing[:20], "...")

# Newer API: method in {"softmax", "sparsemax", "normalized_exponential"}
adata_gv = cell_embed.phenotype_probability(
    adata_gv,
    PHENOTYPE_MARKERS,
    method="normalized_exponential",
    target_col="genevector",
    temperature=0.05,
)

prob_cols = [c for c in adata_gv.obs.columns if c.endswith("Pseudo-probability")]
sc.pl.umap(adata_gv, color=["genevector"] + prob_cols[:6], wspace=0.35, ncols=3)

# %% Optional: optional ground-truth column if present
for truth_key in ("cell_type", "phenotype", "spatial_context", "leiden"):
    if truth_key in adata_gv.obs:
        sc.pl.umap(adata_gv, color=["genevector", truth_key], wspace=0.35)
        break
