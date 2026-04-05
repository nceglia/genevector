# GeneVector

[![PyPI version](https://badge.fury.io/py/genevector.svg)](https://badge.fury.io/py/genevector)
[![Documentation Status](https://readthedocs.org/projects/genevector/badge/?version=latest)](https://genevector.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GeneVector Logo](https://github.com/nceglia/genevector/blob/main/logo.png?raw=true)

GeneVector is a Python library for single-cell RNA sequencing analysis that learns distributed gene representations using a neural embedding approach. It enables gene co-expression analysis, cell type annotation, and metagene discovery through vector arithmetic operations.

## Key Features

- **Gene Embeddings**: Learn distributed representations of genes based on co-expression patterns
- **Cell Type Annotation**: Automated cell type assignment using marker gene sets
- **Metagene Discovery**: Identify functionally related gene modules through clustering
- **Vector Arithmetic**: Perform gene relationship analysis using vector operations
- **Batch Correction**: Fast batch effect correction for multi-sample datasets
- **Pluggable Targets**: Train on mutual information, Pearson, Spearman, Jaccard, cosine, or custom metrics
- **High Performance**: Numba JIT, Rust native, and CUDA GPU backends for MI computation
- **Smart Caching**: Computed scores cached to disk, instant reload on re-runs

![Framework Overview](https://github.com/nceglia/genevector/blob/main/framework.png?raw=true)

## Installation

### From PyPI (Recommended)
```bash
pip install genevector
```

### With Numba Acceleration (Recommended)
```bash
pip install genevector[fast]
```

### From Source
```bash
git clone https://github.com/nceglia/genevector.git
cd genevector
pip install -e .

# with numba support
pip install -e ".[fast]"

# with rust backend (requires Rust toolchain)
pip install maturin
maturin develop --release
```

### Dependencies
- Python ≥ 3.9
- PyTorch
- Scanpy
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn

Optional: `numba` for JIT-accelerated MI computation (strongly recommended).

## Quick Start

```python
import scanpy as sc
from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Create dataset (auto-selects fastest MI backend)
dataset = GeneVectorDataset(adata)

# Train the model
model = GeneVector(dataset, output_file="genes.vec", emb_dimension=100)
model.train(1000, threshold=1e-6)

# Load gene embeddings
gene_embed = GeneEmbedding("genes.vec", dataset, vector="average")

# Generate cell embeddings
cell_embed = CellEmbedding(dataset, gene_embed)
adata_embedded = cell_embed.get_adata()

# Analyze gene similarities
similarities = gene_embed.compute_similarities("CD8A")
gene_embed.plot_similarities("CD8A", n_genes=10)
```

## MI Computation Backends

GeneVector computes pairwise co-expression scores between all genes as the training target. For mutual information (the default target), multiple backends are available with automatic dispatch.

### Dispatch Priority

When `mi_backend="auto"` (the default), GeneVector selects the fastest available backend:

| Priority | Backend | Speedup vs Legacy | How It Works | Availability |
|----------|---------|-------------------|--------------|--------------|
| 1 | **CUDA GPU** | ~200-1000x | PyTorch scatter-based histograms on GPU | Requires `device="cuda"` and NVIDIA GPU |
| 2 | **Rust** | ~100-800x | Native compiled with rayon parallelism | Requires `maturin develop --release` build |
| 3 | **Numba** | ~100-500x | JIT-compiled, parallel across CPU cores | Requires `pip install numba` |
| 4 | **NumPy** | ~10-30x | Vectorized discretization, Python pair loop | Always available (default fallback) |

### Selecting a Backend

```python
# Auto-select (recommended): picks the fastest available
dataset = GeneVectorDataset(adata, mi_backend="auto")

# Force a specific backend
dataset = GeneVectorDataset(adata, mi_backend="rust")
dataset = GeneVectorDataset(adata, mi_backend="numba")
dataset = GeneVectorDataset(adata, mi_backend="numpy")
dataset = GeneVectorDataset(adata, mi_backend="gpu", device="cuda")
```

### Check Available Backends

```python
from genevector.metrics import HAS_NUMBA

# Rust availability (if built)
try:
    from genevector._rust import compute_mi_pairs
    print("Rust backend: available")
except ImportError:
    print("Rust backend: not installed")

print(f"Numba backend: {'available' if HAS_NUMBA else 'not installed'}")

import torch
print(f"CUDA backend: {'available' if torch.cuda.is_available() else 'not available'}")
```

### Practical Runtimes (Approximate)

For 2,000 genes × 5,000 cells (~2M gene pairs):

| Backend | Time |
|---------|------|
| Legacy (v0.2) | ~26 hours |
| NumPy | ~3 hours |
| Numba | ~2-3 minutes |
| Rust | ~1-2 minutes |

Scores are cached to `~/.genevector/cache/` after the first run. Subsequent runs with the same data load instantly.

## Co-expression Targets

GeneVector can train on different co-expression metrics beyond mutual information:

```python
# Default: signed mutual information
dataset = GeneVectorDataset(adata, target="mi", signed_mi=True)

# Pearson correlation
dataset = GeneVectorDataset(adata, target="pearson")

# Spearman rank correlation
dataset = GeneVectorDataset(adata, target="spearman")

# Jaccard index (binarized co-detection)
dataset = GeneVectorDataset(adata, target="jaccard")

# Cosine similarity between gene expression vectors
dataset = GeneVectorDataset(adata, target="cosine")

# Custom callable
def my_metric(X, gene_names, **kwargs):
    # must return dict[str, dict[str, float]]
    ...

dataset = GeneVectorDataset(adata, target=my_metric)
```

The `mi_backend` parameter only applies when `target="mi"`. The matrix-based targets (Pearson, Spearman, Jaccard, cosine) compute in seconds via BLAS regardless of gene count.

## Tutorials

See the `example/` directory for comprehensive workflows:

1. **PBMC workflow**: Identification of interferon stimulated metagene and cell type annotation
2. **TICA workflow**: Cell type assignment
3. **SPECTRUM workflow**: Vector arithmetic for site specific metagenes
4. **FITNESS workflow**: Identifying increasing metagenes in time series

## Detailed Usage Guide

### 1. Data Loading

GeneVector uses Scanpy AnnData objects and requires raw count data in the `.X` matrix. It's recommended to subset genes using the `seurat_v3` flavor in Scanpy or GeneVector's entropy-based quality control.

```python
from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding
import scanpy as sc

# Load and preprocess data
adata = sc.read_h5ad("your_data.h5ad")

# Option A: Scanpy's variable gene selection
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

# Option B: GeneVector's entropy-based gene filtering
adata = GeneVectorDataset.quality_control(adata, entropy_threshold=1.0)

# Create dataset
dataset = GeneVectorDataset(adata)
```

### 2. Model Training

Creating a `GeneVector` object computes the co-expression target (MI by default) and prepares training batches. Training time varies by dataset size — a 10k PBMC dataset trains in under 5 minutes. The `emb_dimension` parameter controls vector size (minimum 50 recommended).

```python
# Initialize model (triggers MI computation + batch preparation)
model = GeneVector(
    dataset,
    output_file="genes.vec",
    emb_dimension=100,
    device="cpu"
)

# Train for 1000 iterations or until convergence
model.train(1000, threshold=1e-6)

# Visualize training progress
model.plot()
```

### 3. Gene Embeddings Analysis

Training produces two vector files (input and output weights). Using the average of both weights is recommended for best results.

```python
# Load gene embeddings
gene_embed = GeneEmbedding("genes.vec", dataset, vector="average")

# Compute gene similarities
similarities_df = gene_embed.compute_similarities("CD8A")
gene_embed.plot_similarities("CD8A", n_genes=10)

# Generate metagenes through clustering
gene_adata = gene_embed.get_adata(resolution=40)
metagenes = gene_embed.get_metagenes(gene_adata)

# Visualize specific metagenes
gene_embed.plot_metagene(gene_adata, mg=metagenes[0])
```

### 4. Cell Embeddings

Generate cell embeddings using the trained gene vectors. The embeddings are stored in AnnData format with automatic UMAP generation.

```python
# Create cell embeddings
cell_embed = CellEmbedding(dataset, gene_embed)
adata_embedded = cell_embed.get_adata()

# Visualize with Scanpy
sc.pl.umap(adata_embedded)

# Optional: Batch correction
cell_embed.batch_correct(column="sample", reference="control")
adata_corrected = cell_embed.get_adata()
```

### 5. Cell Type Assignment

```python
# Define marker genes for cell types
markers = {
    "T Cell": ["CD3D", "CD3G", "CD3E", "TRAC", "IL32", "CD2"],
    "B/Plasma": ["CD79A", "CD79B", "MZB1", "CD19", "BANK1"],
    "Myeloid": ["LYZ", "CST3", "AIF1", "CD68", "C1QA", "C1QB", "C1QC"]
}

# Perform automated cell type assignment
annotated_adata = cell_embed.phenotype_probability(adata_embedded, markers)

# Visualize results
prob_cols = [col for col in annotated_adata.obs.columns if "Pseudo-probability" in col]
sc.pl.umap(annotated_adata, color=prob_cols + ["genevector"], size=25)
```

### 6. Caching

Computed co-expression scores are cached automatically. Control this with:

```python
# Enable caching (default)
dataset = GeneVectorDataset(adata, use_cache=True)

# Disable caching (always recompute)
dataset = GeneVectorDataset(adata, use_cache=False)

# Clear the cache
from genevector.cache import clear_cache
clear_cache()
```

Cache location: `~/.genevector/cache/`. Cache keys incorporate the expression matrix, gene list, target function, and all parameters, so different configurations never collide.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GeneVector in your research, please cite:

> Ceglia, N., Sethna, Z., Freeman, S.S. et al. Identification of transcriptional programs using dense vector representations defined by mutual information with GeneVector. *Nat Commun* **14**, 4400 (2023). https://doi.org/10.1038/s41467-023-39985-2

```bibtex
@article{ceglia2023genevector,
  title={Identification of transcriptional programs using dense vector representations defined by mutual information with GeneVector},
  author={Ceglia, Nicholas and Sethna, Zachary and Freeman, Samuel S and others},
  journal={Nature Communications},
  volume={14},
  pages={4400},
  year={2023},
  doi={10.1038/s41467-023-39985-2}
}
```

## Documentation

For detailed documentation and examples, visit: https://genevector.readthedocs.io