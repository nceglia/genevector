# GeneVector

[![PyPI version](https://badge.fury.io/py/genevector.svg)](https://badge.fury.io/py/genevector)
[![Documentation Status](https://readthedocs.org/projects/genevector/badge/?version=latest)](https://genevector.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GeneVector Logo](https://github.com/nceglia/genevector/blob/main/logo.png?raw=true)

GeneVector is a powerful Python library for single-cell RNA sequencing analysis that learns distributed gene representations using a neural embedding approach. It enables advanced gene co-expression analysis, cell type annotation, and metagene discovery through vector arithmetic operations.

## Key Features

- **Gene Embeddings**: Learn distributed representations of genes based on co-expression patterns
- **Cell Type Annotation**: Automated cell type assignment using marker gene sets
- **Metagene Discovery**: Identify functionally related gene modules through clustering
- **Vector Arithmetic**: Perform intuitive gene relationship analysis using vector operations
- **Batch Correction**: Fast batch effect correction for multi-sample datasets
- **GPU Acceleration**: CUDA support for efficient training on large datasets

![Framework Overview](https://github.com/nceglia/genevector/blob/main/framework.png?raw=true)

## Installation

### From PyPI (Recommended)
```bash
pip install genevector
```

### From Source
```bash
git clone https://github.com/nceglia/genevector.git
cd genevector
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Dependencies
- Python ≥ 3.7
- PyTorch
- Scanpy
- NumPy
- Pandas
- Matplotlib

For GPU acceleration, ensure you have CUDA-compatible PyTorch installed.

## Quick Start

```python
import scanpy as sc
from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Create GeneVector dataset
dataset = GeneVectorDataset(adata, device="cuda")  # Use "cpu" if no GPU

# Train the model
model = GeneVector(dataset, output_file="genes.vec", emb_dimension=100)
model.train(1000)

# Load gene embeddings
gene_embed = GeneEmbedding("genes.vec", dataset, vector="average")

# Generate cell embeddings
cell_embed = CellEmbedding(dataset, gene_embed)
adata_embedded = cell_embed.get_adata()

# Analyze gene similarities
similarities = gene_embed.compute_similarities("CD8A")
gene_embed.plot_similarities("CD8A", n_genes=10)
```

## Tutorials

See the `example/` directory for comprehensive workflows:

1. **PBMC workflow**: Identification of interferon stimulated metagene and cell type annotation
2. **TICA workflow**: Cell type assignment  
3. **SPECTRUM workflow**: Vector arithmetic for site specific metagenes
4. **FITNESS workflow**: Identifying increasing metagenes in time series

## Detailed Usage Guide

### 1. Data Loading

GeneVector uses Scanpy AnnData objects and requires raw count data in the `.X` matrix. It's recommended to subset genes using the `seurat_v3` flavor in Scanpy.

```python
from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding
import scanpy as sc

# Load and preprocess data
adata = sc.read_h5ad("your_data.h5ad")
sc.pp.highly_variable_genes(adata, flavor="seurat_v3")

# Create GeneVector dataset
dataset = GeneVectorDataset(adata, device="cuda")  # Use "cpu" if no GPU
```

### 2. Model Training

Creating a GeneVector object computes mutual information between genes (up to 15 minutes for 250k cells). Training times vary by dataset size - a 10k PBMC dataset trains in under 5 minutes. The `emb_dimension` parameter controls vector size (minimum 50 recommended).

```python
# Initialize and train model
model = GeneVector(
    dataset,
    output_file="genes.vec",
    emb_dimension=100,
    threshold=1e-6,
    device="cuda"
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
import scanpy as sc
sc.pl.umap(adata_embedded)

# Optional: Batch correction (fast operation)
cell_embed.batch_correct(column="sample", reference="control")
adata_corrected = cell_embed.get_adata()
```

### 5. Metagene Scoring and Cell Type Assignment

```python
# Score cells by metagenes
gene_embed.score_metagenes(adata_embedded, metagenes)
gene_embed.plot_metagenes_scores(adata_embedded, metagenes, "cell_type")

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GeneVector in your research, please cite:

```bibtex
@software{genevector,
  author = {Nicholas Ceglia},
  title = {GeneVector: Single Cell Gene Embedding Library},
  url = {https://github.com/nceglia/genevector},
  version = {0.3.0},
  year = {2024}
}
```

## Documentation

For detailed documentation and examples, visit: https://genevector.readthedocs.io

---

*Additional analyses described in the manuscript, including comparisons to LDVAE and CellAssign, can be found in Jupyter notebooks in the examples directory. Results were computed for v0.0.1.*
