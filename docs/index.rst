GeneVector
==========

.. image:: https://badge.fury.io/py/genevector.svg
   :target: https://badge.fury.io/py/genevector
.. image:: https://github.com/nceglia/genevector/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/nceglia/genevector/actions/workflows/tests.yml
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

GeneVector is a Python library for single-cell RNA sequencing analysis that learns
distributed gene representations using a neural embedding approach. It enables gene
co-expression analysis, cell type annotation, and metagene discovery through vector
arithmetic operations.

Key Features
------------

- **Gene embeddings** from co-expression patterns using mutual information, correlation, or custom metrics
- **Automated cell type annotation** using marker gene sets with probabilistic assignment
- **Metagene discovery** through embedding clustering
- **Graph-aware targets** for spatial transcriptomics and TCR/immune profiling data
- **High-performance backends**: Rust (PyO3), Numba JIT, CUDA GPU, and vectorized NumPy
- **Score caching** to disk for instant re-runs
- **Vector arithmetic** for intuitive gene relationship analysis

.. code-block:: python

   from genevector.data import GeneVectorDataset
   from genevector.model import GeneVector
   from genevector.embedding import GeneEmbedding, CellEmbedding

   dataset = GeneVectorDataset(adata)
   model = GeneVector(dataset, output_file="genes.vec", emb_dimension=100)
   model.train(1000, threshold=1e-6)

   gene_embed = GeneEmbedding("genes.vec", dataset, vector="average")
   cell_embed = CellEmbedding(dataset, gene_embed)
   adata = cell_embed.get_adata()

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/installation
   source/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   source/backends
   source/targets

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   source/api/data
   source/api/model
   source/api/embedding
   source/api/metrics
   source/api/cache
   source/api/aggregation

.. toctree::
   :maxdepth: 1
   :caption: About

   source/citation
   source/changelog
