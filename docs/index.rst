Welcome to GeneVector's documentation!
======================================

GeneVector is a Python library for single-cell RNA sequencing analysis that learns
distributed gene representations using a neural embedding approach. It enables gene
co-expression analysis, cell type annotation, and metagene discovery through vector
arithmetic operations.

Key features:

- Gene embeddings from co-expression patterns (mutual information, correlation, or custom metrics)
- Automated cell type annotation using marker gene sets
- Metagene discovery through embedding clustering
- Graph-aware spatial and TCR targets for spatially-resolved and immune profiling data
- Multiple MI computation backends: Rust, Numba, GPU (CUDA), and NumPy
- Score caching for instant re-runs

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/genevector

API Reference
=============

* :ref:`genindex`
* :ref:`modindex`
