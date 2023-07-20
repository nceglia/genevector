.. GeneVector documentation master file, created by
   sphinx-quickstart on Thu Jul 20 00:11:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GeneVector's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Overview
=============

GeneVector is a Python library for generating gene embeddings from mutual information between genes in single cell RNA-seq datasets. The library is built on top of PyTorch and Scanpy, making use of the AnnData format for representing single cell RNA-seq datasets. After generating gene-based embeddings by training a model, these embeddings can be loaded into GeneEmbedding and CellEmbedding objects to perform analyses include the identification of transcriptional programs (metagenes), cell type annotation, and batch correction. Visualization methods are included that make use of Scanpy's core plotting features.


API Reference
=============

* :ref:`genindex`
* :ref:`modindex`