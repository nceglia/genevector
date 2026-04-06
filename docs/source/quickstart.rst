Quick Start
===========

This guide walks through a minimal GeneVector workflow: loading data, training
gene embeddings, and annotating cell types.

Loading Data
------------

GeneVector uses Scanpy AnnData objects with raw count data in ``.X``:

.. code-block:: python

   import scanpy as sc
   from genevector.data import GeneVectorDataset

   adata = sc.read_h5ad("your_data.h5ad")

   # Filter to informative genes
   adata = GeneVectorDataset.quality_control(adata, entropy_threshold=1.0)

   # Create dataset (auto-selects fastest MI backend)
   dataset = GeneVectorDataset(adata)

Training
--------

.. code-block:: python

   from genevector.model import GeneVector

   model = GeneVector(dataset, output_file="genes.vec", emb_dimension=100)
   model.train(1000, threshold=1e-6)
   model.plot()  # visualize convergence

Gene Embeddings
---------------

.. code-block:: python

   from genevector.embedding import GeneEmbedding

   embed = GeneEmbedding("genes.vec", dataset, vector="average")

   # Find similar genes
   embed.compute_similarities("CD8A")
   embed.plot_similarities("CD8A", n_genes=10)

   # Discover metagenes
   gdata = embed.get_adata(resolution=40)
   metagenes = embed.get_metagenes(gdata)

Cell Embeddings & Annotation
-----------------------------

.. code-block:: python

   from genevector.embedding import CellEmbedding

   cell_embed = CellEmbedding(dataset, embed)
   adata = cell_embed.get_adata()

   # Define markers and annotate
   markers = {
       "T Cell": ["CD3D", "CD3G", "CD3E"],
       "B Cell": ["CD79A", "CD79B", "MZB1"],
       "Myeloid": ["LYZ", "CST3", "AIF1"],
   }
   adata = cell_embed.phenotype_probability(adata, markers)

   # Visualize
   import scanpy as sc
   sc.pl.umap(adata, color=["genevector"], size=25)
