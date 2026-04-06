Co-expression Targets
=====================

GeneVector can train gene embeddings on different co-expression metrics. The
target function defines what relationship between gene pairs the model learns
to reproduce with dot products in the embedding space.

Built-in Targets
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Target
     - Speed
     - Description
   * - ``mi``
     - varies
     - Mutual information (default). Captures nonlinear statistical dependence. Multiple backends available.
   * - ``pearson``
     - instant
     - Pearson correlation coefficient. Linear co-expression.
   * - ``spearman``
     - instant
     - Spearman rank correlation. Monotonic co-expression, robust to outliers.
   * - ``jaccard``
     - instant
     - Jaccard index on binarized expression (detected / not detected).
   * - ``cosine``
     - instant
     - Cosine similarity between gene expression vectors across cells.

Usage
-----

.. code-block:: python

   from genevector.data import GeneVectorDataset

   # Default: signed mutual information
   dataset = GeneVectorDataset(adata, target="mi", signed_mi=True)

   # Pearson correlation
   dataset = GeneVectorDataset(adata, target="pearson")

   # Spearman rank correlation
   dataset = GeneVectorDataset(adata, target="spearman")

   # Jaccard index
   dataset = GeneVectorDataset(adata, target="jaccard")

   # Cosine similarity
   dataset = GeneVectorDataset(adata, target="cosine")

The ``mi_backend`` parameter only applies when ``target="mi"``. The matrix-based
targets (Pearson, Spearman, Jaccard, cosine) compute in seconds via BLAS
regardless of gene count.

Graph-Aware Targets
-------------------

Graph-aware targets measure co-expression across graph neighbors rather than
within individual cells. The ``graph`` parameter accepts any scipy sparse
adjacency matrix — spatial neighbors, TCR similarity, or custom graphs.

.. code-block:: python

   import squidpy as sq

   # Build a spatial neighbor graph
   sq.gr.spatial_neighbors(adata, n_neighs=10, coord_type="generic")
   graph = adata.obsp["spatial_connectivities"]

   # Cross-correlation between self-expression and neighbor-aggregated expression
   dataset = GeneVectorDataset(adata, target="graph_xcorr",
       target_kwargs={"graph": graph, "aggr": "mean"})

The graph is **domain-agnostic** — the same target works on spatial, TCR, or any
graph topology:

.. code-block:: python

   from genevector.graphs import build_clonotype_graph

   # Same target, different graph
   clone_graph = build_clonotype_graph(adata, clone_key="clone_id")
   dataset = GeneVectorDataset(adata, target="graph_xcorr",
       target_kwargs={"graph": clone_graph})

Custom Targets
--------------

Register a custom target function:

.. code-block:: python

   from genevector.metrics import register_target

   @register_target("my_metric")
   def my_target(X, gene_names, **kwargs):
       # Compute pairwise scores
       # Must return dict[str, dict[str, float]]
       scores = {}
       # ... your computation ...
       return scores

   dataset = GeneVectorDataset(adata, target="my_metric")

Or pass a callable directly without registration:

.. code-block:: python

   dataset = GeneVectorDataset(adata,
       target=lambda X, names, **kw: my_score_function(X, names))

Caching
-------

All computed target scores are cached automatically to ``~/.genevector/cache/``.
Cache keys incorporate the expression matrix, gene list, target function name,
and all parameters — different configurations never collide.

.. code-block:: python

   # Disable caching
   dataset = GeneVectorDataset(adata, use_cache=False)

   # Clear the cache
   from genevector.cache import clear_cache
   clear_cache()
