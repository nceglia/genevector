Synthetic dataset templates
============================

GeneVector ships with a small catalog of synthetic spatial transcriptomics
datasets, each modeling a different biological pathology. They share a common
output contract (AnnData + ground-truth dict) and are useful for benchmarking,
testing, and feature development across spatial-omics tools.

.. code-block:: python

   from genevector.benchmarks.synthetic import (
       build_paracrine_dataset,
       build_niche_dataset,
       build_gradient_dataset,
       build_pathology,
       list_templates,
   )

   print(list_templates())  # name → description map

   adata, ground_truth = build_paracrine_dataset(seed=42)

The full reference, including the ground-truth schema and per-template
parameter tables, lives in
`docs/synthetic_templates.md <https://github.com/nceglia/genevector/blob/main/docs/synthetic_templates.md>`_.

Templates
---------

- ``build_paracrine_dataset`` — two intermixed cell types with N
  ligand-receptor pairs and configurable mixing.
- ``build_niche_dataset`` — tumor blob with surrounding T cells; niche genes
  induced by local tumor density.
- ``build_gradient_dataset`` — 1D axial pathology, cells along a line with
  smooth monotone and peaked gene expression.
- ``build_pathology`` — full grafiti-derived FOV with paracrine, niche,
  T-rare, and housekeeping overlays composed.

Shared contract
---------------

Every builder returns ``(adata, ground_truth)`` where:

- ``adata.X`` is a ``scipy.sparse.csr_matrix`` of float counts.
- ``adata.obs["phenotype"]`` is categorical.
- ``adata.obsm["spatial"]`` is a ``(n_cells, 2)`` ``float64`` array.
- ``adata.var_names`` are uppercase, unique strings.

Ground-truth is a JSON-serializable dict with keys ``template``, ``version``,
``seed``, ``params``, ``phenotypes``, ``genes``, and ``pairs``. Schema
version is ``"2.0"``.
