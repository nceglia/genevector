MI Computation Backends
=======================

GeneVector computes pairwise co-expression scores between all genes as training
targets. For mutual information — the default and most principled target — this
involves computing MI for every gene pair, which is O(G² × N) where G is the
number of genes and N is the number of cells.

To make this tractable for real datasets (2,000+ genes, 50,000+ cells), GeneVector
provides multiple computation backends with automatic dispatch.

How MI Computation Works
------------------------

**Step 1: Discretization**

Continuous gene expression values are discretized into integer bins per gene.
Zero expression maps to bin 0. Nonzero values are split into quantile-based
bins (default 10). This is done once for all genes via ``discretize_genes()``.

**Step 2: Joint Histogram**

For each gene pair (A, B), a joint histogram is built by counting cells in each
(bin_A, bin_B) combination. Only cells where at least one gene is expressed
contribute.

**Step 3: MI from Joint Distribution**

.. math::

   MI(A, B) = \sum_{a,b} p(a,b) \log_2 \frac{p(a,b)}{p(a) \cdot p(b)}

where :math:`p(a,b)` is the joint probability, and :math:`p(a)`, :math:`p(b)` are
marginals.

**Step 4: Signing (optional)**

When ``signed_mi=True`` (default), MI is multiplied by the sign of the Pearson
correlation between the two genes. This produces directional MI: positive for
co-expressed genes, negative for anti-correlated genes.

Backend Dispatch
----------------

When ``mi_backend="auto"`` (the default), GeneVector selects the fastest available
backend in this order:

.. list-table::
   :header-rows: 1
   :widths: 10 15 20 25 30

   * - Priority
     - Backend
     - Speedup
     - How It Works
     - Availability
   * - 1
     - **CUDA GPU**
     - ~200-1000×
     - PyTorch scatter-based histograms
     - Requires ``device="cuda"``
   * - 2
     - **Rust**
     - ~100-800×
     - Native compiled, rayon parallelism
     - Requires ``maturin develop --release``
   * - 3
     - **Numba**
     - ~100-500×
     - JIT compiled, parallel across CPU cores
     - Requires ``pip install numba``
   * - 4
     - **NumPy**
     - ~10-30×
     - Vectorized discretization, Python pair loop
     - Always available

All backends produce identical results (within floating-point precision) — the
same discretization, the same histogram, the same MI formula. The only difference
is speed.

Selecting a Backend
-------------------

.. code-block:: python

   # Auto-select (recommended)
   dataset = GeneVectorDataset(adata, mi_backend="auto")

   # Force a specific backend
   dataset = GeneVectorDataset(adata, mi_backend="rust")
   dataset = GeneVectorDataset(adata, mi_backend="numba")
   dataset = GeneVectorDataset(adata, mi_backend="numpy")
   dataset = GeneVectorDataset(adata, mi_backend="gpu", device="cuda")

Checking Available Backends
---------------------------

.. code-block:: python

   from genevector.metrics import HAS_NUMBA

   try:
       from genevector._rust import compute_mi_pairs
       print("Rust backend: available")
   except ImportError:
       print("Rust backend: not installed")

   print(f"Numba backend: {'available' if HAS_NUMBA else 'not installed'}")

   import torch
   print(f"CUDA backend: {'available' if torch.cuda.is_available() else 'not available'}")

Practical Runtimes
------------------

For 2,000 genes × 5,000 cells (~2M gene pairs):

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Backend
     - Time
   * - Legacy (v0.2)
     - ~26 hours
   * - NumPy
     - ~3 hours
   * - Numba
     - ~2-3 minutes
   * - Rust
     - ~1-2 minutes

.. note::

   Scores are cached to ``~/.genevector/cache/`` after the first run.
   Subsequent runs with the same data and parameters load instantly.

Architecture
------------

The backend dispatch lives in ``genevector/metrics.py``. Each backend implements
the same interface:

.. code-block:: python

   def compute_mi_BACKEND(X, gene_names, n_bins=10, signed=False) -> dict[str, dict[str, float]]

All backends share the ``discretize_genes()`` function for the binning step.
The Rust backend (``src/lib.rs``) receives the pre-discretized integer matrix
from Python via zero-copy numpy arrays (PyO3 + numpy-rs), then parallelizes
the histogram + MI computation across gene pairs using rayon.

.. tip::

   For datasets larger than 2,000 genes, use the entropy-based gene filter
   ``GeneVectorDataset.quality_control(adata, entropy_threshold=1.0)`` to
   reduce gene count before MI computation. The matrix-based targets
   (Pearson, Spearman, Jaccard, cosine) compute in seconds regardless of
   gene count since they use BLAS-backed matrix operations.
