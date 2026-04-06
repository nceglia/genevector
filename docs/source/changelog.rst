Changelog
=========

v1.0.0 (2026)
-------------

Major refactor and first stable release.

- **Pluggable co-expression targets**: MI, Pearson, Spearman, Jaccard, cosine, and custom callables
- **MI backend dispatch**: automatic selection from Rust > Numba > NumPy, with GPU (CUDA) support
- **Rust backend**: PyO3 + rayon for native parallel MI computation
- **Score caching**: computed targets cached to ``~/.genevector/cache/`` with hash-based keys
- **Graph-aware targets**: ``graph_xcorr`` for cross-correlation on any graph (spatial, TCR, custom)
- **Aggregation registry**: composable graph aggregations (mean, with more planned)
- **Vectorized tensor construction**: replaced O(G²) Python loop with numpy meshgrid
- **NaN handling**: ``corrcoef`` results sanitized across all backends
- **GPU float32**: MPS compatibility for Apple Silicon
- **CI/CD**: GitHub Actions for testing (Python 3.9-3.12) and automated PyPI releases
- **Documentation**: Sphinx with Monokai theme, full API docs, NumPy-style docstrings

v0.3.0
------

- Added signed MI (directional co-expression)
- PyPI packaging with setuptools

v0.2.0
------

- Initial public release
- Mutual information computation
- Gene and cell embeddings
- Cell type annotation via phenotype_probability
