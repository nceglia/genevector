"""Synthetic spatial dataset utilities for GeneVector benchmarks."""
from .layout import generate_synthetic_data, create_anndata_from_synthetic
from .overlay import apply_overlay
from .pathology import build_pathology

__all__ = [
    "generate_synthetic_data",
    "create_anndata_from_synthetic",
    "apply_overlay",
    "build_pathology",
]
