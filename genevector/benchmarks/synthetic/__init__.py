"""Synthetic spatial dataset templates for GeneVector benchmarks.

See docs/synthetic_templates.md for descriptions of each template.
"""

from __future__ import annotations

from ._shared import GROUND_TRUTH_VERSION, GroundTruth
from .gradient import build_gradient_dataset
from .layout import create_anndata_from_synthetic, generate_synthetic_data
from .niche import build_niche_dataset
from .overlay import apply_overlay
from .paracrine import build_paracrine_dataset
from .pathology import build_pathology


_TEMPLATES = {
    "pathology": {
        "builder": build_pathology,
        "description": (
            "Full grafiti-derived layout with paracrine, niche, T-rare, and "
            "housekeeping overlays."
        ),
    },
    "paracrine": {
        "builder": build_paracrine_dataset,
        "description": (
            "Two intermixed cell types with N ligand-receptor pairs and "
            "configurable mixing."
        ),
    },
    "niche": {
        "builder": build_niche_dataset,
        "description": (
            "Tumor blob with surrounding T cells; niche genes induced by "
            "local tumor density."
        ),
    },
    "gradient": {
        "builder": build_gradient_dataset,
        "description": (
            "1D axial pathology — cells along a line with smooth gradient "
            "and peaked gene expression."
        ),
    },
}


def list_templates() -> dict[str, str]:
    """Return template names mapped to one-line descriptions."""
    return {name: meta["description"] for name, meta in _TEMPLATES.items()}


__all__ = [
    "generate_synthetic_data",
    "create_anndata_from_synthetic",
    "apply_overlay",
    "build_pathology",
    "build_paracrine_dataset",
    "build_niche_dataset",
    "build_gradient_dataset",
    "list_templates",
    "GroundTruth",
    "GROUND_TRUTH_VERSION",
]
