"""Convenience wrapper combining layout + AnnData conversion + diagnostic overlay."""

from __future__ import annotations

from .layout import generate_synthetic_data, create_anndata_from_synthetic
from .overlay import apply_overlay


_DEFAULT_LAYOUT_KWARGS = dict(
    num_cells=10000,
    tumor_fraction=0.6,
    stromal_fraction=0.1,
    t_fraction=0.1,
    b_fraction=0.1,
    myeloid_fraction=0.1,
    aggregates_fraction=0.5,
    infiltrated_fraction=0.1,
    scattered_tumor_fraction=0.05,
    tumor_clustered_fraction=0,
    border_width=5,
    num_markers=15,
    num_tumor_gradient=5,
    rare_t_cell=True,
)


def build_pathology(
    layout_kwargs: dict | None = None,
    overlay_kwargs: dict | None = None,
    seed: int = 42,
):
    """
    Build a synthetic spatial AnnData and apply the diagnostic overlay.

    Parameters
    ----------
    layout_kwargs : dict, optional
        Overrides for ``generate_synthetic_data``. Defaults to the FOV1
        seed-42 invocation.
    overlay_kwargs : dict, optional
        Overrides for ``apply_overlay`` (excluding ``seed``).
    seed : int
        Seed shared by the layout RNG and the overlay RNG (independent
        seeded streams).

    Returns
    -------
    adata : AnnData
        AnnData with extended ``var_names`` and overlaid phenotypes.
    ground_truth : dict
        JSON-serializable record of the overlay (see ``apply_overlay``).
    """
    merged_layout = dict(_DEFAULT_LAYOUT_KWARGS)
    if layout_kwargs:
        merged_layout.update(layout_kwargs)
    merged_layout["seed"] = seed

    merged_overlay = dict(overlay_kwargs) if overlay_kwargs else {}
    merged_overlay["seed"] = seed

    df, _ = generate_synthetic_data(**merged_layout)
    adata = create_anndata_from_synthetic(df)
    return apply_overlay(adata, **merged_overlay)
