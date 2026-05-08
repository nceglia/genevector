"""Render one PNG per synthetic template at default kwargs.

Outputs:
    docs/_images/template_paracrine.png
    docs/_images/template_niche.png
    docs/_images/template_gradient.png
    docs/_images/template_pathology.png

Each PNG is a single 2D scatter colored by phenotype.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from genevector.benchmarks.synthetic import (
    build_gradient_dataset,
    build_niche_dataset,
    build_paracrine_dataset,
    build_pathology,
)


SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "_images"


def _scatter(adata, title, out_path):
    coords = adata.obsm["spatial"]
    pheno = adata.obs["phenotype"].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    for label in sorted(set(pheno)):
        mask = pheno == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=4, alpha=0.6, label=f"{label} (n={int(mask.sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, frameon=False, markerscale=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    a, _ = build_paracrine_dataset(n_cells_per_type=800, seed=SEED)
    _scatter(a, "build_paracrine_dataset (defaults)", OUT_DIR / "template_paracrine.png")

    a, _ = build_niche_dataset(n_tumor_cells=800, n_t_cells=800, seed=SEED)
    _scatter(a, "build_niche_dataset (defaults)", OUT_DIR / "template_niche.png")

    a, _ = build_gradient_dataset(n_cells=1500, seed=SEED)
    _scatter(a, "build_gradient_dataset (defaults)", OUT_DIR / "template_gradient.png")

    a, _ = build_pathology(layout_kwargs={"num_cells": 3000}, seed=SEED)
    _scatter(a, "build_pathology (defaults, n=3000)", OUT_DIR / "template_pathology.png")


if __name__ == "__main__":
    main()
