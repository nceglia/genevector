"""Tests for genevector.benchmarks.synthetic.layout — vendored from grafiti."""

import numpy as np
import pandas as pd
import pytest

from genevector.benchmarks.synthetic import (
    generate_synthetic_data,
    create_anndata_from_synthetic,
)

try:
    import grafiti as gf  # noqa: F401
    HAS_GRAFITI = True
except ImportError:
    HAS_GRAFITI = False


FOV1_KWARGS = dict(
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
    seed=42,
)


def test_basic_generation():
    df, marker_assignments = generate_synthetic_data(num_cells=500, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(marker_assignments, dict)
    assert len(df) > 0
    assert len(marker_assignments) > 0
    for col in ("x", "y", "phenotype"):
        assert col in df.columns


def test_determinism():
    df1, ma1 = generate_synthetic_data(num_cells=500, seed=42)
    df2, ma2 = generate_synthetic_data(num_cells=500, seed=42)

    pd.testing.assert_frame_equal(df1, df2)
    assert ma1 == ma2


@pytest.mark.skipif(not HAS_GRAFITI, reason="grafiti not installed")
def test_grafiti_parity():
    import grafiti as gf

    df_vendor, ma_vendor = generate_synthetic_data(**FOV1_KWARGS)
    df_grafiti, ma_grafiti = gf.ds.generate_synthetic_data(**FOV1_KWARGS)

    pd.testing.assert_frame_equal(df_vendor, df_grafiti)
    assert ma_vendor == ma_grafiti

    adata_vendor = create_anndata_from_synthetic(df_vendor.copy())
    adata_grafiti = gf.ds.create_anndata_from_synthetic(df_grafiti.copy())

    X_v = adata_vendor.X.toarray() if hasattr(adata_vendor.X, "toarray") else np.asarray(adata_vendor.X)
    X_g = adata_grafiti.X.toarray() if hasattr(adata_grafiti.X, "toarray") else np.asarray(adata_grafiti.X)
    assert np.array_equal(X_v, X_g)

    pd.testing.assert_frame_equal(adata_vendor.obs, adata_grafiti.obs)
    assert list(adata_vendor.var.index) == list(adata_grafiti.var.index)
