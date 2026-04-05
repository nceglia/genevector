def compute_mi_pairs(
    x_disc: "numpy.ndarray",
    n_bins_per_gene: "numpy.ndarray",
    corr_signs: "numpy.ndarray | None" = None,
) -> list[tuple[int, int, float]]: ...
