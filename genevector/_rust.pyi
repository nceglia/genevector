def compute_mi_pairs(
    x_disc: "numpy.ndarray",
    n_bins_per_gene: "numpy.ndarray",
    corr_signs: "numpy.ndarray | None" = None,
) -> list[tuple[int, int, float]]: ...


def compute_cross_mi_pairs(
    a_disc: "numpy.ndarray",
    na_bins: "numpy.ndarray",
    b_disc: "numpy.ndarray",
    nb_bins: "numpy.ndarray",
    corr_signs: "numpy.ndarray | None" = None,
) -> list[tuple[int, int, float]]: ...
