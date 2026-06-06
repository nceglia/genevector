use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute MI for all upper-triangle gene pairs.
///
/// Takes the pre-discretized integer matrix (from Python's discretize_genes)
/// and returns a flat Vec of (gene_i, gene_j, mi_value) triples.
#[pyfunction]
#[pyo3(signature = (x_disc, n_bins_per_gene, corr_signs=None))]
fn compute_mi_pairs(
    x_disc: PyReadonlyArray2<i32>,
    n_bins_per_gene: PyReadonlyArray1<i32>,
    corr_signs: Option<PyReadonlyArray2<f32>>,
) -> Vec<(usize, usize, f64)> {
    let x = x_disc.as_array();
    let bins = n_bins_per_gene.as_array();
    let n_genes = x.ncols();
    let n_cells = x.nrows();

    // optional correlation sign matrix for signed MI
    let signs: Option<Vec<Vec<f32>>> = corr_signs.map(|arr| {
        let a = arr.as_array();
        (0..n_genes)
            .map(|i| (0..n_genes).map(|j| *a.get((i, j)).unwrap_or(&0.0)).collect())
            .collect()
    });

    // build list of pairs to process
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_genes {
        for j in (i + 1)..n_genes {
            if bins[i] > 1 && bins[j] > 1 {
                pairs.push((i, j));
            }
        }
    }

    // parallel MI computation via rayon
    pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let na = bins[i] as usize;
            let nb = bins[j] as usize;

            // build joint histogram
            let mut joint = vec![0u32; na * nb];
            let mut count = 0u32;
            for c in 0..n_cells {
                let a = x[[c, i]] as usize;
                let b = x[[c, j]] as usize;
                if a > 0 || b > 0 {
                    joint[a * nb + b] += 1;
                    count += 1;
                }
            }

            if count == 0 {
                return None;
            }

            let total = count as f64;

            // marginals
            let mut px = vec![0.0f64; na];
            let mut py = vec![0.0f64; nb];
            let mut joint_f = vec![0.0f64; na * nb];

            for ai in 0..na {
                for bi in 0..nb {
                    let p = joint[ai * nb + bi] as f64 / total;
                    joint_f[ai * nb + bi] = p;
                    px[ai] += p;
                    py[bi] += p;
                }
            }

            // MI
            let mut mi = 0.0f64;
            for ai in 0..na {
                for bi in 0..nb {
                    let pxy = joint_f[ai * nb + bi];
                    let px_py = px[ai] * py[bi];
                    if pxy > 0.0 && px_py > 0.0 {
                        mi += pxy * (pxy / px_py).log2();
                    }
                }
            }

            // apply correlation sign if provided
            if let Some(ref s) = signs {
                let sign = if s[i][j] >= 0.0 { 1.0 } else { -1.0 };
                mi *= sign;
            }

            Some((i, j, mi))
        })
        .collect()
}

/// Compute cross mutual information between every self gene and every neighbour gene.
///
/// `a_disc` is the discretized self-expression (cells x genes), `b_disc` the discretized
/// graph-neighbour-aggregated expression (cells x genes). Returns flat (i, j, mi) triples
/// where mi = MI(self gene i, neighbour gene j), over all ordered pairs i != j. Cells where
/// both bins are zero are dropped ((a>0)||(b>0) mask), matching the numpy/torch paths.
/// rayon-parallel across the P*P pairs — fast multi-core CPU graph_mi.
#[pyfunction]
#[pyo3(signature = (a_disc, na_bins, b_disc, nb_bins, corr_signs=None))]
fn compute_cross_mi_pairs(
    a_disc: PyReadonlyArray2<i32>,
    na_bins: PyReadonlyArray1<i32>,
    b_disc: PyReadonlyArray2<i32>,
    nb_bins: PyReadonlyArray1<i32>,
    corr_signs: Option<PyReadonlyArray2<f32>>,
) -> Vec<(usize, usize, f64)> {
    let a = a_disc.as_array();
    let b = b_disc.as_array();
    let abins = na_bins.as_array();
    let bbins = nb_bins.as_array();
    let n_genes = a.ncols();
    let n_cells = a.nrows();

    let signs: Option<Vec<Vec<f32>>> = corr_signs.map(|arr| {
        let m = arr.as_array();
        (0..n_genes)
            .map(|i| (0..n_genes).map(|j| *m.get((i, j)).unwrap_or(&0.0)).collect())
            .collect()
    });

    // all ordered self/neighbour pairs (diagonal is zeroed downstream)
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_genes {
        for j in 0..n_genes {
            if i != j && abins[i] > 1 && bbins[j] > 1 {
                pairs.push((i, j));
            }
        }
    }

    pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let na = abins[i] as usize;
            let nb = bbins[j] as usize;

            let mut joint = vec![0u32; na * nb];
            let mut count = 0u32;
            for c in 0..n_cells {
                let av = a[[c, i]] as usize;
                let bv = b[[c, j]] as usize;
                if av > 0 || bv > 0 {
                    joint[av * nb + bv] += 1;
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let total = count as f64;

            let mut px = vec![0.0f64; na];
            let mut py = vec![0.0f64; nb];
            let mut joint_f = vec![0.0f64; na * nb];
            for ai in 0..na {
                for bi in 0..nb {
                    let p = joint[ai * nb + bi] as f64 / total;
                    joint_f[ai * nb + bi] = p;
                    px[ai] += p;
                    py[bi] += p;
                }
            }

            let mut mi = 0.0f64;
            for ai in 0..na {
                for bi in 0..nb {
                    let pxy = joint_f[ai * nb + bi];
                    let px_py = px[ai] * py[bi];
                    if pxy > 0.0 && px_py > 0.0 {
                        mi += pxy * (pxy / px_py).log2();
                    }
                }
            }

            if let Some(ref s) = signs {
                let sign = if s[i][j] >= 0.0 { 1.0 } else { -1.0 };
                mi *= sign;
            }
            Some((i, j, mi))
        })
        .collect()
}

/// Python module definition
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mi_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cross_mi_pairs, m)?)?;
    Ok(())
}
