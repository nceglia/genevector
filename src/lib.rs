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

/// Python module definition
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mi_pairs, m)?)?;
    Ok(())
}
