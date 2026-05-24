//! Closed-form batched Pearson chi^2 (Cramer's V + Tschuprow's T) for binary tables.
//!
//! Mirrors `_chi2_assoc_batch` in `binary_combination_evaluators.py` cell-for-cell:
//!
//! * `+tol` shift on every in-range observed cell;
//! * Yates correction applied iff `n_groups == 2`;
//! * `round(x / tol) * tol` quantisation on both cramerv and tschuprowt;
//! * `tschuprowt == cramerv` when `n_groups < 2`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::assign::build_assignment;

#[inline]
fn chi2_pearson_2col(
    row_assign: &[i32],
    row_seen: &[u8],
    n_groups: usize,
    n0_per_mod: &[f64],
    n1_per_mod: &[f64],
    n_mod: usize,
    tol: f64,
) -> f64 {
    let mut n0_g = vec![0.0_f64; n_mod];
    let mut n1_g = vec![0.0_f64; n_mod];

    for m in 0..n_mod {
        let gi = row_assign[m] as usize;
        n0_g[gi] += n0_per_mod[m];
        n1_g[gi] += n1_per_mod[m];
    }

    // Observed = (n0_g, n1_g) + tol on every in-range (== seen) cell.
    // Sums of marginals are over seen groups only.
    let mut col0: f64 = 0.0;
    let mut col1: f64 = 0.0;
    let mut row_marg = vec![0.0_f64; n_mod];
    let mut seen_count: usize = 0;
    for g in 0..n_mod {
        if row_seen[g] == 0 {
            continue;
        }
        seen_count += 1;
        let o0 = n0_g[g] + tol;
        let o1 = n1_g[g] + tol;
        col0 += o0;
        col1 += o1;
        row_marg[g] = o0 + o1;
    }
    debug_assert_eq!(seen_count, n_groups);

    let n_table = col0 + col1;
    let yates = n_groups == 2;

    let mut chi2 = 0.0_f64;
    for g in 0..n_mod {
        if row_seen[g] == 0 {
            continue;
        }
        let r = row_marg[g];
        let exp0 = r * col0 / n_table;
        let exp1 = r * col1 / n_table;
        let mut o0 = n0_g[g] + tol;
        let mut o1 = n1_g[g] + tol;

        if yates {
            let d0 = exp0 - o0;
            o0 += d0.signum() * d0.abs().min(0.5);
            let d1 = exp1 - o1;
            o1 += d1.signum() * d1.abs().min(0.5);
        }

        let d0 = o0 - exp0;
        let d1 = o1 - exp1;
        chi2 += d0 * d0 / exp0 + d1 * d1 / exp1;
    }

    chi2
}

#[inline]
fn quantise(x: f64, tol: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        (x / tol).round() * tol
    }
}

#[pyfunction]
#[pyo3(signature = (py_index_to_groupby, mod_to_pos, n_mod, n0_per_mod, n1_per_mod, n_obs, tol))]
pub fn chi2_assoc_batch<'py>(
    py: Python<'py>,
    py_index_to_groupby: &Bound<'py, PyList>,
    mod_to_pos: &Bound<'py, PyDict>,
    n_mod: usize,
    n0_per_mod: PyReadonlyArray1<'py, f64>,
    n1_per_mod: PyReadonlyArray1<'py, f64>,
    n_obs: f64,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let assignment = build_assignment(py, py_index_to_groupby, mod_to_pos, n_mod)?;
    let b_len = assignment.n_groups.len();

    let n0_slice = n0_per_mod.as_slice()?;
    let n1_slice = n1_per_mod.as_slice()?;

    let results: (Vec<f64>, Vec<f64>) = py.allow_threads(|| {
        (0..b_len)
            .into_par_iter()
            .map(|b| {
                let n_groups = assignment.n_groups[b] as usize;
                let row_a = &assignment.assign[b * n_mod..(b + 1) * n_mod];
                let row_s = &assignment.seen[b * n_mod..(b + 1) * n_mod];
                let chi2 =
                    chi2_pearson_2col(row_a, row_s, n_groups, n0_slice, n1_slice, n_mod, tol);

                let cramerv_raw = (chi2 / n_obs).sqrt();
                let cramerv_q = quantise(cramerv_raw, tol);

                let tschuprowt_q = if n_groups > 1 {
                    let denom = ((n_groups as f64) - 1.0).sqrt().sqrt();
                    quantise(cramerv_q / denom, tol)
                } else {
                    cramerv_q
                };

                (cramerv_q, tschuprowt_q)
            })
            .unzip()
    });

    let (cramerv, tschuprowt) = results;

    Ok((
        cramerv.into_pyarray_bound(py),
        tschuprowt.into_pyarray_bound(py),
        assignment.n_groups.into_pyarray_bound(py),
    ))
}
