//! Closed-form batched Kruskal-Wallis H.
//!
//! Mirrors `_kruskal_h_batch` in `continuous_combination_evaluators.py` exactly:
//!
//! ```text
//!   ssbn = sum_g R_g^2 / n_g          # NaN propagates when any n_g == 0
//!   H = (12 / (N*(N+1))) * ssbn - 3*(N+1)
//!   H /= tie_corr                     # NaN if tie_corr == 0
//!   H = NaN sentinel iff n_groups < 2 (Python wrapper maps NaN-from-sentinel to None)
//! ```
//!
//! Per-group accumulators are sized to `n_mod` (max possible gid in our sparse scheme);
//! the `seen` bitmap from `build_assignment` tells us which gids are real.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::assign::build_assignment;

#[inline]
fn h_for_row(
    row_assign: &[i32],
    row_seen: &[u8],
    n_groups: usize,
    r_per_mod: &[f64],
    n_per_mod: &[f64],
    n_mod: usize,
    n_total: f64,
    tie_corr: f64,
) -> f64 {
    if n_groups < 2 {
        return f64::NAN;
    }

    let mut r_g = vec![0.0_f64; n_mod];
    let mut n_g = vec![0.0_f64; n_mod];

    for m in 0..n_mod {
        let gi = row_assign[m] as usize;
        r_g[gi] += r_per_mod[m];
        n_g[gi] += n_per_mod[m];
    }

    // ssbn over only the gids actually seen; NaN propagates from any n_g == 0.
    let mut ssbn = 0.0_f64;
    for g in 0..n_mod {
        if row_seen[g] == 0 {
            continue;
        }
        if n_g[g] == 0.0 {
            ssbn = f64::NAN;
            break;
        }
        ssbn += r_g[g] * r_g[g] / n_g[g];
    }

    let h_raw = (12.0_f64 / (n_total * (n_total + 1.0))) * ssbn - 3.0_f64 * (n_total + 1.0);
    if tie_corr == 0.0 {
        f64::NAN
    } else {
        h_raw / tie_corr
    }
}

#[pyfunction]
#[pyo3(signature = (py_index_to_groupby, mod_to_pos, n_mod, r_per_mod, n_per_mod, n_total, tie_corr))]
pub fn kruskal_h_batch<'py>(
    py: Python<'py>,
    py_index_to_groupby: &Bound<'py, PyList>,
    mod_to_pos: &Bound<'py, PyDict>,
    n_mod: usize,
    r_per_mod: PyReadonlyArray1<'py, f64>,
    n_per_mod: PyReadonlyArray1<'py, f64>,
    n_total: f64,
    tie_corr: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let assignment = build_assignment(py, py_index_to_groupby, mod_to_pos, n_mod)?;
    let b_len = assignment.n_groups.len();

    let r_slice = r_per_mod.as_slice()?;
    let n_slice = n_per_mod.as_slice()?;

    let h_out: Vec<f64> = py.allow_threads(|| {
        (0..b_len)
            .into_par_iter()
            .map(|b| {
                let row_a = &assignment.assign[b * n_mod..(b + 1) * n_mod];
                let row_s = &assignment.seen[b * n_mod..(b + 1) * n_mod];
                h_for_row(
                    row_a,
                    row_s,
                    assignment.n_groups[b] as usize,
                    r_slice,
                    n_slice,
                    n_mod,
                    n_total,
                    tie_corr,
                )
            })
            .collect()
    });

    Ok((
        h_out.into_pyarray_bound(py),
        assignment.n_groups.into_pyarray_bound(py),
    ))
}
