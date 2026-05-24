//! Native kernels for AutoCarver.
//!
//! Exposes batched closed-form Kruskal-Wallis H and Pearson chi^2 to Python
//! as `AutoCarver._kernels`. See `SPEEDUP_PLAN.md` §7 for the design rationale.

use pyo3::prelude::*;

mod assign;
mod chi2;
mod kruskal;

#[pymodule]
fn _kernels(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kruskal::kruskal_h_batch, m)?)?;
    m.add_function(wrap_pyfunction!(chi2::chi2_assoc_batch, m)?)?;
    Ok(())
}
