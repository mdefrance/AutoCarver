//! Shared dict-walking: turn a list of `index_to_groupby` dicts into a
//! `(B, n_mod)` `i32` assignment matrix + per-combination `n_groups`.
//!
//! Faithful to the Python `_kruskal_h_batch` / `_chi2_assoc_batch` shape, with
//! one *equivalent* simplification: because `combination_formatter` always sets
//! the leader to one of the modalities (it's `group[0]`), we can use
//! `mod_to_pos[leader]` directly as a canonical group id instead of building a
//! per-combination `leaders → 0..n_groups` map and equality-checking leaders.
//!
//! This means:
//!
//! * group ids live in `[0, n_mod)` (sparse — empty buckets contribute zero);
//! * `n_groups` is the count of *distinct* gids actually seen in the row;
//! * unassigned modalities (present in `mod_to_pos` but missing from
//!   `index_to_groupby`) get `assign[pos] = pos`, i.e. their own singleton group.
//!
//! The math kernels (`kruskal.rs`, `chi2.rs`) size their per-group accumulators
//! to `n_mod` and walk only the gids in the `seen` bitmap, so the dense-id
//! difference is invisible to the closed-form output.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Per-combination assignment data, owned on the Rust side so the GIL can be released
/// before the math runs.
pub struct BatchAssignment {
    /// Flat `(B * n_mod)` row-major `i32` matrix. `assign[b * n_mod + m]` is the group id
    /// of modality at position `m` in combination `b`. Group ids are sparse (any value
    /// in `[0, n_mod)`); see `seen` for which ones are actually populated.
    pub assign: Vec<i32>,
    /// Flat `(B * n_mod)` bitmap as `u8` (0 or 1). `seen[b * n_mod + g] == 1` iff group `g`
    /// is populated in combination `b`. Used by the math kernels to skip empty buckets.
    pub seen: Vec<u8>,
    /// `n_groups[b]`: number of distinct gids in combination `b` (== `seen[b * n_mod ..].sum()`).
    pub n_groups: Vec<i32>,
}

/// Walk the `(py_index_to_groupby, mod_to_pos)` Python objects under the GIL and produce
/// a Rust-owned `BatchAssignment`. After this call, math can run with the GIL released.
pub fn build_assignment(
    _py: Python<'_>,
    py_index_to_groupby: &Bound<'_, PyList>,
    mod_to_pos: &Bound<'_, PyDict>,
    n_mod: usize,
) -> PyResult<BatchAssignment> {
    let b_len = py_index_to_groupby.len();
    let mut assign = vec![0_i32; b_len * n_mod];
    let mut seen = vec![0_u8; b_len * n_mod];
    let mut n_groups = vec![0_i32; b_len];

    for (b, item) in py_index_to_groupby.iter().enumerate() {
        let idx2gb: &Bound<'_, PyDict> = item.downcast()?;
        let row_start = b * n_mod;

        // Default: every modality is its own singleton group (gid == pos).
        for pos in 0..n_mod {
            assign[row_start + pos] = pos as i32;
        }
        let mut assigned = vec![false; n_mod];
        let mut row_seen = vec![false; n_mod];

        for (py_mod, py_leader) in idx2gb.iter() {
            let pos: usize = mod_to_pos
                .get_item(&py_mod)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!("modality {} not in mod_to_pos", py_mod))
                })?
                .extract()?;
            let gid: usize = mod_to_pos
                .get_item(&py_leader)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "leader {} not in mod_to_pos (combination_formatter should keep leader == group[0])",
                        py_leader
                    ))
                })?
                .extract()?;

            assign[row_start + pos] = gid as i32;
            assigned[pos] = true;
            row_seen[gid] = true;
        }

        // Singleton groups for any pos not visited: gid is `pos`, mark seen.
        for pos in 0..n_mod {
            if !assigned[pos] {
                row_seen[pos] = true;
            }
        }

        let mut count: i32 = 0;
        for (g, &s) in row_seen.iter().enumerate() {
            if s {
                seen[row_start + g] = 1;
                count += 1;
            }
        }
        n_groups[b] = count;
    }

    Ok(BatchAssignment { assign, seen, n_groups })
}
