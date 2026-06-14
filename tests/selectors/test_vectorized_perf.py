"""Performance sanity check: the batched ``compute_all`` must be much faster
than the per-feature ``compute_association`` loop on a wide frame.

Threshold is deliberately conservative (>= 3x) to stay robust across machines
and against the worst case for the tie-correction kernel (heavily-tied integer
columns); on low-tie/continuous data the speedup is an order of magnitude. The
same parity the correctness suite checks is re-asserted here so a "fast but
wrong" kernel fails.
"""

from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pytest import approx

from AutoCarver.selectors.measures import KruskalEtaSquaredMeasure


def test_compute_all_is_much_faster_than_scalar_loop():
    rng = np.random.default_rng(0)
    n_rows, n_cols = 5000, 400
    X = pd.DataFrame({f"q{j}": rng.integers(0, 12, n_rows).astype(float) for j in range(n_cols)})
    y = pd.Series(rng.integers(0, 3, n_rows))
    features = [SimpleNamespace(version=col) for col in X.columns]

    scalar = KruskalEtaSquaredMeasure()
    batch = KruskalEtaSquaredMeasure()

    def _scalar():
        return {col: scalar.compute_association(X[col], y) for col in X.columns}

    def _batch():
        return batch.compute_all(X, y, features)

    # best-of-N (min) timing is robust to transient CPU contention from the
    # parallel test runner — at least one of the repeats runs near-uncontended.
    def _best_time(fn, repeats=5):
        best = float("inf")
        result = None
        for _ in range(repeats):
            t0 = perf_counter()
            result = fn()
            best = min(best, perf_counter() - t0)
        return best, result

    scalar_time, scalar_values = _best_time(_scalar)
    batch_time, batch_results = _best_time(_batch)

    # fast ...
    assert batch_time * 3 < scalar_time, f"vectorized {batch_time:.3f}s vs scalar {scalar_time:.3f}s"

    # ... and correct
    for col in X.columns:
        assert batch_results[col]["value"] == approx(scalar_values[col], rel=1e-9, abs=1e-9)
