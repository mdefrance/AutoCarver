"""Golden end-to-end snapshot: refactors must not change carved output."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from AutoCarver import BinaryCarver, ContinuousCarver, MulticlassCarver, OneVsRestCarver, OrdinalCarver
from AutoCarver.features import Features

GOLDEN_DIR = Path(__file__).parent / "golden"


def _dataset(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 3000
    return pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.gamma(2.0, size=n),
            "cat_a": rng.choice(list("abcdefgh"), size=n),
            "cat_b": rng.choice(["x", "y", "z"], size=n, p=[0.7, 0.2, 0.1]),
        }
    )


def _signal(X: pd.DataFrame, rng) -> np.ndarray:
    return X["num_a"] + 0.5 * X["num_b"] + (X["cat_a"] < "d") * 1.5 + rng.normal(scale=0.5, size=len(X))


def _target(target: str, X: pd.DataFrame, rng) -> pd.Series:
    signal = _signal(X, rng)
    if target == "binary":
        return (signal > np.median(signal)).astype(int)
    if target == "continuous":
        return pd.Series(signal)
    if target == "ordinal":
        return pd.qcut(signal, 4, labels=[1, 2, 3, 4]).astype(int)
    if target == "multiclass":
        return pd.qcut(signal, 3, labels=["lo", "mid", "hi"]).astype(str)
    raise ValueError(target)


@pytest.mark.parametrize(
    "carver_cls, target",
    [
        (BinaryCarver, "binary"),
        (ContinuousCarver, "continuous"),
        (OrdinalCarver, "ordinal"),
        (MulticlassCarver, "multiclass"),
        (OneVsRestCarver, "multiclass"),
    ],
)
def test_golden_carving(carver_cls, target):
    """Carve a fixed dataset; compare `summary` against a stored snapshot."""
    rng = np.random.default_rng(0)
    X = _dataset(seed=0)
    y = _target(target, X, rng)

    features = Features(categoricals=["cat_a", "cat_b"], numericals=["num_a", "num_b"])
    carver = carver_cls(features, min_freq=0.05, max_n_mod=5)
    carver.fit(X, y)

    summary = carver.summary
    snapshot = {
        "index_names": list(summary.index.names),
        "columns": list(summary.columns),
        "records": summary.reset_index().to_dict("records"),
    }

    golden_path = GOLDEN_DIR / f"{carver_cls.__name__}.json"
    expected = json.loads(golden_path.read_text())

    assert snapshot["index_names"] == expected["index_names"]
    assert snapshot["columns"] == expected["columns"]
    assert len(snapshot["records"]) == len(expected["records"])
    for actual_row, expected_row in zip(snapshot["records"], expected["records"]):
        assert list(actual_row.keys()) == list(expected_row.keys())
        for key in actual_row:
            actual_value = actual_row[key]
            expected_value = expected_row[key]
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(expected_value, rel=1e-12)
            else:
                assert actual_value == expected_value
