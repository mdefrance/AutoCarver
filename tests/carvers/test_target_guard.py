"""The carver drops a target column that leaked into the features (e.g. via from_dataframe)."""

import pandas as pd
from pytest import warns

from AutoCarver.carvers import BinaryCarver, ContinuousCarver, MulticlassCarver
from AutoCarver.features import Features


def _frame(target_values: list) -> tuple[pd.DataFrame, pd.Series]:
    """A DataFrame whose 'target' column is also mapped as a feature by from_dataframe."""
    X = pd.DataFrame(
        {
            "f1": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"],
            "f2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": target_values,
        }
    )
    return X, X["target"]


def _assert_drops_target(carver_cls, target_values: list) -> None:
    X, y = _frame(target_values)
    features = Features.from_dataframe(X)
    assert "target" in features  # from_dataframe mapped the target column too

    carver = carver_cls(features=features, min_freq=0.1, max_n_mod=5)
    with warns(UserWarning, match="dropping target column 'target'"):
        carver.fit(X, y)
    assert "target" not in carver.features


def test_binary_carver_drops_target():
    _assert_drops_target(BinaryCarver, [0, 1, 0, 1, 1, 0, 1, 0, 1, 0])


def test_continuous_carver_drops_target():
    _assert_drops_target(ContinuousCarver, [0.1, 1.2, 0.3, 1.4, 1.5, 0.6, 1.7, 0.8, 1.9, 0.2])


def test_multiclass_carver_drops_target():
    _assert_drops_target(MulticlassCarver, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
