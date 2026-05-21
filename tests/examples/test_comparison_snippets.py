"""Runs the code snippets from docs/source/comparison.rst.

The AutoCarver block is exercised unconditionally; the optbinning and
KBinsDiscretizer blocks are guarded by ``pytest.importorskip`` so the test
file still collects when those libraries are missing (optbinning is in the
optional ``compare`` extra; scikit-learn is a runtime dep so KBins is
expected to be importable, but the guard keeps the test self-contained).
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

TITANIC_URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
TARGET = "Survived"
NUMERIC_COLS = ["Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"]


@pytest.fixture(scope="module")
def titanic_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, dev) split of the Titanic dataset, stratified on the target."""
    data = pd.read_csv(TITANIC_URL)
    return train_test_split(data, test_size=0.33, random_state=42, stratify=data[TARGET])


@pytest.mark.network
def test_autocarver_snippet(titanic_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """AutoCarver block from comparison.rst — one call covers all dtypes."""
    from AutoCarver import BinaryCarver, Features

    train, dev = titanic_split
    features = Features(
        categoricals=["Sex"],
        quantitatives=NUMERIC_COLS,
        ordinals={"Pclass": ["1", "2", "3"]},
    )
    carver = BinaryCarver(features=features, min_freq=0.05, max_n_mod=5)
    carver.fit(train, train[TARGET], X_dev=dev, y_dev=dev[TARGET])
    train_binned = carver.transform(train)

    assert set(train_binned.columns) >= {"Sex", "Pclass", *NUMERIC_COLS}
    assert len(train_binned) == len(train)


@pytest.mark.network
def test_optbinning_snippet(titanic_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """optbinning block from comparison.rst — one binner per feature."""
    optbinning = pytest.importorskip("optbinning")

    train, _ = titanic_split
    columns = {
        "Age": "numerical",
        "Fare": "numerical",
        "Siblings/Spouses Aboard": "numerical",
        "Parents/Children Aboard": "numerical",
        "Sex": "categorical",
        "Pclass": "categorical",
    }
    train_binned = pd.DataFrame(index=train.index)
    for name, dtype in columns.items():
        ob = optbinning.OptimalBinning(name=name, dtype=dtype, solver="cp")
        ob.fit(train[name].to_numpy(), train[TARGET].to_numpy())
        train_binned[name] = ob.transform(train[name].to_numpy(), metric="bins")

    assert set(train_binned.columns) == set(columns)
    assert len(train_binned) == len(train)


@pytest.mark.network
def test_kbins_snippet(titanic_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """KBinsDiscretizer block from comparison.rst — unsupervised, numeric-only."""
    preprocessing = pytest.importorskip("sklearn.preprocessing")

    train, _ = titanic_split
    train_numeric = train[NUMERIC_COLS].fillna(train[NUMERIC_COLS].median())

    kbd = preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    train_binned = pd.DataFrame(kbd.fit_transform(train_numeric), columns=NUMERIC_COLS, index=train.index)

    assert list(train_binned.columns) == NUMERIC_COLS
    assert len(train_binned) == len(train)
    # quantile strategy should yield <= n_bins distinct values per feature
    for col in NUMERIC_COLS:
        assert train_binned[col].nunique() <= 5
