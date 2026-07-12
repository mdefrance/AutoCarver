"""Post-fit manual bin adjustment (feature.group) must be reflected by transform:
users who audit and fix a boundary themselves must get consistent output."""

import numpy as np
import pandas as pd

from AutoCarver import BinaryCarver
from AutoCarver.combinations import CramervCombinations
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import Features


def _six_modality_data() -> tuple[pd.DataFrame, pd.Series]:
    """6 categorical modalities, 300 rows each, distinct exact target rates 0.1..0.6."""
    labels = ["A", "B", "C", "D", "E", "F"]
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    feature, y = [], []
    for label, rate in zip(labels, rates):
        n_pos = round(rate * 300)
        feature += [label] * 300
        y += [1] * n_pos + [0] * (300 - n_pos)
    return pd.DataFrame({"feature": feature}), pd.Series(y)


def test_manual_group_categorical_string_labels():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=6,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    a, b = feature.labels[0], feature.labels[1]
    assert a != b

    n_labels_before = len(set(feature.labels))
    assert n_labels_before >= 3

    out_before = carver.transform(X)
    mask_a = out_before["feature"] == a
    mask_b = out_before["feature"] == b
    mask_other = ~(mask_a | mask_b)

    feature.group([a], b)
    out_after = carver.transform(X)

    # a's rows and b's rows now share a single new label, distinct from all others
    assert out_after.loc[mask_a, "feature"].nunique() == 1
    assert out_after.loc[mask_a, "feature"].iloc[0] == out_after.loc[mask_b, "feature"].iloc[0]
    assert out_after["feature"].nunique() == n_labels_before - 1
    # untouched rows keep their exact former label
    assert (out_after.loc[mask_other, "feature"] == out_before.loc[mask_other, "feature"]).all()
    assert len(out_after) == len(X)


def test_manual_group_categorical_ordinal_encoding():
    X, y = _six_modality_data()
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=6,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    carver.fit(X, y)

    feature = carver.features("feature")
    n_codes_before = len(set(feature.labels))
    a, b = feature.labels[0], feature.labels[1]
    assert a != b

    feature.group([a], b)
    out = carver.transform(X)

    codes = sorted(out["feature"].unique())
    assert codes == list(range(n_codes_before - 1))


def test_manual_group_quantitative_adjacent_intervals():
    """4 quartiles with sharply distinct, monotone target rates -> carving should keep
    all 4 apart (max_n_mod=4); merging two adjacent ones must leave the others alone."""
    rng = np.random.default_rng(0)
    n_per_quartile = 1000
    feature = np.concatenate(
        [
            rng.uniform(0, 25, n_per_quartile),
            rng.uniform(25, 50, n_per_quartile),
            rng.uniform(50, 75, n_per_quartile),
            rng.uniform(75, 100, n_per_quartile),
        ]
    )
    rates = [0.1, 0.3, 0.5, 0.7]
    y = np.concatenate([(rng.random(n_per_quartile) < rate).astype(int) for rate in rates])
    X = pd.DataFrame({"feature": feature})
    y = pd.Series(y)

    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=True),
    )
    carver.fit(X, y)

    feature_obj = carver.features("feature")
    n_labels_before = len(set(feature_obj.labels))
    assert n_labels_before >= 3
    label_a, label_b = feature_obj.labels[0], feature_obj.labels[1]

    out_before = carver.transform(X)
    mask_a = out_before["feature"] == label_a
    mask_b = out_before["feature"] == label_b
    mask_other = ~(mask_a | mask_b)

    feature_obj.group([label_a], label_b)
    out_after = carver.transform(X)

    assert out_after.loc[mask_a, "feature"].nunique() == 1
    assert out_after.loc[mask_a, "feature"].iloc[0] == out_after.loc[mask_b, "feature"].iloc[0]
    assert out_after["feature"].nunique() == n_labels_before - 1
    assert (out_after.loc[mask_other, "feature"] == out_before.loc[mask_other, "feature"]).all()
    assert len(out_after) == len(X)
