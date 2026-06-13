"""Set of tests for the nested_discretizer module."""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver import BinaryCarver
from AutoCarver.combinations import BinaryCombinationEvaluator, TschuprowtCombinations
from AutoCarver.discretizers import DiscretizerConfig
from AutoCarver.discretizers.qualitatives import NestedDiscretizer, QualitativeDiscretizer
from AutoCarver.features import Features

# col_c (finest) -> col_b -> col_a (coarsest)
C_TO_B = {"A1x": "A1", "A1y": "A1", "A1z": "A1", "A2x": "A2", "A2y": "A2", "B1p": "B1", "B1q": "B1"}
B_TO_A = {"A1": "A", "A2": "A", "B1": "B"}

# exact counts (n=1000) chosen so that, at min_freq=0.10:
#  - A2x (400) and A2y (340) are frequent and kept as-is
#  - A1x/A1y/A1z (40/40/50) are each rare -> roll up to A1 (sum 130) which is then kept
#  - B1p/B1q (70/60) are each rare -> roll up to B1 (sum 130) which is then kept
COUNTS = {"A2x": 400, "A2y": 340, "A1x": 40, "A1y": 40, "A1z": 50, "B1p": 70, "B1q": 60}

# distinct target rates per final bucket so the carver finds a robust combination
BUCKET_TARGET_RATE = {"A2x": 0.50, "A2y": 0.45, "A1": 0.15, "B1": 0.85}


def _final_bucket(col_c_value: str) -> str:
    """The bucket each finest modality is expected to roll up into."""
    if col_c_value in ("A2x", "A2y"):
        return col_c_value
    return C_TO_B[col_c_value]


def build_nested_df(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Builds a deterministic nested DataFrame with a binary target."""
    rng = np.random.default_rng(seed)
    col_c = np.concatenate([np.repeat(value, count) for value, count in COUNTS.items()])
    rng.shuffle(col_c)
    col_b = np.array([C_TO_B[value] for value in col_c])
    col_a = np.array([B_TO_A[value] for value in col_b])

    rates = np.array([BUCKET_TARGET_RATE[_final_bucket(value)] for value in col_c])
    y = pd.Series((rng.random(len(col_c)) < rates).astype(int), name="target")

    X = pd.DataFrame({"col_c": col_c, "col_b": col_b, "col_a": col_a})
    return X, y


def test_nested_discretizer_rolls_rare_modalities_to_parents() -> None:
    """Rare finest modalities roll up to their data-derived parent; frequent ones are kept."""
    X, y = build_nested_df(seed=0)
    min_freq = 0.10

    features = Features(nested={"col_c": ["col_b", "col_a"]})
    discretizer = NestedDiscretizer(nesteds=features.nested, min_freq=min_freq)
    X_transformed = discretizer.fit_transform(X, y)

    feature = features("col_c")

    # rare modalities grouped under their parent, frequent ones kept on their own
    # (the always-present empty default bucket is excluded from this comparison)
    content_sets = {leader: set(members) for leader, members in feature.content.items() if leader != feature.default}
    assert content_sets == {
        "A1": {"A1x", "A1y", "A1z", "A1"},
        "B1": {"B1p", "B1q", "B1"},
        "A2x": {"A2x"},
        "A2y": {"A2y"},
    }

    # a default (__OTHER__) bucket is always present as the unseen-modality fallback, empty here
    assert feature.has_default
    assert feature.content[feature.default] == [feature.default]

    # labels are the bucket (leader) values themselves (+ the default)
    assert set(feature.labels) == {"A1", "B1", "A2x", "A2y", feature.default}

    # output is a single column whose every surviving modality is frequent enough
    frequencies = X_transformed["col_c"].value_counts(normalize=True)
    assert set(frequencies.index) == {"A1", "B1", "A2x", "A2y"}
    assert (frequencies >= min_freq).all(), "All surviving modalities should reach min_freq"

    # raw finest values map straight through to their bucket label
    assert all(feature.label_per_value[value] == _final_bucket(value) for value in COUNTS)


def test_nested_discretizer_transform_maps_unseen_rows() -> None:
    """A fitted discretizer maps fresh raw rows to the learned buckets."""
    X, y = build_nested_df(seed=1)
    features = Features(nested={"col_c": ["col_b", "col_a"]})
    discretizer = NestedDiscretizer(nesteds=features.nested, min_freq=0.10)
    discretizer.fit(X, y)

    # a fresh frame containing only the finest column (parents not needed at transform time)
    fresh = pd.DataFrame(
        {"col_c": ["A1x", "A2x", "B1q", "A2y"], "col_b": ["A1", "A2", "B1", "A2"], "col_a": ["A", "A", "B", "A"]}
    )
    out = discretizer.transform(fresh)
    assert out["col_c"].tolist() == ["A1", "A2x", "B1", "A2y"]


def test_nested_discretizer_raises_on_unclean_nesting() -> None:
    """A finest modality nested within several parents is rejected."""
    X, y = build_nested_df(seed=2)
    # corrupting the nesting: some A1x rows now sit under A2 instead of A1
    mask = X["col_c"] == "A1x"
    corrupt_index = X[mask].index[:5]
    X.loc[corrupt_index, "col_b"] = "A2"

    features = Features(nested={"col_c": ["col_b", "col_a"]})
    discretizer = NestedDiscretizer(nesteds=features.nested, min_freq=0.10)
    with raises(ValueError):
        discretizer.fit(X, y)


def test_nested_discretizer_single_parent() -> None:
    """Works with a single parent level (col_c -> col_b only)."""
    X, y = build_nested_df(seed=3)
    features = Features(nested={"col_c": ["col_b"]})
    discretizer = NestedDiscretizer(nesteds=features.nested, min_freq=0.10)
    X_transformed = discretizer.fit_transform(X, y)

    frequencies = X_transformed["col_c"].value_counts(normalize=True)
    assert set(frequencies.index) == {"A1", "B1", "A2x", "A2y"}
    assert (frequencies >= 0.10).all()


def test_nested_discretizer_within_qualitative_discretizer() -> None:
    """A nested feature is rolled up by the QualitativeDiscretizer alongside other features."""
    X, y = build_nested_df(seed=4)
    X["plain"] = np.where(np.arange(len(X)) % 2 == 0, "p", "q")  # a frequent categorical

    features = Features(nested={"col_c": ["col_b", "col_a"]}, categoricals=["plain"])
    discretizer = QualitativeDiscretizer(qualitatives=features.qualitatives, min_freq=0.10)
    X_transformed = discretizer.fit_transform(X, y)

    # nested feature collapsed to a single robust column
    assert set(X_transformed["col_c"].unique()).issubset(set(features("col_c").labels))
    assert X_transformed["col_c"].value_counts(normalize=True).min() >= 0.10


def _build_two_level(counts: dict, b_to_a: dict, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic 2-level frame (col_b nested in col_a) with a binary target."""
    rng = np.random.default_rng(seed)
    col_b = np.concatenate([np.repeat(value, count) for value, count in counts.items()])
    rng.shuffle(col_b)
    col_a = np.array([b_to_a[value] for value in col_b])
    y = pd.Series((rng.random(len(col_b)) < 0.5).astype(int))
    return pd.DataFrame({"col_b": col_b, "col_a": col_a}), y


def test_nested_discretizer_pools_terminal_rare_into_other() -> None:
    """A finest modality whose whole lineage stays rare is pooled into __OTHER__."""
    # Z-branch (Z1/Z2/Z3, all under col_a 'Z') totals 6% — rare even at the coarsest level
    counts = {"M1": 700, "M2": 240, "Z1": 20, "Z2": 20, "Z3": 20}
    b_to_a = {"M1": "M", "M2": "M", "Z1": "Z", "Z2": "Z", "Z3": "Z"}
    X, y = _build_two_level(counts, b_to_a, seed=0)

    features = Features(nested={"col_b": ["col_a"]})
    discretizer = NestedDiscretizer(nesteds=features.nested, min_freq=0.10)
    X_transformed = discretizer.fit_transform(X, y)
    feature = features("col_b")

    # the rare Z-branch lands in __OTHER__
    assert set(feature.content[feature.default]) >= {"Z1", "Z2", "Z3"}
    assert all(feature.label_per_value[value] == feature.default for value in ("Z1", "Z2", "Z3"))

    # standalone (non-default) buckets are all frequent enough
    standalone = X_transformed["col_b"].value_counts(normalize=True).drop(feature.default, errors="ignore")
    assert (standalone >= 0.10).all()


def test_nested_discretizer_unseen_modality_parent_aware() -> None:
    """Unseen finest modalities at transform roll up to a known parent bucket, else __OTHER__."""
    # 3-level data where the whole A-branch is rare and rolls up into a single 'A' bucket
    c_to_b = {"A1a": "A1", "A1b": "A1", "A2a": "A2", "B1a": "B1", "B1b": "B1"}
    b_to_a = {"A1": "A", "A2": "A", "B1": "B"}
    counts = {"A1a": 90, "A1b": 90, "A2a": 120, "B1a": 1350, "B1b": 1350}
    rng = np.random.default_rng(3)
    col_c = np.concatenate([np.repeat(v, c) for v, c in counts.items()])
    rng.shuffle(col_c)
    X = pd.DataFrame({"col_c": col_c})
    X["col_b"] = X["col_c"].map(c_to_b)
    X["col_a"] = X["col_b"].map(b_to_a)
    y = pd.Series(rng.integers(0, 2, len(col_c)))

    features = Features(nested={"col_c": ["col_b", "col_a"]})
    discretizer = NestedDiscretizer(nesteds=features.nested, min_freq=0.10)
    discretizer.fit(X, y)
    feature = features("col_c")
    assert "A" in feature.values, "the rare A-branch should have rolled up into an 'A' bucket"

    # unseen A1c is nested (col_a == 'A') under the known 'A' bucket -> resolves to 'A';
    # unseen Q9 has no known ancestor bucket -> __OTHER__
    fresh = pd.DataFrame({"col_c": ["A1c", "B1a", "Q9"], "col_b": ["A1", "B1", "Q9"], "col_a": ["A", "B", "Q"]})
    out = discretizer.transform(fresh)  # must not raise
    assert out["col_c"].tolist() == ["A", "B1a", feature.default]


def test_nested_feature_within_binary_carver_unseen_dev() -> None:
    """The user's scenario: X_dev has a finest modality unseen at fit -> no crash."""
    X_train, y_train = build_nested_df(seed=7)
    X_dev, y_dev = build_nested_df(seed=8)
    # inject an unseen finest modality 'A1w' nested under the known A1 bucket (col_b 'A1')
    extra = pd.DataFrame({"col_c": ["A1w"] * 40, "col_b": ["A1"] * 40, "col_a": ["A"] * 40})
    X_dev = pd.concat([X_dev, extra], ignore_index=True)
    y_dev = pd.concat([y_dev, pd.Series([0] * 40)], ignore_index=True)

    features = Features(nested={"col_c": ["col_b", "col_a"]})
    carver = BinaryCarver(
        min_freq=0.10,
        max_n_mod=4,
        combination_evaluator=TschuprowtCombinations(),
        features=features,
        config=DiscretizerConfig(verbose=False),
    )
    # fit transforms X_dev internally — would crash without the parent-aware remap
    carver.fit_transform(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    out = carver.transform(X_dev)  # must not raise
    assert out["col_c"].nunique() <= carver.max_n_mod


def test_nested_feature_within_binary_carver() -> None:
    """A nested feature flows through a BinaryCarver end-to-end (carved like a categorical)."""
    X_train, y_train = build_nested_df(seed=5)
    X_dev, y_dev = build_nested_df(seed=6)

    features = Features(nested={"col_c": ["col_b", "col_a"]})
    evaluator: BinaryCombinationEvaluator = TschuprowtCombinations()
    carver = BinaryCarver(
        min_freq=0.10,
        max_n_mod=4,
        combination_evaluator=evaluator,
        features=features,
        config=DiscretizerConfig(verbose=False),
    )
    X_carved = carver.fit_transform(X_train, y_train, X_dev=X_dev, y_dev=y_dev)

    # the nested feature survived carving and yields a bounded number of buckets
    assert "col_c" in carver.features.versions
    assert X_carved["col_c"].nunique() <= carver.max_n_mod
    # the dev set transforms with the same buckets
    X_dev_carved = carver.transform(X_dev)
    assert set(X_dev_carved["col_c"].unique()) == set(X_carved["col_c"].unique())
