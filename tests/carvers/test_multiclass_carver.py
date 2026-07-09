"""Set of tests for the joint multiclass_carver module (one carving per feature,
against the full K-class target — see test_one_vs_rest_carver.py for the
one-vs-rest alternative)."""

from pathlib import Path

import numpy as np
import pandas as pd
from pytest import FixtureRequest, fixture, raises
from sklearn.utils.validation import check_is_fitted

from AutoCarver.carvers.multiclass_carver import MulticlassCarver
from AutoCarver.carvers.utils.base_carver import Sample, Samples
from AutoCarver.combinations import (
    CombinationEvaluator,
    CramervMulticlassCombinations,
    KruskalCombinations,
    TschuprowtMulticlassCombinations,
)
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import Features


@fixture(params=[CramervMulticlassCombinations, TschuprowtMulticlassCombinations])
def evaluator(request: FixtureRequest) -> CombinationEvaluator:
    """Evaluator instance fixture, passed as combination_evaluator= to the carver."""
    return request.param()


def test_multiclass_carver_initialization():
    """Test MulticlassCarver initialization."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, max_n_mod=5, features=features)
    assert carver.min_freq == 0.1
    assert carver.features == features
    assert carver.config.dropna is True
    assert isinstance(carver.combination_evaluator, TschuprowtMulticlassCombinations)
    assert carver.max_n_mod == 5

    carver = MulticlassCarver(
        min_freq=0.1, features=features, max_n_mod=8, combination_evaluator=CramervMulticlassCombinations()
    )
    assert isinstance(carver.combination_evaluator, CramervMulticlassCombinations)

    with raises(ValueError):
        MulticlassCarver(min_freq=0.1, features=features, max_n_mod=5, combination_evaluator=KruskalCombinations())


def test_multiclass_carver_prepare_samples(evaluator: CombinationEvaluator):
    """Test MulticlassCarver _prepare_samples method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    X = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]})

    # binary y -> rejected
    y = pd.Series([0, 1, 0])
    with raises(ValueError, match="BinaryCarver"):
        carver._prepare_samples(Samples(train=Sample(X, y)))

    # 3-class y -> accepted
    y = pd.Series([0, 1, 2])
    prepared = carver._prepare_samples(Samples(train=Sample(X, y)))
    assert isinstance(prepared, Samples)

    # mismatched dev classes -> rejected
    y_dev = pd.Series([0, 1, 0])
    with raises(ValueError, match="Mismatched classes"):
        carver._prepare_samples(Samples(train=Sample(X, y), dev=Sample(X, y_dev)))

    # NaN in y -> rejected (must not silently become a "nan" class via astype(str))
    y_nan = pd.Series([0, 1, np.nan])
    with raises(ValueError, match="should not contain numpy.nan"):
        carver._prepare_samples(Samples(train=Sample(X, y_nan)))

    # NaN in y_dev -> rejected
    y = pd.Series([0, 1, 2])
    y_dev_nan = pd.Series([0, 1, np.nan])
    with raises(ValueError, match="should not contain numpy.nan"):
        carver._prepare_samples(Samples(train=Sample(X, y), dev=Sample(X, y_dev_nan)))


def test_quantitative_feature_with_rare_modality_and_numeric_target(evaluator: CombinationEvaluator):
    """Regression test: a quantitative feature with a dominant repeated value produces a
    rare quantile modality, which routes through OrdinalDiscretizer's rare-modality merge
    during the shared pre-discretization pass (see QuantitativeDiscretizer
    ._fit_continuous_with_rare_modalities). Under MulticlassCarver, y is cast to str there;
    for a numeric-coded target (e.g. 0/1/2 class codes) this used to crash with
    `TypeError: unsupported operand type(s) for /: 'str' and 'int'` since the target-rate
    merge sums y, and summing strings silently concatenates instead of erroring until the
    later division."""
    rng = np.random.default_rng(0)
    n = 2000
    values = np.concatenate([np.zeros(1600), rng.normal(10, 3, n - 1600)])
    rng.shuffle(values)
    X = pd.DataFrame({"num1": values})
    y = pd.Series(rng.choice([0, 1, 2], size=n), name="target")

    carver = MulticlassCarver(
        features=Features(numericals=["num1"]),
        min_freq=0.05,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=False, copy=True, verbose=False, n_jobs=1, min_freq_alpha=0.05),
    )
    carver.fit_transform(X, y)  # must not raise


def test_quantitative_feature_with_rare_modality_and_nominal_target(evaluator: CombinationEvaluator):
    """Same as above, but with a genuinely nominal (non-numeric) target — this crashed
    even before the target was ever cast to str, since sum()/count() of nominal labels
    is never meaningful."""
    rng = np.random.default_rng(0)
    n = 2000
    values = np.concatenate([np.zeros(1600), rng.normal(10, 3, n - 1600)])
    rng.shuffle(values)
    X = pd.DataFrame({"num1": values})
    y = pd.Series(rng.choice(["cat", "dog", "bird"], size=n), name="target")

    carver = MulticlassCarver(
        features=Features(numericals=["num1"]),
        min_freq=0.05,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=False, copy=True, verbose=False, n_jobs=1, min_freq_alpha=0.05),
    )
    carver.fit_transform(X, y)  # must not raise


def _multiclass_dataset(n_classes: int, n: int = 900, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A signal feature strongly associated with an n_classes-way target, plus a
    pure-noise feature — enough rows per (modality, class) cell to clear a
    reasonable min_freq."""
    rng = np.random.default_rng(seed)
    classes = [f"c{i}" for i in range(n_classes)]
    signal = rng.choice(classes, size=n)  # signal modality == "true" class name
    y = np.array([s if rng.random() < 0.8 else rng.choice(classes) for s in signal])
    noise = rng.choice(["x", "y", "z"], size=n)
    X = pd.DataFrame({"signal": signal, "noise": noise})
    return X, pd.Series(y, name="target")


def test_end_to_end_k3_single_version_per_feature(evaluator: CombinationEvaluator):
    """K=3: fit/transform produces exactly one carved version per feature (no
    per-class __y= explosion), within max_n_mod."""
    X, y = _multiclass_dataset(3)
    carver = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False, copy=False),
    )
    X_transformed = carver.fit_transform(X, y)

    assert all(feature.version == feature.name for feature in carver.features)
    assert not any("__y=" in col for col in X_transformed.columns)
    for feature in carver.features:
        assert X_transformed[feature.version].nunique() <= carver.max_n_mod


def test_end_to_end_k5_with_dev(evaluator: CombinationEvaluator):
    """K=5, with a dev set: fit succeeds and stays within max_n_mod on both samples."""
    X, y = _multiclass_dataset(5, n=1500, seed=1)
    X_dev, y_dev = _multiclass_dataset(5, n=1500, seed=2)
    carver = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.05,
        max_n_mod=6,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False, copy=False),
    )
    X_transformed = carver.fit_transform(X, y, X_dev=X_dev, y_dev=y_dev)
    X_dev_transformed = carver.transform(X_dev)

    for feature in carver.features:
        assert X_transformed[feature.version].nunique() <= carver.max_n_mod
        assert X_dev_transformed[feature.version].nunique() <= carver.max_n_mod


def test_string_and_integer_class_labels_give_identical_groupings(evaluator: CombinationEvaluator):
    """The target's own label text (int-coded vs string-coded classes) must not
    affect the resulting grouping."""
    X, y_str = _multiclass_dataset(3, n=900, seed=7)
    class_to_int = {c: i for i, c in enumerate(sorted(y_str.unique()))}
    y_int = y_str.map(class_to_int)

    carver_str = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
        config=ProcessingConfig(dropna=True, verbose=False),
    )
    carver_str.fit(X, y_str)

    carver_int = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
        config=ProcessingConfig(dropna=True, verbose=False),
    )
    carver_int.fit(X, y_int)

    for feature_name in ("signal", "noise"):
        assert carver_str.features(feature_name).content == carver_int.features(feature_name).content


def test_ordering_determinism_fit_twice_and_permuted_rows(evaluator: CombinationEvaluator):
    """Fitting twice gives identical feature content; permuting input rows does too."""
    X, y = _multiclass_dataset(3, n=900, seed=3)

    carver1 = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
    )
    carver1.fit(X, y)

    carver2 = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
    )
    carver2.fit(X, y)

    for feature_name in ("signal", "noise"):
        assert carver1.features(feature_name).content == carver2.features(feature_name).content

    # permuted rows -> identical content
    perm = np.random.default_rng(0).permutation(len(X))
    X_perm = X.iloc[perm].reset_index(drop=True)
    y_perm = y.iloc[perm].reset_index(drop=True)

    carver3 = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
    )
    carver3.fit(X_perm, y_perm)

    for feature_name in ("signal", "noise"):
        assert carver1.features(feature_name).content == carver3.features(feature_name).content


def test_label_independence_renamed_modalities(evaluator: CombinationEvaluator):
    """Renaming a feature's modality labels must not change the resulting
    grouping structure (only the labels used inside it)."""
    X, y = _multiclass_dataset(3, n=900, seed=4)
    rename = {"c0": "zeta", "c1": "alpha", "c2": "mu"}
    X_renamed = X.copy()
    X_renamed["signal"] = X_renamed["signal"].map(rename)

    carver = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
    )
    carver.fit(X, y)

    carver_renamed = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
    )
    carver_renamed.fit(X_renamed, y)

    # same number of groups, and group membership matches modulo the rename
    content = carver.features("signal").content
    content_renamed = carver_renamed.features("signal").content
    assert len(content) == len(content_renamed)
    renamed_content = {
        rename.get(leader, leader): sorted(rename.get(m, m) for m in members) for leader, members in content.items()
    }
    assert {frozenset(v) for v in renamed_content.values()} == {frozenset(v) for v in content_renamed.values()}


def test_dev_veto_fires_for_a_feature_whose_ca_ordering_inverts_on_dev():
    """A feature whose per-modality class association on dev is the reverse of
    train's must trip the train/dev rank-preservation veto for its finest
    (all-distinct) grouping — recorded as a non-viable "Inversion of target
    rates" candidate in the feature's history, even if the carver eventually
    falls back to a coarser, accidentally-monotone grouping."""
    rng = np.random.default_rng(9)
    n = 600
    classes = ["c0", "c1", "c2"]
    modalities = ["m0", "m1", "m2", "m3"]
    # train: modality i strongly predicts a class; dev: the reverse mapping
    train_class_by_mod = {"m0": "c0", "m1": "c1", "m2": "c2", "m3": "c0"}
    dev_class_by_mod = {"m0": "c2", "m1": "c1", "m2": "c0", "m3": "c2"}

    def _make(class_by_mod: dict[str, str]) -> tuple[pd.DataFrame, pd.Series]:
        signal = rng.choice(modalities, size=n)
        y = np.array([class_by_mod[m] if rng.random() < 0.9 else rng.choice(classes) for m in signal])
        return pd.DataFrame({"signal": signal}), pd.Series(y)

    X, y = _make(train_class_by_mod)
    X_dev, y_dev = _make(dev_class_by_mod)

    carver = MulticlassCarver(
        features=Features(categoricals=["signal"]),
        min_freq=0.05,
        max_n_mod=4,
        config=ProcessingConfig(dropna=True, verbose=False),
    )
    carver.fit(X, y, X_dev=X_dev, y_dev=y_dev)

    assert "signal" in carver.features or len(carver.dropped_features) == 1
    history = carver.features("signal").history if "signal" in carver.features else carver.dropped_features[0].history
    dev_blocks = [row for row in history["dev"] if isinstance(row, dict)]
    assert any(
        not block["viable"] and block.get("info") == "Inversion of target rates per modality" for block in dev_blocks
    ), "expected at least one candidate rejected for train/dev rank inversion"


def test_save_load_roundtrip(tmp_path: Path, evaluator: CombinationEvaluator):
    """Save/load on a fitted carver preserves the carved content and transform output."""
    X, y = _multiclass_dataset(3, n=900, seed=5)
    carver = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False),
    )
    X_transformed = carver.fit_transform(X, y)

    carver_file = tmp_path / "multiclass_carver.json"
    carver.save(carver_file)
    loaded = MulticlassCarver.load(carver_file)

    assert carver.min_freq == loaded.min_freq
    assert carver.combination_evaluator.__class__ == loaded.combination_evaluator.__class__
    for feature in carver.features:
        assert feature in loaded.features
        assert feature.content == loaded.features[feature.name].content

    X_loaded_transformed = loaded.transform(X)
    assert X_transformed.equals(X_loaded_transformed)


def test_n_jobs_1_vs_2_parity(evaluator: CombinationEvaluator):
    """n_jobs=1 and n_jobs=2 produce the same carved content and transform output."""
    X, y = _multiclass_dataset(3, n=900, seed=6)

    carver_serial = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
        config=ProcessingConfig(dropna=True, verbose=False, n_jobs=1),
    )
    X_serial = carver_serial.fit_transform(X, y)

    carver_parallel = MulticlassCarver(
        features=Features(categoricals=["signal", "noise"]),
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=type(evaluator)(),
        config=ProcessingConfig(dropna=True, verbose=False, n_jobs=2),
    )
    X_parallel = carver_parallel.fit_transform(X, y)

    for feature_name in ("signal", "noise"):
        assert carver_serial.features(feature_name).content == carver_parallel.features(feature_name).content
    assert X_serial.equals(X_parallel)


def test_class_absent_from_dev_raises():
    """A class present in train but absent from dev raises at fit time."""
    X, y = _multiclass_dataset(3, n=300, seed=8)
    # dev only has two of the three classes
    mask = y != y.unique()[0]
    X_dev, y_dev = X[mask], y[mask]

    carver = MulticlassCarver(features=Features(categoricals=["signal", "noise"]), min_freq=0.1, max_n_mod=4)
    with raises(ValueError, match="Mismatched classes"):
        carver.fit(X, y, X_dev=X_dev, y_dev=y_dev)


def test_fit_is_fitted_after_fit():
    X, y = _multiclass_dataset(3, n=300, seed=10)
    carver = MulticlassCarver(features=Features(categoricals=["signal", "noise"]), min_freq=0.1, max_n_mod=4)
    carver.fit(X, y)
    check_is_fitted(carver)
