"""Set of tests for ordinal_carver module."""

import numpy as np
import pandas as pd
from pytest import mark, raises

from AutoCarver import OrdinalCarver
from AutoCarver.carvers.utils.base_carver import Sample, Samples
from AutoCarver.combinations import (
    KendallTauBCombinations,
    KendallTauCCombinations,
    KruskalCombinations,
    SomersDCombinations,
    TschuprowtCombinations,
)
from AutoCarver.combinations.ordinal.ordinal_target_rates import TargetMeanLevel, TargetMeanRidit
from AutoCarver.features import Features

ORDINAL_EVALUATORS = [KendallTauCCombinations, KendallTauBCombinations, SomersDCombinations]


def test_ordinal_carver_initialization():
    """Default evaluator is KendallTauCCombinations; non-ordinal evaluators are rejected."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features)
    assert isinstance(carver.combination_evaluator, KendallTauCCombinations)
    assert carver.is_y_ordinal is True
    assert carver.max_n_mod == 5

    # all ordinal evaluators are accepted
    for evaluator in ORDINAL_EVALUATORS:
        carver = OrdinalCarver(min_freq=0.1, features=features, max_n_mod=5, combination_evaluator=evaluator())
        assert carver.combination_evaluator.is_y_ordinal is True

    # non-ordinal evaluators are rejected
    with raises(ValueError):
        OrdinalCarver(min_freq=0.1, features=features, max_n_mod=5, combination_evaluator=KruskalCombinations())
    with raises(ValueError):
        OrdinalCarver(min_freq=0.1, features=features, max_n_mod=5, combination_evaluator=TschuprowtCombinations())


def test_ordinal_carver_prepare_samples():
    """_prepare_samples accepts integer-encoded ordinal y, rejects binary/object/non-integer."""
    features = Features(categoricals=["feature1"], numericals=["feature3"])
    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features)
    X = pd.DataFrame({"feature1": list("ABABC"), "feature3": [1, 2, 3, 4, 5]})

    # binary target -> use BinaryCarver instead
    with raises(ValueError):
        carver._prepare_samples(Samples(train=Sample(X, pd.Series([0, 1, 1, 0, 1]))))

    # object/string target
    with raises(ValueError):
        carver._prepare_samples(Samples(train=Sample(X, pd.Series(["1", "2", "3", "4", "5"]))))

    # non-integer (truly continuous) target
    with raises(ValueError):
        carver._prepare_samples(Samples(train=Sample(X, pd.Series([0.2, 1.5, 2.3, 3.1, 4.4]))))

    # integer-encoded ordinal target (1..5)
    prepared = carver._prepare_samples(Samples(train=Sample(X, pd.Series([1, 2, 3, 4, 5]))))
    assert isinstance(prepared, Samples)


def test_ordinal_carver_aggregator():
    """_aggregator builds one crosstab per feature: modalities x ordinal target levels."""
    features = Features(categoricals=["feature1"])
    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features)
    X = pd.DataFrame({"feature1": ["A", "B", "A", "B", "C", "C"]})
    y = pd.Series([1, 2, 1, 3, 2, 3])
    carver.features.fit(X, y)

    xaggs = carver._aggregator(X, y)
    xagg = xaggs[carver.features[0].version]
    assert isinstance(xagg, pd.DataFrame)
    assert list(xagg.columns) == [1, 2, 3]  # ordinal levels, ascending
    assert int(xagg.to_numpy().sum()) == len(X)


# the symmetric Kendall taus reward the genuine 3-cluster structure; the asymmetric
# Somers' D collapses to the coarsest split (its documented behaviour).
EXPECTED_MODALITIES = {
    KendallTauCCombinations: 3,
    KendallTauBCombinations: 3,
    SomersDCombinations: 2,
}


@mark.parametrize("evaluator", ORDINAL_EVALUATORS)
def test_ordinal_carver_fit_recovers_cluster_structure(evaluator):
    """A 6-level feature with 3 latent clusters: Kendall taus -> 3 buckets, Somers' D -> 2."""
    rng = np.random.default_rng(7)
    n = 3000
    base = rng.integers(0, 6, size=n)
    cluster = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    g = np.vectorize(cluster.get)(base)
    y = pd.Series(g * 3 + rng.integers(1, 4, size=n), name="target")  # ordinal levels track clusters
    X = pd.DataFrame({"q": [str(b) for b in base]})

    features = Features(ordinals={"q": ["0", "1", "2", "3", "4", "5"]})
    carver = OrdinalCarver(min_freq=0.03, max_n_mod=6, features=features, combination_evaluator=evaluator())
    carver.fit(X, y)

    feature = carver.features[0]
    assert len(feature.labels) == EXPECTED_MODALITIES[evaluator]
    # groups are ordered by increasing mean train-ridit (the default target rate)
    mean_ridit = feature.statistics["target_mean_ridit"]
    assert list(mean_ridit) == sorted(mean_ridit)


@mark.parametrize("evaluator", ORDINAL_EVALUATORS)
def test_ordinal_carver_save_load(tmp_path, evaluator):
    """Each ordinal evaluator round-trips through save/load (regression for sort_by dispatch)."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator())
    carver_file = tmp_path / "ordinal_carver.json"  # exercise save/load with a Path, not str
    carver.save(carver_file)
    loaded_carver = OrdinalCarver.load(carver_file)

    assert carver.combination_evaluator.__class__ == loaded_carver.combination_evaluator.__class__
    assert carver.combination_evaluator.sort_by == loaded_carver.combination_evaluator.sort_by
    assert carver.min_freq == loaded_carver.min_freq
    assert carver.max_n_mod == loaded_carver.max_n_mod


# ---------------------------------------------------------------------------
# target_scale: ridit (default) / level / {level: value}
# ---------------------------------------------------------------------------


def test_ordinal_carver_target_scale_resolution():
    """target_scale resolves into the evaluator's rate; conflicts and typos raise."""
    features = Features(categoricals=["feature1"])

    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features)
    assert isinstance(carver.combination_evaluator.target_rate, TargetMeanRidit)

    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features, target_scale="level")
    assert isinstance(carver.combination_evaluator.target_rate, TargetMeanLevel)
    assert carver.combination_evaluator.target_rate.level_values is None

    scale = {1: 0.01, 2: 0.05, 3: 0.2}
    carver = OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features, target_scale=scale)
    assert isinstance(carver.combination_evaluator.target_rate, TargetMeanLevel)
    assert carver.combination_evaluator.target_rate.level_values == scale

    # an explicitly chosen TargetMeanLevel rate is kept under the default target_scale
    rate = TargetMeanLevel()
    carver = OrdinalCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=KendallTauCCombinations(target_rate=rate),
    )
    assert carver.combination_evaluator.target_rate is rate

    # ... but a non-default target_scale on top of it is ambiguous
    with raises(ValueError):
        OrdinalCarver(
            min_freq=0.1,
            max_n_mod=5,
            features=features,
            combination_evaluator=KendallTauCCombinations(target_rate=TargetMeanLevel()),
            target_scale="level",
        )
    with raises(ValueError):
        OrdinalCarver(
            min_freq=0.1,
            max_n_mod=5,
            features=features,
            combination_evaluator=KendallTauCCombinations(target_rate=TargetMeanLevel()),
            target_scale={1: 0.1, 2: 0.2, 3: 0.3},
        )

    # unknown mode
    with raises(ValueError):
        OrdinalCarver(min_freq=0.1, max_n_mod=5, features=features, target_scale="midrank")


def _make_encoded_data(top_level: int) -> tuple[pd.DataFrame, pd.Series]:
    """Categorical feature vs y over levels {1, 2, 3, top_level}.

    Modality 'b' is a mixture of the extreme levels, so its mean *encoded* level
    crosses modality 'a''s (concentrated on 3) when level 4 is re-encoded to 10 —
    flipping the target-mean pre-sort while all rank statistics are unchanged.
    """
    rng = np.random.default_rng(3)
    n = 4000
    level_probs = {
        "a": [0.05, 0.15, 0.60, 0.20],
        "b": [0.45, 0.05, 0.05, 0.45],
        "c": [0.60, 0.25, 0.10, 0.05],
        "d": [0.55, 0.28, 0.11, 0.06],
        "e": [0.10, 0.20, 0.30, 0.40],
    }
    x = rng.choice(list(level_probs), size=n)
    levels = np.array([1, 2, 3, top_level])
    y = np.array([levels[rng.choice(4, p=level_probs[modality])] for modality in x])
    return pd.DataFrame({"cat": x}), pd.Series(y, name="target")


def _fit_encoded(top_level: int, **kwargs) -> dict:
    X, y = _make_encoded_data(top_level)
    carver = OrdinalCarver(min_freq=0.05, max_n_mod=3, features=Features(categoricals=["cat"]), **kwargs)
    carver.fit(X, y)
    return dict(carver.features[0].content)


def test_ordinal_carver_ridit_default_is_encoding_invariant():
    """Headline regression: y encoded {1,2,3,4} vs {1,2,3,10} carve identically with
    the "ridit" default (the raw target-mean pre-sort differs between the two)."""
    assert _fit_encoded(4) == _fit_encoded(10)


def test_ordinal_carver_level_scale_differs_across_encodings():
    """target_scale="level" reads the encoding numerically: the same data under the
    two encodings pre-sorts (and here groups) differently — the documented count
    behaviour, and the reason "ridit" is the ordinal default."""
    assert _fit_encoded(4, target_scale="level") != _fit_encoded(10, target_scale="level")


def test_ordinal_carver_level_scale_pins_previous_default():
    """Count regression: target_scale="level" is byte-identical to the previous
    release's default carving (content pinned from the pre-ridit code)."""
    rng = np.random.default_rng(42)
    n = 3000
    base = rng.integers(0, 6, size=n)
    cluster = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    g = np.vectorize(cluster.get)(base)
    level_probs = {
        0: [0.60, 0.30, 0.08, 0.02],
        1: [0.35, 0.35, 0.20, 0.10],
        2: [0.10, 0.20, 0.30, 0.40],
    }
    levels = np.array([0, 1, 2, 10])  # skewed count-like spacing
    y = pd.Series(np.array([levels[rng.choice(4, p=level_probs[c])] for c in g]), name="target")
    X = pd.DataFrame({"q": [str(b) for b in base]})

    features = Features(ordinals={"q": ["0", "1", "2", "3", "4", "5"]})
    carver = OrdinalCarver(min_freq=0.03, max_n_mod=6, features=features, target_scale="level")
    carver.fit(X, y)

    # pinned from the previous default (TargetMeanLevel) on this exact scenario
    assert carver.features[0].content == {"0": ["1", "2", "3", "0"], "4": ["5", "4"]}
    assert "target_mean_level" in carver.features[0].statistics.columns


def test_ordinal_carver_dict_scale_end_to_end():
    """A {level: value} scale carves exactly like re-encoding y with those values and
    using target_scale="level" — the dict reaches both the pre-sort and viability."""
    dict_content = _fit_encoded(4, target_scale={1: 1.0, 2: 2.0, 3: 3.0, 4: 10.0})
    reencoded_content = _fit_encoded(10, target_scale="level")
    assert dict_content == reencoded_content


def test_ordinal_carver_dict_scale_missing_level_raises():
    """A scale not covering every observed train level raises at fit."""
    X, y = _make_encoded_data(4)
    carver = OrdinalCarver(
        min_freq=0.05,
        max_n_mod=3,
        features=Features(categoricals=["cat"]),
        target_scale={1: 1.0, 2: 2.0, 3: 3.0},  # level 4 missing
    )
    with raises(ValueError):
        carver.fit(X, y)


def test_ordinal_carver_custom_level_rate_drives_presort():
    """An explicit TargetMeanLevel() evaluator rate with the default target_scale is
    kept, and the pre-sort follows it (no ridit/level mismatch): the carving equals
    target_scale="level"'s."""
    explicit_content = _fit_encoded(10, combination_evaluator=KendallTauCCombinations(target_rate=TargetMeanLevel()))
    assert explicit_content == _fit_encoded(10, target_scale="level")


@mark.parametrize("evaluator", ORDINAL_EVALUATORS)
def test_ordinal_carver_fit_cv_runs(evaluator):
    """``fit(cv=...)`` runs end to end and transform works (ordinal target -> plain KFold)."""
    rng = np.random.default_rng(0)
    n = 400
    base = rng.integers(0, 4, size=n)
    y = pd.Series(base * 2 + rng.integers(0, 2, size=n), name="target")
    X = pd.DataFrame({"q": [str(b) for b in base]})

    features = Features(ordinals={"q": ["0", "1", "2", "3"]})
    carver = OrdinalCarver(min_freq=0.05, max_n_mod=4, features=features, combination_evaluator=evaluator())
    carver.fit(X, y, cv=3)
    X_transformed = carver.transform(X)

    assert "q" in carver.features
    assert isinstance(X_transformed, pd.DataFrame)
