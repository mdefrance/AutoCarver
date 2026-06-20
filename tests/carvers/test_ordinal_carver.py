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
    # groups are ordered by increasing mean ordinal level
    mean_level = feature.statistics["target_mean_level"]
    assert list(mean_level) == sorted(mean_level)
