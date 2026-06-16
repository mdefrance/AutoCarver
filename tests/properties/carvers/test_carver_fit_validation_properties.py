"""Property-based tests for carver fit-time target/evaluator validation.

Source: ``carvers/{binary,continuous,multiclass}_carver.py``. Each carver guards
its target shape and the type of its combination evaluator; these properties
assert the guards fire for every malformed input the strategies produce.
"""

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strategies import dataframe_and_features

from AutoCarver.carvers import BinaryCarver, ContinuousCarver, MulticlassCarver
from AutoCarver.combinations import CramervCombinations, KruskalCombinations, TschuprowtCombinations
from AutoCarver.features import Features

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --------------------------------------------------------------------------
# BinaryCarver
# --------------------------------------------------------------------------
@given(dataframe_and_features("binary"), st.data())
@SETTINGS
def test_binary_carver_rejects_non_binary_y(prob, data):
    """A target that is not exactly {0, 1} is rejected."""
    X, features, _ = prob
    n = len(X)
    # a 3-class target (too many classes)
    bad_multiclass = pd.Series([0, 1, 2] + data.draw(st.lists(st.integers(0, 2), min_size=n - 3, max_size=n - 3)))
    # a 2-value target that is not {0, 1}
    bad_labels = pd.Series([2, 3] + data.draw(st.lists(st.sampled_from([2, 3]), min_size=n - 2, max_size=n - 2)))

    for bad_y in (bad_multiclass, bad_labels):
        with pytest.raises(ValueError):
            BinaryCarver(features, min_freq=0.2, max_n_mod=4).fit(X, bad_y)


def test_binary_carver_rejects_continuous_evaluator():
    """BinaryCarver refuses an evaluator that isn't suited to a binary target."""
    features = Features(numericals=["a", "b"])
    with pytest.raises(ValueError):
        BinaryCarver(features, min_freq=0.2, max_n_mod=4, combination_evaluator=KruskalCombinations())


# --------------------------------------------------------------------------
# ContinuousCarver
# --------------------------------------------------------------------------
@given(dataframe_and_features("continuous"), st.data())
@SETTINGS
def test_continuous_carver_rejects_string_target(prob, data):
    """A string/categorical target is rejected."""
    X, features, _ = prob
    n = len(X)
    bad_y = pd.Series(data.draw(st.lists(st.sampled_from(["a", "b", "c"]), min_size=n, max_size=n)))
    with pytest.raises(ValueError):
        ContinuousCarver(features, min_freq=0.2, max_n_mod=4).fit(X, bad_y)


@given(dataframe_and_features("continuous"), st.data())
@SETTINGS
def test_continuous_carver_rejects_binary_target(prob, data):
    """A target with two or fewer distinct values is rejected (use BinaryCarver)."""
    X, features, _ = prob
    n = len(X)
    bad_y = pd.Series([0, 1] + data.draw(st.lists(st.sampled_from([0, 1]), min_size=n - 2, max_size=n - 2)))
    with pytest.raises(ValueError):
        ContinuousCarver(features, min_freq=0.2, max_n_mod=4).fit(X, bad_y)


def test_continuous_carver_rejects_binary_evaluator():
    """ContinuousCarver refuses an evaluator not suited to a continuous target."""
    features = Features(numericals=["a", "b"])
    for evaluator in (TschuprowtCombinations(), CramervCombinations()):
        with pytest.raises(ValueError):
            ContinuousCarver(features, min_freq=0.2, max_n_mod=4, combination_evaluator=evaluator)


# --------------------------------------------------------------------------
# MulticlassCarver
# --------------------------------------------------------------------------
@given(dataframe_and_features("binary"), st.data())
@SETTINGS
def test_multiclass_carver_rejects_binary_target(prob, data):
    """A target with two or fewer classes is rejected (use BinaryCarver)."""
    X, features, _ = prob
    n = len(X)
    bad_y = pd.Series([0, 1] + data.draw(st.lists(st.sampled_from([0, 1]), min_size=n - 2, max_size=n - 2)))
    with pytest.raises(ValueError):
        MulticlassCarver(features, min_freq=0.2, max_n_mod=4).fit(X, bad_y)


@given(dataframe_and_features("multiclass"), st.data())
@SETTINGS
def test_multiclass_carver_rejects_dev_class_mismatch(prob, data):
    """A dev target missing one of the train classes is rejected."""
    X, features, y = prob
    n = len(X)
    # collapse the dev target to two classes so a train class is absent from dev
    y_dev = pd.Series([0, 1] + data.draw(st.lists(st.sampled_from([0, 1]), min_size=n - 2, max_size=n - 2)))
    with pytest.raises(ValueError):
        MulticlassCarver(features, min_freq=0.2, max_n_mod=4).fit(X, y, X_dev=X, y_dev=y_dev)
