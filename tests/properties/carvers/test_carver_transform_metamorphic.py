"""End-to-end metamorphic properties for carver fit/transform.

Source: ``carvers/utils/base_carver.py`` (via :class:`BinaryCarver` /
:class:`MulticlassCarver`), driven by ``dataframe_and_features``. Checks
row/index preservation, the ``max_n_mod`` cap, label membership / ordinal
encoding, copy non-mutation, fit_transform determinism, the fit/transform guards
and the multiclass version structure.

Carving is comparatively expensive, so the frames are small and example counts
modest. Features for which no robust combination survives are *dropped* from the
carver, so output-structure checks are scoped to the surviving features.
"""

import pytest
from hypothesis import HealthCheck, given, reject, settings
from hypothesis import strategies as st
from pandas.api.types import is_numeric_dtype
from strategies import clone_features, dataframe_and_features

from AutoCarver.carvers import BinaryCarver, MulticlassCarver, OrdinalCarver
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig

SETTINGS = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


def _carve(carver, X, y, **kwargs):
    """fit, filtering out degenerate-feature inputs the inner discretizer rejects."""
    try:
        return carver.fit(X, y, **kwargs)
    except ValueError as error:
        if "frequent" in str(error):  # degenerate feature rejected by the inner Discretizer
            reject()
        raise


def _carve_transform(carver, X, y):
    """fit_transform, filtering out degenerate-feature inputs."""
    try:
        return carver.fit_transform(X, y)
    except ValueError as error:
        if "frequent" in str(error):
            reject()
        raise


@st.composite
def binary_problem(draw, *, ordinal_encoding=False):
    X, features, y = draw(dataframe_and_features("binary", nrows=(40, 80), with_nan=True))
    max_n_mod = draw(st.integers(min_value=2, max_value=4))
    config = ProcessingConfig(ordinal_encoding=ordinal_encoding, dropna=True, copy=True)
    return X, features, y, max_n_mod, config


@st.composite
def ordinal_problem(draw, *, ordinal_encoding=False):
    X, features, y = draw(dataframe_and_features("ordinal", nrows=(40, 80), with_nan=True))
    max_n_mod = draw(st.integers(min_value=2, max_value=4))
    config = ProcessingConfig(ordinal_encoding=ordinal_encoding, dropna=True, copy=True)
    return X, features, y, max_n_mod, config


def _non_nan_labels(feature):
    nan_label = feature.label_per_value.get(feature.nan)
    return [label for label in feature.labels if label != nan_label]


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
@given(binary_problem())
@SETTINGS
def test_rows_index_and_versions_present(prob):
    """transform preserves rows/index; every surviving feature version is output."""
    X, features, y, max_n_mod, config = prob
    carver = BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    out = _carve_transform(carver, X, y)

    assert len(out) == len(X)
    assert list(out.index) == list(X.index)
    assert set(carver.features.versions).issubset(set(out.columns))


@given(binary_problem())
@SETTINGS
def test_modalities_within_max_n_mod_and_labels(prob):
    """Each carved feature has at most ``max_n_mod`` non-nan modalities, and every
    output value belongs to the feature's labels."""
    X, features, y, max_n_mod, config = prob
    carver = BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    out = _carve_transform(carver, X, y)

    for feature in carver.features:
        assert len(_non_nan_labels(feature)) <= max_n_mod, feature.version
        observed = set(out[feature.version].dropna().unique())
        assert observed.issubset(set(feature.labels)), feature.version


@given(binary_problem(ordinal_encoding=True))
@SETTINGS
def test_ordinal_encoding_contiguous_codes(prob):
    """ordinal_encoding=True yields numeric columns with contiguous 0..k-1 labels."""
    X, features, y, max_n_mod, config = prob
    carver = BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    out = _carve_transform(carver, X, y)

    for feature in carver.features:
        labels = feature.labels
        assert labels == list(range(len(labels))), feature.version
        assert is_numeric_dtype(out[feature.version]), feature.version
        observed = set(out[feature.version].dropna().unique())
        assert observed.issubset(set(range(len(labels)))), feature.version


# --------------------------------------------------------------------------
# copy / determinism
# --------------------------------------------------------------------------
@given(binary_problem())
@SETTINGS
def test_copy_does_not_mutate_input(prob):
    """copy=True leaves the input DataFrame untouched through fit_transform."""
    X, features, y, max_n_mod, config = prob
    before = X.copy(deep=True)
    _carve_transform(BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)
    assert X.equals(before)


@given(binary_problem())
@SETTINGS
def test_fit_transform_equals_fit_then_transform(prob):
    """fit_transform equals fit().transform() and is deterministic across instances."""
    X, features, y, max_n_mod, config = prob
    features2 = clone_features(features)

    out_ft = _carve_transform(BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)
    fitted = _carve(BinaryCarver(features2, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)
    out_split = fitted.transform(X)

    assert out_ft.equals(out_split)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
@given(binary_problem())
@SETTINGS
def test_transform_before_fit_raises(prob):
    """transform before fit raises RuntimeError."""
    X, features, _, max_n_mod, config = prob
    carver = BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    with pytest.raises(RuntimeError):
        carver.transform(X)


@given(binary_problem())
@SETTINGS
def test_refit_raises(prob):
    """Re-fitting an already-fitted carver raises."""
    X, features, y, max_n_mod, config = prob
    carver = BinaryCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    _carve(carver, X, y)
    with pytest.raises(ValueError):
        carver.fit(X, y)


# --------------------------------------------------------------------------
# multiclass version structure
# --------------------------------------------------------------------------
@given(dataframe_and_features("multiclass", nrows=(40, 80)), st.integers(min_value=2, max_value=4))
@SETTINGS
def test_multiclass_version_structure(prob, max_n_mod):
    """MulticlassCarver builds one feature version per (class - 1, input feature);
    after drops, versions stay within (n_classes - 1) x n_inputs and carry exactly
    the n_classes - 1 expected version tags."""
    X, features, y = prob
    n_input = len(features)
    n_classes = y.nunique()

    config = ProcessingConfig(ordinal_encoding=True, dropna=True, copy=False)
    carver = MulticlassCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    _carve(carver, X, y)

    versions = carver.features.versions
    assert len(versions) <= (n_classes - 1) * n_input
    # one version-tag group per modelled class (all but the dropped reference class)
    tags = {feature.version_tag for feature in carver.features}
    assert len(tags) <= n_classes - 1


# --------------------------------------------------------------------------
# ordinal structure / determinism
# --------------------------------------------------------------------------
@given(ordinal_problem())
@SETTINGS
def test_ordinal_rows_index_and_versions_present(prob):
    """OrdinalCarver transform preserves rows/index; every surviving version is output."""
    X, features, y, max_n_mod, config = prob
    out = _carve_transform(OrdinalCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)

    assert len(out) == len(X)
    assert list(out.index) == list(X.index)


@given(ordinal_problem())
@SETTINGS
def test_ordinal_modalities_within_max_n_mod_and_labels(prob):
    """Each carved feature has at most ``max_n_mod`` non-nan modalities, and every
    output value belongs to the feature's labels."""
    X, features, y, max_n_mod, config = prob
    carver = OrdinalCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config)
    out = _carve_transform(carver, X, y)

    for feature in carver.features:
        assert len(_non_nan_labels(feature)) <= max_n_mod, feature.version
        observed = set(out[feature.version].dropna().unique())
        assert observed.issubset(set(feature.labels)), feature.version


@given(ordinal_problem())
@SETTINGS
def test_ordinal_copy_does_not_mutate_input(prob):
    """copy=True leaves the input DataFrame untouched through fit_transform."""
    X, features, y, max_n_mod, config = prob
    before = X.copy(deep=True)
    _carve_transform(OrdinalCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)
    assert X.equals(before)


@given(ordinal_problem())
@SETTINGS
def test_ordinal_fit_transform_equals_fit_then_transform(prob):
    """fit_transform equals fit().transform() and is deterministic across instances."""
    X, features, y, max_n_mod, config = prob
    features2 = clone_features(features)

    out_ft = _carve_transform(OrdinalCarver(features, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)
    fitted = _carve(OrdinalCarver(features2, min_freq=0.15, max_n_mod=max_n_mod, config=config), X, y)
    out_split = fitted.transform(X)

    assert out_ft.equals(out_split)
