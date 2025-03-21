""" Set of tests for base carver module."""

from pandas import DataFrame, Series
from pytest import FixtureRequest, fixture, raises

from AutoCarver.carvers.utils.base_carver import BaseCarver, Samples, discretize
from AutoCarver.combinations import CramervCombinations, KruskalCombinations, TschuprowtCombinations
from AutoCarver.discretizers.utils.base_discretizer import Sample
from AutoCarver.features import Features
from AutoCarver.utils.dependencies import has_idisplay

_has_idisplay = has_idisplay()

# removing abstractmethod
BaseCarver.__abstractmethods__ = set()


@fixture
def sample_data():
    """Fixture for sample data used in tests."""
    X_train = DataFrame({"feature1": [1, 2, 3, float("nan")], "feature2": [4, 5, 6, 7]})
    y_train = Series([0, 1, 0, 0])
    X_dev = DataFrame(
        {"feature1": [float("nan"), 2, 1, float("nan"), 2, 1], "feature2": [6, 5, 4, 6, 5, 4]}
    )
    y_dev = Series([1, 0, 1, 1, 0, 1])
    train_sample = Sample(X=X_train, y=y_train)
    dev_sample = Sample(X=X_dev, y=y_dev)
    return train_sample, dev_sample


def test_initialization_default():
    """Test default initialization of Samples."""
    samples = Samples()
    assert samples.train.X is None
    assert samples.dev.X is None
    assert samples.train.y is None
    assert samples.dev.y is None


def test_initialization_with_values(sample_data):
    """Test initialization of Samples with provided values."""
    train_sample, dev_sample = sample_data
    samples = Samples(train=train_sample, dev=dev_sample)
    assert samples.train.X.equals(train_sample.X)
    assert samples.train.y.equals(train_sample.y)
    assert samples.dev.X.equals(dev_sample.X)
    assert samples.dev.y.equals(dev_sample.y)


def test_attributes(sample_data):
    """Test attributes of Samples."""
    train_sample, dev_sample = sample_data
    samples = Samples(train=train_sample, dev=dev_sample)
    assert isinstance(samples.train, Sample)
    assert isinstance(samples.dev, Sample)
    assert samples.train.X.equals(train_sample.X)
    assert samples.train.y.equals(train_sample.y)
    assert samples.dev.X.equals(dev_sample.X)
    assert samples.dev.y.equals(dev_sample.y)


@fixture
def samples(sample_data):
    """Fixture for Samples used in tests."""
    train_sample, dev_sample = sample_data
    return Samples(train=train_sample, dev=dev_sample)


@fixture(params=[KruskalCombinations, CramervCombinations, TschuprowtCombinations])
def evaluator(request: FixtureRequest):
    """Fixture for evaluator used in tests."""
    return request.param()


@fixture
def features():
    """Fixture for features used in tests."""
    return Features(["feature1", "feature2"])


def test_initialization(features, evaluator):
    """Test initialization of BaseCarver."""
    carver = BaseCarver(
        features=features,
        min_freq=0.1,
        combinations=evaluator,
        dropna=True,
        verbose=True,
        n_jobs=2,
    )
    assert carver.min_freq == 0.1
    assert carver.dropna is True
    assert carver.verbose is True
    assert carver.n_jobs == 2
    assert carver.combinations == evaluator
    assert carver.combinations.min_freq == 0.1
    assert carver.combinations.verbose is True
    assert carver.combinations.dropna is True


def test_pretty_print(features, evaluator):
    """Test pretty_print property of BaseCarver."""
    carver = BaseCarver(features=features, min_freq=0.1, combinations=evaluator, verbose=True)

    assert carver.pretty_print == (carver.verbose and _has_idisplay)


def test_prepare_data_raises_value_error(features, evaluator, samples):
    """Test _prepare_data method raises ValueError when y is None."""
    carver = BaseCarver(features=features, min_freq=0.1, combinations=evaluator, verbose=True)
    samples.train.y = None
    with raises(ValueError):
        carver._prepare_data(samples)


def test_prepare_data(features, evaluator, samples):
    """Test _prepare_data method of BaseCarver."""
    carver = BaseCarver(features=features, min_freq=0.1, combinations=evaluator, verbose=True)
    prepared_samples = carver._prepare_data(samples)
    print(prepared_samples.train.X)
    print(samples.train.X)

    expected_train = DataFrame(
        {"feature1": ["1", "2", "3", features[0].nan], "feature2": ["4", "5", "6", "7"]}
    )
    expected_dev = DataFrame(
        {
            "feature1": [features[0].nan, "2", "1", features[0].nan, "2", "1"],
            "feature2": ["6", "5", "4", "6", "5", "4"],
        }
    )
    y_train = Series([0, 1, 0, 0])
    y_dev = Series([1, 0, 1, 1, 0, 1])

    assert expected_train.equals(samples.train.X)
    assert expected_dev.equals(samples.dev.X)

    assert y_train.equals(samples.train.y)
    assert y_dev.equals(samples.dev.y)

    assert carver.features.dropna is True
    for feature in features:
        assert feature.dropna is True


def test_discretize_train(features, samples):
    """Test discretize function for train samples."""
    discretizer_min_freq = 0.1
    samples.dev = Sample(X=None)
    samples = discretize(features, samples, discretizer_min_freq)
    assert samples.train.X is not None
    assert samples.dev.X is None


def test_discretize_dev(features, samples):
    """Test discretize function for dev samples."""
    discretizer_min_freq = 0.1
    samples = discretize(features, samples, discretizer_min_freq)
    assert samples.train.X is not None
    assert samples.dev.X is not None
