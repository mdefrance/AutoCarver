"""Set of tests for quantitative_discretizers module."""

from numpy import inf
from pandas import DataFrame

from AutoCarver.discretizers import ContinuousDiscretizer
from AutoCarver.features import Features


def test_continuous_discretizer(x_train: DataFrame):
    """Tests ContinuousDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    features = Features(quantitatives=quantitatives)
    min_freq = 0.1

    discretizer = ContinuousDiscretizer(
        features,
        min_freq,
        copy=True,
    )
    x_discretized = discretizer.fit(x_train)
    features.dropna = True
    x_discretized = discretizer.transform(x_train)
    features.dropna = False

    assert all(
        x_discretized.Quantitative.value_counts(normalize=True) == min_freq
    ), "Wrong quantiles"

    assert features("Discrete_Quantitative_highnan").values == [
        2.0,
        3.0,
        4.0,
        7.0,
        inf,
    ], "NaNs should not be added to the order"

    assert features("Discrete_Quantitative_highnan").has_nan, "Should have nan"

    assert features("Discrete_Quantitative_lownan").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        inf,
    ], "NaNs should not be grouped whatsoever"

    assert features("Discrete_Quantitative_rarevalue").values == [
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        inf,
    ], "Wrongly associated rare values"
