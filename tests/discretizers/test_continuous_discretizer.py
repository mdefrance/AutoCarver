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
    for feature in features:
        feature.dropna = True
    min_freq = 0.1

    discretizer = ContinuousDiscretizer(
        features,
        min_freq,
        copy=True,
    )
    x_discretized = discretizer.fit_transform(x_train)

    assert all(
        x_discretized.Quantitative.value_counts(normalize=True) == min_freq
    ), "Wrong quantiles"
    assert discretizer.features("Discrete_Quantitative_highnan").values == [
        2.0,
        3.0,
        4.0,
        7.0,
        inf,
    ], "NaNs should not be added to the order"

    assert discretizer.features(
        "Discrete_Quantitative_highnan"
    ).has_nan, "NaNs should fitted per feature"

    assert discretizer.features("Discrete_Quantitative_lownan").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        inf,
    ], "NaNs should not be grouped whatsoever"

    assert discretizer.features("Discrete_Quantitative_rarevalue").values == [
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        inf,
    ], "Wrongly associated rare values"
