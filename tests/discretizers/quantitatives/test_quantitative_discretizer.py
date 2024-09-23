"""Set of tests for discretizers module."""

from numpy import inf
from pandas import DataFrame

from AutoCarver import Features
from AutoCarver.discretizers import QuantitativeDiscretizer


def test_quantitative_discretizer(x_train: DataFrame, target: str):
    """Tests QuantitativeDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    min_freq = 0.1
    features = Features(quantitatives=quantitatives)

    discretizer = QuantitativeDiscretizer(quantitatives=features, min_freq=min_freq)
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    assert not features("Discrete_Quantitative_lownan").values.contains(
        features("Discrete_Quantitative_lownan").nan
    ), "Missing order should not be grouped with ordinal_discretizer"

    assert all(
        x_discretized["Quantitative"].value_counts(normalize=True) >= min_freq
    ), "Non-nan value was not grouped"

    print(x_train.Discrete_Quantitative_rarevalue.value_counts(dropna=False, normalize=True))

    print(features("Discrete_Quantitative_rarevalue").content)
    assert features("Discrete_Quantitative_rarevalue").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        inf,
    ], (
        "Rare values should be grouped to the closest one and inf should be kept whatsoever "
        "(OrdinalDiscretizer)"
    )
