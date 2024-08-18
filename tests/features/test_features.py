"""Set of tests for features module.
"""

from pytest import raises

from AutoCarver.features import Features, GroupedList


def _features(
    quantitative_features: list[str],
    qualitative_features: list[str],
    ordinal_features: list[str],
    values_orders: dict[str, list[str]],
) -> None:
    """Tests Features

    Parameters
    ----------
    quantitative_features : list[str]
        List of quantitative raw features
    qualitative_features : list[str]
        List of qualitative raw features
    ordinal_features : list[str]
        List of ordinal raw features
    values_orders : dict[str, list[str]]
        values_orders of raw features
    """
    features = Features(
        categoricals=qualitative_features,
        quantitatives=quantitative_features,
        ordinals=ordinal_features,
        ordinal_values=values_orders,
    )
    # checking for initiation of ordinal features
    assert len(features(ordinal_features[0]).values) > 0, "non initiated ordinal values"
    assert (
        "High+" in features("Qualitative_Ordinal").label_per_value
    ), "non initiated ordinal labels"

    # checking for updates of values
    features.update(
        {"Qualitative_Ordinal": GroupedList(["High-", "High", "High+", "High++++"])}, replace=True
    )
    assert (
        features.ordinals[0].values == features("Qualitative_Ordinal").values
    ), "reference issue, not same Feature object"
    assert (
        "High++++" in features("Qualitative_Ordinal").values
    ), "reference issue, not same Feature object"

    # checking that an ordinal feature needs its values
    with raises(ValueError):
        Features(ordinals=["test"])

    # checking that a feature can not be both ordinal and categorical
    with raises(ValueError):
        Features(categoricals=["test"], ordinals=["test"], ordinal_values={"test": ["test2"]})
