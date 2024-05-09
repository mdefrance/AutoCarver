"""Set of tests for features module.
"""

from AutoCarver.discretizers import GroupedList
from AutoCarver.features import Features


def test_features(
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
    features.update({"Qualitative_Ordinal": GroupedList(["High-", "High", "High+", "High++++"])})
    assert (
        features.ordinals[0].values == features("Qualitative_Ordinal").values
    ), "reference issue, not same Feature object"
    assert "High++++" in features("Qualitative_Ordinal").values
