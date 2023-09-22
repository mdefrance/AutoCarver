"""Set of tests for FeatureSelector module."""

from pandas import DataFrame

from AutoCarver.feature_selection import FeatureSelector


def test_feature_selector(x_train: DataFrame) -> None:
    """Tests FeatureSelector

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    quantitative_features = [
        "Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative",
        "Discrete_Quantitative_rarevalue",
    ]
    qualitative_features = [
        "Qualitative",
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Discrete_Qualitative_noorder",
        "Discrete_Qualitative_lownan_noorder",
        "Discrete_Qualitative_rarevalue_noorder",
    ]
    ordinal_features = [
        "Qualitative_Ordinal",
        "Qualitative_Ordinal_lownan",
        "Discrete_Qualitative_highnan",
    ]

    # select the best 5 most target associated qualitative features
    quali_selector = FeatureSelector(
        n_best=5,
        qualitative_features=qualitative_features + ordinal_features,
    )
    best_features = quali_selector.select(x_train, x_train["quali_ordinal_target"])

    expected = [
        "Qualitative_Ordinal_lownan",
        "Qualitative_Ordinal",
        "Qualitative_highnan",
        "Qualitative_grouped",
        "Qualitative_lownan",
    ]
    assert best_features == expected, "Not correctly selected qualitative features"

    # select the best 5 most target associated qualitative features
    quanti_selector = FeatureSelector(
        n_best=5,
        quantitative_features=quantitative_features,
    )
    best_features = quanti_selector.select(x_train, x_train["quali_ordinal_target"])

    expected = [
        "Discrete_Quantitative_highnan",
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    assert best_features == expected, "Not correctly selected qualitative features"