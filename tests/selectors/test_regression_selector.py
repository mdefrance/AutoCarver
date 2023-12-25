"""Set of tests for RegressionSelector module."""

from pandas import DataFrame

from AutoCarver.selectors import RegressionSelector


def test_regression_selector(
    x_train: DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    ordinal_features: list[str],
    n_best: int,
) -> None:
    """Tests RegressionSelector

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    quantitative_features : list[str]
        List of quantitative raw features to be carved
    qualitative_features : list[str]
        List of qualitative raw features to be carved
    ordinal_features : list[str]
        List of ordinal raw features to be carved
    n_best: int
        Number of features to be selected
    """

    target = "continuous_target"

    # select the n_best most target associated qualitative features
    feature_selector = RegressionSelector(
        n_best=n_best,
        qualitative_features=qualitative_features + ordinal_features,
        quantitative_features=quantitative_features,
        verbose=False,
    )
    best_features = feature_selector.select(x_train, x_train[target])

    expected = {
        3: [
            "Discrete_Quantitative_highnan",
            "Discrete_Quantitative",
            "Discrete_Quantitative_lownan",
            "Qualitative_Ordinal",
            "Discrete_Qualitative_noorder",
            "Discrete_Qualitative_rarevalue_noorder",
        ],
        5: [
            "Discrete_Quantitative_highnan",
            "Discrete_Quantitative",
            "Discrete_Quantitative_lownan",
            "Discrete_Quantitative_rarevalue",
            "Quantitative",
            "Qualitative_Ordinal",
            "Discrete_Qualitative_noorder",
            "Discrete_Qualitative_rarevalue_noorder",
            "Qualitative",
            "Qualitative_grouped",
        ],
    }
    assert all(feature in best_features for feature in expected[n_best]) and all(
        feature in expected[n_best] for feature in best_features
    ), "Not correctly selected features"
    # checking for correctly selected number of features -> not possible
    # assert len(list(f for f in best_features if f in quantitative_features)) <= n_best
