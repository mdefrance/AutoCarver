"""Set of tests for ClassificationSelector module."""

from pandas import DataFrame

from AutoCarver.selectors import ClassificationSelector


def test_classification_selector(x_train: DataFrame) -> None:
    """Tests ClassificationSelector

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    target = "binary_target"

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
    quali_selector = ClassificationSelector(
        n_best=5,
        qualitative_features=qualitative_features + ordinal_features,
    )
    best_features = quali_selector.select(x_train, x_train[target])

    expected = {
        "binary_target": [
            "Qualitative_Ordinal_lownan",
            "Qualitative_Ordinal",
            "Qualitative_highnan",
            "Qualitative_grouped",
            "Qualitative_lownan",
        ],
        "continuous_target": [
            "Discrete_Qualitative_rarevalue_noorder",
            "Discrete_Qualitative_noorder",
            "Qualitative_Ordinal",
            "Discrete_Qualitative_lownan_noorder",
            "Discrete_Qualitative_highnan",
        ],
    }
    assert all(
        feature in best_features for feature in expected[target]
    ), "Not correctly selected qualitative features"

    # select the best 5 most target associated qualitative features
    quanti_selector = ClassificationSelector(
        n_best=5,
        quantitative_features=quantitative_features,
    )
    best_features = quanti_selector.select(x_train, x_train[target])

    expected = {
        "binary_target": [
            "Discrete_Quantitative_highnan",
            "Quantitative",
            "Discrete_Quantitative",
            "Discrete_Quantitative_lownan",
            "Discrete_Quantitative_rarevalue",
        ],
        "continuous_target": [],
    }
    assert all(
        feature in best_features for feature in expected[target]
    ), "Not correctly selected qualitative features"
