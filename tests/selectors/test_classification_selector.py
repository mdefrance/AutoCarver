"""Set of tests for ClassificationSelector module."""

from pandas import DataFrame
from pytest import FixtureRequest, fixture

# from AutoCarver.selectors import ClassificationSelector


@fixture(params=["binary_target", "multiclass_target"])
def target(request: type[FixtureRequest]) -> str:
    return request.param


def _classification_selector(
    x_train: DataFrame,
    target: str,
    quantitative_features: list[str],
    qualitative_features: list[str],
    ordinal_features: list[str],
    n_best: int,
) -> None:
    """Tests ClassificationSelector

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    quantitative_features : list[str]
        List of quantitative raw features to be carved
    qualitative_features : list[str]
        List of qualitative raw features to be carved
    ordinal_features : list[str]
        List of ordinal raw features to be carved
    n_best: int
        Number of features to be selected
    """

    # select the n_best most target associated qualitative features
    feature_selector = ClassificationSelector(
        n_best=n_best,
        qualitative_features=qualitative_features + ordinal_features,
        quantitative_features=quantitative_features,
        verbose=False,
    )
    best_features = feature_selector.select(x_train, x_train[target])

    # checking that the right features where selected
    expected = {
        3: {
            "binary_target": [
                "Discrete_Quantitative_highnan",
                "Quantitative",
                "Discrete_Quantitative",
                "Qualitative_Ordinal_lownan",
                "Qualitative_Ordinal",
                "Qualitative_highnan",
            ],
            "multiclass_target": [
                "Discrete_Quantitative_rarevalue",
                "Discrete_Quantitative_highnan",
                "Quantitative",
                "Discrete_Qualitative_highnan",
                "Discrete_Qualitative_rarevalue_noorder",
                "Discrete_Qualitative_noorder",
            ],
        },
        5: {
            "binary_target": [
                "Discrete_Quantitative_highnan",
                "Quantitative",
                "Discrete_Quantitative",
                "Discrete_Quantitative_lownan",
                "Discrete_Quantitative_rarevalue",
                "Qualitative_Ordinal_lownan",
                "Qualitative_Ordinal",
                "Qualitative_highnan",
                "Qualitative_grouped",
                "Qualitative_lownan",
            ],
            "multiclass_target": [
                "Discrete_Quantitative_rarevalue",
                "Discrete_Quantitative_highnan",
                "Quantitative",
                "Discrete_Quantitative_lownan",
                "Discrete_Quantitative",
                "Discrete_Qualitative_highnan",
                "Discrete_Qualitative_noorder",
                "Discrete_Qualitative_rarevalue_noorder",
                "Qualitative_Ordinal",
                "Qualitative_Ordinal_lownan",
            ],
        },
    }
    assert all(feature in best_features for feature in expected[n_best][target]) and all(
        feature in expected[n_best][target] for feature in best_features
    ), "Not correctly selected features"
    # checking for correctly selected number of features -> not possible
    # assert len(list(feature for feature in best_features if feature in quantitative_features)) <= n_best
