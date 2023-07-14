"""Set of tests for auto_carver module."""

from AutoCarver.auto_carver import *
from pytest import fixture

@fixture(scope="module", params=["float", "str"])
def output_dtype(request) -> str:
    return request.param

@fixture(scope="module", params=[True, False])
def dropna(request) -> bool:
    return request.param

@fixture(scope="module", params=["tschuprowt", "cramerv"])
def sort_by(request) -> str:
    return request.param

def test_auto_carver(x_train: DataFrame, x_test_1: DataFrame, output_dtype: str, dropna: bool, sort_by: str) -> None:
    """Tests AutoCarver

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    x_test_1 : DataFrame
        Simulated Test DataFrame
    """

    quantitative_features = ['Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative', 'Discrete_Quantitative_rarevalue']
    qualitative_features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan", "Discrete_Qualitative_noorder", "Discrete_Qualitative_lownan_noorder", "Discrete_Qualitative_rarevalue_noorder"]
    ordinal_features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan", "Discrete_Qualitative_highnan"]
    values_orders = {
        "Qualitative_Ordinal": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
        "Qualitative_Ordinal_lownan": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
        "Discrete_Qualitative_highnan" : ["1", "2", "3", "4", "5", "6", "7"],
    }



    # minimum frequency per modality
    min_freq = 0.06
    max_n_mod = 4

    # tests with 'tschuprowt' measure
    auto_carver = AutoCarver(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        ordinal_features=ordinal_features,
        values_orders=values_orders,
        min_freq=min_freq,
        max_n_mod=max_n_mod,
        sort_by=sort_by,
        output_dtype=output_dtype,
        dropna=dropna,
        copy=True,
        verbose=False,
    )
    x_discretized = auto_carver.fit_transform(x_train, x_train["quali_ordinal_target"], X_test=x_test_1, y_test=x_test_1["quali_ordinal_target"])
    x_test_discretized = auto_carver.transform(x_test_1)

    assert all(x_discretized[auto_carver.features].nunique() <= max_n_mod), "Too many values after carving of train sample"
    assert all(x_test_discretized[auto_carver.features].nunique() <= max_n_mod), "Too many values after carving of test sample"
    assert all(x_discretized[auto_carver.features].nunique() == x_test_discretized[auto_carver.features].nunique()), "More values in train or test samples"

    
    # test that all values still are in the values_orders
    for feature in auto_carver.qualitative_features:
        fitted_values = auto_carver.values_orders[feature].values()
        init_values = x_train[feature].fillna('__NAN__').unique()
        assert all(value in fitted_values for value in init_values), f"Missing value in output! Some values are been dropped for qualitative feature: {feature}"
    
    # testing output of nans
    if not dropna:
        assert all(x_train[auto_carver.features].isna().mean()  == x_discretized[auto_carver.features].isna().mean()), "Some Nans are being dropped (grouped) or more nans than expected"
    else:
        assert all(x_discretized[auto_carver.features].isna().mean() == 0), "Some Nans are not dropped (grouped)"

# def test_auto_carver_copy(x_train: DataFrame, x_test_1: DataFrame, output_dtype: str, dropna: bool, sort_by: str) -> None:

    # TODO test missing values in test
    # TODO try with copy = False
    # TODO: test avec chained_discretizer
