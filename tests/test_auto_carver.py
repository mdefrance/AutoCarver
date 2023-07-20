"""Set of tests for auto_carver module.
"""

from json import dumps, loads

from pandas import DataFrame
from pytest import fixture, raises

from AutoCarver import AutoCarver, load_carver
from AutoCarver.discretizers import ChainedDiscretizer


@fixture(scope="module", params=["float", "str"])
def output_dtype(request) -> str:
    return request.param


@fixture(scope="module", params=[True, False])
def dropna(request) -> bool:
    return request.param


@fixture(scope="module", params=["tschuprowt", "cramerv"])
def sort_by(request) -> str:
    return request.param


@fixture(scope="module", params=[True, False])
def copy(request) -> bool:
    return request.param


def test_auto_carver(
    x_train: DataFrame,
    x_train_wrong_1: DataFrame,
    x_train_wrong_2: DataFrame,
    x_dev_1: DataFrame,
    x_dev_wrong_1: DataFrame,
    x_dev_wrong_2: DataFrame,
    x_dev_wrong_3: DataFrame,
    output_dtype: str,
    dropna: bool,
    sort_by: str,
    copy: bool,
) -> None:
    """Tests AutoCarver

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    x_train_wrong_1 : DataFrame
        Simulated Train DataFrame with unknown values (without nans)
    x_train_wrong_2 : DataFrame
        Simulated Train DataFrame with unknown values (with nans)
    x_dev_1 : DataFrame
        Simulated Dev DataFrame
    x_dev_wrong_1 : DataFrame
        Simulated wrong Dev DataFrame with unexpected modality
    x_dev_wrong_2 : DataFrame
        Simulated wrong Dev DataFrame with unexpected nans
    output_dtype : str
        Output type 'str' or 'float'
    dropna : bool
        Whether or note to drop nans
    sort_by : str
        Sorting measure 'tschuprowt' or 'cramerv'
    copy : bool
        Whether or not to copy the input dataset
    """
    # copying x_train for comparison purposes
    raw_x_train = x_train.copy()

    # list of feaures
    quantitative_features = [
        "Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative",
        "Discrete_Quantitative_rarevalue",
        "one",
        "one_nan",
    ]
    qualitative_features = [
        "Qualitative",
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Discrete_Qualitative_noorder",
        "Discrete_Qualitative_lownan_noorder",
        "Discrete_Qualitative_rarevalue_noorder",
        "nan",
        "ones",
        "ones_nan",
    ]
    ordinal_features = [
        "Qualitative_Ordinal",
        "Qualitative_Ordinal_lownan",
        "Discrete_Qualitative_highnan",
    ]
    values_orders = {
        "Qualitative_Ordinal": [
            "Low-",
            "Low",
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
        "Qualitative_Ordinal_lownan": [
            "Low-",
            "Low",
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
        "Discrete_Qualitative_highnan": ["1", "2", "3", "4", "5", "6", "7"],
    }
    chained_features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]

    level0_to_level1 = {
        "Lows": ["Low-", "Low", "Low+", "Lows"],
        "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
        "Highs": ["High-", "High", "High+", "Highs"],
    }
    level1_to_level2 = {
        "Worst": ["Lows", "Mediums", "Worst"],
        "Best": ["Highs", "Best"],
    }

    min_freq = 0.15

    chained_discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        unknown_handling="drop",
        copy=copy,
    )
    x_discretized = chained_discretizer.fit_transform(x_train)
    values_orders.update(chained_discretizer.values_orders)

    # minimum frequency per modality
    min_freq = 0.1
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
        copy=copy,
        verbose=False,
    )
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train["quali_ordinal_target"],
        X_dev=x_dev_1,
        y_dev=x_dev_1["quali_ordinal_target"],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)

    assert all(
        x_discretized[auto_carver.features].nunique() <= max_n_mod
    ), "Too many values after carving of train sample"
    assert all(
        x_dev_discretized[auto_carver.features].nunique() <= max_n_mod
    ), "Too many values after carving of test sample"
    assert all(
        x_discretized[auto_carver.features].nunique()
        == x_dev_discretized[auto_carver.features].nunique()
    ), "More values in train or test samples"

    # test that all values still are in the values_orders
    for feature in auto_carver.qualitative_features:
        fitted_values = auto_carver.values_orders[feature].values()
        init_values = raw_x_train[feature].fillna("__NAN__").unique()
        assert all(
            value in fitted_values for value in init_values
        ), f"Missing value in output! Some values are been dropped for qualitative feature: {feature}"

    # testing output of nans
    if not dropna:
        assert all(
            raw_x_train[auto_carver.features].isna().mean()
            == x_discretized[auto_carver.features].isna().mean()
        ), "Some Nans are being dropped (grouped) or more nans than expected"
    else:
        assert all(
            x_discretized[auto_carver.features].isna().mean() == 0
        ), "Some Nans are not dropped (grouped)"

    # testing copy functionnality
    if copy:
        assert all(
            x_discretized[auto_carver.features].fillna("__NAN__")
            == x_train[auto_carver.features].fillna("__NAN__")
        ), "Not copied correctly"

    # testing json serialization
    json_serialized_auto_carver = dumps(auto_carver.to_json())
    loaded_carver = load_carver(loads(json_serialized_auto_carver))

    assert all(
        loaded_carver.summary() == auto_carver.summary()
    ), "Non-identical AutoCarver when loading JSON"

    # testing to transform dev set with unexpected modality for a feature that passed through DefaultDiscretizer
    auto_carver.transform(x_dev_wrong_1)

    # testing to transform dev set with unexpected nans for a feature that passed through DefaultDiscretizer
    with raises(AssertionError):
        auto_carver.transform(x_dev_wrong_2)

    # testing to transform dev set with unexpected modality for a feature that did not pass through DefaultDiscretizer
    with raises(AssertionError):
        auto_carver.transform(x_dev_wrong_3)

    # testing with unknown values in chained discretizer
    chained_features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    values_orders = {
        "Qualitative_Ordinal_lownan": [
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
        "Qualitative_Ordinal_highnan": [
            "Low-",
            "Low",
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
        "Discrete_Qualitative_highnan": ["1", "2", "3", "4", "5", "6", "7"],
    }

    level0_to_level1 = {
        "Lows": ["Low-", "Low", "Low+", "Lows"],
        "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
        "Highs": ["High-", "High", "High+", "Highs"],
        "ALONE": ["ALONE"],
    }
    level1_to_level2 = {
        "Worst": ["Lows", "Mediums", "Worst"],
        "Best": ["Highs", "Best"],
        "BEST": ["ALONE", "BEST"],
    }

    min_freq = 0.15
    max_n_mod = 4
    chained_discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        unknown_handling="drop",
        copy=True,
    )
    x_discretized = chained_discretizer.fit_transform(
        x_train_wrong_2, x_train_wrong_2["quali_ordinal_target"]
    )
    values_orders.update(chained_discretizer.values_orders)

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
        copy=copy,
        verbose=False,
    )
    x_discretized = auto_carver.fit_transform(
        x_train_wrong_2, x_train_wrong_2["quali_ordinal_target"]
    )

    if not dropna and sort_by == "cramerv":
        expected = {
            "Mediums": [
                "Low+",
                "Low",
                "Low-",
                "Lows",
                "Worst",
                "Medium+",
                "Medium",
                "Medium-",
                "Mediums",
            ],
            "High+": ["High", "High-", "Highs", "Best", "ALONE", "BEST", "High+"],
            "__NAN__": ["unknown", "__NAN__"],
        }
        assert (
            auto_carver.values_orders["Qualitative_Ordinal_lownan"].content == expected
        ), "Unknown modalities should be kept in the order"

    elif dropna and sort_by == "tschuprowt":
        expected = {
            "Mediums": [
                "Low+",
                "Low",
                "Low-",
                "Lows",
                "Worst",
                "Medium+",
                "Medium",
                "Medium-",
                "Mediums",
            ],
            "High+": ["High", "High-", "Highs", "Best", "ALONE", "BEST", "High+"],
            "__NAN__": ["unknown", "__NAN__"],
        }
        assert (
            auto_carver.values_orders["Qualitative_Ordinal_lownan"].content == expected
        ), "Unknown modalities should be kept in the order"
