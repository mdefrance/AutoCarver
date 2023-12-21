"""Set of tests for continuous_carver module.
"""

from json import dumps, loads

from pandas import DataFrame
from pytest import raises

from auto_carver import ContinuousCarver, load_carver
from auto_carver.discretizers import ChainedDiscretizer


def test_continuous_carver(
    x_train: DataFrame,
    x_train_wrong_2: DataFrame,
    x_dev_1: DataFrame,
    x_dev_wrong_1: DataFrame,
    x_dev_wrong_2: DataFrame,
    x_dev_wrong_3: DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    ordinal_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    min_freq_mod: float,
    output_dtype: str,
    dropna: bool,
    copy: bool,
) -> None:
    """Tests ContinuousCarver

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
    x_dev_wrong_3 : DataFrame
        Simulated wrong Dev DataFrame
    quantitative_features : list[str]
        List of quantitative raw features to be carved
    qualitative_features : list[str]
        List of qualitative raw features to be carved
    ordinal_features : list[str]
        List of ordinal raw features to be carved
    values_orders : dict[str, list[str]]
        values_orders of raw features to be carved
    chained_features : list[str]
        List of chained raw features to be chained
    level0_to_level1 : dict[str, list[str]]
        Chained orders level0 to level1 of features to be chained
    level1_to_level2 : dict[str, list[str]]
        Chained orders level1 to level2 of features to be chained
    min_freq_mod : float
        Minimum frequency per carved modalities
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

    target = "continuous_target"

    min_freq = 0.15

    chained_discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        unknown_handling="drop",
        copy=copy,
        n_jobs=1,
    )
    x_discretized = chained_discretizer.fit_transform(x_train)
    values_orders.update(chained_discretizer.values_orders)

    # minimum frequency per modality
    min_freq = 0.1
    max_n_mod = 4

    # tests with 'tschuprowt' measure
    auto_carver = ContinuousCarver(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        ordinal_features=ordinal_features,
        values_orders=values_orders,
        min_freq=min_freq,
        max_n_mod=max_n_mod,
        output_dtype=output_dtype,
        min_freq_mod=min_freq_mod,
        dropna=dropna,
        copy=copy,
        verbose=False,
        n_jobs=1,
    )
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train[target],
        X_dev=x_dev_1,
        y_dev=x_dev_1[target],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)

    # testing that attributes where correctly used
    assert all(
        x_discretized[auto_carver.features].nunique() <= max_n_mod
    ), "Too many buckets after carving of train sample"
    assert all(
        x_dev_discretized[auto_carver.features].nunique() <= max_n_mod
    ), "Too many buckets after carving of test sample"

    # testing for differences between train and dev
    assert all(
        x_discretized[auto_carver.features].nunique()
        == x_dev_discretized[auto_carver.features].nunique()
    ), "More buckets in train or test samples"
    for feature in auto_carver.features:
        train_target_rate = x_discretized.groupby(feature)[target].mean().sort_values()
        dev_target_rate = x_dev_discretized.groupby(feature)[target].mean().sort_values()
        assert all(
            train_target_rate.index == dev_target_rate.index
        ), f"Not robust feature {feature} was not dropped, or robustness test not working"

        # checking for final modalities less frequent than min_freq_mod
        train_frequency = x_discretized[feature].value_counts(normalize=True, dropna=True)
        assert not any(
            train_frequency.values < auto_carver.min_freq_mod
        ), f"Some modalities of {feature} are less frequent than min_freq_mod in train"
        dev_frequency = x_dev_discretized[feature].value_counts(normalize=True, dropna=True)
        assert not any(
            dev_frequency.values < auto_carver.min_freq_mod
        ), f"Some modalities {feature} are less frequent than min_freq_mod in dev"

    # test that all values still are in the values_orders
    for feature in auto_carver.qualitative_features:
        fitted_values = auto_carver.values_orders[feature].values()
        init_values = raw_x_train[feature].fillna("__NAN__").unique()
        assert all(value in fitted_values for value in init_values), (
            "Missing value in output! Some values are been dropped for qualitative "
            f"feature: {feature}"
        )

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

    # transform dev with unexpected modal for a feature that passed through CategoricalDiscretizer
    auto_carver.transform(x_dev_wrong_1)

    # transform dev with unexpected nans for a feature that passed through CategoricalDiscretizer
    with raises(AssertionError):
        auto_carver.transform(x_dev_wrong_2)

    # transform dev with unexpected modal for a feature that didnt go through CategoricalDiscretizer
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
        n_jobs=1,
    )
    x_discretized = chained_discretizer.fit_transform(x_train_wrong_2, x_train_wrong_2[target])
    values_orders.update(chained_discretizer.values_orders)

    auto_carver = ContinuousCarver(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        ordinal_features=ordinal_features,
        values_orders=values_orders,
        min_freq=min_freq,
        max_n_mod=max_n_mod,
        min_freq_mod=min_freq_mod,
        output_dtype=output_dtype,
        dropna=dropna,
        copy=copy,
        verbose=False,
        n_jobs=1,
    )
    x_discretized = auto_carver.fit_transform(x_train_wrong_2, x_train_wrong_2[target])

    if not dropna:
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

    elif dropna:
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
