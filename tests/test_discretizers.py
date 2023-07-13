"""Set of tests for discretizers module."""

from AutoCarver.discretizers import discretizers
from pandas import DataFrame
from numpy import inf

def test_quantitative_discretizer(x_train: DataFrame):
    """Tests QuantitativeDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    features = ['Quantitative', 'Discrete_Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative_rarevalue']
    min_freq = 0.1

    discretizer = discretizers.QuantitativeDiscretizer(features, min_freq=min_freq)
    x_discretized = discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

    assert '__NAN__' in discretizer.values_orders["Discrete_Quantitative_lownan"], "Missing order should not be grouped with ordinal_discretizer"
    assert all(x_discretized.Quantitative.value_counts(normalize=True) >= min_freq), "Non-nan value was not grouped"
    assert discretizer.values_orders["Discrete_Quantitative_lownan"] == [1.0, 2.0, 3.0, inf, '__NAN__'], "NaNs should not be grouped whatsoever"
    assert discretizer.values_orders["Discrete_Quantitative_rarevalue"] == [1.0, 2.0, 3.0, inf], "Rare values should be grouped to the closest one (OrdinalDiscretizer)"


def test_qualitative_discretizer(x_train: DataFrame):
    """Tests QualitativeDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """


    features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan"]
    ordinal_features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    values_orders = {
        "Qualitative_Ordinal": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
        "Qualitative_Ordinal_lownan": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
    }

    min_freq = 0.1

    discretizer = discretizers.QualitativeDiscretizer(features, min_freq, ordinal_features=ordinal_features, values_orders=values_orders, copy=True, verbose=True)
    x_discretized = discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])


    quali_expected = {
        '__OTHER__': ['Category A', 'Category D', 'Category F', '__OTHER__'],
         'Category C': ['Category C'],
         'Category E': ['Category E'],
    }
    assert discretizer.values_orders['Qualitative'].contained == quali_expected, "Values less frequent than min_freq should be grouped into default_value"
    quali_lownan_expected = {
        '__NAN__': ['__NAN__'],
        '__OTHER__': ['Category D', 'Category F', '__OTHER__'],
        'Category C': ['Category C'],
        'Category E': ['Category E'],
    }
    assert discretizer.values_orders['Qualitative_lownan'].contained == quali_lownan_expected, "If any, NaN values should be put into str_nan and kept by themselves"

    expected_ordinal = {
        'Low+': ['Low-', 'Low', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['Medium+', 'High-', 'High'],
        'High+': ['High+']
    }
    expected_ordinal_lownan = {
        'Low+': ['Low', 'Low-', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['Medium+', 'High-', 'High'],
        'High+': ['High+'],
        '__NAN__': ['__NAN__']
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal, "Values not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan, "NaNs should stay by themselves."

def test_discretizer(x_train: DataFrame, x_test_1: DataFrame):
    """Tests Discretizer

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



    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.1

    # discretizing features
    discretizer = discretizers.Discretizer(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        ordinal_features=ordinal_features,
        values_orders=values_orders,
        min_freq=min_freq,
        copy=True,
    )
    x_discretized = discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])
    x_test_discretized = discretizer.transform(x_test_1)



    assert all(x_discretized.Quantitative.value_counts(normalize=True) >= min_freq), "Non-nan value was not grouped"
    assert discretizer.values_orders["Discrete_Quantitative_lownan"] == [1.0, 2.0, 3.0, inf, '__NAN__'], "NaNs should not be grouped whatsoever"
    assert discretizer.values_orders["Discrete_Quantitative_rarevalue"] == [1.0, 2.0, 3.0, inf], "Rare values should be grouped to the closest one (OrdinalDiscretizer)"


    quali_expected = {
        '__OTHER__': ['Category A', 'Category D', 'Category F', '__OTHER__'],
        'Category C': ['Category C'],
        'Category E': ['Category E'],
    }
    assert discretizer.values_orders['Qualitative'].contained == quali_expected, "Values less frequent than min_freq should be grouped into default_value"
    quali_lownan_expected = {
        '__NAN__' : ['__NAN__'],
        '__OTHER__': ['Category D', 'Category F', '__OTHER__'],
        'Category C': ['Category C'],
        'Category E': ['Category E'],
    }
    assert discretizer.values_orders['Qualitative_lownan'].contained == quali_lownan_expected, "If any, NaN values should be put into str_nan and kept by themselves"

    expected_ordinal = {
        'Low+': ['Low-', 'Low', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['Medium+', 'High-', 'High'],
        'High+': ['High+']
    }
    expected_ordinal_lownan = {
        'Low+': ['Low', 'Low-', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['Medium+', 'High-', 'High'],
        'High+': ['High+'],
        '__NAN__': ['__NAN__'],
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal, "Values not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan, "NaNs should stay by themselves."

    # Testing out qualitative with int/float values inside -> StringDiscretizer
    expected = {
        '2': [2.0, '2'],
        '4': [4.0, '4'],
        '1': [1.0, '1'],
        '3': [3.0, '3'],
        '__OTHER__': [0.5, '0.5', 6.0, '6', 5.0, '5', '__OTHER__']
    }
    assert discretizer.values_orders["Discrete_Qualitative_rarevalue_noorder"].contained == expected, "Qualitative features with float values should be converted to string and there values stored in the values_orders"
    expected = {
        '2': [2, '2'],
        '4': [4, '4'],
        '1': [1, '1'],
        '3': [3, '3'],
        '__OTHER__': [7, '7', 6, '6', 5, '5', '__OTHER__']
    }
    assert discretizer.values_orders["Discrete_Qualitative_noorder"].contained == expected, "Qualitative features with int values should be converted to string and there values stored in the values_orders"
    expected = {
        '2': ['1', 2.0, '2'],
        '3': [3.0, '3'],
        '4': [4.0, '4'],
        '5': [6.0, '6', 7.0, '7', 5.0, '5'],
        '__NAN__': ['__NAN__']
    }
    assert discretizer.values_orders["Discrete_Qualitative_highnan"].contained == expected, "Ordinal qualitative features with int or float values that contain nan should be converted to string and there values stored in the values_orders"

    # checking for inconsistancies in tranform
    for feature in discretizer.features:
        test_unique = x_test_discretized[feature].unique()
        train_unique = x_discretized[feature].unique()
        assert all(value in test_unique for value in train_unique), "Missing value from test (at transform step)"
        assert all(value in train_unique for value in test_unique), "Missing value from train (at transform step)"
