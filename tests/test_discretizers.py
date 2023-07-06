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

    discretizer = discretizers.QuantitativeDiscretizer(features, min_freq, copy=True)
    x_discretized = discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

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
        '__OTHER__': ['__NAN__', 'Category D', 'Category F', '__OTHER__'],
        'Category C': ['Category C'],
        'Category E': ['Category E'],
    }
    assert discretizer.values_orders['Qualitative_lownan'].contained == quali_lownan_expected, "If any, NaN values should be put into str_nan"

    expected_ordinal = {
        'Low+': ['Low', 'Low-', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['High-', 'Medium+', 'High'],
        'High+': ['High+']
    }
    expected_ordinal_lownan = {
        'Low+': ['Low-', 'Low', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['High-', 'Medium+', 'High'],
        'High+': ['High+']
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal, "Values not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan, "Values not correctly grouped or introduced nans."

def test_discretizer(x_train: DataFrame):
    # TODO
    pass