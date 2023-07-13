"""Set of tests for quantitative_discretizers module."""

from AutoCarver.discretizers.utils.type_discretizers import *
from pandas import DataFrame

def test_string_discretizer(x_train: DataFrame) -> None:
    """Tests StringDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    qualitative_features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan", "Discrete_Qualitative_noorder", "Discrete_Qualitative_lownan_noorder", "Discrete_Qualitative_rarevalue_noorder"]
    ordinal_features = ["Qualitative_Ordinal_lownan", "Discrete_Qualitative_highnan"]
    values_orders = {
        "Qualitative_Ordinal": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
        "Qualitative_Ordinal_lownan": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
        "Discrete_Qualitative_highnan" : ["1", "2", "3", "4", "5", "6", "7"],
    }


    discretizer = StringDiscretizer(
        features=list(set(ordinal_features + qualitative_features)), values_orders=values_orders
    )
    x_discretized = discretizer.fit_transform(x_train)

    expected = {'2': [2, '2'],
    '4': [4, '4'],
    '3': [3, '3'],
    '7': [7, '7'],
    '1': [1, '1'],
    '5': [5, '5'],
    '6': [6, '6']}
    assert discretizer.values_orders['Discrete_Qualitative_noorder'].contained == expected, "Not correctly converted for qualitative with integers"

    expected = {'2': [2.0, '2'],
    '4': [4.0, '4'],
    '3': [3.0, '3'],
    '1': [1.0, '1'],
    '5': [5.0, '5'],
    '6': [6.0, '6']}
    assert discretizer.values_orders['Discrete_Qualitative_lownan_noorder'].contained == expected, "Not correctly converted for qualitative with integers and nans"

    expected = {'2': [2.0, '2'],
    '4': [4.0, '4'],
    '3': [3.0, '3'],
    '0.5': [0.5, '0.5'],
    '1': [1.0, '1'],
    '5': [5.0, '5'],
    '6': [6.0, '6']}
    assert discretizer.values_orders['Discrete_Qualitative_rarevalue_noorder'].contained == expected, "Not correctly converted for qualitative with integers and floats"

    expected = {'Low-': ['Low-'],
    'Low': ['Low'],
    'Low+': ['Low+'],
    'Medium-': ['Medium-'],
    'Medium': ['Medium'],
    'Medium+': ['Medium+'],
    'High-': ['High-'],
    'High': ['High'],
    'High+': ['High+']}
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected, "No conversion for already string features"
    
    expected = {'Low-': ['Low-'],
    'Low': ['Low'],
    'Low+': ['Low+'],
    'Medium-': ['Medium-'],
    'Medium': ['Medium'],
    'Medium+': ['Medium+'],
    'High-': ['High-'],
    'High': ['High'],
    'High+': ['High+']}
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected, "No conversion for not specified featues"

    expected = {'1': ['1'], '2': [2.0, '2'], '3': [3.0, '3'], '4': [4.0, '4'], '5': [5.0, '5'], '6': [6.0, '6'], '7': [7.0, '7']}
    assert discretizer.values_orders['Discrete_Qualitative_highnan'].contained == expected, "Original order should be kept for ordinal features"
