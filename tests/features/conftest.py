""" Defines fixtures for features pytests"""

from pytest import fixture


@fixture
def quantitative_features() -> list[str]:
    """List of quantitative raw features to be carved"""
    return [
        "Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative",
        "Discrete_Quantitative_rarevalue",
        "one",
        "one_nan",
    ]


@fixture
def qualitative_features() -> list[str]:
    """List of qualitative raw features to be carved"""
    return [
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


@fixture
def ordinal_features() -> list[str]:
    """List of ordinal raw features to be carved"""
    return [
        "Qualitative_Ordinal",
        "Qualitative_Ordinal_lownan",
        "Discrete_Qualitative_highnan",
    ]


@fixture
def values_orders() -> dict[str, list[str]]:
    """values_orders of raw features to be carved"""
    return {
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
