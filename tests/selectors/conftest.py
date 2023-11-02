""" Defines fixtures for carvers pytests"""

from pytest import FixtureRequest, fixture


@fixture
def quantitative_features() -> list[str]:
    """List of quantitative raw features to be carved"""
    return [
        "Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative",
        "Discrete_Quantitative_rarevalue",
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
    ]


@fixture
def ordinal_features() -> list[str]:
    """List of ordinal raw features to be carved"""
    return [
        "Qualitative_Ordinal",
        "Qualitative_Ordinal_lownan",
        "Discrete_Qualitative_highnan",
    ]


@fixture(params=[3, 5])
def n_best(request: type[FixtureRequest]) -> int:
    """Number of features to be selected"""
    return request.param
