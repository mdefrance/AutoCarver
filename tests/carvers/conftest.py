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


@fixture
def chained_features() -> list[str]:
    """List of chained raw features to be chained"""
    return ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]


@fixture
def level0_to_level1() -> dict[str, list[str]]:
    """Chained orders level0 to level1 of features to be chained"""
    return {
        "Lows": ["Low-", "Low", "Low+", "Lows"],
        "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
        "Highs": ["High-", "High", "High+", "Highs"],
    }


@fixture
def level1_to_level2() -> dict[str, list[str]]:
    """Chained orders level1 to level2 of features to be chained"""
    return {
        "Worst": ["Lows", "Mediums", "Worst"],
        "Best": ["Highs", "Best"],
    }


@fixture(scope="module", params=[True, False])
def ordinal_encoding(request: FixtureRequest) -> bool:
    return request.param


@fixture(scope="module", params=[True, False])
def dropna(request: FixtureRequest) -> bool:
    return request.param


@fixture(scope="module", params=["tschuprowt", "cramerv"])
def sort_by(request: FixtureRequest) -> str:
    return request.param


@fixture(scope="module", params=[True, False])
def copy(request: FixtureRequest) -> bool:
    return request.param


@fixture(scope="module", params=[None, 0.12])
def discretizer_min_freq(request: FixtureRequest) -> float:
    return request.param
