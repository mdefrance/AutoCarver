""" Defines fixtures for carvers pytests"""

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


@fixture(scope="module", params=[True, False])
def copy(request) -> bool:
    return request.param
