""" Set of tools used for multiprocessing"""

from functools import partial
from multiprocessing import Pool
from typing import Any, Callable

from pandas import DataFrame


def imap_unordered_function(fun: Callable, elements: list[Any], n_jobs: int, **kwargs):
    """converts a function to a multiprocessing imap_unordered format or list comprehension"""

    # initiating list of function results
    results = []

    # no multiprocessing: list comprehension
    if n_jobs <= 1:
        results = [fun(elt, **kwargs) for elt in elements]
    # multiprocessing
    else:
        with Pool(processes=n_jobs) as pool:
            # feature processing
            results += pool.imap_unordered(partial(fun, **kwargs), elements)

    return results


def apply_async_function(fun: Callable, elements: list[Any], n_jobs: int, X: DataFrame, *args):
    """converts a function to a multiprocessing apply_async format or list comprehension"""

    # no multiprocessing
    if n_jobs <= 1:
        results = [fun(elt, X[elt], *args) for elt in elements]

    # asynchronous transform of each feature
    else:
        with Pool(processes=n_jobs) as pool:
            results_async = [pool.apply_async(fun, (elt, X[elt], *args)) for elt in elements]

            #  waiting for the results
            results = [result.get() for result in results_async]

    return results
