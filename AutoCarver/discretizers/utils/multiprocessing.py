""" Set of tools used for multiprocessing"""

from functools import partial
from multiprocessing import Pool
from typing import Callable

from pandas import DataFrame

from ...features import BaseFeature


def imap_unordered_function(fun: Callable, features: list[str], n_jobs: int, **kwargs):
    """converts a function to a multiprocessing imap_unordered format or list comprehension"""

    # initiating list of function results
    results = []

    # no multiprocessing: list comprehension
    if n_jobs <= 1:
        results = [fun(f, **kwargs) for f in features]
    # multiprocessing
    else:
        with Pool(processes=n_jobs) as pool:
            # feature processing
            results += pool.imap_unordered(partial(fun, **kwargs), features)

    return results


def apply_async_function(
    fun: Callable, features: list[BaseFeature], n_jobs: int, X: DataFrame, *args: list
):
    """converts a function to a multiprocessing apply_async format or list comprehension"""

    # no multiprocessing
    if n_jobs <= 1:
        results = [fun(f, X[f.version], *args) for f in features]

    # asynchronous transform of each feature
    else:
        with Pool(processes=n_jobs) as pool:
            results_async = [pool.apply_async(fun, (f, X[f.version], *args)) for f in features]

            #  waiting for the results
            results = [result.get() for result in results_async]

    return results
