"""Set of tools used for multiprocessing"""

from functools import partial
from multiprocessing import Pool
from typing import Callable

from ...features import BaseFeature, Features

# def parallelize(fun: Callable, features: list[str], n_jobs: int, **kwargs):
#     """converts a function to a multiprocessing imap_unordered format or list comprehension"""

#     # initiating list of function results
#     results = []

#     # no multiprocessing: list comprehension
#     if n_jobs <= 1:
#         results = [fun(f, **kwargs) for f in features]
#     # multiprocessing
#     else:
#         with Pool(processes=n_jobs) as pool:
#             # feature processing
#             results += pool.imap_unordered(partial(fun, **kwargs), features)

#     return results


def parallelize(
    carver: Callable,
    features: Features,
    n_jobs: int,
    xaggs: dict,
    xaggs_dev: dict,
    applier: Callable,
):
    """converts a function to a multiprocessing apply_async format or list comprehension"""

    # no multiprocessing
    if n_jobs <= 1:
        for feature in features:
            out = carver(feature, xagg=xaggs[feature.version], xagg_dev=xaggs_dev[feature.version])
            applier(out)

    # asynchronous transform of each feature
    else:
        with Pool(processes=n_jobs) as pool:
            results_async = [
                pool.apply_async(
                    carver, (feature, xaggs[feature.version], xaggs_dev[feature.version])
                )
                for features in features
            ]

            #  waiting for the results
            for result in results_async:
                out = result.get()
                applier(out)
