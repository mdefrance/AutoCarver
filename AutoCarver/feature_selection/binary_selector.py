"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle
from typing import Any, Callable
from warnings import warn

from pandas import DataFrame, Series

from .base_selector import BaseSelector
from .filters import spearman_filter, thresh_filter, tschuprowt_filter
from .measures import (
    kruskal_measure,
    tschuprowt_measure,
)



class QualitativeSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a binary target.

    * Best features are the n_best of each measure
    * Get your best features with ``FeatureSelector.select()``!
    """

    def __init__(
        self,
        n_best: int,
        qualitative_features: list[str],
        *,
        measures: list[Callable] = None,
        filters: list[Callable] = None,
        colsample: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Initiates a ``FeatureSelector``.

        Parameters
        ----------
        n_best : int
            Number of features to select.

        quantitative_features : list[str], optional
            List of column names of quantitative features to chose from, by default ``None``
            Must be set if ``qualitative_features=None``.

        qualitative_features : list[str], optional
            List of column names of qualitative features to chose from, by default ``None``
            Must be set if ``quantitative_features=None``.

        measures : list[Callable], optional
            List of association measures to be used, by default ``None``.
            Ranks features based on last provided measure of the list.
            See :ref:`Measures`.
            Implemented measures are:

            * [Quantitative Features] For association evaluation: ``kruskal_measure`` (default), ``R_measure``
            * [Quantitative Features] For outlier detection: ``zscore_measure``, ``iqr_measure``
            * [Qualitative Features] For association evaluation: ``chi2_measure``, ``cramerv_measure``, ``tschuprowt_measure`` (default)

        filters : list[Callable], optional
            List of filters to be used, by default ``None``.
            See :ref:`Filters`.
            Implemented filters are:

            * [Quantitative Features] For linear correlation: ``spearman_filter`` (default), ``pearson_filter``
            * [Qualitative Features] For correlation: ``cramerv_filter``, ``tschuprowt_filter`` (default)

        colsample : float, optional
            Size of sampled list of features for sped up computation, between 0 and 1, by default ``1.0``
            By default, all features are used.

            For colsample=0.5, FeatureSelector will search for the best features in
            ``features[:len(features)//2]`` and then in ``features[len(features)//2:]``.

            **Tip:** for better performance, should be set such as ``len(features)//2 < 200``.

        verbose : bool, optional
            * ``True``, without IPython installed: prints raw feature selection steps for X, by default ``False``
            * ``True``, with IPython installed: adds HTML tables to the output.

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``.

        **kwargs
            Sets thresholds for ``measures`` and ``filters``, as long as ``pretty_print``, passed as keyword arguments.

        Examples
        --------
        See `FeatureSelector examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # default measure
        if measures is None:
            measures = [kruskal_measure]
            measures = [tschuprowt_measure]

        # default filter
        if filters is None:
            filters = [spearman_filter]
            filters = [tschuprowt_filter]

        # initiating BaseSelector with the corresponding list of measures
        super().__init__(
            n_best,
            features=qualitative_features,
            input_dtypes="str",
            measures=measures,
            filters=filters,
            colsample=colsample,
            verbose=verbose,
            **kwargs,
        )
