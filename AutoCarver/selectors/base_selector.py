"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle, seed
from typing import Any, Callable, Union
from warnings import warn

from numpy import unique
from pandas import DataFrame, Series
from math import ceil
from .filters import ValidFilter, BaseFilter, SpearmanFilter, TschuprowtFilter
from .measures import (
    ModeMeasure,
    NanMeasure,
    TschuprowtMeasure,
    SpearmanMeasure,
    BaseMeasure,
)
from ..features import Features, BaseFeature, QuantitativeFeature, QualitativeFeature

# trying to import extra dependencies
try:
    from IPython.display import display_html
except ImportError:
    _has_idisplay = False
else:
    _has_idisplay = True

from abc import ABC, abstractmethod


class BaseSelector(ABC):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a binary target.

    * Best features are the n_best of each measure
    * Get your best features with ``Selector.select()``!
    """

    __name__ = "BaseSelector"

    def __init__(
        self,
        n_best: int,
        features: Features,
        *,
        measures: list[BaseMeasure] = None,
        filters: list[BaseFilter] = None,
        max_num_features_per_chunk: int = 100,
        **kwargs: dict,
    ) -> None:
        # measures : list[Callable], optional
        #     List of association measures to be used, by default ``None``.
        #     Ranks features based on last provided measure of the list.
        #     See :ref:`Measures`.
        #     Implemented measures are:

        #     * [Quantitative Features] For association evaluation: ``kruskal_measure`` (default),
        # ``R_measure``
        #     * [Quantitative Features] For outlier detection: ``zscore_measure``, ``iqr_measure``
        #     * [Qualitative Features] For association evaluation: ``chi2_measure``,
        # ``cramerv_measure``, ``tschuprowt_measure`` (default)

        # filters : list[Callable], optional
        #     List of filters to be used, by default ``None``.
        #     See :ref:`Filters`.
        #     Implemented filters are:

        #     * [Quantitative Features] For linear correlation: ``spearman_filter`` (default),
        # ``pearson_filter``
        #     * [Qualitative Features] For correlation: ``cramerv_filter``, ``tschuprowt_filter``
        #  (default)

        """
        Parameters
        ----------
        n_best : int
            Number of best features to select. Best features are:

            * The first ``n_best`` of each provided data types as set in:
                * ``quantitative_features``
                * ``qualitative_features``
            * The first ``n_best`` for each provided measures as set in:
                * ``quantitative_measures``
                * ``qualitative_measures``

        quantitative_features : list[str]
            List of column names of quantitative features to chose from, by default ``None``.
            Must be set if ``qualitative_features=None``.

        qualitative_features : list[str]
            List of column names of qualitative features to chose from, by default ``None``.
            Must be set if ``quantitative_features=None``.

        quantitative_measures : list[Callable], optional
            List of association measures to be used for ``quantitative_features``.
            :ref:`Implemented measures <Measures>` are:

            * For association evaluation:
                * :ref:`Kruskal-Wallis' H <kruskal>` (default)
                * :ref:`R`
            * For outlier detection:
                * :ref:`Standard score <zscore>`
                * :ref:`Interquartile range <iqr>`

        qualitative_measures : list[Callable], optional
            List of association measures to be used for ``qualitative_features``.
            :ref:`Implemented measures <Measures>` are:

            * For association evaluation:
                * :ref:`Pearson's chi² <chi2>`
                * :ref:`cramerv`
                * :ref:`tschuprowt` (default)

        quantitative_filters : list[Callable], optional
            List of filters to be used for ``quantitative_features``.
            :ref:`Implemented filters <Filters>` are:

            * For linear correlation:
                * :ref:`pearson_filter`
                * :ref:`Spearman's rho <spearman_filter>` (default)

        qualitative_filters : list[Callable], optional
            List of filters to be used for ``qualitative_features``.
            :ref:`Implemented filters <Filters>` are:

            * For correlation:
                * :ref:`cramerv_filter`
                * :ref:`tschuprowt_filter` (default)

        colsample : float, optional
            Size of sampled list of features for sped up computation, between ``0`` and ``1``,
            by default ``1.0``, all features are used.

            For ``colsample=0.5``, Selector will search for the best features in
            ``features[:len(features)//2]`` and then in ``features[len(features)//2:]``.

            **Tip:** for better performance, should be set such as ``len(features)//2 < 200``.

        verbose : bool, optional
            * ``True``, without IPython: prints raw selection steps for X, by default ``False``
            * ``True``, with IPython: adds HTML tables to the output.

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``.

        **kwargs
            Allows one to set thresholds for provided ``quantitative_measures``/
            ``qualitative_measures`` and ``quantitative_filters``/``qualitative_filters``
            (see :ref:`Measures` and :ref:`Filters`) passed as keyword arguments.

        Examples
        --------
        See `Selectors examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # TODO print a config that says what the user has selectes (measures+filters)
        # features and values
        self.features = features
        if isinstance(features, list):
            self.features = Features(features)

        # number of features selected
        self.n_best = n_best
        if not (0 < int(self.n_best) <= len(self.features)):
            raise ValueError("Must set 0 < n_best <= len(features)")

        # feature sample size per iteration
        # maximum number of features per chunk
        self.max_num_features_per_chunk = max_num_features_per_chunk
        if max_num_features_per_chunk <= 2:
            raise ValueError("Must set 2 < max_num_features_per_chunk")

        # random state for chunk shuffling
        self.random_state = kwargs.get("random_state", 0)

        # adding default measures
        self.measures = self._initiate_measures(measures)

        # adding default filter
        self.filters = self._initiate_filters(filters)

        # whether to print info
        self.verbose = bool(max(kwargs.get("verbose", True), kwargs.get("pretty_print", False)))
        self.pretty_print = False
        if self.verbose and kwargs.get("pretty_print", True):
            if _has_idisplay:  # checking for installed dependencies
                self.pretty_print = True
            else:
                warn(
                    "Package not found: IPython. Defaulting to raw verbose. "
                    "Install extra dependencies with pip install autocarver[jupyter]",
                    UserWarning,
                )

        # keyword arguments
        self.kwargs = kwargs

    @abstractmethod
    def _initiate_measures(self, requested_measures: list[BaseMeasure] = None) -> list[BaseMeasure]:
        """initiates the list of measures with default values"""
        pass

    @abstractmethod
    def _initiate_filters(self, requested_filters: list[BaseFilter] = None) -> list[BaseFilter]:
        """initiates the list of measures with default values"""
        pass

    def select(self, X: DataFrame, y: Series) -> list[BaseFeature]:
        """Selects the ``n_best`` features of the DataFrame, by association with the target

        Parameters
        ----------
        X : DataFrame
            Dataset to determine optimal features.
            Needs to have columns has specified in ``features`` attribute.

        y : Series
            Target with wich the association is evaluated.

        Returns
        -------
        list[str]
            List of selected features
        """

        # apply default measures to features
        apply_measures(self.features, X, y, get_default_measures(self.measures))

        # apply default filters to features
        features = apply_filters(self.features, X, get_default_filters(self.filters))

        # checking for quantitative features before selection
        quantitatives = [feature for feature in features if feature.is_quantitative]
        if len(quantitatives) > 0:
            best_quantitative_features = self._select_quantitatives(quantitatives, X, y)

        # checking for qualitative features before selection
        qualitatives = [feature for feature in features if feature.is_qualitative]
        if len(qualitatives) > 0:
            best_qualitative_features = self._select_qualitatives(qualitatives, X, y)

        return best_quantitative_features, best_qualitative_features

    def _select_quantitatives(
        self, quantitatives: list[BaseFeature], X: DataFrame, y: Series
    ) -> list[QuantitativeFeature]:
        """selects amongst quantitative features"""

        # getting measures to sort features
        measures = get_quantitative_measures(self.measures)

        # getting filters
        filters = get_quantitative_filters(self.filters)

        # splitting features in chunks and getting best per-chunk set of features
        return self._get_best_features_across_chunks(quantitatives, X, y, measures, filters)

    def _select_qualitatives(
        self, features: list[BaseFeature], X: DataFrame, y: Series
    ) -> list[QualitativeFeature]:
        """selects amongst qualitative features"""

        # iterating over qualitative features
        qualitatives = [feature for feature in features if feature.is_qualitative]

        # getting measures to sort features
        measures = get_qualitative_measures(self.measures)

        # getting filters
        filters = get_qualitative_filters(self.filters)

        # splitting features in chunks and getting best per-chunk set of features
        return self._get_best_features_across_chunks(qualitatives, X, y, measures, filters)

    def _get_best_features_across_chunks(
        self,
        features: list[BaseFeature],
        X: DataFrame,
        y: Series,
        measures: list[BaseMeasure],
        filters: list[BaseFilter],
    ) -> list[BaseFeature]:
        """gets best features per chunk of maximum size as set by max_num_features_per_chunk"""

        # keeping all features per default
        best_features = features[:]

        # too many features: chunking and selecting best amongst chunks
        if len(features) > self.max_num_features_per_chunk:

            # making random chunks of features
            chunks = make_random_chunks(
                features, self.max_num_features_per_chunk, self.random_state
            )

            # iterating over each feature samples
            best_features = []
            for chunk in chunks:
                # fitting association on features
                best_features += get_best_features(
                    chunk, X, y, measures, filters, int(self.n_best // len(chunks))
                )

        return get_best_features(best_features, X, y, measures, filters, self.n_best)

    # def _print_associations(self, association: DataFrame, message: str = "") -> None:
    #     """EDA of fitted associations

    #     Parameters
    #     ----------
    #     association : DataFrame
    #         _description_
    #     pretty_print : bool, optional
    #         _description_, by default False
    #     """
    #     # checking for verbose
    #     if self.verbose:
    #         print(f"\n - [{self.__name__}] Association between X and y{message}")

    #         # printing raw DataFrame
    #         if not self.pretty_print:
    #             print(association)

    #         # adding colors and displaying DataFrame as html
    #         else:
    #             # finding columns with indicators to colorize
    #             subset = [
    #                 column
    #                 for column in association.columns
    #                 # checking for an association indicator
    #                 if any(indic in column for indic in ["pct_", "_measure", "_filter"])
    #             ]

    #             # adding coolwarm color gradient
    #             nicer_association = association.style.background_gradient(
    #                 cmap="coolwarm", subset=subset
    #             )
    #             # printing inline notebook
    #             nicer_association = nicer_association.set_table_attributes("style='display:inline'")

    #             # lower precision
    #             nicer_association = nicer_association.format(precision=4)

    #             # displaying html of colored DataFrame
    #             display_html(nicer_association._repr_html_(), raw=True)  # pylint: disable=W0212


def make_random_chunks(elements: list, max_chunk_sizes: int, random_state: int = None) -> list:
    """makes a specific number of random chunks of of a list"""

    # copying in order to not moidy initial list
    shuffled_elements = elements[:]

    # shuffling features to get random samples of features
    seed(random_state)
    shuffle(shuffled_elements)

    # number of chunks
    num_chunks = ceil(len(elements) / max_chunk_sizes)

    # getting size of each chunk
    chunk_size, remainder = divmod(len(elements), num_chunks)

    # getting all chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(shuffled_elements[start:end])
        start = end

    return chunks


def get_quantitative_measures(measures: list[BaseMeasure]) -> list[BaseMeasure]:
    """returns filtered list of measures that apply on quantitative features"""
    return [measure for measure in measures if measure.is_x_quantitative and not measure.is_default]


def get_qualitative_measures(measures: list[BaseMeasure]) -> list[BaseMeasure]:
    """returns filtered list of measures that apply on qualitative features"""
    return [measure for measure in measures if measure.is_x_qualitative and not measure.is_default]


def get_default_measures(measures: list[BaseMeasure]) -> list[BaseMeasure]:
    """returns filtered list of measures that apply on qualitative features"""
    return [measure for measure in measures if measure.is_default]


def get_quantitative_filters(filters: list[BaseFilter]) -> list[BaseFilter]:
    """returns filtered list of filters that apply on quantitative features"""
    return [filter for filter in filters if filter.is_x_quantitative and not filter.is_default]


def get_qualitative_filters(filters: list[BaseFilter]) -> list[BaseFilter]:
    """returns filtered list of filters that apply on qualitative features"""
    return [filter for filter in filters if filter.is_x_qualitative and not filter.is_default]


def get_default_filters(filters: list[BaseFilter]) -> list[BaseFilter]:
    """returns filtered list of filters that apply on qualitative features"""
    return [filter for filter in filters if filter.is_default]


def apply_measures(
    features: list[BaseFeature], X: DataFrame, y: Series, measures: list[BaseMeasure]
) -> DataFrame:
    """Measures association between columns of X and y"""

    # iterating over each feature
    for feature in features:

        # iterating over each measure
        for measure in measures:

            # computing association for feature
            measure.compute_association(X[feature.version], y)

            # updating feature accordingly
            measure.update_feature(feature)


def apply_filters(
    features: list[BaseFeature], X: DataFrame, filters: list[BaseFilter]
) -> DataFrame:
    """Filters out too correlated features (least relevant first)"""

    # keeping track of remaining features
    filtered = features[:]

    # iterating over each filter
    for filter_ in filters:

        # applying filter
        filtered = filter_.filter(X, filtered)

    return filtered


def get_best_features(
    features: list[BaseFeature],
    X: DataFrame,
    y: Series,
    measures: list[BaseMeasure],
    filters: list[BaseFilter],
    n_best: int,
) -> list[BaseFeature]:
    """gives best features according to provided measures"""

    # check for sortable measures
    if not all(measure.is_sortable for measure in measures):
        raise ValueError("All provided measures should be sortable")

    # applying measures to all features
    apply_measures(features, X, y, measures)

    # getting best feature for each sortable measure
    best_features = []
    for measure in measures:

        # sorting features according to measure
        sorted_features = sort_features_per_measure(features, measure)

        # applying filters
        filtered_features = apply_filters(sorted_features, X, filters)

        # getting best features according to measure
        best_features += filtered_features[:n_best]

    # deduplicating best features
    return remove_duplicates(best_features)


def remove_duplicates(features: list[BaseFeature]) -> list[BaseFeature]:
    """removes duplicated features, keeping its first appearance"""
    return [features[i] for i in range(len(features)) if features[i] not in features[:i]]


def sort_features_per_measure(
    features: list[BaseFeature], measure: BaseMeasure
) -> list[BaseFeature]:
    """sorts features according to specified measure"""
    return sorted(features, key=lambda feature: get_measure_value(feature, measure))


def get_measure_value(feature: BaseFeature, measure: BaseMeasure) -> float:
    """gives value of measure for specified feature"""
    return feature.statistics.get("measures").get(measure.__name__).get("value")
