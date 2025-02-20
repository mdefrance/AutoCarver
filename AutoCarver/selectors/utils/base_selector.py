"""Tools to select the best Quantitative and Qualitative features."""

from abc import ABC, abstractmethod
from math import ceil
from random import seed, shuffle
from typing import Union

from pandas import DataFrame, Series

from ...features import BaseFeature, Features
from ...utils import get_bool_attribute, has_idisplay
from ..filters import BaseFilter
from ..measures import BaseMeasure
from .pretty_print import format_ranked_features, prettier_measures

# trying to import extra dependencies
_has_idisplay = has_idisplay()
if _has_idisplay:
    from IPython.display import display_html


class BaseSelector(ABC):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a binary target.

    Examples
    --------
    See `Selectors examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
    """

    __name__ = "BaseSelector"

    def __init__(
        self,
        features: Features,
        n_best_per_type: int,
        # *,
        # measures: list[BaseMeasure] = None,
        # filters: list[BaseFilter] = None,
        # max_num_features_per_chunk: int = 100,
        **kwargs,
    ) -> None:
        # quantitative_measures : list[Callable], optional
        #     List of association measures to be used for ``quantitative_features``.
        #     :ref:`Implemented measures <Measures>` are:

        #     * For association evaluation:
        #         * :ref:`Kruskal-Wallis' H <kruskal>` (default)
        #         * :ref:`R`
        #     * For outlier detection:
        #         * :ref:`Standard score <zscore>`
        #         * :ref:`Interquartile range <iqr>`

        # qualitative_measures : list[Callable], optional
        #     List of association measures to be used for ``qualitative_features``.
        #     :ref:`Implemented measures <Measures>` are:

        #     * For association evaluation:
        #         * :ref:`Pearson's chi² <chi2>`
        #         * :ref:`cramerv`
        #         * :ref:`tschuprowt` (default)

        # quantitative_filters : list[Callable], optional
        #     List of filters to be used for ``quantitative_features``.
        #     :ref:`Implemented filters <Filters>` are:

        #     * For linear correlation:
        #         * :ref:`pearson_filter`
        #         * :ref:`Spearman's rho <spearman_filter>` (default)

        # qualitative_filters : list[Callable], optional
        #     List of filters to be used for ``qualitative_features``.
        #     :ref:`Implemented filters <Filters>` are:

        #     * For correlation:
        #         * :ref:`cramerv_filter`
        #         * :ref:`tschuprowt_filter` (default)

        # **kwargs
        #     Allows one to set thresholds for provided ``quantitative_measures``/
        #     ``qualitative_measures`` and ``quantitative_filters``/``qualitative_filters``
        #     (see :ref:`Measures` and :ref:`Filters`) passed as keyword arguments.
        """
        Parameters
        ----------
        features : Features
            A set of :class:`Features` to select from

        n_best_per_type : int
            Number of quantitative and/or qualitative :class:`Features` to  select


        Keyword Arguments
        -----------------
        measures : list[BaseMeasure], optional
            List of association measures to be used, by default ``None``.

            Selects :attr:`n_best_per_type` features for each measure provided.
            Implemented measures are:

            * :class:`QuantitativeFeature`: see available :ref:`QuantiMeasures`
            * :class:`QualitativeFeature`: see available :ref:`QualiMeasures`

        filters : list[BaseFilter], optional
            List of filters to be used, by default ``None``.

            Filters out features that do not pass the threshold of each filter.
            Implemented filters are:

            * :class:`QuantitativeFeature`: see available :ref:`QuantiFilters`
            * :class:`QualitativeFeature`: see available :ref:`QualiFilters`

        max_num_features_per_chunk : int, optional
            Maximum number of features per chunk, by default ``100``.

            Chunking is used to speed up the selection process for large numbers
            of :class:`Features`.

            1. :class:`Features` are split in ``n_chunks`` of :attr:`max_num_features_per_chunk`
            2. ``n_best_per_type//n_chunks`` of each chunk are selected
            3. best features are selected from the remaining features

        verbose : bool, optional
            * ``True``, without ``IPython``: prints raw statitics
            * ``True``, with ``IPython``: prints HTML statistics, by default ``False``
        """
        # TODO print a config that says what the user has selected (measures+filters)
        # features and values
        self.features = features
        if isinstance(features, list):
            self.features = Features(features)

        # number of features selected
        self.n_best_per_type = n_best_per_type
        if not (0 < int(self.n_best_per_type) <= len(self.features)):
            raise ValueError("Must set 0 < n_best_per_type <= len(features)")

        # feature sample size per iteration
        # maximum number of features per chunk
        self.max_num_features_per_chunk = kwargs.get("max_num_features_per_chunk", 100)
        if self.max_num_features_per_chunk <= 2:
            raise ValueError("Must set 2 < max_num_features_per_chunk")

        # random state for chunk shuffling
        self.random_state = kwargs.get("random_state", 0)

        # adding default measures
        self.measures = self._initiate_measures(kwargs.get("measures"))

        # adding default filter
        self.filters = self._initiate_filters(kwargs.get("filters"))

        # whether to print info
        self.verbose = get_bool_attribute(kwargs, "verbose", False)
        self._message = ""

        # target name
        self.target_name = None

    def __repr__(self) -> str:
        """Returns the name of the selector"""
        return self.__name__

    @property
    def pretty_print(self) -> bool:
        """Returns the pretty_print attribute"""
        return self.verbose and _has_idisplay

    @abstractmethod
    def _initiate_measures(self, requested_measures: list[BaseMeasure] = None) -> list[BaseMeasure]:
        """initiates the list of measures with default values"""
        return requested_measures

    @abstractmethod
    def _initiate_filters(self, requested_filters: list[BaseFilter] = None) -> list[BaseFilter]:
        """initiates the list of measures with default values"""
        return requested_filters

    def _select_quantitatives(
        self, quantitatives: list[BaseFeature], X: DataFrame, y: Series
    ) -> list[BaseFeature]:
        """Selects the best quantitative features"""

        # checking for quantitative features before selection
        best_quantitatives: list[BaseFeature] = []
        if len(quantitatives) > 0:
            # getting qualitative measures and filters
            measures = get_quantitative_metrics(self.measures)
            filters = get_quantitative_metrics(self.filters)

            # setting message for pretty print
            self._message = "Quantitative "

            # getting best features
            best_quantitatives = self._select_features(quantitatives, X, y, measures, filters)

        return best_quantitatives

    def _select_qualitatives(
        self, qualitatives: list[BaseFeature], X: DataFrame, y: Series
    ) -> list[BaseFeature]:
        """Selects the best qualitative features"""

        # checking for qualitative features before selection
        best_qualitatives: list[BaseFeature] = []
        if len(qualitatives) > 0:
            # getting qualitative measures and filters
            measures = get_qualitative_metrics(self.measures)
            filters = get_qualitative_metrics(self.filters)

            # setting message for pretty print
            self._message = "Qualitative "

            # getting best features
            best_qualitatives = self._select_features(qualitatives, X, y, measures, filters)

        return best_qualitatives

    def _initiate_features_measures(
        self, features: list[BaseFeature], remove_default: bool = True
    ) -> None:
        """initiates the list of measures with default values"""
        # iterating over each feature
        for feature in features:
            # removing all measures
            if remove_default:
                feature.measures = {}
                feature.filters = {}

            # keeping only default measures
            else:
                remove_non_default_metrics_from_features(feature)

    def select(self, X: DataFrame, y: Series) -> Features:
        """Selects the :attr:`n_best_per_type` :class:`Features` of ``X``

        Parameters
        ----------
        X : DataFrame
            Dataset to determine optimal features.

        y : Series
            Target with wich the association is evaluated.

        Returns
        -------
        Features
            Selected :class:`Features`
        """

        # getting target name
        if isinstance(y, Series):
            self.target_name = y.name

        # initiating features measures and filters
        self._initiate_features_measures(self.features, remove_default=True)

        # checking for quantitative and qualitative features
        features = get_typed_features(self.features)

        # selecting quantitative features
        best_features = self._select_quantitatives(features.get("quantitatives"), X, y)

        # selecting qualitative features
        best_features += self._select_qualitatives(features.get("qualitatives"), X, y)

        # converting to Features
        if len(best_features) > 0:
            return Features(best_features)
        return best_features

    def _select_features(
        self,
        features: list[BaseFeature],
        X: DataFrame,
        y: Series,
        measures: list[BaseMeasure],
        filters: list[BaseFilter],
    ) -> list[BaseFeature]:
        """selects amongst features"""

        # apply default measures to features
        apply_measures(features, X, y, measures, default_measures=True)

        # apply default filters to features
        features = apply_filters(features, X, filters, default_filters=True)

        # getting non-default measures and filters
        measures = remove_default_metrics(measures)
        filters = remove_default_metrics(filters)

        # splitting features in chunks and getting best per-chunk set of features
        return self._get_best_features_across_chunks(features, X, y, measures, filters)

    def _get_best_features_across_chunks(
        self,
        features: list[BaseFeature],
        X: DataFrame,
        y: Series,
        measures: list[BaseMeasure],
        filters: list[BaseFilter],
    ) -> list[BaseFeature]:
        """gets best features per chunk of maximum size as set by max_num_features_per_chunk"""

        # too many features: chunking and selecting best amongst chunks
        if len(features) > self.max_num_features_per_chunk:
            # making random chunks of features
            chunks = make_random_chunks(
                features, self.max_num_features_per_chunk, self.random_state
            )

            # iterating over each feature samples
            features: list[BaseFeature] = []
            for n_chunk, chunk in enumerate(chunks):
                # fitting association on features
                features += get_best_features(
                    chunk, X, y, measures, filters, max(int(self.n_best_per_type // len(chunks)), 1)
                )

                # printing association
                self._print_measures(chunk, f"from chunk {n_chunk}/{len(chunks)}")

        # initiating features measures and filters for final prediction
        self._initiate_features_measures(features, remove_default=False)

        # selecting from remaining features
        best_features = get_best_features(features, X, y, measures, filters, self.n_best_per_type)

        # printing association
        self._print_measures(features)

        return best_features

    def _print_measures(self, features: list[BaseFeature], message: str = "") -> None:
        """Prints crosstabs' target rates and frequencies per modality, in raw or html format

        Parameters
        ----------
        xagg : Union[DataFrame, Series]
            Train crosstab
        xagg_dev : Union[DataFrame, Series]
            Dev crosstab, by default None
        pretty_print : bool, optional
            Whether to output html or not, by default False
        """
        if self.verbose:
            # formatting measures to print
            formatted_measures = format_ranked_features(features)

            if not formatted_measures.empty:
                print(f" [{self.__name__}] Selected {self._message}Features {message}")

            if not self.pretty_print:  # no pretty hmtl printing
                self._print_raw(formatted_measures)
            else:  # pretty html printing
                self._print_html(formatted_measures)

    def _print_raw(self, formatted_measures: DataFrame) -> None:
        """Prints raw XAGG DataFrames."""
        print(formatted_measures, "\n")

    def _print_html(self, formatted_measures: DataFrame) -> None:
        """Prints XAGG DataFrames in HTML format."""
        # getting prettier xtabs
        nicer_association = prettier_measures(formatted_measures)

        # displaying html of colored DataFrame
        display_html(nicer_association, raw=True)  # pylint: disable=W0212


def get_typed_features(features: Features) -> dict[str, list[BaseFeature]]:
    """returns quantitative and qualitative features from list of features"""
    return {
        "quantitatives": [feature for feature in features if is_quantitative(feature)],
        "qualitatives": [feature for feature in features if is_qualitative(feature)],
    }


def is_quantitative(feature: BaseFeature) -> bool:
    """checks if feature is quantitative"""
    return feature.is_quantitative and not feature.is_fitted


def is_qualitative(feature: BaseFeature) -> bool:
    """checks if feature is qualitative"""
    return feature.is_qualitative or feature.is_fitted


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


def get_qualitative_metrics(
    metrics: list[Union[BaseMeasure, BaseFilter]]
) -> list[Union[BaseMeasure, BaseFilter]]:
    """returns filtered list of measures/filters that apply on qualitative features"""
    return [metric for metric in metrics if metric.is_x_qualitative]


def get_quantitative_metrics(
    metrics: list[Union[BaseMeasure, BaseFilter]]
) -> list[Union[BaseMeasure, BaseFilter]]:
    """returns filtered list of measures/filters that apply on quantitative features"""
    return [metric for metric in metrics if metric.is_x_quantitative]


def get_default_metrics(
    metrics: list[Union[BaseMeasure, BaseFilter]]
) -> list[Union[BaseMeasure, BaseFilter]]:
    """returns filtered list of measures/filters that are default"""
    return [metric for metric in metrics if metric.is_default]


def remove_default_metrics(
    metrics: list[Union[BaseMeasure, BaseFilter]]
) -> list[Union[BaseMeasure, BaseFilter]]:
    """returns filtered list of measures/filters that are non-default"""
    return [metric for metric in metrics if not metric.is_default]


def remove_non_default_metrics_from_features(feature: BaseFeature) -> None:
    """removes default measures from feature"""
    # removing non default measures
    measures = dict(feature.measures)
    for measure_name, measure in feature.measures.items():
        if not measure.get("info").get("is_default"):
            measures.pop(measure_name)

    # removing non default filters
    filters = dict(feature.filters)
    for filter_name, measure in feature.filters.items():
        if not measure.get("info").get("is_default"):
            filters.pop(filter_name)

    # updating feature
    feature.measures = measures
    feature.filters = filters


def remove_duplicates(features: list[BaseFeature]) -> list[BaseFeature]:
    """removes duplicated features, keeping its first appearance"""
    return [features[i] for i in range(len(features)) if features[i] not in features[:i]]


def sort_features_per_measure(
    features: list[BaseFeature], measure: BaseMeasure
) -> list[BaseFeature]:
    """sorts features according to specified measure"""
    # checking if features are already ranked
    ranked = False
    for feature in features:
        if make_rank_name(measure) in feature.measures:
            ranked = True

    # setting reverse mode
    reverse = not measure.info.get("higher_is_better")
    if ranked:
        reverse = False

    # sorting features according to measure
    return sorted(features, key=lambda feature: get_feature_rank(feature, measure), reverse=reverse)


def get_feature_rank(feature: BaseFeature, measure: BaseMeasure) -> float:
    """gives rank of feature according to measure"""
    if make_rank_name(measure) not in feature.measures:
        return get_measure_value(feature, measure)
    return get_measure_rank(feature, measure)


def get_measure_rank(feature: BaseFeature, measure: BaseMeasure) -> int:
    """gives rank of feature according to measure"""
    return feature.measures.get(make_rank_name(measure)).get("value")


def get_measure_value(feature: BaseFeature, measure: BaseMeasure) -> float:
    """gives value of measure for specified feature"""
    return feature.measures.get(measure.__name__).get("value")


def apply_measures(
    features: list[BaseFeature],
    X: DataFrame,
    y: Series,
    measures: list[BaseMeasure],
    default_measures: bool = False,
) -> list[BaseMeasure]:
    """Measures association between columns of X and y, returns remaining_measures (not used)"""

    # keeping only default measures or non default measures
    used_measures = remove_default_metrics(measures)
    if default_measures:
        used_measures = get_default_metrics(measures)

    # iterating over each feature
    for feature in features:
        # iterating over each measure
        for measure in used_measures:
            # checking for mismatched data types
            check_measure_mismatch(feature, measure)

            # computing association for feature
            measure.compute_association(X[feature.version], y)

            # updating feature accordingly
            measure._update_feature(feature)


def apply_filters(
    features: list[BaseFeature],
    X: DataFrame,
    filters: list[BaseFilter],
    default_filters: bool = False,
) -> list[BaseFeature]:
    """Filters out too correlated features (least relevant first)"""

    # keeping only default filters or non default filters
    used_filters = remove_default_metrics(filters)
    if default_filters:
        used_filters = get_default_metrics(filters)

    # keeping track of remaining features
    filtered = features[:]

    # iterating over each filter
    for measure in used_filters:
        # checking for mismatched data types
        for feature in features:
            check_measure_mismatch(feature, measure)

        # applying filter
        filtered = measure.filter(X, filtered)

    return filtered


def check_measure_mismatch(feature: BaseFeature, measure: Union[BaseMeasure, BaseFilter]) -> None:
    """checks for mismatched data types between feature and measure"""
    if not (
        (is_quantitative(feature) and measure.is_x_quantitative)
        or (is_qualitative(feature) and measure.is_x_qualitative)
    ):
        raise TypeError(
            f"Type mismatch, provided feature {feature}, with {measure} that has "
            f"is_x_quantitative={measure.is_x_quantitative}"
        )


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
        # getting best features according to measure
        best_features += select_with_measure(X, features, measure, filters, n_best)

    # deduplicating best features
    return remove_duplicates(best_features)


def select_with_measure(
    X: DataFrame,
    features: list[BaseFeature],
    measure: BaseMeasure,
    filters: list[BaseFilter],
    n_best: int,
) -> list[BaseFeature]:
    """Selects the ``n_best`` features of the DataFrame, by association with the target"""

    # sorting features according to measure
    sorted_features = sort_features_per_measure(features, measure)

    # reversing features
    sorted_features.reverse()

    # applying filters
    filtered_features = apply_filters(sorted_features, X, filters)

    # adding info to features
    for rank, feature in enumerate(filtered_features):
        feature.measures.update(make_rank_info(rank, measure, n_best, len(filtered_features)))

    # getting best features according to measure
    return select_from_rank(filtered_features, measure)


def select_from_rank(features: list[BaseFeature], measure: BaseMeasure) -> list[BaseFeature]:
    """Selects the ``n_best`` features of the DataFrame, by association with the target"""
    return [
        feature
        for feature in features
        if feature.measures.get(make_rank_name(measure)).get("valid")
    ]


def make_rank_name(measure: BaseMeasure) -> str:
    """makes a name for the rank info"""
    return f"{measure.__name__.replace('Measure', '')}Rank"


def make_rank_info(rank: int, measure: BaseMeasure, n_best: int, n_features: int) -> dict:
    """makes a dict with rank and measure info"""
    return {
        make_rank_name(measure): {
            "value": rank,
            "threshold": n_features - n_best,
            "valid": rank < n_best,
            "info": {"is_default": False, "higher_is_better": False},
        }
    }
