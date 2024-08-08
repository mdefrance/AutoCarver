"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle
from typing import Any, Callable, Union
from warnings import warn

from numpy import unique
from pandas import DataFrame, Series
from math import ceil
from .filters import thresh_filter
from .measures import dtype_measure, make_measure, mode_measure, nans_measure
from ..features import Features, BaseFeature

# trying to import extra dependencies
try:
    from IPython.display import display_html
except ImportError:
    _has_idisplay = False
else:
    _has_idisplay = True


class BaseSelector:
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a binary target.

    * Best features are the n_best of each measure
    * Get your best features with ``Selector.select()``!
    """

    __name__ = "Selector"

    def __init__(
        self,
        n_best: int,
        features: Features = None,
        *,
        measures: Union[list[Callable], dict[str, list[Callable]]] = None,
        filters: Union[list[Callable], dict[str, list[Callable]]] = None,
        max_num_features_per_chunk: int = 100,
        verbose: bool = False,
        **kwargs,
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
        if not (2 < max_num_features_per_chunk):
            raise ValueError("Must set 2 < max_num_features_per_chunk")

        # adding default measures
        self.measures = {
            dtype: [dtype_measure, nans_measure, mode_measure] + requested_measures[:]
            for dtype, requested_measures in measures.items()
        }

        # adding default filter
        self.filters = {
            dtype: [thresh_filter] + requested_filters[:]
            for dtype, requested_filters in filters.items()
        }

        # names of measures to sort by
        self.measure_names = {
            dtype: [measure.__name__ for measure in requested_measures[::-1]]
            for dtype, requested_measures in self.measures.items()
        }

        # wether or not to print tables
        self.verbose = bool(max(verbose, kwargs.get("pretty_print", False)))
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

    def _select_features(
        self,
        X: DataFrame,
        y: Series,
        features: list[BaseFeature],
        n_best: int,
    ) -> list[BaseFeature]:
        """Selects the n_best features amongst the specified ones"""
        if self.verbose:  # verbose if requested
            print(f"------\n[{self.__name__}] Selecting from: {features}\n---")

        # Computes association between X and y
        initial_associations = apply_measures(
            X, y, measures=self.measures[dtype], features=features, **self.kwargs
        )

        # sorting statistics
        measure_names = evaluated_measure_names(initial_associations, self.measure_names[dtype])
        initial_associations = initial_associations.sort_values(measure_names, ascending=False)

        # displaying non-filterd associations
        self._print_associations(association=initial_associations)

        # applying filtering for each measure
        all_best_features: dict[str, Any] = {}
        for measure_name in measure_names:
            # sorting association for each measure
            associations = initial_associations.sort_values(measure_name, ascending=False)

            # filtering for each measure, as each measure ranks the features differently
            filtered_association = apply_filters(
                X, associations, filters=self.filters[dtype], **self.kwargs
            )

            # selected features for the measure
            selected_features = [
                feature
                for feature in initial_associations.index
                if (len(filtered_association) > 0)
                and (feature in filtered_association.index[:n_best])
            ]

            # saving results
            all_best_features.update(
                {
                    measure_name: {
                        "selected": selected_features,
                        "association": filtered_association,
                    }
                }
            )

        # list of unique best_featues per measure_name
        best_features = [
            # ordering according target association
            feature
            for feature in initial_associations.index
            # checking that feature has been selected by a measure
            if any(feature in all_best_features[measure]["selected"] for measure in measure_names)
        ]

        # displaying association measure
        self._print_associations(
            association=initial_associations.reindex(best_features),
            message=", filtered for inter-feature assocation",
        )

        return best_features

    def select(self, X: DataFrame, y: Series) -> list[str]:
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
        # iterating over each type of feature
        all_best_features = []
        for features in [self.features.get_qualitatives(), self.features.get_quantitatives()]:

            # splitting features in chunks and getting best chunk-features if needed
            best_features = self._get_best_features_per_chunk(features, X, y)

            # selecting amongst best features per chunk
            if any(best_features):
                best_features = self._select_features(X, y, best_features, self.n_best)

                all_best_features += best_features
                if self.verbose:  # verbose if requested
                    print(f"\n - [{self.__name__}] Selected: {best_features}------\n")

        return all_best_features

    def _get_best_features_per_chunk(
        self, features: list[BaseFeature], X: DataFrame, y: Series
    ) -> list[BaseFeature]:
        """gets best features per chunk of maximum size as set by max_num_features_per_chunk"""

        # keeping all features per default
        best_features = features[:]

        # too many features: chunking and selecting best amongst chunks
        if len(features) > self.max_num_features_per_chunk:

            # making random chunks of features
            chunks = make_random_chunks(features, self.max_num_features_per_chunk)

            # iterating over each feature samples
            best_features = []
            for chunk in chunks:
                # fitting association on features
                best_features += self._select_features(X, y, chunk, int(self.n_best // 2))

        return best_features

    def _print_associations(self, association: DataFrame, message: str = "") -> None:
        """EDA of fitted associations

        Parameters
        ----------
        association : DataFrame
            _description_
        pretty_print : bool, optional
            _description_, by default False
        """
        # checking for verbose
        if self.verbose:
            print(f"\n - [{self.__name__}] Association between X and y{message}")

            # printing raw DataFrame
            if not self.pretty_print:
                print(association)

            # adding colors and displaying DataFrame as html
            else:
                # finding columns with indicators to colorize
                subset = [
                    column
                    for column in association.columns
                    # checking for an association indicator
                    if any(indic in column for indic in ["pct_", "_measure", "_filter"])
                ]

                # adding coolwarm color gradient
                nicer_association = association.style.background_gradient(
                    cmap="coolwarm", subset=subset
                )
                # printing inline notebook
                nicer_association = nicer_association.set_table_attributes("style='display:inline'")

                # lower precision
                nicer_association = nicer_association.format(precision=4)

                # displaying html of colored DataFrame
                display_html(nicer_association._repr_html_(), raw=True)  # pylint: disable=W0212


def make_random_chunks(elements: list, max_chunk_sizes: int) -> list:
    """makes a specific number of random chunks of of a list"""

    # copying in order to not moidy initial list
    shuffled_elements = elements[:]

    # shuffling features to get random samples of features
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


def feature_association(x: Series, y: Series, measures: list[Callable], **kwargs) -> dict[str, Any]:
    """Measures association between x and y

    Parameters
    ----------
    x : Series
        Sample of a feature.
    y : Series
        Binary target feature with wich the association is evaluated.

    Returns
    -------
    dict[str, Any]
        Association metrics' values
    """
    # measures keep going only if previous basic tests are passed
    passed = True
    association = {}

    # iterating over each measure
    for measure in measures:
        passed, association = make_measure(
            measure, passed, association, x, y, **kwargs, **association
        )

    return association


def apply_measures(
    X: DataFrame, y: Series, measures: list[Callable], features: list[BaseFeature], **kwargs
) -> DataFrame:
    """Measures association between columns of X and y"""
    # applying association measure to each column
    associations = (
        X[features]
        .apply(feature_association, y=y, measures=measures, **kwargs, result_type="expand", axis=0)
        .T
    )

    return associations


def evaluated_measure_names(associations: DataFrame, measure_names: list[str]) -> list[str]:
    """_summary_

    Parameters
    ----------
    associations : DataFrame
        _description_
    measure_names : list[str]
        _description_

    Returns
    -------
    list[str]
        _description_
    """
    # Getting evaluated measures (filtering out non-measures: pct_zscore, pct_iqr...)
    sort_by = [
        measure_name
        for measure_name in measure_names
        if measure_name in associations and "_measure" in measure_name
    ]

    return sort_by


def apply_filters(
    X: DataFrame, associations: DataFrame, filters: list[Callable], **kwargs
) -> DataFrame:
    """Filters out too correlated features (least relevant first)

    Parameters
    ----------
    X : DataFrame
        _description_
    associations : DataFrame
        _description_
    filters : list[Callable]
        _description_
    measure_name : str
        _description_

    Returns
    -------
    DataFrame
        _description_
    """
    # applying successive filters
    filtered_associations = associations.copy()
    for filtering in filters:
        filtered_associations = filtering(X, filtered_associations, **kwargs)

    return filtered_associations
