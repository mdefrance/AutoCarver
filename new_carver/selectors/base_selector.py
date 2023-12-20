"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle
from typing import Any, Callable, Union
from warnings import warn

from numpy import unique
from pandas import DataFrame, Series

from .filters import thresh_filter
from .measures import dtype_measure, make_measure, mode_measure, nans_measure

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

    def __init__(
        self,
        n_best: int,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        *,
        measures: Union[list[Callable], dict[str, list[Callable]]] = None,
        filters: Union[list[Callable], dict[str, list[Callable]]] = None,
        colsample: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        # measures : list[Callable], optional
        #     List of association measures to be used, by default ``None``.
        #     Ranks features based on last provided measure of the list.
        #     See :ref:`Measures`.
        #     Implemented measures are:

        #     * [Quantitative Features] For association evaluation: ``kruskal_measure`` (default), ``R_measure``
        #     * [Quantitative Features] For outlier detection: ``zscore_measure``, ``iqr_measure``
        #     * [Qualitative Features] For association evaluation: ``chi2_measure``, ``cramerv_measure``, ``tschuprowt_measure`` (default)

        # filters : list[Callable], optional
        #     List of filters to be used, by default ``None``.
        #     See :ref:`Filters`.
        #     Implemented filters are:

        #     * [Quantitative Features] For linear correlation: ``spearman_filter`` (default), ``pearson_filter``
        #     * [Qualitative Features] For correlation: ``cramerv_filter``, ``tschuprowt_filter`` (default)

        """
        Parameters
        ----------
        n_best : int
            Number of features to select.

            * Best features are the ``n_best`` of each provided data types (set in ``quantitative_features`` and/or ``qualitative_features``)
            * Best features are the ``n_best`` for each provided measures (set in ``quantitative_measures`` and/or ``qualitative_measures``)

        quantitative_features : list[str]
            List of column names of quantitative features to chose from, by default ``None``.
            Must be set if ``qualitative_features=None``.

        qualitative_features : list[str]
            List of column names of qualitative features to chose from, by default ``None``.
            Must be set if ``quantitative_features=None``.

        quantitative_measures : list[Callable], optional
            List of association measures to be used for ``quantitative_features``.
            :ref:`Implemented measures <Measures>` are:

            * For association evaluation: :ref:`Kruskal-Wallis' H <kruskal>` (default), :ref:`R`
            * For outlier detection: :ref:`Standard score <zscore>`, :ref:`Interquartile range <iqr>`

        qualitative_measures : list[Callable], optional
            List of association measures to be used for ``qualitative_features``.
            :ref:`Implemented measures <Measures>` are:

            * For association evaluation: :ref:`Pearson's chi² <chi2>`, :ref:`cramerv`, :ref:`tschuprowt` (default)

        quantitative_filters : list[Callable], optional
            List of filters to be used for ``quantitative_features``.
            :ref:`Implemented filters <Filters>` are:

            * For linear correlation: :ref:`pearson_filter`, :ref:`Spearman's rho <spearman_filter>` (default)

        qualitative_filters : list[Callable], optional
            List of filters to be used for ``qualitative_features``.
            :ref:`Implemented filters <Filters>` are:

            * For correlation: :ref:`cramerv_filter`, :ref:`tschuprowt_filter` (default)

        colsample : float, optional
            Size of sampled list of features for sped up computation, between ``0`` and ``1``, by default ``1.0``
            By default, all features are used.

            For ``colsample=0.5``, Selector will search for the best features in
            ``features[:len(features)//2]`` and then in ``features[len(features)//2:]``.

            **Tip:** for better performance, should be set such as ``len(features)//2 < 200``.

        verbose : bool, optional
            * ``True``, without IPython installed: prints raw feature selection steps for X, by default ``False``
            * ``True``, with IPython installed: adds HTML tables to the output.

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``.

        **kwargs
            Allows one to set thresholds for provided ``quantitative_measures``/``qualitative_measures`` and ``quantitative_filters``/``qualitative_filters`` (see :ref:`Measures` and :ref:`Filters`) passed as keyword arguments.

        Examples
        --------
        See `Selectors examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # settinp up list of features
        if quantitative_features is None:
            quantitative_features = []
        self.quantitative_features = list(set(quantitative_features))
        if qualitative_features is None:
            qualitative_features = []
        self.qualitative_features = list(set(qualitative_features))
        assert len(quantitative_features) > 0 or len(qualitative_features) > 0, (
            "No feature passed as input. Pleased provided column names to Carver by setting "
            "qualitative_features or quantitative_features."
        )
        self.features = list(set(self.qualitative_features + self.quantitative_features))
        self.input_dtypes = {"float": self.quantitative_features, "str": self.qualitative_features}

        # number of features selected
        self.n_best = n_best
        assert (
            0 < int(self.n_best) <= len(self.features) + 1
        ), "Must set 0 < n_best <= len(features)"

        # feature sample size per iteration
        self.colsample = colsample

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
        features: list[str],
        n_best: int,
        dtype: str,
    ) -> list[str]:
        """Selects the n_best features amongst the specified ones"""
        if self.verbose:  # verbose if requested
            data_type = "quantitative"
            if dtype != "float":
                data_type = "qualitative"
            print(f"------\n[Selector] Selecting from {data_type} features: {str(features)}\n---")

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
        for dtype in unique(list(self.input_dtypes.keys())):
            # getting features of the specific data type
            features = self.input_dtypes[dtype][:]

            # splitting features in chunks
            if self.colsample < 1:
                # shuffling features to get random samples of features
                shuffle(features)

                # number of features per sample
                chunks = int(len(self.features) // (1 / self.colsample))

                # splitting feature list in samples
                feature_samples = [
                    features[chunks * i : chunks * (i + 1)]
                    for i in range(int(1 / self.colsample) - 1)
                ]

                # adding last sample with all remaining features
                feature_samples += [features[chunks * (int(1 / self.colsample) - 1) :]]

                # iterating over each feature samples
                best_features = []
                for feature_sample in feature_samples:
                    # fitting association on features
                    best_features += self._select_features(
                        X, y, feature_sample, int(self.n_best // 2), dtype
                    )

            # splitting in chunks not requested
            else:
                best_features = features[:]

            # final selection with all best_features selected
            if any(best_features):
                best_features = self._select_features(X, y, best_features, self.n_best, dtype)
                all_best_features += best_features
                if self.verbose:  # verbose if requested
                    data_type = "quantitative"
                    if dtype != "float":
                        data_type = "qualitative"
                    print(f"\n - [Selector] Selected {data_type} features: {str(best_features)}")
                    print("------\n")

        return all_best_features

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
            print(f"\n - [Selector] Association between X and y{message}")

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
                display_html(nicer_association._repr_html_(), raw=True)


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
    X: DataFrame, y: Series, measures: list[Callable], features: list[str], **kwargs
) -> DataFrame:
    """Measures association between columns of X and y

    Parameters
    ----------
    X : DataFrame
        _description_
    y : Series
        _description_
    features : list[str]
        _description_
    measure_names : list[str]
        _description_

    Returns
    -------
    DataFrame
        _description_
    """
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
