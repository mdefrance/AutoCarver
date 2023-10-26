"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle
from typing import Any, Callable, Union
from warnings import warn

from pandas import DataFrame, Series

from .filters import spearman_filter, thresh_filter, tschuprowt_filter
from .measures import (
    dtype_measure,
    kruskal_measure,
    make_measure,
    mode_measure,
    nans_measure,
    tschuprowt_measure,
)

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
    * Get your best features with ``FeatureSelector.select()``!
    """

    def __init__(
        self,
        n_best: int,
        features: list[str],
        *,
        input_dtypes: Union(str, dict[str, str]) = "float",  # TODO
        measures: Union(list[Callable], dict[str, list[Callable]]) = None,
        filters: Union(list[Callable], dict[str, list[Callable]]) = None,
        colsample: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        n_best : int
            Number of features to select.

        features : list[str], optional
            List of column names to chose from, by default ``None``

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
        # setting up features
        self.features = list(set(features))

        # number of features selected
        self.n_best = n_best
        assert (
            0 < int(n_best // 2) <= len(self.features) + 1
        ), "Must set 0 < n_best // 2 <= len(features)"

        # feature sample size per iteration
        self.colsample = colsample

        # checking for unique input_dtypes (str)
        if isinstance(input_dtypes, str):
            input_dtypes = {feature: input_dtypes for feature in features}
        self.input_dtypes = input_dtypes

        # initiating measures
        if isinstance(measures, list):
            assert isinstance(input_dtypes, str), (
                " - [BaseSelector] Provide a unique input data type corresponding to the provided "
                "list of measures: input_dtypes='str' or input_dtypes='float'"
            )
            measures = {input_dtypes: measures}
        # adding default measures
        self.measures = {
            dtype: [dtype_measure, nans_measure, mode_measure] + requested_measures[:]
            for dtype, requested_measures in measures.items()
        }

        # initiating filters
        if isinstance(filters, list):
            assert isinstance(input_dtypes, str), (
                " - [BaseSelector] Provide a unique input data type corresponding to the provided "
                "list of measures: input_dtypes='str' or input_dtypes='float'"
            )
            filters = {input_dtypes: filters[:]}
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
            print(f"------\n[FeatureSelector] Selecting from Features: {str(features)}\n---")

        # Computes association between X and y
        initial_associations = apply_measures(
            X, y, measures=self.measures[dtype], features=features, **self.kwargs
        )

        # sorting statistics
        measure_names = evaluated_measure_names(initial_associations, self.measure_names[dtype])
        initial_associations = initial_associations.sort_values(measure_names, ascending=False)

        if self.verbose:  # displaying association measure
            print("\n - [FeatureSelector] Association between X and y")
            print_associations(initial_associations, self.pretty_print)

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

        if self.verbose:  # displaying association measure
            print(
                "\n - [FeatureSelector] Association between X and y, filtered for inter-feature assocation"
            )
            print_associations(initial_associations.reindex(best_features), self.pretty_print)
            print("------\n")

        return best_features

    def select(self, X: DataFrame, y: Series) -> list[str]:
        """Selects the ``n_best`` features of the DataFrame, by association with the binary target

        Parameters
        ----------
        X : DataFrame
            Dataset used to measure association between features and target.
            Needs to have columns has specified in ``FeatureSelector.features``.
        y : Series
            Binary target feature with wich the association is maximized.

        Returns
        -------
        list[str]
            List of selected features
        """
        # iterating over each type of feature
        all_best_features = []
        for dtype in self.input_dtypes.values().unique():
            if self.verbose:  # verbose if requested
                if dtype == "float":
                    print(f"------\n[FeatureSelector] Selecting from Quantitative Features\n---")
                else:
                    print(f"------\n[FeatureSelector] Selecting from Qualitative Features\n---")

            # getting features of the specific data type
            features = [feature for feature in self.features if self.input_dtypes[feature] == dtype]

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
                    if dtype == "float":
                        print(
                            f"------\n[FeatureSelector] Selected Quantitative Features: {str(features)}\n---"
                        )
                    else:
                        print(
                            f"------\n[FeatureSelector] Selected Qualitative Features: {str(features)}\n---"
                        )

        return all_best_features


def print_associations(association: DataFrame, pretty_print: bool = False) -> None:
    """EDA of fitted associations

    Parameters
    ----------
    association : DataFrame
        _description_
    pretty_print : bool, optional
        _description_, by default False
    """
    # printing raw DataFrame
    if not pretty_print:
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
        # getting prettier association table
        nicer_association = association.style.background_gradient(cmap="coolwarm", subset=subset)
        nicer_association = nicer_association.set_table_attributes("style='display:inline'")

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
