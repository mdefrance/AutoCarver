"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle
from typing import Any, Callable

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


class FeatureSelector:
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a binary target.

    * Best features are the n_best of each measure
    * Get your best features with ``FeatureSelector.select()``!
    """

    def __init__(
        self,
        n_best: int,
        *,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        measures: list[Callable] = None,
        filters: list[Callable] = None,
        colsample: float = 1.0,
        verbose: bool = False,
        pretty_print: bool = False,
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
            If ``True``, prints raw Discretizers Fit and Transform steps, as long as
            information on AutoCarver's processing and tables of target rates and frequencies for
            X, by default ``False``

        pretty_print : bool, optional
            If ``True``, adds to the verbose some HTML tables of target rates and frequencies for X and, if provided, X_dev.
            Overrides the value of ``verbose``, by default ``False``

        **kwargs
            Sets thresholds for ``measures`` and ``filters``, passed as keyword arguments.

        Examples
        --------
        See `FeatureSelector examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # settinp up list of features
        if quantitative_features is None:
            quantitative_features = []
        if qualitative_features is None:
            qualitative_features = []
        assert (
            len(quantitative_features) > 0 or len(qualitative_features) > 0
        ), "No feature passed as input. Pleased provided column names to Carver by setting qualitative_features or quantitative_features."
        assert (len(quantitative_features) > 0 and len(qualitative_features) == 0) or (
            len(qualitative_features) > 0 and len(quantitative_features) == 0
        ), "Mixed quantitative and qualitative features. One only of quantitative_features and qualitative_features should be set."
        self.features = list(set(qualitative_features + quantitative_features))

        # number of features selected
        self.n_best = n_best
        assert (
            0 < int(n_best // 2) <= len(self.features) + 1
        ), "Must set 0 < n_best // 2 <= len(features)"

        # feature sample size per iteration
        self.colsample = colsample

        # initiating measures
        if measures is None:
            if any(quantitative_features):  # quantitative feature association measure
                measures = [kruskal_measure]
            else:  # qualitative feature association measure
                measures = [tschuprowt_measure]
        self.measures = [dtype_measure, nans_measure, mode_measure] + measures[:]

        # initiating filters
        if filters is None:
            if any(quantitative_features):  # quantitative feature association measure
                filters = [spearman_filter]
            else:  # qualitative feature association measure
                filters = [tschuprowt_filter]
        self.filters = [thresh_filter] + filters[:]

        # names of measures to sort by
        self.measure_names = [measure.__name__ for measure in measures[::-1]]

        # wether or not to print tables
        self.verbose = bool(max(verbose, pretty_print))
        if pretty_print:
            if _has_idisplay:  # checking for installed dependencies
                self.pretty_print = pretty_print
            else:
                self.verbose = True
                print(
                    "Package not found: ipython. Defaulting to verbose=True. Install extra dependencies with pip install autocarver[jupyter]"
                )

        # keyword arguments
        self.kwargs = kwargs

    def _select_features(
        self, X: DataFrame, y: Series, features: list[str], n_best: int
    ) -> list[str]:
        """Selects the n_best features amongst the specified ones

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_
        features : list[str]
            _description_
        n_best : int
            _description_

        Returns
        -------
        list[str]
            _description_
        """
        if self.verbose:  # verbose if requested
            print(f"------\n[FeatureSelector] Selecting from Features: {str(features)}\n---")

        # Computes association between X and y
        initial_associations = apply_measures(
            X, y, measures=self.measures, features=features, **self.kwargs
        )

        # sorting statistics
        measure_names = evaluated_measure_names(initial_associations, self.measure_names)
        initial_associations = initial_associations.sort_values(measure_names, ascending=False)

        if self.verbose:  # displaying association measure
            print("\n - Association between X and y")
            print_associations(initial_associations, self.pretty_print)

        # applying filtering for each measure
        all_best_features: dict[str, Any] = {}
        for measure_name in measure_names:
            # sorting association for each measure
            associations = initial_associations.sort_values(measure_name, ascending=False)

            # filtering for each measure, as each measure ranks the features differently
            filtered_association = apply_filters(
                X, associations, filters=self.filters, **self.kwargs
            )

            # selected features for the measure
            selected_features = [
                feature
                for feature in initial_associations.index
                if feature in filtered_association.index[:n_best]
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
            print("\n - Association between X and y, filtered for inter-feature assocation")
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
        # splitting features in chunks
        if self.colsample < 1:
            # shuffling features to get random samples of features
            shuffle(self.features)

            # number of features per sample
            chunks = int(len(self.features) // (1 / self.colsample))

            # splitting feature list in samples
            feature_samples = [
                self.features[chunks * i : chunks * (i + 1)]
                for i in range(int(1 / self.colsample) - 1)
            ]

            # adding last sample with all remaining features
            feature_samples += [self.features[chunks * (int(1 / self.colsample) - 1) :]]

            # iterating over each feature samples
            best_features = []
            for features in feature_samples:
                # fitting association on features
                best_features += self._select_features(X, y, features, int(self.n_best // 2))

        # splitting in chunks not requested
        else:
            best_features = self.features[:]

        # final selection with all best_features selected
        if any(best_features):
            best_features = self._select_features(X, y, best_features, self.n_best)

        return best_features


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
