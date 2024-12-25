"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from pandas import DataFrame, Series

from ...combinations import (
    CombinationEvaluator,
    CramervCombinations,
    KruskalCombinations,
    TschuprowtCombinations,
)
from ...discretizers import BaseDiscretizer, Discretizer, Sample
from ...features import BaseFeature, Features, GroupedList
from ...utils import get_attribute, get_bool_attribute
from .pretty_print import index_mapper, prettier_xagg

# trying to import extra dependencies
try:
    from IPython.display import display_html
except ImportError:
    _has_idisplay = False
else:
    _has_idisplay = True


@dataclass
class Samples:
    """
    A container for storing training and development samples.

    Attributes:
        train (Sample): The training sample, containing features (X) and target (y).
        dev (Sample): The development sample, containing features (X) and target (y).

    Example:
        >>> import pandas as pd
        >>> from base_carver import Sample, Samples
        >>> X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> y_train = pd.Series([0, 1, 0])
        >>> X_dev = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [10, 11, 12]})
        >>> y_dev = pd.Series([1, 0, 1])
        >>> train_sample = Sample(X=X_train, y=y_train)
        >>> dev_sample = Sample(X=X_dev, y=y_dev)
        >>> samples = Samples(train=train_sample, dev=dev_sample)
        >>> print(samples.train.X)
           feature1  feature2
        0         1         4
        1         2         5
        2         3         6
        >>> print(samples.dev.y)
        0    1
        1    0
        2    1
        dtype: int64
    """

    train: Sample = Sample(X=None)
    dev: Sample = Sample(X=None)


class BaseCarver(BaseDiscretizer, ABC):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary or continuous target.

    First fits a :class:`Discretizer`. Raw data should be provided as input (not a result of
    ``Discretizer.transform()``).
    """

    __name__ = "AutoCarver"
    is_y_binary = False
    is_y_continuous = False
    is_y_multiclass = False

    def __init__(
        self,
        features: Features,
        min_freq: float,
        combinations: CombinationEvaluator,
        dropna: bool = True,
        **kwargs,
    ) -> None:
        """
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between ``0.01`` (slower, preciser, less robust) and ``0.2``
            (faster, more robust)

        quantitative_features : list[str], optional
            List of column names of quantitative features (continuous and discrete) to be carved,
            by default ``None``

        qualitative_features : list[str], optional
            List of column names of qualitative features (non-ordinal) to be carved,
            by default ``None``

        ordinal_features : list[str], optional
            List of column names of ordinal features to be carved. For those features a list of
            values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a :class:`GroupedList` will automatically be initiated,
            by default ``None``

        max_n_mod : int, optional
            Maximum number of modality per feature, by default ``5``

            All combinations of modalities for groups of modalities of sizes from 1 to
            ``max_n_mod`` will be tested.
            The combination with the best association will be selected.

            **Tip**: set between ``3`` (faster, more robust) and ``7`` (slower, less robust)

        min_freq_mod : float, optional
            Minimum frequency per final modality, by default ``None`` for ``min_freq/2``

        ordinal_encoding : bool, optional
            Whether or not to ordinal encode features, by default ``True``

            * ``True``, sets the rank of modalities as label.
            * ``False``, sets one modality of group as label.

        dropna : bool, optional
            * ``True``, try to group ``numpy.nan`` with other modalities.
            * ``False``, ``numpy.nan`` are ignored (not grouped), by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided
            DataFrame, by default ``False``

        verbose : bool, optional
            * ``True``, without IPython: prints raw steps for X, by default ``False``
            * ``True``, with IPython: adds HTML tables of target rates for X and X_dev

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``

        n_jobs : int, optional
            Number of processes used by multiprocessing, by default ``1``

        **kwargs: dict
            Pass values for ``str_default`` and ``str_nan`` (default string values),
            as long as ``pretty_print`` to turn off IPython
        """

        # minimum frequency for discretizer
        self.discretizer_min_freq = get_attribute(kwargs, "discretizer_min_freq", min_freq / 2)

        # Initiating BaseDiscretizer
        super().__init__(
            features,
            **dict(
                kwargs,
                verbose=get_bool_attribute(kwargs, "verbose", False),
                min_freq=min_freq,
                dropna=dropna,
                discretizer_min_freq=self.discretizer_min_freq,
            ),
        )

        # setting combinations evaluator
        self.combinations = combinations
        self.combinations.min_freq = self.min_freq
        self.combinations.verbose = self.verbose
        self.combinations.dropna = self.dropna

    @property
    def pretty_print(self) -> bool:
        """Returns the pretty_print attribute"""
        return self.verbose and _has_idisplay

    @property
    def max_n_mod(self) -> int:
        """Returns the max_n_mod attribute"""
        return self.combinations.max_n_mod

    @max_n_mod.setter
    def max_n_mod(self, value: int) -> None:
        """Sets the max_n_mod attribute"""
        self.combinations.max_n_mod = value

    def _prepare_data(self, samples: Samples) -> Samples:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``BaseCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default ``None``
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated,
            by default ``None``

        Returns
        -------
        tuple[DataFrame, DataFrame]
            Copies of (X, X_dev)
        """
        # checking for not provided y
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided, got {samples.train.y}")

        # Checking for binary target and copying X
        samples.train = super()._prepare_data(samples.train)
        samples.dev = super()._prepare_data(samples.dev)

        # discretizing features according to min_freq
        samples = discretize(self.features, samples, **self.kwargs)

        # setting dropna to True for filling up nans
        self.features.dropna = True

        # filling up nans
        samples.train.X = self.features.fillna(samples.train.X)
        if samples.dev.X is not None:
            samples.dev.X = self.features.fillna(samples.dev.X)

        return samples

    def fit(  # pylint: disable=W0222
        self,
        X: DataFrame,
        y: Series,
        *,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> None:
        """Finds the combination of modalities of X that provides the best association with y.

        Parameters
        ----------
        X : DataFrame
            Training dataset, to determine features' optimal carving.
            Needs to have columns has specified in ``features`` attribute.

        y : Series
            Target with wich the association is maximized.

        X_dev : DataFrame, optional
            Development dataset, to evaluate robustness of carved features, by default ``None``
            Should have the same distribution as X.

        y_dev : Series, optional
            Target of the development dataset, by default ``None``
            Should have the same distribution as y.
        """

        # initiating samples
        samples = Samples(Sample(X, y), Sample(X_dev, y_dev))

        # preparing datasets and checking for wrong values
        samples = self._prepare_data(samples)

        # logging if requested
        super().log_if_verbose("---------\n------")

        # computing crosstabs for each feature on train/test
        xaggs = self._aggregator(**samples.train)
        xaggs_dev = self._aggregator(**samples.dev)

        # getting all features to carve (features are removed from self.features)
        all_features = self.features.versions

        # carving each feature
        for n, feature in enumerate(all_features):
            num_iter = f"{n+1}/{len(all_features)}"  # logging iteration number
            self._carve_feature(self.features(feature), xaggs, xaggs_dev, num_iter)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    @abstractmethod
    def _aggregator(self, X: DataFrame, y: Series) -> Union[Series, DataFrame]:
        """Helper that aggregates X by y into crosstab or means (carver specific)"""

    def _carve_feature(
        self,
        feature: BaseFeature,
        xaggs: dict[str, Union[Series, DataFrame]],
        xaggs_dev: dict[str, Union[Series, DataFrame]],
        num_iter: str,
    ) -> dict[str, GroupedList]:
        """Carves a feature into buckets that maximize association with the target"""

        # verbose if requested
        if self.verbose:
            print(f"--- [{self.__name__}] Fit {feature} ({num_iter})")

        # getting xtabs on train/test
        xagg = xaggs[feature.version]
        xagg_dev = xaggs_dev[feature.version]

        # printing raw distribution
        self._print_xagg(feature, xagg=xagg, xagg_dev=xagg_dev, message="Raw distribution")

        # getting best combination
        best_combination = self.combinations.get_best_combination(feature, xagg, xagg_dev=xagg_dev)

        # printing carved distribution, for found, suitable combination
        if best_combination is not None:
            self._print_xagg(
                feature,
                xagg=self.combinations.samples.train.xagg,
                xagg_dev=self.combinations.samples.dev.xagg,
                message="Carved distribution",
            )

        # no suitable combination has been found -> removing feature
        else:
            print(
                f"WARNING: No robust combination for {feature}. Consider increasing the size of "
                "X_dev or dropping the feature (X not representative of X_dev for this feature)."
            )
            self.features.remove(feature.version)

    def _print_xagg(
        self,
        feature: BaseFeature,
        xagg: Union[DataFrame, Series],
        message: str,
        *,
        xagg_dev: Union[DataFrame, Series] = None,
    ) -> None:
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
            print(f" [{self.__name__}] {message}")

            formatted_xagg, formatted_xagg_dev = self._format_xagg(feature, xagg, xagg_dev)

            nice_xagg, nice_xagg_dev = self._pretty_print(formatted_xagg, formatted_xagg_dev)

            if not self.pretty_print:  # no pretty hmtl printing
                self._print_raw(nice_xagg, nice_xagg_dev, xagg_dev)
            else:  # pretty html printing
                self._print_html(nice_xagg, nice_xagg_dev)

    def _format_xagg(
        self, feature: BaseFeature, xagg: DataFrame, xagg_dev: DataFrame = None
    ) -> tuple[DataFrame, DataFrame]:
        """Formats the XAGG DataFrame."""
        formatted_xagg = index_mapper(feature, xagg)
        formatted_xagg_dev = index_mapper(feature, xagg_dev)
        return formatted_xagg, formatted_xagg_dev

    def _pretty_print(
        self, formatted_xagg: DataFrame, formatted_xagg_dev: DataFrame
    ) -> tuple[str, str]:
        """Returns pretty-printed XAGG DataFrames."""
        nice_xagg = self.combinations._compute_target_rates(formatted_xagg)
        nice_xagg_dev = self.combinations._compute_target_rates(formatted_xagg_dev)
        return nice_xagg, nice_xagg_dev

    def _print_raw(self, nice_xagg: str, nice_xagg_dev: str, xagg_dev: DataFrame = None) -> None:
        """Prints raw XAGG DataFrames."""
        print(nice_xagg, "\n")
        if xagg_dev is not None:
            print("X_dev distribution\n", nice_xagg_dev, "\n")

    def _print_html(self, nice_xagg: str, nice_xagg_dev: str) -> None:
        """Prints XAGG DataFrames in HTML format."""
        # getting prettier xtabs
        nicer_xagg = prettier_xagg(nice_xagg, caption="X distribution")
        nicer_xagg_dev = prettier_xagg(nice_xagg_dev, caption="X_dev distribution", hide_index=True)

        # merging outputs
        nicer_xaggs = nicer_xagg + "          " + nicer_xagg_dev

        # displaying html of colored DataFrame
        display_html(nicer_xaggs, raw=True)

    @classmethod
    def load(cls, file_name: str) -> "BaseCarver":
        """Allows one to load a Carver saved as a .json file.

        The Carver has to be saved with ``Carver.save()``, otherwise there
        can be no guarantee for it to be restored.

        Parameters
        ----------
        file_name : str
            String of saved Carver's .json file name.

        Returns
        -------
        BaseDiscretizer
            A fitted Carver.
        """
        # reading file
        with open(file_name, "r", encoding="utf-8") as json_file:
            carver_json = json.load(json_file)

        # deserializing features
        features = Features.load(carver_json.pop("features"))

        # deserializing Combinations
        combinations = carver_json.pop("combinations")
        if combinations["sort_by"] == "tschuprowt":
            combinations = TschuprowtCombinations.load(combinations)
        elif combinations["sort_by"] == "cramerv":
            combinations = CramervCombinations.load(combinations)
        elif combinations["sort_by"] == "kruskal":
            combinations = KruskalCombinations.load(combinations)
        else:
            combinations = CombinationEvaluator.load(combinations)

        # initiating BaseDiscretizer
        return cls(features=features, combinations=combinations, **carver_json)


def discretize(
    features: Features,
    samples: Samples,
    discretizer_min_freq: float,
    **kwargs: dict,
) -> Samples:
    """Discretizes X and X_dev according to the frequency of each feature's modalities."""

    # discretizing all features, always copying, to keep discretization from start to finish
    discretizer = Discretizer(
        features=features,
        **dict(
            kwargs,
            dropna=False,
            copy=True,
            ordinal_encoding=False,
            min_freq=discretizer_min_freq,
        ),
    )

    # fitting discretizer on X
    samples.train.X = discretizer.fit_transform(**samples.train)

    # applying discretizer on X_dev if provided
    if samples.dev.X is not None:
        samples.dev.X = discretizer.transform(**samples.dev)

    return samples
