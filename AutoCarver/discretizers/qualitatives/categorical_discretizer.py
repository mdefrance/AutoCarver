"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series, notna

from ...features import CategoricalFeature
from ...utils import extend_docstring
from ..utils.base_discretizer import BaseDiscretizer


class CategoricalDiscretizer(BaseDiscretizer):
    """Automatic discretization of categorical features, building simple groups frequent enough.

    Groups a qualitative features' values less frequent than ``min_freq`` into a ``str_default``
    string.

    NaNs are left untouched.

    Only use for qualitative non-ordinal features.
    """

    __name__ = "CategoricalDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        categoricals: list[CategoricalFeature],
        min_freq: float,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: set between ``0.02`` (slower, less robust) and ``0.05`` (faster, more robust)
        """
        # Initiating BaseDiscretizer
        super().__init__(categoricals, **dict(kwargs, min_freq=min_freq))

    def _prepare_data(self, X: DataFrame, y: Series) -> DataFrame:  # pylint: disable=W0222
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``CategoricalDiscretizer.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for binary target
        x_copy = super()._prepare_data(X, y)

        # fitting features
        self.features.fit(x_copy, y)

        # filling up nans for features that have some
        x_copy = self.features.fillna(x_copy)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # copying dataframe and checking data before bucketization
        x_copy = self._prepare_data(X, y)

        self._verbose()  # verbose if requested

        # grouping modalities less frequent than min_freq into feature.default
        x_copy = self._group_rare_modalities(x_copy)

        # sorting features' values by target rate
        self._target_sort(x_copy, y)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def _group_feature_rare_modalities(
        self, feature: CategoricalFeature, X: DataFrame, frequencies: DataFrame
    ) -> DataFrame:
        """Groups modalities less frequent than min_freq into feature.default"""

        # checking for rare values
        values_to_group = [
            value
            for value, freq in frequencies[feature.version].items()
            if freq < self.min_freq and value != feature.nan and notna(value)
        ]

        # checking for completly missing values (no frequency observed in X)
        missing_values = [
            value for value in feature.values if value not in frequencies[feature.version]
        ]
        if len(missing_values) > 0:
            raise ValueError(f"[{self.__name__}] Unexpected values {missing_values} for {feature}.")

        # grouping values to str_default if any
        if any(values_to_group):
            # adding default value to the order
            feature.has_default = True

            # grouping rare values in default value
            feature.group(values_to_group, feature.default)
            X.loc[X[feature.version].isin(values_to_group), feature.version] = feature.default

        return X

    def _group_rare_modalities(self, X: DataFrame) -> DataFrame:
        """Groups modalities less frequent than min_freq into feature.default"""

        # computing frequencies of each modality
        frequencies = X[self.features.versions].apply(series_value_counts, axis=0)
        print("frequencies", frequencies)

        # grouping rare modalities for each feature
        for feature in self.features:
            X = self._group_feature_rare_modalities(feature, X, frequencies)

        return X

    def _target_sort(self, X: DataFrame, y: Series) -> None:
        """Sorts features' values by target rate"""

        # computing target rate per modality for ordering
        target_rates = X[self.features.versions].apply(series_target_rate, y=y, axis=0)
        print(target_rates)

        # sorting features' values based on target rates
        self.features.update(
            {feature: list(sorted_values) for feature, sorted_values in target_rates.items()},
            sorted_values=True,
        )
        print({feature: list(sorted_values) for feature, sorted_values in target_rates.items()})


def series_target_rate(x: Series, y: Series, dropna: bool = True, ascending=True) -> dict:
    """Target y rate per modality of x into a dictionnary"""

    rates = y.groupby(x, dropna=dropna).mean().sort_values(ascending=ascending)

    return rates.to_dict()


def series_value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> dict:
    """Counts the values of each modality of a series into a dictionnary"""

    values = x.value_counts(dropna=dropna, normalize=normalize)

    return values.to_dict()
