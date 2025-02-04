"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series, notna

from ...features import CategoricalFeature
from ...utils import extend_docstring
from ..utils.base_discretizer import BaseDiscretizer, Sample


class CategoricalDiscretizer(BaseDiscretizer):
    """Automatic discretization of categorical features, building simple groups frequent enough.

    Groups a qualitative features' values less frequent than ``min_freq`` into a ``str_default``
    string.

    NaNs are left untouched.

    Only use for qualitative non-ordinal features.
    """

    __name__ = "CategoricalDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, append=False, exclude=["features"])
    def __init__(
        self,
        categoricals: list[CategoricalFeature],
        min_freq: float,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        categoricals : list[CategoricalFeature]
            Categorical features to process
        """
        # Initiating BaseDiscretizer
        super().__init__(categoricals, **dict(kwargs, min_freq=min_freq))

    def _prepare_data(self, sample: Sample) -> Sample:
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
        sample = super()._prepare_data(sample)

        # fitting features
        self.features.fit(**sample)

        # filling up nans for features that have some
        sample.X = self.features.fillna(sample.X)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # copying dataframe and checking data before bucketization
        sample = self._prepare_data(Sample(X, y))

        self._log_if_verbose()  # verbose if requested

        # grouping modalities less frequent than min_freq into feature.default
        sample.X = self._group_rare_modalities(sample.X)

        # sorting features' values by target rate
        self._target_sort(**sample)

        # discretizing features based on each feature's values_order
        super().fit(**sample)

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

        # grouping rare modalities for each feature
        for feature in self.features:
            X = self._group_feature_rare_modalities(feature, X, frequencies)

        return X

    def _target_sort(self, X: DataFrame, y: Series) -> None:
        """Sorts features' values by target rate"""

        # computing target rate per modality for ordering
        target_rates = X[self.features.versions].apply(series_target_rate, y=y, axis=0)

        # sorting features' values based on target rates
        self.features.update(
            {feature: list(sorted_values) for feature, sorted_values in target_rates.items()},
            sorted_values=True,
        )


def series_target_rate(x: Series, y: Series, dropna: bool = True, ascending=True) -> dict:
    """Target y rate per modality of x into a dictionnary"""

    rates = y.groupby(x, dropna=dropna).mean().sort_index().sort_values(ascending=ascending)

    return rates.to_dict()


def series_value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> dict:
    """Counts the values of each modality of a series into a dictionnary"""

    values = x.value_counts(dropna=dropna, normalize=normalize)

    return values.to_dict()
