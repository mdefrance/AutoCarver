"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from typing import Union

import numpy as np
from numpy import argmin, nan, select
from pandas import DataFrame, Series, isna, unique, notna

from ..features import BaseFeature, CategoricalFeature, Features, GroupedList, OrdinalFeature
from .utils.base_discretizers import BaseDiscretizer, extend_docstring
from .utils.type_discretizers import StringDiscretizer


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
        super().__init__(categoricals, **kwargs)
        self.min_freq = min_freq

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
        x_copy = self._group_defaults(x_copy)

        # sorting features' values by target rate
        self._target_sort(x_copy, y)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def _group_defaults(self, X: DataFrame) -> DataFrame:
        """Groups modalities less frequent than min_freq into feature.default"""

        # computing frequencies of each modality
        frequencies = X[self.features.get_names()].apply(series_value_counts, axis=0)

        # iterating over each feature
        for feature in self.features:
            # checking for rare values
            values_to_group = [
                value
                for value, freq in frequencies[feature.name].items()
                if freq < self.min_freq and value != feature.nan and notna(value)
            ]

            # checking for completly missing values (no frequency observed in X)
            missing_values = [
                value for value in feature.values if value not in frequencies[feature.name]
            ]
            if len(missing_values) > 0:
                raise ValueError(
                    f" - [{self.__name__}] Unexpected values {missing_values} for {feature}."
                )

            # grouping values to str_default if any
            if any(values_to_group):
                # adding default value to the order
                feature.set_has_default(True)

                # grouping rare values in default value
                feature.group_list(values_to_group, feature.default)
                X.loc[X[feature.name].isin(values_to_group), feature.name] = feature.default

        return X

    def _target_sort(self, X: DataFrame, y: Series) -> None:
        """Sorts features' values by target rate"""

        # computing target rate per modality for ordering
        target_rates = X[self.features.get_names()].apply(series_target_rate, y=y, axis=0)

        # sorting features' values based on target rates
        self.features.update(
            {feature: list(sorted_values) for feature, sorted_values in target_rates.items()},
            sorted_values=True,
        )


class OrdinalDiscretizer(BaseDiscretizer):
    """Automatic discretization of ordinal features, grouping less frequent modalities with the
    closest modlity in target rate or by frequency.

    NaNs are left untouched.

    Only use for qualitative ordinal features.

    Fisrt fits :ref:`StringDiscretizer` if neccesary.
    """

    __name__ = "OrdinalDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        ordinals: list[OrdinalFeature],
        min_freq: float,
        **kwargs: dict,
    ):
        """
        Parameters
        ----------
        ordinal_features : list[str]
            List of column names of ordinal features to be discretized. For those features a list
            of values has to be provided in the ``values_orders`` dict.

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < than ``min_freq`` won't be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between ``0.02`` (slower, preciser, less robust) and ``0.05``
            (faster, more robust)

        input_dtypes : Union[str, dict[str, str]], optional
            Input data type, converted to a dict of the provided type for each feature,
            by default ``"str"``

            * ``"str"``, features are considered as qualitative.
            * ``'float"``, features are considered as quantitative.
        """
        # Initiating BaseDiscretizer
        super().__init__(ordinals, **kwargs)

        # class specific attributes
        self.min_freq = min_freq

    def _prepare_data(self, X: DataFrame, y: Series) -> DataFrame:  # pylint: disable=W0222
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``OrdinalDiscretizer.features``.

        y : Series
            Binary target feature.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for binary target and copying X
        x_copy = super()._prepare_data(X, y)

        # fitting features
        self.features.fit(x_copy, y)

        # filling up nans for features that have some
        x_copy = self.features.fillna(x_copy)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222

        # checking values orders
        x_copy = self._prepare_data(X, y)
        self._verbose()  # verbose if requested

        # grouping rare modalities for each feature
        common_modalities = {
            feature.name: find_common_modalities(
                x_copy[feature.name],
                y,
                min_freq=self.min_freq,
                order=[label for label in feature.labels if label != feature.nan],
            )
            for feature in self.features
        }

        # updating features accordingly
        self.features.update(common_modalities, convert_labels=True)

        # discretizing features based on each feature's values_order
        super().fit(x_copy, y)

        return self


class ChainedDiscretizer(BaseDiscretizer):
    """Automatic discretization of categorical features, joining rare modalities into higher
    level groups.

    For each provided :class:`GroupedList` from ``chained_orders`` attribute, values less frequent
    than ``min_freq`` are grouped in there respective group, as defined by :class:`GroupedList`.
    """

    __name__ = "ChainedDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        min_freq: float,
        features: list[BaseFeature],
        chained_orders: list[GroupedList],
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        qualitative_features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized

        chained_orders : list[GroupedList]
            A list of interlocked higher level groups for each modalities of each ordinal feature.
            Values of ``chained_orders[0]`` have to be grouped in ``chained_order[1]`` etc.

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < than ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: set between ``0.02`` (slower, less robust) and ``0.05`` (faster, more robust)

        unknown_handling : str, optional
            Whether or not to remove unknown values, by default ``'raise'``.

            * ``'raise'``, unknown values raise an ``AssertionError``.
            * ``'drop'``, unknown values are grouped with ``str_nan``.
        """
        # not dropping nans whatsoever
        kwargs = dict(kwargs, dropna=False)
        super().__init__(features, **kwargs)  # Initiating BaseDiscretizer

        # class specific attributes
        self.min_freq = min_freq
        self.chained_orders = [GroupedList(values) for values in chained_orders]

        # known_values: all ordered values describe in each level of the chained_orders
        # starting off with first level
        known_values = self.chained_orders[0].values()
        # adding each level
        for n, next_level in enumerate(self.chained_orders[1:]):
            # iterating over each group of the next level
            for next_group, next_values in next_level.content.items():
                # looking for known and unknwon values in next_level

                # checking for unknown values
                next_unknown = [
                    value
                    for value in next_values
                    if value not in known_values and value != next_group
                ]
                if len(next_unknown) > 0:
                    raise ValueError(
                        f" - [{self.__name__}] Values {str(next_unknown)}, provided in "
                        f"chained_orders[{n+1}] are missing from chained_orders[{n}]. Please make "
                        "sure values are kept trhough each level."
                    )

                # checking for known values
                next_known = [
                    value for value in next_values if value in known_values and value != next_group
                ]
                if len(next_known) == 0:
                    raise ValueError(
                        f" - [{self.__name__}] For key '{next_group}', the provided chained_orders"
                        f"[{n+1}] has no values from chained_orders[:{n+1}]. Please provide some"
                        " existing values."
                    )

                # index of the highest ranked known value of the next_group
                highest_index = known_values.index(next_known[-1])

                # adding next_group to the order at the right place in the amongst known_values
                known_values = (
                    known_values[: highest_index + 1]
                    + [next_group]
                    + known_values[highest_index + 1 :]
                )

        # saving resulting known values
        self.known_values = known_values

        # adding known_values to each feature's order
        for feature in self.features:
            # checking for already known values of the feature
            order = feature.values
            # no known values for the feature
            if order is None:
                order = GroupedList([])

            # checking that all values from the order are in known_values
            for value in order:
                if value not in self.known_values:
                    raise ValueError(
                        f" - [{self.__name__}] Value {value} from feature {feature} provided in "
                        "values_orders is missing from levels of chained_orders. Add value to a "
                        "level of chained_orders or adapt values_orders."
                    )
            # adding known values if missing from the order
            for value in self.known_values:
                if value not in order.values():
                    order.append(value)
            # sorting in case an ordering was provided
            order = order.sort_by(self.known_values)
            # updating feature
            feature.update(order, replace=True)

    def _prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Validates format and content of X and y. Converts non-string columns into strings.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``ChainedDiscretizer.features``.

        y : Series
            Binary target feature, not used, by default None.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # copying dataframe
        x_copy = X.copy()

        # checking for binary target and previous fit
        x_copy = super()._prepare_data(x_copy, y)

        # checking feature values' frequencies
        check_frequencies(self.features, x_copy, self.min_freq, self.__name__)

        # converting non-str columns
        x_copy = check_dtypes(self.features, x_copy, **self.kwargs)

        # fitting features
        self.features.fit(x_copy, y)

        # filling up nans for features that have some
        x_copy = self.features.fillna(x_copy)

        # checking for unexpected values
        self.features.check_values(x_copy)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222

        # preprocessing data
        x_copy = self._prepare_data(X, y)
        self._verbose()  # verbose if requested

        # iterating over each feature
        for feature in self.features:

            # iterating over each specified orders
            for level_order in self.chained_orders:
                # computing frequencies of each modality
                frequencies = (
                    x_copy[feature.name]
                    .value_counts(normalize=True, dropna=False)
                    .drop(nan, errors="ignore")
                )
                values, frequencies = frequencies.index, frequencies.values

                # values that are frequent enough
                to_keep = list(values[frequencies >= self.min_freq])

                # values from the order to group (not frequent enough or absent)
                values_to_group = [value for value in level_order.values() if value not in to_keep]

                # values to group into discarded values
                groups_value = [level_order.get_group(value) for value in values_to_group]

                # values of the feature to input (needed for next levels of the order)
                df_to_input = [x_copy[feature.name] == discarded for discarded in values_to_group]

                # inputing non frequent values
                x_copy[feature.name] = select(
                    df_to_input, groups_value, default=x_copy[feature.name]
                )

                # historizing in the feature's order
                order = GroupedList(feature.values)
                for discarded, kept in zip(values_to_group, groups_value):
                    order.group(discarded, kept)

                # updating feature accordingly
                feature.update(order, replace=True)

        super().fit(X, y)

        return self


def series_target_rate(x: Series, y: Series, dropna: bool = True, ascending=True) -> dict:
    """Target y rate per modality of x into a dictionnary"""

    rates = y.groupby(x, dropna=dropna).mean().sort_values(ascending=ascending)

    return rates.to_dict()


def series_value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> dict:
    """Counts the values of each modality of a series into a dictionnary"""

    values = x.value_counts(dropna=dropna, normalize=normalize)

    return values.to_dict()


def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    order: list[str],
) -> dict[str, Union[str, float]]:
    """finds common modalities of a ordinal feature"""
    # making a groupedlist of ordered labels
    order = GroupedList(order)

    # total size
    len_df = len(df_feature)

    # computing frequencies and target rate of each modality
    not_nans = ~isna(df_feature)  # pylint: disable=E1130
    stats = np.vstack(
        (
            df_feature[not_nans]
            .value_counts(dropna=False, normalize=False)
            .reindex(order, fill_value=0)
            .values,
            y[not_nans].groupby(df_feature[not_nans]).sum().reindex(order).values,
        )
    )

    # case 2.1: there are underrepresented modalities/values
    while any(stats[0, :] / len_df < min_freq) & (stats.shape[1] > 1):
        # identifying the underrepresented value
        discarded_idx = argmin(stats[0, :])

        # choosing amongst previous and next modality (by volume and target rate)
        kept_idx = find_closest_modality(
            discarded_idx,
            stats[0, :] / len_df,
            stats[1, :] / stats[0, :],
            min_freq,
        )

        # removing the value from the initial ordering
        order.group(order[discarded_idx], order[kept_idx])

        # adding up grouped frequencies and target counts
        stats[:, kept_idx] += stats[:, discarded_idx]

        # removing discarded modality
        stats = stats[:, np.arange(stats.shape[1]) != discarded_idx]

    # case 2.2 : no underrepresented value
    return order


def find_closest_modality(
    idx: int, frequencies: np.array, target_rates: np.array, min_freq: float
) -> int:
    """Finds the closest modality in terms of frequency and target rate"""
    # case 1: lowest ranked modality
    if idx == 0:
        return 1

    # case 2: highest ranked modality
    if idx == frequencies.shape[0] - 1:
        return idx - 1

    # case 3: modality ranked in the middle
    # modalities frequencies and target rates
    (
        (previous_freq, current_freq, next_freq),
        (previous_target, current_target, next_target),
        idx_closest_modality,
    ) = (
        frequencies[idx - 1 : idx + 2],
        target_rates[idx - 1 : idx + 2],
        idx - 1,
    )  # by default previous value

    # cases when idx + 1 is the closest
    if (
        # case 1: next modality is the only below min_freq -> underrepresented
        (next_freq < min_freq <= previous_freq)
        or (
            (
                # case 2: both are below min_freq
                ((next_freq < min_freq) and (previous_freq < min_freq))
                or
                # case 3: both are above min_freq -> representative modalities
                ((next_freq >= min_freq) and (previous_freq >= min_freq))
            )
            and (
                # no target to differentiate -> least frequent modality
                ((current_freq == 0) and (next_freq < previous_freq))
                or
                # differentiate by target -> closest target rate
                (
                    (current_target > 0)
                    and (abs(previous_target - current_target) > abs(next_target - current_target))
                )
            )
        )
    ):
        # identifying smallest modality in terms of frequency
        idx_closest_modality = idx + 1

    # finding the closest value
    return idx_closest_modality


def check_frequencies(features: Features, X: DataFrame, min_freq: float, name: str) -> None:
    """Checks features' frequencies compared to min_freq"""

    # computing features' max modality frequency (mode frequency)
    max_freq = X[features.get_names()].apply(
        lambda u: u.value_counts(normalize=True, dropna=False).max(),
        axis=0,
    )

    # features with no common modality (biggest value less frequent than min_freq)
    non_common = [f.name for f in features if max_freq[f.name] < min_freq]

    # features with too common modality (biggest value more frequent than 1-min_freq)
    too_common = [f.name for f in features if max_freq[f.name] > 1 - min_freq]

    # raising
    if len(too_common + non_common) > 0:
        # building error message
        error_msg = (
            f" - [{name}] Features {str(too_common + non_common)} contain a too frequent modality "
            "or no frequent enough modalities. Consider decreasing min_freq or removing these "
            "features.\nINFO:\n"
        )

        # adding features with no common values
        non_common = [
            (
                f" - {features(feature)}: most frequent value has "
                f"freq={max_freq[feature]:2.2%} < min_freq={min_freq:2.2%}"
            )
            for feature in non_common
        ]

        # adding features with too common values
        too_common = [
            (
                f" - {features(feature)}: most frequent value has "
                f"freq={max_freq[feature]:2.2%} > (1-min_freq)={1-min_freq:2.2%}"
            )
            for feature in too_common
        ]
        error_msg += "\n".join(too_common + non_common)

        raise ValueError(error_msg)


def check_dtypes(features: Features, X: DataFrame, **kwargs: dict) -> DataFrame:
    """Checks features' data types and converts int/float to str"""

    # getting per feature data types
    dtypes = (
        X.fillna({feature.name: feature.nan for feature in features if feature.has_nan})[
            features.get_names()
        ]
        .map(type)
        .apply(unique, result_type="reduce")
    )

    # identifying features that are not str
    not_object = dtypes.apply(lambda u: any(dtype != str for dtype in u))

    # converting detected non-string features
    if any(not_object):
        # converting non-str features into qualitative features
        to_convert = [feat for feat in features if feat.name in not_object.index[not_object]]
        string_discretizer = StringDiscretizer(features=to_convert, **kwargs)
        X = string_discretizer.fit_transform(X)

    return X
