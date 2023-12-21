"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""
from typing import Union
from warnings import warn

import numpy as np
from numpy import argmin, nan, select
from pandas import DataFrame, Series, isna, unique

from .base_discretizers import (
    BaseDiscretizer,
    convert_to_labels,
    convert_to_values,
    extend_docstring,
    nan_unique,
    target_rate,
    value_counts,
)
from .grouped_list import GroupedList
from .type_discretizers import StringDiscretizer


class CategoricalDiscretizer(BaseDiscretizer):
    """Automatic discretization of categorical features, building simple groups frequent enough.

    Groups a qualitative features' values less frequent than ``min_freq`` into a ``str_default``
    string.

    NaNs are left untouched.

    Only use for qualitative non-ordinal features.
    """

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        qualitative_features: list[str],
        min_freq: float,
        *,
        values_orders: dict[str, GroupedList] = None,
        copy: bool = False,
        verbose: bool = False,
        n_jobs: int = 1,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        qualitative_features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: set between ``0.02`` (slower, less robust) and ``0.05`` (faster, more robust)
        """
        # Initiating BaseDiscretizer
        super().__init__(
            features=qualitative_features,
            values_orders=values_orders,
            input_dtypes="str",
            output_dtype="str",
            str_nan=kwargs.get("str_nan", "__NAN__"),
            str_default=kwargs.get("str_default", "__OTHER__"),
            copy=copy,
            verbose=verbose,
            n_jobs=n_jobs,
        )

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

        # checks and initilizes values_orders
        for feature in self.qualitative_features:
            # initiating features missing from values_orders
            if feature not in self.values_orders:
                self.values_orders.update({feature: GroupedList(nan_unique(x_copy[feature]))})

        # checking that all unique values in X are in values_orders
        self._check_new_values(x_copy, features=self.qualitative_features)

        # adding NANS
        for feature in self.qualitative_features:
            order = self.values_orders[feature]
            if any(x_copy[feature].isna()):
                if self.str_nan not in order.values():
                    order.append(self.str_nan)
                    self.values_orders.update({feature: order})

        # filling up NaNs
        x_copy[self.qualitative_features] = x_copy[self.qualitative_features].fillna(self.str_nan)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # copying dataframe and checking data before bucketization
        x_copy = self._prepare_data(X, y)

        if self.verbose:  # verbose if requested
            print(f" - [CategoricalDiscretizer] Fit {str(self.qualitative_features)}")

        # computing frequencies of each modality
        frequencies = x_copy[self.qualitative_features].apply(value_counts, normalize=True, axis=0)

        for feature in self.qualitative_features:
            # sorting orders based on target rates
            order = self.values_orders[feature]

            # checking for rare values
            values_to_group = [
                val
                for val, freq in frequencies[feature].items()
                if freq < self.min_freq and val != self.str_nan
            ]

            # adding values that are completly missing (no frequency in X)
            values_to_group += [value for value in order if value not in frequencies[feature]]

            # grouping values to str_default if any
            if any(values_to_group):
                # adding default value to the order
                order.append(self.str_default)

                # grouping rare values in default value
                order.group_list(values_to_group, self.str_default)
                x_copy.loc[x_copy[feature].isin(values_to_group), feature] = self.str_default

                # updating values_orders
                self.values_orders.update({feature: order})

        # computing target rate per modality for ordering
        target_rates = x_copy[self.qualitative_features].apply(
            target_rate, y=y, ascending=True, axis=0
        )

        # sorting orders based on target rates
        for feature in self.qualitative_features:
            order = self.values_orders[feature]

            # new ordering according to target rate
            new_order = list(target_rates[feature])

            # checking for default but no value observed, enable to group this modal, raising error
            assert (self.str_default in order and self.str_default in new_order) or (
                self.str_default not in order and self.str_default not in new_order
            ), (
                f"Some values from values_orders['{feature}'] are never observed. Can not fit a "
                f"distribution without any observation. Please remove following values "
                f"{str([v for v in order.content[self.str_default] if v != self.str_default])}"
                f" from values_orders['{feature}']."
            )
            # leaving NaNs at the end of the list
            if self.str_nan in new_order:
                new_order.remove(self.str_nan)
                new_order += [self.str_nan]

            # sorting order updating values_orders
            self.values_orders.update({feature: order.sort_by(new_order)})

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


class OrdinalDiscretizer(BaseDiscretizer):
    """Automatic discretization of ordinal features, grouping less frequent modalities with the
    closest modlity in target rate or by frequency.

    NaNs are left untouched.

    Only use for qualitative ordinal features.

    Fisrt fits :ref:`StringDiscretizer` if neccesary.
    """

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        ordinal_features: list[str],
        min_freq: float,
        values_orders: dict[str, GroupedList],
        *,
        input_dtypes: Union[str, dict[str, str]] = "str",
        copy: bool = False,
        verbose: bool = False,
        n_jobs: int = 1,
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
        super().__init__(
            features=ordinal_features,
            values_orders=values_orders,
            input_dtypes=input_dtypes,
            output_dtype="str",
            str_nan=kwargs.get("str_nan", "__NAN__"),
            copy=copy,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        # checking for missong orders
        no_order_provided = [
            feature for feature in self.features if feature not in self.values_orders
        ]
        assert len(no_order_provided) == 0, (
            " - [OrdinalDiscretizer] No ordering was provided for following features:"
            f" {str(no_order_provided)}. Please make sure you defined ``values_orders`` correctly."
        )

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
        # adding NANS
        for feature in self.features:
            if any(x_copy[feature].isna()):
                values = self.values_orders[feature]
                if not values.contains(self.str_nan):
                    values.append(self.str_nan)
                    self.values_orders.update({feature: values})

        # removing NaNs if any already imputed -> grouping only non-nan values
        if self.str_nan:
            x_copy = x_copy.replace(self.str_nan, nan)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        if self.verbose:  # verbose if requested
            print(f" - [OrdinalDiscretizer] Fit {str(self.features)}")

        # checking values orders
        x_copy = self._prepare_data(X, y)

        # converting potential quantiles into there labels
        known_orders = convert_to_labels(
            features=self.features,
            quantitative_features=self.quantitative_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            dropna=True,
        )

        # grouping rare modalities for each feature
        common_modalities = {
            feature: find_common_modalities(
                x_copy[feature], y, min_freq=self.min_freq, order=known_orders[feature]
            )
            for feature in self.features
        }

        # converting potential labels into there respective values (quantiles)
        self.values_orders.update(
            convert_to_values(
                features=self.features,
                quantitative_features=self.quantitative_features,
                values_orders=self.values_orders,
                label_orders=common_modalities,
                str_nan=self.str_nan,
            )
        )

        # discretizing features based on each feature's values_order
        super().fit(x_copy, y)

        return self


class ChainedDiscretizer(BaseDiscretizer):
    """Automatic discretization of categorical features, joining rare modalities into higher
    level groups.

    For each provided :class:`GroupedList` from ``chained_orders`` attribute, values less frequent
    than ``min_freq`` are grouped in there respective group, as defined by :class:`GroupedList`.
    """

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        qualitative_features: list[str],
        min_freq: float,
        chained_orders: list[GroupedList],
        *,
        values_orders: dict[str, GroupedList] = None,
        unknown_handling: str = "raise",
        copy: bool = False,
        verbose: bool = False,
        n_jobs: int = 1,
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
        # Initiating BaseDiscretizer
        super().__init__(
            features=qualitative_features,
            values_orders=values_orders,
            input_dtypes="str",
            output_dtype="str",
            str_nan=kwargs.get("str_nan", "__NAN__"),
            dropna=False,
            copy=copy,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        # class specific attributes
        self.min_freq = min_freq
        self.chained_orders = [GroupedList(values) for values in chained_orders]
        assert unknown_handling in [
            "drop",
            "raise",
        ], " - [ChainedDiscretizer] Wrong value for unknown_handling. Choose from 'drop', 'raise'."
        self.unknown_handling = unknown_handling

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
                assert len(next_unknown) == 0, (
                    f" - [ChainedDiscretizer] Values {str(next_unknown)}, provided in chained_orde"
                    f"rs[{n+1}] are missing from chained_orders[{n}]. Please make sure values are "
                    "kept trhough each level."
                )
                # checking for known values
                next_known = [
                    value for value in next_values if value in known_values and value != next_group
                ]
                assert len(next_known) > 0, (
                    f" - [ChainedDiscretizer] For key '{next_group}', the provided chained_orders["
                    f"{n+1}] has no values from chained_orders[:{n+1}]. Please provide some values"
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
            if feature in self.values_orders:
                order = self.values_orders[feature]
            # no known values for the feature
            else:
                order = GroupedList([])

            # checking that all values from the order are in known_values
            for value in order:
                assert value in self.known_values, (
                    f" - [ChainedDiscretizer] Value {value} from feature {feature} provided in val"
                    "ues_orders is missing from levels of chained_orders. Add value to a level of "
                    "chained_orders or adapt values_orders."
                )
            # adding known values if missing from the order
            for value in self.known_values:
                if value not in order.values():
                    order.append(value)
            order = order.sort_by(self.known_values)
            self.values_orders.update({feature: order})

        self.verbose = verbose

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

        # checking for ids (unique value per row)
        max_frequencies = x_copy[self.features].apply(
            lambda u: u.value_counts(normalize=True, dropna=False).drop(nan, errors="ignore").max(),
            axis=0,
        )
        # for each feature, checking that at least one value is more frequent than min_freq
        all_features = self.features[:]
        for feature in all_features:
            if max_frequencies[feature] < self.min_freq:
                warn(
                    f" - [ChainedDiscretizer] For feature '{feature}', the largest modality"
                    f" has {max_frequencies[feature]:2.2%} observations which is lower than "
                    "min_freq={self.min_freq:2.2%}. This feature will not be Discretized. Consider"
                    " decreasing parameter min_freq or removing this feature.",
                    UserWarning,
                )
                super()._remove_feature(feature)

        # checking for columns containing floats or integers even with filled nans
        dtypes = x_copy[self.features].fillna(self.str_nan).map(type).apply(unique)
        not_object = dtypes.apply(lambda u: any(typ != str for typ in u))

        # non qualitative features detected
        if any(not_object):
            features_to_convert = list(not_object.index[not_object])
            if self.verbose:
                unexpected_dtypes = [
                    typ for dtyp in dtypes[not_object] for typ in dtyp if typ != str
                ]
                warn(
                    f" - [ChainedDiscretizer] Non-string features: {str(features_to_convert)}. Try"
                    "ing to convert them using type_discretizers.StringDiscretizer, otherwise conv"
                    f"ert them manually. Unexpected data types: {str(list(unexpected_dtypes))}.",
                    UserWarning,
                )

            # converting specified features into qualitative features
            stringer = StringDiscretizer(
                qualitative_features=features_to_convert,
                values_orders=self.values_orders,
                n_jobs=self.n_jobs,
            )
            x_copy = stringer.fit_transform(x_copy, y)

            # updating values_orders accordingly
            self.values_orders.update(stringer.values_orders)

        # filling nans
        x_copy[self.features] = x_copy[self.features].fillna(self.str_nan)

        # adding nans and unknown values
        for feature in self.features:
            order = self.values_orders[feature]
            # checking for unknown values (missing from known_values)
            unknown_values = [
                value
                for value in x_copy[feature].unique()
                if value not in self.known_values and value != self.str_nan
            ]

            # converting unknown values to NaN
            if len(unknown_values) > 0:
                # raising an error
                if self.unknown_handling == "raise":
                    assert not len(unknown_values) > 0, (
                        f" - [ChainedDiscretizer] Order for feature '{feature}' needs to be provid"
                        f"ed for values: {str(unknown_values)}, otherwise set remove_unknown='drop"
                        "' (policy unknown_handling='raise')"
                    )
                # dropping unknown value
                else:  # unknown_handling='drop'
                    # alerting user
                    print(
                        f" - [ChainedDiscretizer] Order for feature '{feature}' was not provided "
                        f"for values:  {str(unknown_values)}, these values will be converted to '"
                        f"{self.str_nan}' (policy unknown_handling='drop')"
                    )

                    # adding unknown to the order
                    for unknown_value in unknown_values:
                        order.append(unknown_value)
                        order.append(self.str_nan)
                        # grouping unknown value with str_nan
                        order.group(unknown_value, self.str_nan)

            # updating values_orders accordingly
            self.values_orders.update({feature: order})

        # adding up NAN for all features of values_orders for seamless integration
        # when GroupedList._tranform_qualitative is called nans are replaced by str_nan
        for feature, order in self.values_orders.items():
            # adding NaNs to the order if any
            if any(x_copy[feature].isna()) or any(x_copy[feature] == self.str_nan):
                if self.str_nan not in order:
                    order.append(self.str_nan)
                    self.values_orders.update({feature: order})

        # checking that all unique values in X are in values_orders
        self._check_new_values(x_copy, features=self.features)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        # filling nans
        x_copy = self._prepare_data(X, y)

        if self.verbose:  # verbose if requested
            print(f" - [ChainedDiscretizer] Fit {str(self.features)}")

        # iterating over each feature
        for feature in self.features:
            # computing frequencies of each modality
            frequencies = x_copy[feature].value_counts(normalize=True)

            # iterating over each specified orders
            for level_order in self.chained_orders:
                values, frequencies = frequencies.index, frequencies.values

                # values that are frequent enough
                to_keep = list(values[frequencies >= self.min_freq]) + [
                    self.str_nan,
                ]

                # values from the order to group (not frequent enough or absent)
                values_to_group = [value for value in level_order.values() if value not in to_keep]

                # values to group into discarded values
                groups_value = [level_order.get_group(value) for value in values_to_group]

                # values of the feature to input (needed for next levels of the order)
                df_to_input = [x_copy[feature] == discarded for discarded in values_to_group]

                # inputing non frequent values
                x_copy[feature] = select(df_to_input, groups_value, default=x_copy[feature])

                # historizing in the feature's order
                for discarded, kept in zip(values_to_group, groups_value):
                    self.values_orders.get(feature).group(discarded, kept)

                # updating frequencies of each modality for the next ordering
                frequencies = x_copy[feature].value_counts(normalize=True)

        super().fit(X, y)

        return self


def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    order: GroupedList,
) -> dict[str, Union[str, float]]:
    """finds common modalities of a ordinal feature"""
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
