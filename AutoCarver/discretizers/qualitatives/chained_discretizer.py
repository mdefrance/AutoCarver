"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from numpy import nan, select
from pandas import DataFrame, Series, unique

from ...features import BaseFeature, Features, GroupedList
from ...utils import extend_docstring
from ..utils.base_discretizer import BaseDiscretizer, Sample
from ..utils.type_discretizers import StringDiscretizer


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
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------

        chained_orders : list[GroupedList]
            A list of interlocked higher level groups for each modalities of each ordinal feature.
            Values of ``chained_orders[0]`` have to be grouped in ``chained_order[1]`` etc.
        """
        # not dropping nans whatsoever
        kwargs = dict(kwargs, dropna=False)
        super().__init__(features, **kwargs)  # Initiating BaseDiscretizer

        # class specific attributes
        self.min_freq = min_freq
        self.chained_orders = [GroupedList(values) for values in chained_orders]

        # known_values: all ordered values describe in each level of the chained_orders
        # starting off with first level
        known_values = self.chained_orders[0].values
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
                        f"[{self.__name__}] Values {str(next_unknown)}, provided in "
                        f"chained_orders[{n+1}] are missing from chained_orders[{n}]. Please make "
                        "sure values are kept trhough each level."
                    )

                # checking for known values
                next_known = [
                    value for value in next_values if value in known_values and value != next_group
                ]
                if len(next_known) == 0:
                    raise ValueError(
                        f"[{self.__name__}] For key '{next_group}', the provided chained_orders"
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
                        f"[{self.__name__}] Value {value} from feature {feature} provided in "
                        "values_orders is missing from levels of chained_orders. Add value to a "
                        "level of chained_orders or adapt values_orders."
                    )
            # adding known values if missing from the order
            for value in self.known_values:
                if value not in order.values:
                    order.append(value)
            # sorting in case an ordering was provided
            order = order.sort_by(self.known_values)
            # updating feature
            feature.update(order, replace=True)

    def _prepare_data(self, sample: Sample) -> Sample:
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
        sample.X = sample.X.copy()

        # checking for binary target and previous fit
        sample = super()._prepare_data(sample)

        # checking feature values' frequencies
        check_frequencies(self.features, sample.X, self.min_freq, self.__name__)

        # converting non-str columns
        sample.X = ensure_qualitative_dtypes(self.features, sample.X, **self.kwargs)

        # fitting features
        self.features.fit(**sample)

        # filling up nans for features that have some
        sample.X = self.features.fillna(sample.X)

        # checking for unexpected values
        self.features.check_values(sample.X)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        # preprocessing data
        sample = self._prepare_data(Sample(X, y))
        self._log_if_verbose()  # verbose if requested

        # iterating over each feature
        for feature in self.features:
            # iterating over each specified orders
            for level_order in self.chained_orders:
                # computing frequencies of each modality
                frequencies = (
                    sample.X[feature.version]
                    .value_counts(normalize=True, dropna=False)
                    .drop(nan, errors="ignore")
                )
                values, frequencies = frequencies.index, frequencies.values

                # values that are frequent enough
                to_keep = list(values[frequencies >= self.min_freq])

                # values from the order to group (not frequent enough or absent)
                values_to_group = [value for value in level_order.values if value not in to_keep]

                # values to group into discarded values
                groups_value = [level_order.get_group(value) for value in values_to_group]

                # values of the feature to input (needed for next levels of the order)
                df_to_input = [
                    sample.X[feature.version] == discarded for discarded in values_to_group
                ]

                # inputing non frequent values
                sample.X[feature.version] = select(
                    df_to_input, groups_value, default=sample.X[feature.version]
                )

                # historizing in the feature's order
                order = GroupedList(feature.values)
                for discarded, kept in zip(values_to_group, groups_value):
                    order.group(discarded, kept)

                # updating feature accordingly
                feature.update(order, replace=True)

        super().fit(X, y)

        if self.verbose:  # verbose if requested
            print("\n")

        return self


def check_frequencies(features: Features, X: DataFrame, min_freq: float, name: str) -> None:
    """Checks features' frequencies compared to min_freq"""

    # computing features' max modality frequency (mode frequency)
    max_freq = X[features.versions].apply(
        lambda u: u.value_counts(normalize=True, dropna=False).max(),
        axis=0,
    )

    # features with no common modality (biggest value less frequent than min_freq)
    non_common = [f.version for f in features if max_freq[f.version] < min_freq]

    # features with too common modality (biggest value more frequent than 1-min_freq)
    too_common = [f.version for f in features if max_freq[f.version] > 1 - min_freq]

    # raising
    if len(too_common + non_common) > 0:
        # building error message
        error_msg = (
            f"[{name}] Features {str(too_common + non_common)} contain a too frequent modality "
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


def ensure_qualitative_dtypes(features: Features, X: DataFrame, **kwargs) -> DataFrame:
    """Checks features' data types and converts int/float to str"""

    # getting per feature data types
    dtypes = (
        X.fillna({feature.version: feature.nan for feature in features})[features.versions]
        .map(type)
        .apply(unique, result_type="reduce")
    )

    # identifying features that are not str
    not_object = dtypes.apply(lambda u: any(dtype != str for dtype in u))

    # converting detected non-string features
    if any(not_object):
        # converting non-str features into qualitative features
        to_convert = [
            feature for feature in features if feature.version in not_object.index[not_object]
        ]
        string_discretizer = StringDiscretizer(features=to_convert, **kwargs)
        X = string_discretizer.fit_transform(X)

    return X
