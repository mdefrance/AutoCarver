"""Base tools to convert values into specific types.
"""

from pandas import DataFrame, Series

from ...features import BaseFeature, Features, GroupedList
from ...features.qualitatives import nan_unique
from ...utils import extend_docstring
from .base_discretizer import BaseDiscretizer, Sample
from .multiprocessing import apply_async_function


class StringDiscretizer(BaseDiscretizer):
    """Converts specified columns of a DataFrame into strings.
    First step of a Qualitative discretization pipeline.

    * Keeps NaN inplace
    * Converts floats of int to int
    """

    __name__ = "StringDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, exclude=["min_freq"])
    def __init__(self, features: Features, **kwargs) -> None:
        # initiating features
        features = Features(features, **kwargs)

        # Initiating BaseDiscretizer
        super().__init__(features=features, **kwargs)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        self._log_if_verbose()  # verbose if requested

        # checking for binary target and copying X
        sample = self._prepare_data(Sample(X, y))

        # transforming all features
        all_orders = apply_async_function(fit_feature, self.features, self.n_jobs, sample.X)

        # updating features accordingly
        self.features.update(dict(all_orders), replace=True)
        self.features.fit(**sample)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


def fit_feature(feature: BaseFeature, df_feature: Series):
    """fits one feature"""

    # unique feature values
    unique_values = nan_unique(df_feature)

    # initiating feature values
    order = GroupedList(unique_values)
    if feature.values is not None:
        order = GroupedList(feature.values)

    # formatting values
    for value in unique_values:
        # case 0: the value is an integer
        if isinstance(value, float) and float.is_integer(value):
            str_value = str(int(value))  # converting value to string

        # case 1: converting to string
        else:
            str_value = str(value)

        # adding missing str_value to the order (for provided orders)
        if str_value not in order:
            order.append(str_value)

        # adding float/int values to the order (for provided orders)
        if value not in order:
            order.append(value)

        # grouping float/int value into the str value
        order.group(value, str_value)

    return feature.version, order


class TimedeltaDiscretizer(BaseDiscretizer):
    """TODO Converts specified columns of a DataFrame into float timedeltas.

    * Keeps NaN inplace
    """

    __name__ = "TimedeltaDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(self, features: list[BaseFeature], **kwargs) -> None:
        """
        Parameters
        ----------
        features : list[str]
            List of column names of qualitative features to be converted as string
        """
        # initiating features
        features = Features(features, **kwargs)

        # Initiating BaseDiscretizer
        super().__init__(features=features, **kwargs)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        self._log_if_verbose()  # verbose if requested

        # checking for binary target and copying X
        sample = self._prepare_data(Sample(X, y))

        # transforming all features
        all_orders = apply_async_function(fit_feature, self.features, self.n_jobs, sample.X)

        # updating features accordingly
        self.features.update(dict(all_orders), replace=True)
        self.features.fit(**sample)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self
