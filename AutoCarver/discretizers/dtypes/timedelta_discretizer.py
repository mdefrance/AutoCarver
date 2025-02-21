"""Base tools to convert datetime into timedelta."""

from pandas import DataFrame, Series

from ...features import Features, get_versions
from ...features.quantitatives import DatetimeFeature, DatetimeUnit
from ...utils import extend_docstring
from ..utils import BaseDiscretizer, Sample, imap_unordered_function


class TimedeltaDiscretizer(BaseDiscretizer):
    """Converts specified columns of a DataFrame into float timedeltas.

    * Keeps NaN inplace
    """

    __name__ = "TimedeltaDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, exclude=["min_freq"])
    def __init__(self, features: Features, **kwargs) -> None:
        # initiating features
        features = Features(features, **kwargs)

        # Initiating BaseDiscretizer
        super().__init__(features=features, **kwargs)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:
        self._log_if_verbose()  # verbose if requested

        # checking for binary target and copying X
        sample = self._prepare_data(Sample(X, y))

        # getting needed columns
        timedelta_features = get_versions(self.features.datetimes)

        # adding reference to columns
        timedelta_features += [feature.reference for feature in self.features.datetimes]

        # deduplicating columns
        timedelta_features = list(set(timedelta_features))

        # fitting each feature
        imap_unordered_function(
            fit_feature,
            self.features.datetimes,
            self.n_jobs,
            X=sample.X[timedelta_features],
        )

        return self


def fit_feature(feature: DatetimeFeature, X: Series) -> None:
    """
    Determine the optimal timedelta unit for a pandas Series of timedeltas.
    To be used during preprocessing of the data.
    """
    # converting to timedelta
    td_series = feature.convert_to_timedelta(X)

    # setting unit
    feature.unit = get_optimal_unit(td_series)

    # fitting feature
    feature.fit(DataFrame({feature.version: td_series}))
    print(
        f"{feature.version} has been fitted with unit {feature.unit}, {feature.has_nan}, {td_series.isnull().sum()}"
    )


def get_optimal_unit(td_series: Series) -> DatetimeUnit:
    """returns the optimal unit for timedelta"""

    # setting unit
    unit = DatetimeUnit.YEARS.value
    if td_series.max() < 60:
        unit = DatetimeUnit.SECONDS.value
    elif td_series.max() < 3600:
        unit = DatetimeUnit.MINUTES.value
    elif td_series.max() < 86400:
        unit = DatetimeUnit.HOURS.value
    elif td_series.max() < 604800:
        unit = DatetimeUnit.DAYS.value
    elif td_series.max() < 2628000:
        unit = DatetimeUnit.WEEKS.value
    elif td_series.max() < 31536000:
        unit = DatetimeUnit.MONTHS.value
    return unit
