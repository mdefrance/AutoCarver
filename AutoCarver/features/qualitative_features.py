""" Defines a categorical feature"""

from pandas import DataFrame, Series, notna, unique

from .utils.base_feature import BaseFeature
from .utils.grouped_list import GroupedList


class CategoricalFeature(BaseFeature):
    __name__ = "Categorical"

    def __init__(self, name: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_categorical = True
        self.is_qualitative = True

        # TODO adding stats
        self.n_unique = None
        self.mode = None
        self.pct_mode = None

    def __repr__(self):
        return f"{self.__name__}('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """TODO fit stats"""

        # checking that feature is not ordinal (already set values)
        if self.values is None:
            # checking for feature's unique non-nan values
            unique_values = nan_unique(X[self.name])

            # initiating feature with its unique non-nan values
            self.update(GroupedList(unique_values))

        super().fit(X, y)

        # checking for unexpected values
        self.check_values(X)

    def check_values(self, X: DataFrame) -> None:
        """checks for unexpected values from unique values in DataFrame"""

        # computing unique values if not provided
        unique_values = unique(X[self.name])

        # unexpected values for this feature
        unexpected = [
            value for value in unique_values if not self.values.contains(value) and notna(value)
        ]
        if len(unexpected) > 0:
            # feature does not have a default value
            if not self.has_default:
                raise ValueError(
                    f" - [Features] Unexpected values for "
                    f"{self.__name__}('{self.name}'). Unexpected values: {str(list(unexpected))}."
                )

            # feature has default value: adding unexpected to default
            default_group = self.values.get_group(self.default)
            self.group_list(unexpected, default_group)

        super().check_values(X)

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
    ) -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(values, convert_labels, sorted_values, replace)

        # for qualitative feature -> by default, labels are values
        super().update_labels()
        # TODO make better labels


class OrdinalFeature(CategoricalFeature):
    __name__ = "Ordinal"

    def __init__(self, name: str, values: list[str], **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_ordinal = True
        self.is_categorical = False

        # checking for values
        if values is None or len(values) == 0:
            raise ValueError(
                " - [Features] Please make sure to provide values for "
                f"{self.__name__}('{self.name}')"
            )

        # setting values and labels
        super().update(GroupedList(values))


def nan_unique(x: Series) -> list[str]:
    """Unique non-NaN values.

    Parameters
    ----------
    x : Series
        Values to be deduplicated.

    Returns
    -------
    list[str]
        List of unique non-nan values
    """

    # unique values
    uniques = unique(x)

    # filtering out nans
    uniques = [value for value in uniques if notna(value)]

    return uniques
