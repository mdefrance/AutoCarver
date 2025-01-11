""" Filters based on association measures between Qualitative features.
"""

from pandas import DataFrame

from ...features import BaseFeature, get_versions
from ..measures import CramervMeasure, TschuprowtMeasure
from .base_filters import BaseFilter


class QualitativeFilter(BaseFilter):
    """Computes max association between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).
    """

    __name__ = "QualitativeFilter"

    is_x_qualitative = True

    def filter(self, X: DataFrame, ranks: list[BaseFeature]) -> list[BaseFeature]:
        # filtering ranks to avoid correlation with already removed features
        filtered_ranks = ranks[:]

        # iterating over each feature by target association order
        filtered: list[BaseFeature] = []
        for feature in ranks:
            # maximum correlation with a better feature
            correlation_with, worst_correlation = self._compute_worst_correlation(
                X, feature, filtered_ranks
            )

            # checking for too much correlation
            valid = self._validate(worst_correlation)

            # update feature accordingly (update stats)
            self._update_feature(
                feature, worst_correlation, valid, info={"correlation_with": correlation_with}
            )

            # keeping feature
            if valid:
                filtered += [feature]

            # removing feature from ranks
            else:
                filtered_ranks.remove(feature)

        return filtered

    def _compute_worst_correlation(
        self, X: DataFrame, feature: BaseFeature, rank: list[BaseFeature]
    ) -> tuple[str, float]:
        """Computes maximum association between a feature and features
        more associated to the target (according to ranks)
        """

        # initiating worst correlation
        correlation_with, worst_correlation = "itself", 0.0

        # features more associated with target
        current_feature_index = get_versions(rank).index(feature.version)
        better_features = rank[:current_feature_index]

        # iterating over each better feature
        for better_feature in better_features:
            # computing association with the better feature
            correlation = self.measure.compute_association(
                X[feature.version], X[better_feature.version]
            )

            # updating association if it's greater than previous better features
            if correlation > worst_correlation:
                worst_correlation, correlation_with = correlation, better_feature.version

            # breaking loop if too correlated
            if not self._validate(worst_correlation):
                break

        return correlation_with, worst_correlation

    def _validate(self, worst_correlation: float) -> bool:
        """Checks if the worst correlation of a feature is above specified threshold"""
        # dropping the feature if it was too correlated to a better feature
        valid = True
        if worst_correlation > self.threshold:
            valid = False

        return valid


class CramervFilter(QualitativeFilter):
    """Computes maximum Cramer's V between ``X`` and ``X`` (qualitative).
    Features too correlated to a feature more associated with the target
    are excluded (according to provided ``ranks``).

    Parameters
    ----------
    X : DataFrame
        Contains columns named after ``ranks``'s index (feature names)
    ranks : DataFrame
        Ranked features as index of the association table
    thresh_corr : float, optional
        Maximum Cramér's V bewteen features, by default ``1``

    Returns
    -------
    dict[str, Any]
        Maximum Cramér's V with a better feature
    """

    __name__ = "CramerV"

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)
        self.measure = CramervMeasure(threshold)


class TschuprowtFilter(QualitativeFilter):
    """Computes max Tschuprow's T between X and X (qualitative).
    Features too correlated to a feature more associated with the target
    are excluded (according to provided ``ranks``).

    Parameters
    ----------
    X : DataFrame
        Contains columns named after ``ranks``'s index (feature names)
    ranks : DataFrame
        Ranked features as index of the association table
    thresh_corr : float, optional
        Maximum Tschuprow's T bewteen features, by default ``1``

    Returns
    -------
    dict[str, Any]
        Maximum Tschuprow's T with a better feature
    """

    __name__ = "TschuprowT"

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)
        self.measure = TschuprowtMeasure(threshold)
