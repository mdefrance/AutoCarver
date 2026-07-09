"""Tool to build optimized buckets out of Quantitative and Qualitative features
for multiclass classification tasks — one carving per feature, against the
full K-class target (see :class:`~AutoCarver.carvers.one_vs_rest_carver.OneVsRestCarver`
for the one-vs-rest alternative: one carving per (class, feature) pair).
"""

import pandas as pd

from AutoCarver.carvers.binary_carver import get_crosstab
from AutoCarver.carvers.utils.base_carver import BaseCarver, Samples, parallel_aggregate
from AutoCarver.combinations import CombinationEvaluator, TschuprowtMulticlassCombinations
from AutoCarver.combinations.multiclass.multiclass_target_rates import MulticlassTargetRate
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.features import BaseFeature, Features
from AutoCarver.utils import extend_docstring


class MulticlassCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a multiclass (unordered,
    :math:`K > 2` classes) target — **one carving per feature**, against the
    full ``n x K`` crosstab.

    Sibling of :class:`~AutoCarver.carvers.ordinal_carver.OrdinalCarver` (both
    sit directly on :class:`BaseCarver` and aggregate a ``feature-groups x
    target-levels`` crosstab): the K target classes are unordered here, so
    qualitative modalities are ordered by their correspondence-analysis
    first-axis score (see :mod:`AutoCarver.discretizers.utils.correspondence_analysis`)
    instead of a numeric target-rate mean, and the association measure is a
    chi²-family statistic (Tschuprow's T or Cramér's V) generalised to a
    ``(B, K)`` table instead of Kendall's tau-c.

    A feature is carved **once**: unlike
    :class:`~AutoCarver.carvers.one_vs_rest_carver.OneVsRestCarver` (which
    fits ``K - 1`` separate :class:`~AutoCarver.carvers.binary_carver.BinaryCarver`
    instances — one per class, producing ``K - 1`` versions of every
    feature), this carver produces a single bucket set per feature and
    supports ``copy=False``.
    """

    __name__ = "MulticlassCarver"
    is_y_multiclass = True

    @extend_docstring(BaseCarver.__init__, exclude=["combination_evaluator"])
    def __init__(
        self,
        features: Features,
        min_freq: float,
        max_n_mod: int,
        *,
        combination_evaluator: CombinationEvaluator | None = None,
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        combination_evaluator : CombinationEvaluator, optional
            Pre-built evaluator instance measuring association between
            :class:`Features` and a multiclass target. Defaults to
            :class:`~AutoCarver.combinations.multiclass.multiclass_combination_evaluators.TschuprowtMulticlassCombinations`.

            Choose from:
            :class:`~AutoCarver.combinations.multiclass.multiclass_combination_evaluators.TschuprowtMulticlassCombinations`
            (default),
            :class:`~AutoCarver.combinations.multiclass.multiclass_combination_evaluators.CramervMulticlassCombinations`.
        """
        if combination_evaluator is None:
            combination_evaluator = TschuprowtMulticlassCombinations()
        if not combination_evaluator.is_y_multiclass:
            raise ValueError(
                f"[{self.__name__}] {type(combination_evaluator).__name__} is not suited for multiclass targets. "
                f"Choose from: TschuprowtMulticlassCombinations, CramervMulticlassCombinations."
            )

        super().__init__(
            features=features,
            min_freq=min_freq,
            max_n_mod=max_n_mod,
            combination_evaluator=combination_evaluator,
            config=config,
        )

    def _prepare_samples(self, samples: Samples) -> Samples:
        """Validates format and content of X and y."""
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided")
        # NaN check must precede astype(str): casting turns NaN into the string "nan",
        # which would silently become a target class (and bypass the base check)
        if samples.train.y.isna().any():
            raise ValueError(f"[{self.__name__}] y should not contain numpy.nan")
        samples.train.y = samples.train.y.astype(str)

        # multiclass target, checking values
        if len(pd.unique(samples.train.y)) <= 2:
            raise ValueError(f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead.")

        # checking for dev target's values
        if samples.dev.y is not None:
            if samples.dev.y.isna().any():
                raise ValueError(f"[{self.__name__}] y_dev should not contain numpy.nan")
            samples.dev.y = samples.dev.y.astype(str)

            unique_y_dev = samples.dev.y.unique()
            unique_y = samples.train.y.unique()
            missing_y = [mod_y for mod_y in unique_y if mod_y not in unique_y_dev]
            missing_y_dev = [mod_y_dev for mod_y_dev in unique_y_dev if mod_y_dev not in unique_y]
            if len(missing_y) > 0 or len(missing_y_dev) > 0:
                raise ValueError(
                    f"[{self.__name__}] Mismatched classes between y and y_dev"
                    f": train({missing_y_dev}), dev({missing_y})"
                )

        return super()._prepare_samples(samples)

    def _aggregator(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series | pd.DataFrame | None]:
        """Computes crosstabs (feature modalities x target classes) for specified
        features, with target classes in canonical (sorted) column order so
        that class-column order never depends on row order of the data.
        Threaded across features when ``n_jobs > 1``."""
        return parallel_aggregate(get_multiclass_crosstab, self.features, X, y, self.config.n_jobs)

    def _print_xagg(
        self,
        feature: BaseFeature,
        xagg: pd.Series | pd.DataFrame | None,
        message: str,
        *,
        xagg_dev: pd.Series | pd.DataFrame | None = None,
    ) -> None:
        """Fits the CA axis from the raw train crosstab before the pre-combination
        "Raw distribution" print.

        Without this, a verbose fit would call ``target_rate.compute`` (via
        :meth:`BaseCarver._pretty_print`) before :meth:`get_best_combination`
        has had a chance to fit the axis (see
        :meth:`~AutoCarver.combinations.multiclass.multiclass_target_rates.MulticlassTargetRate.fit_axis`).
        This fit only serves the print: ``get_best_combination`` refits the
        axis before scoring any candidate — from the same crosstab, minus the
        NaN row when the feature has NaNs (its scores can therefore differ
        slightly from the ones printed here) — so carving never depends on it.
        """
        target_rate = self.combination_evaluator.target_rate
        if message == "Raw distribution" and xagg is not None and isinstance(target_rate, MulticlassTargetRate):
            target_rate.fit_axis(xagg)  # type: ignore
        super()._print_xagg(feature, xagg, message, xagg_dev=xagg_dev)


def get_multiclass_crosstab(X: pd.DataFrame, y: pd.Series, feature: BaseFeature) -> pd.DataFrame:
    """Computes a crosstab between a feature and a multiclass target, with
    target classes in sorted (canonical) column order — ``pd.crosstab``
    already sorts by the classes' (string-cast) values, so this pins that
    order explicitly rather than relying on it implicitly."""
    xtab = get_crosstab(X, y, feature)
    return xtab[sorted(xtab.columns)]
