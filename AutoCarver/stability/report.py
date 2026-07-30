"""Post-fit stability evaluation of carved features against a new sample."""

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from warnings import warn

import pandas as pd

from AutoCarver.combinations.utils.combination_evaluator import CombinationEvaluator
from AutoCarver.combinations.utils.testing import test_viability
from AutoCarver.stability.metrics import (
    chi2_homogeneity,
    population_stability_index,
    to_probability,
    two_proportion_test,
    welch_test,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, importing at runtime would cycle
    from AutoCarver.carvers.utils.base_carver import BaseCarver

# PSI rule-of-thumb cutoffs
MODERATE_PSI = 0.1
SHIFTED_PSI = 0.25

# Cramér's V floor below which a significant chi-square is a negligible effect. The
# chi-square statistic grows with sample size, so on a large production extract
# significance alone flags shifts too small to act on; pairing it with an effect size
# keeps the verdict sample-size-neutral.
NEGLIGIBLE_CRAMERV = 0.1


@dataclass
class StabilityReport:
    """Comparison of carved features between their train reference and a new sample.

    Attributes
    ----------
    per_modality : pd.DataFrame
        One row per ``(feature, label)``: reference and new count / frequency /
        target rate, the modality's PSI contribution, the rate delta and — when
        a target-drift test applies — its p-value.
    per_feature : pd.DataFrame
        One row per feature: ``psi`` and its flag, the chi-square
        goodness-of-fit test, and (with a target) the viability verdict
        produced by the carver's own fit-time robustness tests.
    alpha : float
        Significance level used by every test.
    has_target : bool
        Whether a target was provided — without one, only the frequency-based
        metrics are computed.
    """

    per_modality: pd.DataFrame
    per_feature: pd.DataFrame
    alpha: float
    has_target: bool

    @property
    def summary(self) -> pd.DataFrame:
        """Per-feature verdicts, indexed by feature."""
        if self.per_feature.empty:
            return self.per_feature
        return self.per_feature.set_index("feature")

    @property
    def unstable_features(self) -> list[str]:
        """Features needing attention.

        Flagged when the population shifted (PSI above 0.25), when the
        chi-square test is **both** significant and carries a non-negligible
        effect size (``chi2_cramerv`` at or above 0.1 — significance alone
        would flag nearly everything on a large extract), when the carver's
        viability filter no longer passes, or when the reference is too
        incomplete to judge (``psi_flag == "unknown"``).
        """
        if self.per_feature.empty:
            return []
        shifted = self.per_feature["psi_flag"].eq("shifted") | self.per_feature["psi_flag"].eq("unknown")
        drifted = self.per_feature["chi2_significant"] & self.per_feature["chi2_cramerv"].ge(NEGLIGIBLE_CRAMERV)
        flagged = shifted | drifted
        if self.has_target:
            flagged |= ~self.per_feature["viable"].fillna(True).astype(bool)
        return self.per_feature.loc[flagged, "feature"].tolist()

    def to_json(self) -> dict[str, Any]:
        """Converts to a JSON-serializable dict (for ``json.dump`` or MCP transport).

        Frames round-trip through pandas' own JSON writer so numpy scalars become
        base types and ``NaN`` becomes ``null``.
        """
        return {
            "alpha": self.alpha,
            "has_target": self.has_target,
            "unstable_features": self.unstable_features,
            "per_feature": _records(self.per_feature),
            "per_modality": _records(self.per_modality),
        }

    def __repr__(self) -> str:
        return f"<StabilityReport: {len(self.unstable_features)}/{len(self.per_feature)} features unstable>"


def evaluate_stability(
    carver: "BaseCarver", X: pd.DataFrame, y: pd.Series | None = None, *, alpha: float = 0.05
) -> StabilityReport:
    """Evaluates carved features on a new sample against their train reference.

    The reference is :attr:`~AutoCarver.features.BaseFeature.statistics`, persisted
    on each feature at carving time, so no training data is needed at monitoring
    time. Production rates are recomputed with the carver's own aggregator and
    target rate, which makes them directly comparable, and are then run through
    :func:`~AutoCarver.combinations.utils.testing.test_viability` — the same
    rank-inversion / Wilson ``min_freq`` / distinct-rates suite the carver used
    to accept the combination at fit time.

    Parameters
    ----------
    carver : BaseCarver
        A fitted carver (in memory or reloaded from JSON).
    X : pd.DataFrame
        New (production) sample, with the carver's raw feature columns.
    y : pd.Series, optional
        Target for ``X``. Without it only PSI and the chi-square
        goodness-of-fit on counts are computed, by default ``None``.
    alpha : float, optional
        Significance level for every test, by default ``0.05``.

    Returns
    -------
    StabilityReport

    Notes
    -----
    Ordinal and multiclass carvers get PSI, the chi-square test and the full
    viability block, but **no per-modality target-drift test**: their rate is a
    ridit / correspondence-analysis score whose sampling variance is not
    recoverable from the three stored columns. The rate delta is still reported.
    """
    evaluator = carver.combination_evaluator
    target_rate = evaluator.target_rate
    rate_name = target_rate.__name__

    # carvers configured with copy=False transform in place; never mutate the caller's sample
    X_carved = carver.transform(X if carver.config.copy else X.copy())

    xaggs = _aggregate(carver, X_carved, y) if y is not None else {}

    modality_rows: list[dict] = []
    feature_rows: list[dict] = []

    for feature in carver.features:
        reference = feature.statistics
        if reference is None:
            warn(
                f"[evaluate_stability] {feature} carries no reference statistics (not carved), skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if y is not None:
            target_rate.load_reference(feature.rate_reference)
            new = target_rate.compute(xaggs[feature.version])
        else:
            new = _counts_only(X_carved[feature.version], feature.labels)

        psi, contributions = population_stability_index(reference["frequency"], new["frequency"])
        statistic, chi2_pvalue, _, cramerv = chi2_homogeneity(reference["count"], new["count"])

        viable, info = None, None
        pvalues = pd.Series(float("nan"), index=reference.index)
        rate_delta = pd.Series(float("nan"), index=reference.index)
        if y is not None:
            viability = test_viability(
                new,
                min_freq=carver.min_freq,
                target_rate=rate_name,
                alpha=alpha,
                train_target_rate=reference[rate_name],
            )["dev"]
            viable, info = viability["viable"], viability["info"]
            rate_delta = new[rate_name].reindex(reference.index) - reference[rate_name]
            pvalues = _drift_pvalues(evaluator, rate_name, reference, new)

        significant = pvalues < alpha
        for label in reference.index:
            modality_rows.append(
                {
                    "feature": str(feature),
                    "label": label,
                    "count_ref": reference.loc[label, "count"],
                    "frequency_ref": reference.loc[label, "frequency"],
                    f"{rate_name}_ref": reference.loc[label, rate_name],
                    "count_new": new["count"].get(label, 0),
                    "frequency_new": new["frequency"].get(label, float("nan")),
                    f"{rate_name}_new": new[rate_name].get(label, float("nan")) if y is not None else float("nan"),
                    "psi_contribution": contributions.loc[label],
                    "rate_delta": rate_delta.loc[label],
                    "drift_pvalue": pvalues.loc[label],
                    "drift_significant": bool(significant.loc[label]),
                }
            )

        feature_rows.append(
            {
                "feature": str(feature),
                "psi": psi,
                "psi_flag": _psi_flag(psi),
                "chi2": statistic,
                "chi2_pvalue": chi2_pvalue,
                "chi2_significant": bool(chi2_pvalue < alpha),
                "chi2_cramerv": cramerv,
                "viable": viable,
                "info": info,
                "n_modalities_drifted": int(significant.sum()),
                "n_obs_new": int(new["count"].sum()),
            }
        )

    return StabilityReport(
        per_modality=pd.DataFrame(modality_rows),
        per_feature=pd.DataFrame(feature_rows),
        alpha=alpha,
        has_target=y is not None,
    )


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe records: numpy scalars to base types, ``NaN`` to ``None``."""
    if frame.empty:
        return []
    # to_json only returns None when writing to a buffer, which this call never does
    return json.loads(frame.to_json(orient="records") or "[]")


def _aggregate(carver: "BaseCarver", X_carved: pd.DataFrame, y: pd.Series) -> dict:
    """Aggregates the carved sample against the target, per feature version.

    :class:`~AutoCarver.carvers.one_vs_rest_carver.OneVsRestCarver` carries one
    feature version per target class, each carved against its own one-vs-rest
    binarization — so its versions cannot share a single multiclass ``y`` the
    way every other carver's do.
    """
    from AutoCarver.carvers.binary_carver import get_crosstab
    from AutoCarver.carvers.one_vs_rest_carver import OneVsRestCarver, get_one_vs_rest

    if isinstance(carver, OneVsRestCarver):
        y_str = y.astype(str)
        return {
            feature.version: get_crosstab(X_carved, get_one_vs_rest(y_str, feature.version_tag), feature)
            for feature in carver.features
        }
    return carver._aggregator(X_carved, y)


def _counts_only(column: pd.Series, labels: list | None) -> pd.DataFrame:
    """Builds a ``count`` / ``frequency`` frame from a carved column (no target)."""
    count = column.value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({"frequency": count / count.sum(), "count": count})


def _psi_flag(psi: float) -> str:
    """Maps a PSI to its conventional verdict.

    ``NaN`` means the reference was too incomplete to compute an index — it is
    reported as ``"unknown"``, never as ``"stable"``: an unverifiable feature
    must not read the same as a verified-stable one.
    """
    if math.isnan(psi):
        return "unknown"
    if psi > SHIFTED_PSI:
        return "shifted"
    if psi > MODERATE_PSI:
        return "moderate"
    return "stable"


def _drift_pvalues(
    evaluator: CombinationEvaluator, rate_name: str, reference: pd.DataFrame, new: pd.DataFrame
) -> pd.Series:
    """Per-modality target-drift p-values, dispatched on the rate being compared.

    Only two rates admit a test from the stored statistics: a binary rate
    (invertible to a proportion, tested with a pooled z) and ``target_mean``
    (tested with Welch against the stored ``std``). Everything else gets
    ``nan``:

    * ``target_median`` — the stored ``std`` is a dispersion of *values*, so
      feeding it to a standard-error-of-the-mean formula would test the wrong
      quantity; a median needs its own test.
    * ordinal ``target_mean_ridit`` / ``target_mean_level`` and multiclass
      ``ca_score`` — bounded scores whose sampling variance the three stored
      columns don't carry.

    The rate delta is reported regardless, and the viability block still runs.
    """
    if evaluator.is_y_binary:
        return two_proportion_test(
            to_probability(rate_name, reference[rate_name]),
            reference["count"],
            to_probability(rate_name, new[rate_name]),
            new["count"],
        )
    if rate_name == "target_mean" and not (evaluator.is_y_multiclass or evaluator.is_y_ordinal):  # continuous mean
        if "std" not in reference.columns:
            warn(
                "[evaluate_stability] reference has no 'std' column (carver fitted before it was "
                "persisted): skipping the continuous target-drift test.",
                UserWarning,
                stacklevel=3,
            )
            return pd.Series(float("nan"), index=reference.index)
        return welch_test(
            reference[rate_name],
            reference["std"],
            reference["count"],
            new[rate_name],
            new["std"],
            new["count"],
        )
    return pd.Series(float("nan"), index=reference.index)
