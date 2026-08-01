"""Distributional and target-drift metrics comparing a new sample to a train reference."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, t


def population_stability_index(
    ref_freq: pd.Series, new_freq: pd.Series, *, epsilon: float = 1e-6
) -> tuple[float, pd.Series]:
    """Population Stability Index and its per-modality contributions.

    Both frequencies are floored at ``epsilon`` and renormalized so a modality
    that emptied out on either side yields a large-but-finite contribution
    instead of ``inf``. Conventional reading: below 0.1 stable, 0.1 to 0.25
    moderate shift, above 0.25 significant shift.

    A reference carrying any ``NaN`` bin (a manual split leaves the affected
    bins' statistics unknowable) makes the index undefined: dropping those bins
    would silently renormalize the comparison onto a different support. Both
    the total and every contribution are then ``NaN``.

    Parameters
    ----------
    ref_freq : pd.Series
        Reference (train) frequency per modality — ``feature.statistics["frequency"]``.
    new_freq : pd.Series
        Frequency per modality observed on the new sample.
    epsilon : float, optional
        Floor applied to both frequencies, by default ``1e-6``.

    Returns
    -------
    tuple[float, pd.Series]
        The PSI, and its per-modality contributions (indexed like ``ref_freq``).
    """
    reference = ref_freq.astype(float)
    new = new_freq.reindex(reference.index).fillna(0.0).astype(float)

    # an incomplete reference makes the whole index undefined, not merely partial
    if reference.isna().any():
        return float("nan"), pd.Series(float("nan"), index=reference.index)

    reference = reference.clip(lower=epsilon)
    new = new.clip(lower=epsilon)
    reference = reference / reference.sum()
    new = new / new.sum()

    contributions = (new - reference) * np.log(new / reference)
    return float(contributions.sum()), contributions


def chi2_homogeneity(ref_count: pd.Series, new_count: pd.Series) -> tuple[float, float, int, float]:
    """Chi-square test of homogeneity between the reference and the new sample.

    A **two-sample** test on the ``2 x k`` table of per-modality counts, not a
    goodness-of-fit against fixed frequencies: the reference is itself an
    estimate from a finite train sample, and treating its frequencies as known
    truth would understate the p-value. Expected counts come from the table's
    own margins, so no modality can be compared against a mis-scaled
    expectation. Modalities empty in *both* samples carry no information and
    are dropped.

    The statistic grows with sample size, so a large production extract will
    flag shifts that are real but negligible. Cramér's V is returned alongside
    it as the sample-size-independent effect size — for a ``2 x k`` table
    ``V = sqrt(chi2 / N)``, bounded in ``[0, 1]``, conventionally read as
    negligible below 0.1, small to 0.3, moderate to 0.5, large above.

    Returns
    -------
    tuple[float, float, int, float]
        Statistic, two-sided p-value, degrees of freedom and Cramér's V. All
        ``nan`` / ``0`` when the table is degenerate (an incomplete reference,
        fewer than two informative modalities, or an empty sample).
    """
    reference = ref_count.astype(float)
    new = new_count.reindex(reference.index).fillna(0.0).astype(float)

    # an incomplete reference (manual split) leaves no table to test
    if reference.isna().any():
        return float("nan"), float("nan"), 0, float("nan")

    table = np.vstack([reference.to_numpy(), new.to_numpy()])
    table = table[:, table.sum(axis=0) > 0]  # modalities empty on both sides
    total = float(table.sum())
    if table.shape[1] < 2 or total <= 0 or (table.sum(axis=1) <= 0).any():
        return float("nan"), float("nan"), 0, float("nan")

    statistic, pvalue, dof, _ = chi2_contingency(table)
    return float(statistic), float(pvalue), int(dof), float(np.sqrt(statistic / total))


def two_proportion_test(
    ref_rate: pd.Series, ref_count: pd.Series, new_rate: pd.Series, new_count: pd.Series
) -> pd.Series:
    """Per-modality two-sided p-value for a change in a binary target rate.

    Pooled-proportion z-test. Both rates must already be probabilities — pass
    them through :func:`to_probability` first when the carver's target rate is
    ``woe`` or ``odds_ratio``.
    """
    p_ref = ref_rate.astype(float)
    p_new = new_rate.reindex(p_ref.index).astype(float)
    n_ref = ref_count.astype(float)
    n_new = new_count.reindex(p_ref.index).fillna(0.0).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = (p_ref * n_ref + p_new * n_new) / (n_ref + n_new)
        standard_error = np.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_new))
        z_score = (p_new - p_ref) / standard_error
        pvalues = 2 * norm.sf(np.abs(z_score))

    # a null standard error means no variation to test against (a constant or empty modality)
    pvalues = np.where(np.asarray(standard_error) > 0, pvalues, np.nan)
    return pd.Series(pvalues, index=p_ref.index)


def welch_test(
    ref_mean: pd.Series,
    ref_std: pd.Series,
    ref_count: pd.Series,
    new_mean: pd.Series,
    new_std: pd.Series,
    new_count: pd.Series,
) -> pd.Series:
    """Per-modality two-sided Welch p-value for a change in a continuous target mean.

    Returns ``nan`` wherever an input is ``nan`` — notably for carvers fitted
    before the ``std`` column was persisted, and for singleton modalities.
    """
    mean_ref = ref_mean.astype(float)
    index = mean_ref.index
    std_ref = ref_std.reindex(index).astype(float)
    n_ref = ref_count.reindex(index).astype(float)
    mean_new = new_mean.reindex(index).astype(float)
    std_new = new_std.reindex(index).astype(float)
    n_new = new_count.reindex(index).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        var_ref = std_ref**2 / n_ref
        var_new = std_new**2 / n_new
        standard_error = np.sqrt(var_ref + var_new)
        t_score = (mean_new - mean_ref) / standard_error
        # Welch-Satterthwaite degrees of freedom
        dof = (var_ref + var_new) ** 2 / (var_ref**2 / (n_ref - 1) + var_new**2 / (n_new - 1))
        pvalues = 2 * t.sf(np.abs(t_score), dof)

    pvalues = np.where(np.asarray(standard_error) > 0, pvalues, np.nan)
    return pd.Series(pvalues, index=index)


def to_probability(rate_name: str, values: pd.Series) -> pd.Series:
    """Inverts a binary target rate back to ``P(y=1 | modality)``.

    ``target_mean`` already is that probability; ``odds_ratio`` is ``p/(1-p)``;
    ``woe`` is ``log(P(y=1|mod) / P(y=0|mod))`` (see
    :meth:`~AutoCarver.combinations.binary.binary_target_rates.Woe._compute`),
    i.e. a logit — so ``p = sigmoid(woe)``.

    Raises
    ------
    ValueError
        When ``rate_name`` is not an invertible binary target rate.
    """
    if rate_name == "target_mean":
        return values.astype(float)
    if rate_name == "odds_ratio":
        odds = values.astype(float)
        return odds / (1 + odds)
    if rate_name == "woe":
        return 1 / (1 + np.exp(-values.astype(float)))
    raise ValueError(f"[to_probability] {rate_name!r} is not an invertible binary target rate")
