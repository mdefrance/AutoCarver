.. _Stability:


Stability monitoring
====================

:ref:`Viability testing <Viability>` is a **fit-time** guardrail: it rejects
groupings that don't hold up on a dev sample or CV folds. Once carving is done,
the same question comes back in production — *do these carved features still
hold on data the model has never seen?*

:func:`~AutoCarver.stability.evaluate_stability` answers it without any training
data: each carved feature already stores its own train reference (see
:ref:`ReferenceStatistics`), so a fitted — or reloaded — carver can score a new
sample on its own.

.. code-block:: python

    from AutoCarver import BinaryCarver

    carver = BinaryCarver.load("carver.json")

    report = carver.evaluate_stability(X_prod, y_prod)
    report.summary           # one row per feature
    report.per_modality      # one row per (feature, modality)
    report.unstable_features # features flagged by any check

The target is optional. Labels usually lag in production, so
``carver.evaluate_stability(X_prod)`` computes the population-side metrics
alone (PSI and the chi-square goodness-of-fit); the target-side ones are
reported as ``NaN``.

.. autofunction:: AutoCarver.stability.evaluate_stability

.. autoclass:: AutoCarver.stability.StabilityReport
   :members: summary, unstable_features, to_json


.. _ReferenceStatistics:

The reference
-------------

At carving time the winning combination's per-modality statistics are stored on
the feature itself, as :attr:`~AutoCarver.features.BaseFeature.statistics` — a
frame indexed by final label carrying:

* ``count`` and ``frequency`` — the population reference;
* the evaluator's **target rate** (``target_mean``, ``woe``, ``odds_ratio``,
  ``target_median``, ``target_mean_ridit``, ``target_mean_level`` or
  ``ca_score``) — the target reference;
* ``std`` for continuous targets — the dispersion needed to test a mean shift.

These survive :meth:`~AutoCarver.carvers.utils.base_carver.BaseCarver.save` /
:meth:`~AutoCarver.carvers.utils.base_carver.BaseCarver.load`, together with any
per-feature state the target rate was fit on (the ridit reference marginal, the
correspondence-analysis axis). Features that were never carved — discretized
only — carry no statistics and are skipped with a warning.

Production statistics are recomputed with the carver's **own** aggregator and
target rate, so both sides of every comparison are like-for-like by
construction. Comparing a carver against the very sample it was fitted on
therefore returns a PSI of exactly zero and no drift anywhere — a useful sanity
check.


Population drift
----------------

Two complementary readings of the same shift in modality frequencies.

**PSI** (Population Stability Index) is the industry-standard magnitude:

.. math::

    \text{PSI} \;=\; \sum_{i} (f^{\text{new}}_i - f^{\text{ref}}_i)
                     \, \log \frac{f^{\text{new}}_i}{f^{\text{ref}}_i}

reported per feature (``psi``) and per modality (``psi_contribution``), with the
conventional verdict in ``psi_flag``: ``stable`` below 0.1, ``moderate`` up to
0.25, ``shifted`` above. Both frequencies are floored so a modality that emptied
out contributes a large but finite amount instead of infinity.

A fourth verdict, ``unknown``, means the reference itself was incomplete — a
manual :meth:`~AutoCarver.features.QuantitativeFeature.split` leaves the
affected bins' statistics unknowable (``NaN``), and dropping them would
silently renormalize the comparison onto a different support. The index is
then ``NaN`` rather than a plausible-looking number, and the feature is
reported as needing attention: unverifiable must never read as verified-stable.

**Chi-square homogeneity** is the significance-based counterpart: a two-sample
test on the ``2 x k`` table of reference and production counts (``chi2``,
``chi2_pvalue``, ``chi2_significant``). It is deliberately *not* a
goodness-of-fit against the reference frequencies — the reference is itself an
estimate from a finite train sample, and treating it as known truth would
understate the p-value.

Because any chi-square grows with sample size, **Cramér's V** is reported
beside it (``chi2_cramerv``) as the sample-size-independent effect size:
``V = sqrt(chi2 / N)`` for a ``2 x k`` table, conventionally negligible below
0.1. :attr:`~AutoCarver.stability.StabilityReport.unstable_features` requires
*both* significance and a non-negligible V, so a 300k-row extract doesn't flag
every feature over a shift too small to act on.

.. autofunction:: AutoCarver.stability.population_stability_index

.. autofunction:: AutoCarver.stability.chi2_homogeneity


Target drift
------------

Every modality's rate delta is reported (``rate_delta``). Whether it comes with
a significance test depends on the target:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Carver
     - Test
     - p-value
   * - Binary, one-vs-rest (``target_mean``, ``woe``, ``odds_ratio``)
     - Pooled two-proportion z-test
     - ``drift_pvalue``
   * - Continuous ``target_mean``
     - Welch t-test (uses the stored ``std``)
     - ``drift_pvalue``
   * - Continuous ``target_median``
     - none
     - ``NaN``
   * - Ordinal, multiclass
     - none
     - ``NaN``

Only two rates admit a test from the stored statistics. ``target_median`` does
not: the stored ``std`` describes the spread of *values*, so feeding it to a
standard-error-of-the-mean formula would test the wrong quantity. Ordinal and
multiclass rates are bounded ridit / correspondence-analysis scores whose
sampling variance cannot be recovered from the three stored columns either. In
every such case the rate delta is still reported and the viability block below
still runs — only the p-value is withheld.

A multiclass target carrying a class **unseen at fit time** raises: the
correspondence-analysis axis is fixed at carving time and cannot project a
class it never saw.

.. autofunction:: AutoCarver.stability.two_proportion_test

.. autofunction:: AutoCarver.stability.welch_test


Re-running the viability filter
-------------------------------

The most direct question is also the cheapest: *would this combination still
have been accepted, had production been the dev sample?*

:func:`~AutoCarver.combinations.utils.testing.test_viability` is called with the
stored train rates as the reference and the production rates as the candidate,
so the report's ``viable`` / ``info`` columns are produced by exactly the
machinery described in :ref:`Viability` — rank inversion, Wilson
``min_freq``, distinct target rates — with the same human-readable failure
messages (*"Inversion of target rates per modality"*, *"Non-representative
modality for min_freq=…"*, *"Non-distinct target rates per consecutive
modalities"*).

A rank inversion here is the strongest possible signal: the carved ordering
that the whole model rests on no longer holds.


MCP
---

The same evaluation is exposed as an MCP tool (see :ref:`mcp`)::

    evaluate_stability(path="holdout.csv", target="y")

It returns the report's JSON form: ``unstable_features``, ``per_feature`` and
``per_modality``.
