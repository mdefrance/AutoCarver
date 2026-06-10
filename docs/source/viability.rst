.. _Viability:


Viability testing
=================

The :ref:`combination search <Combinations>` ranks every candidate grouping by
its association with the target, but the highest-scoring grouping is not
necessarily the one that is *kept*. Before a combination is selected it must
pass the **viability filter** — a set of statistical guardrails that reject
groupings which are over-fit, degenerate, or driven by sampling noise. The
:ref:`top-K DP search <DPTopK>` walks candidates in metric-descending order and
the **first** one that is viable on both the train and dev samples wins.

The filter bundles three independent tests:

* :ref:`Minimum-frequency <MinFreqViability>` — every grouped modality is
  frequent enough (Wilson score interval).
* :ref:`Distinct target rates <DistinctRatesViability>` — consecutive modalities
  carry different target rates.
* :ref:`Train/dev rank preservation <RankViability>` — the modality ordering by
  target rate is stable between train and dev (robustness veto).

**Decision rule.** The min-frequency and distinct-rate tests run on **both**
samples; rank preservation runs on **dev** only (it compares dev against train).
A combination is viable iff it passes on train **and** — when a dev sample was
provided — passes on dev:

.. math::

    \text{viable} \;=\;
    \text{viable}_{\text{train}}
    \;\wedge\;
    \big(\text{viable}_{\text{dev}} \;\vee\; \text{no dev sample}\big).

When **no** candidate survives the filter the feature is dropped — see
:ref:`DroppedFeatures`. Each failing test contributes a human-readable reason
(*"Non-representative modality for min_freq=…"*, *"Non-distinct target rates per
consecutive modalities"*, *"Inversion of target rates per modality"*) that
surfaces in :attr:`carver.history` and ``dropped_reason``.

.. autofunction:: AutoCarver.combinations.utils.testing.test_viability

.. autofunction:: AutoCarver.combinations.utils.testing.is_viable


.. _MinFreqViability:

Minimum-frequency test (Wilson score interval)
----------------------------------------------

A candidate combination is *viable* on a sample only if every grouped modality is
sufficiently frequent. Comparing :math:`\hat p = \text{count} / n_{obs}` directly
against ``min_freq`` is noisy on small modalities — a modality with
:math:`\hat p = 4\%` out of :math:`n_{obs}=100` would be rejected against
``min_freq=5%``, even though its 95% confidence interval comfortably straddles
5%. **AutoCarver** instead tests the one-sided question *"is this modality's
true proportion significantly below* ``min_freq`` *?"* at level :math:`\alpha`,
using a Wilson score interval — the small-sample-stable proportion interval
recommended over Wald in Brown, Cai & DasGupta (2001).

**Decision rule.** Modality :math:`m` is declared under-represented iff the
**upper bound** of the two-sided Wilson interval for :math:`\hat p_m` is
strictly below ``min_freq``:

.. math::

    \text{UB}(\hat p, n, \alpha) =
    \frac{\hat p + z^2/(2n)}{1 + z^2/n}
    + \frac{z}{1 + z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}

with :math:`z = \Phi^{-1}(1 - \alpha/2)` (two-sided z-score; :math:`\alpha=0.05`
gives :math:`z \approx 1.96`). Reject iff
:math:`\text{UB}(\hat p_m, n_{obs}, \alpha) < \text{min_freq}`.

**Properties.**

* **Asymptotic equivalence:** as :math:`n_{obs} \to \infty`,
  :math:`\text{UB} \to \hat p`, so the test converges to the legacy strict
  threshold :math:`\hat p < \text{min_freq}`.
* **Small-sample conservativity:** a modality with very few observations cannot
  be rejected (the CI is too wide to fall below ``min_freq``), preventing
  premature merges driven by sampling noise.
* :math:`n_{obs} = 0` returns :math:`\text{UB} = 1.0`, so empty groups are never
  rejected by this test (other checks catch them).

**Where the test fires.**

* Inside each :ref:`Discretizer <Discretizer>` to gate raw modalities **before**
  the combination search. Carvers discretize at ``min_freq / 2`` so this gate
  runs at the halved threshold, giving the combination evaluator a finer
  granularity to recombine.
* Inside :class:`CombinationEvaluator` viability checks on both **train** and
  **dev** samples for every candidate combination during the search.

**Tuning.** Set via :attr:`DiscretizerConfig.min_freq_alpha` (default
``0.05``). Smaller :math:`\alpha` → wider CI → fewer rejections → less merging;
larger :math:`\alpha` → tighter CI → more rejections → more aggressive merging.
:math:`\alpha = 1` recovers the legacy strict-threshold behaviour
(:math:`\text{UB}` collapses to :math:`\hat p`).

.. autofunction:: AutoCarver.discretizers.utils.frequency_ci.wilson_upper_bound

.. autofunction:: AutoCarver.discretizers.utils.frequency_ci.is_significantly_below


.. _DistinctRatesViability:

Distinct-target-rate test
--------------------------

Modalities are searched as **consecutive** groupings of an ordered feature (by
ordinal rank, target rate, or numeric quantile — see :ref:`DPTopK`). Two
*adjacent* groups that end up with the same target rate are statistically
indistinguishable: the split between them carries no information and the two
groups should have been merged into one. A combination is rejected as soon as
**any** consecutive pair shares its target rate:

.. math::

    \text{distinct} \;=\;
    \neg \, \exists\, m \,:\, \tau_m \approx \tau_{m-1},

where :math:`\tau_m` is the target rate of modality :math:`m` and
:math:`\approx` is a floating-point closeness check (``numpy.isclose``). Keeping
the test on **consecutive** modalities (rather than all pairs) matches the
ordered nature of the search: non-adjacent groups are allowed to coincide, only
neighbours that would collapse are forbidden. Failing this test favours the
coarser, more parsimonious combination the search will reach next.


.. _RankViability:

Train/dev rank-preservation test (robustness veto)
--------------------------------------------------

When a dev sample is provided, a viable combination must be **robust**: the
modalities, ranked by their target rate, must keep the *same order* on train and
on dev. A combination whose target-rate ordering flips between the two samples is
over-fit to train and is vetoed:

.. math::

    \text{rank ok} \;=\;
    \big[\, \operatorname{argsort}_m \tau^{\text{train}}_m
          \;=\; \operatorname{argsort}_m \tau^{\text{dev}}_m \,\big].

Both target-rate series are aligned on the same modality index before sorting by
value, so the comparison is purely about *order*, not about the rates' absolute
magnitudes. This is the test that most often drives a feature into
:ref:`dropped_features <DroppedFeatures>`: when every train-viable combination
inverts on dev, it usually signals that **X_dev is too small or not
representative of X** for that feature. The three levers are enlarging the dev
sample, relaxing ``max_n_mod``, or dropping the feature.
