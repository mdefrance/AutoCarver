Statistics
==========

The statistical primitives shared by :doc:`combinations`, :doc:`selectors` and
:doc:`stability`. Every carver, selector and monitoring metric routes its
arithmetic through this module, so a formula stated here is the one that runs.

.. _stats_chi2:

Pearson's :math:`\chi^2`
------------------------

.. autofunction:: AutoCarver.stats.pearson_chi2

.. _stats_cramerv_tschuprowt:

Cramér's :math:`V` and Tschuprow's :math:`T`
--------------------------------------------

Every Cramér's :math:`V` and Tschuprow's :math:`T` in AutoCarver (the binary and
multiclass combination evaluators, the qualitative selector measures and the
redundancy filters) is **bias-corrected** following Bergsma (2013), and is computed
from the **uncorrected** Pearson :math:`\chi^2` (no Yates continuity correction).

.. _stats_bias_correction:

Why a bias correction
^^^^^^^^^^^^^^^^^^^^^

The plain estimates :math:`V = \sqrt{\chi^2 / (N(\min(r, c) - 1))}` and
:math:`T = \sqrt{\chi^2 / (N\sqrt{(r-1)(c-1)})}` are biased **upward** on finite
samples. Under independence :math:`\chi^2` averages its degrees of freedom
:math:`(r-1)(c-1)`, so pure noise scores :math:`\phi^2 = \chi^2/N \approx (r-1)(c-1)/N`,
more with more modalities and fewer rows. AutoCarver compares associations in two
places where that bias decides the outcome:

* **Across group counts, within one feature.** The carving search returns the best
  grouping for every :math:`k \le` ``max_n_mod`` and keeps the one with the highest
  association. Each extra group adds chance association, so the plain estimate
  favours splits that carry nothing.
* **Across features, in the selectors.** A feature observed on 1,000 rows is
  compared with one observed on 100,000. On few rows the plain estimate of a
  feature with no signal can outrank a real, weak signal measured on many rows.

The correction subtracts the chance level and shrinks the table dimensions to match:

.. math::

    \tilde\phi^2_+ = \max\left(0,\ \frac{\chi^2}{N} - \frac{(r-1)(c-1)}{N-1}\right),
    \qquad \tilde r = r - \frac{(r-1)^2}{N-1}, \qquad \tilde c = c - \frac{(c-1)^2}{N-1},

.. math::

    \tilde V = \sqrt{\frac{\tilde\phi^2_+}{\min(\tilde r, \tilde c) - 1}},
    \qquad
    \tilde T = \sqrt{\frac{\tilde\phi^2_+}{\sqrt{(\tilde r - 1)(\tilde c - 1)}}}.

Both are clipped to :math:`[0, 1]`. The correction vanishes as :math:`N` grows, so
large datasets are essentially unaffected. For a fixed table size and :math:`N` it is
monotone in :math:`\chi^2`, so the chi²-maximising DP search (:ref:`DPChi2`) is
unchanged; only candidates with different group counts, and features with different
sample sizes, are re-weighed.

**Measured effect** (synthetic study, one feature, true structure = 2 groups, best
grouping per :math:`k \le 5`): at :math:`N = 200` Tschuprow's :math:`T` picks the
true 2 groups in 75% of runs uncorrected and 84% corrected (61% with Yates); from
:math:`N = 1{,}000` all pick it every time. On pure noise the corrected :math:`T`
reports the least spurious association (0.097 vs 0.134 at :math:`N = 200`). A
5-modality feature with no signal observed on 1,000 rows scores :math:`V = 0.059`
uncorrected, above a real weak signal on 100,000 rows (:math:`V = 0.039`), and
0.018 corrected.

**Limits.**

* A table with about one row per modality carries no evidence: :math:`\tilde r` or
  :math:`\tilde c` drops to 1 and :math:`\tilde V`, :math:`\tilde T` are ``NaN``.
* The correction removes the chance level of **one** table. It does not add a
  penalty per group, so Cramér's :math:`V` on a binary target (denominator
  :math:`\min(\tilde r, \tilde c) - 1 = \tilde c - 1`, the same for every :math:`k`)
  still tends to the largest :math:`k`. Tschuprow's :math:`T`, the binary default,
  carries that penalty through its :math:`\sqrt{\tilde r - 1}` denominator.
* No unbiased estimator of :math:`\phi^2` exists at any sample size (Dwyer, 2026).
  Bergsma's correction is the lowest-error closed form for single tables with few
  degrees of freedom, which is the carving case (:math:`k \le` ``max_n_mod`` groups
  against a binary or few-class target).

**Why not Yates.** Yates' continuity correction (subtract 0.5 from every
:math:`|O - E|`) fixes the **p-value** of a :math:`2 \times 2` independence test,
where the continuous :math:`\chi^2` distribution approximates the discrete statistic
poorly. AutoCarver reads no p-value when carving or selecting. Applied to an effect
size, Yates lowers only the 2-group candidate, which tilts the comparison across
:math:`k` toward more groups (61% vs 75% above) and does nothing for tables larger
than :math:`2 \times 2`. It is kept only where a p-value is read: the drift test of
:doc:`stability`.

**References.**

* W. Bergsma, "A bias-correction for Cramér's V and Tschuprow's T", *Journal of the
  Korean Statistical Society* 42 (2013), 323-328.
  `doi:10.1016/j.jkss.2012.10.002 <https://doi.org/10.1016/j.jkss.2012.10.002>`_
* Dwyer, "An Honest Effect Size for Contingency Tables: Why Nothing Can Be Unbiased,
  Where to Put the Error Instead, and How to Route the Report", arXiv:2609.26211
  (2026, preprint). `arxiv.org/abs/2609.26211 <https://arxiv.org/abs/2609.26211>`_

.. autofunction:: AutoCarver.stats.bergsma_phi2

.. autofunction:: AutoCarver.stats.cramerv_tschuprowt

.. autofunction:: AutoCarver.stats.cramerv_tschuprowt_unrounded

.. _stats_kruskal:

Kruskal-Wallis' :math:`H`
-------------------------

.. autofunction:: AutoCarver.stats.tie_correction

.. autofunction:: AutoCarver.stats.h_from_rank_sums

.. _stats_frequency_ci:

Wilson frequency confidence bound
----------------------------------

.. autofunction:: AutoCarver.stats.wilson_upper_bound

.. autofunction:: AutoCarver.stats.is_significantly_below

.. _stats_ridits:

Ridit scores
------------

.. autofunction:: AutoCarver.stats.ridits_from_counts

.. autofunction:: AutoCarver.stats.ridit_scores_for_levels

.. _stats_rank_association:

Rank association of an ordered table
------------------------------------

For an ordered contingency table :math:`(r \times c)` — :math:`r` feature groups
(rows) × :math:`c` ordinal target levels (cols), both ascending — three
rank-association statistics are built from the same pair counts:

* :math:`C` — **concordant** pairs (both members order the same way on the
  feature and on the target);
* :math:`D` — **discordant** pairs (members order oppositely);
* :math:`P_0 = n(n-1)/2` — all pairs, with :math:`n` the number of observations;
* :math:`T_X`, :math:`T_Y` — pairs **tied** on the feature / on the target
  (equal row / equal column); :math:`P_0 - T_X` and :math:`P_0 - T_Y` are the
  pairs untied on each margin;
* :math:`m = \min(r', c')` — the smaller of the number of **non-empty** grouped
  rows :math:`r'` and target levels :math:`c'`.

The three measures are monotone-comparable transforms of :math:`C - D`. Each is
``None`` for a degenerate table (its denominator vanishes), mirroring the
continuous evaluator's ``None`` convention. Parity against
:func:`scipy.stats.kendalltau` (tau-b) and :func:`scipy.stats.somersd` is pinned
by ``tests/combinations/ordinal/test_ordinal_associations.py`` and the property
suite ``tests/properties/combinations/test_ordinal_combinations_properties.py``.

.. autofunction:: AutoCarver.stats.concordant_minus_discordant

.. autofunction:: AutoCarver.stats.rank_associations

.. autofunction:: AutoCarver.stats.rank_associations_from_counts

.. _tau_c:

Kendall/Stuart's :math:`\tau_c` (ordinal default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stuart's tau-c applies a :math:`\min(r, c)` correction tailored to
**rectangular** tables — exactly our shape (few feature groups × many target
levels):

.. math::

    \tau_c = \frac{2 \, m \, (C - D)}{n^2 \, (m - 1)}.

Because the denominator depends only on :math:`(n, m)` and not on how
observations distribute across groups, its magnitude stays comparable across
combinations with different group counts. It self-balances toward fewer, robust
modalities, only adding one when a split is genuinely discriminative — like
:ref:`Tschuprow's T <Tschuprowt>` and the Kruskal effect sizes. This is the
default for :class:`OrdinalCarver`.

.. _tau_b:

Kendall's :math:`\tau_b`
^^^^^^^^^^^^^^^^^^^^^^^^

Kendall's tau-b normalises :math:`C - D` by the geometric mean of the two
margins' untied pairs:

.. math::

    \tau_b = \frac{C - D}{\sqrt{(P_0 - T_X)(P_0 - T_Y)}}.

It is bit-exact with the ``tau-b`` variant of :func:`scipy.stats.kendalltau` on
the grouped table and tends to retain more modalities on smoothly monotone
signals than :math:`\tau_c`.

.. _somersd:

Somers' D
^^^^^^^^^

The original asymmetric Somers' D ``D(Y|X)`` — concordant minus discordant pairs
over pairs untied on the feature :math:`X`:

.. math::

    D(Y \mid X) = \frac{C - D}{P_0 - T_X}.

It matches ``scipy.stats.somersd(table).statistic``. Being asymmetric it leans
strongly toward the **coarsest** split (its maximum over groupings is typically
two modalities); offered for users who specifically want raw Somers' D rather
than the self-balancing Kendall taus.

.. _stats_correspondence_analysis:

Correspondence analysis
------------------------

.. autofunction:: AutoCarver.stats.fit_ca_axis

.. autofunction:: AutoCarver.stats.ca_row_scores

.. autoclass:: AutoCarver.stats.CAAxis
    :members:
