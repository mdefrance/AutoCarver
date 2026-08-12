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
