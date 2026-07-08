.. _Carvers:

Carvers
=======

The core of **AutoCarver** resides in its **Carvers**, they provide the following Data Optimization steps:

   1. Identifying the most associated combination from all ordered combinations of modalities
   2. Testing all combinations of ``nan``s grouped to one of those modalities

Target-specific tools allow for association optimization per desired task:
 * :ref:`BinaryCarver`
 * :ref:`MulticlassCarver`
 * :ref:`OneVsRestCarver`
 * :ref:`ContinuousCarver`
 * :ref:`OrdinalCarver`

All carvers share the same constructor signature:

* ``features`` (:class:`Features`) — features to carve.
* ``min_freq`` (``float``) — minimum frequency per modality. Tested via the Wilson
  score interval at significance ``min_freq_alpha`` (see :ref:`MinFreqViability`).
* ``max_n_mod`` (``int``) — maximum number of modalities per carved
  feature; forwarded to the configured :class:`CombinationEvaluator`.
* ``combination_evaluator`` (:class:`CombinationEvaluator`, optional) —
  association metric. Defaults to a task-appropriate evaluator (see each subclass).
  The search uses :ref:`progressive top-K interval dynamic programming (DP) <DPTopK>`
  for both Kruskal-H (continuous) and Pearson :math:`\chi^2` (binary); statistically
  equivalent to the legacy enumerate-and-score path.
* ``config`` (:class:`ProcessingConfig`, optional) — behavioral toggles
  (``copy`` / ``ordinal_encoding`` / ``dropna`` / ``verbose`` / ``n_jobs`` /
  ``min_freq_alpha``).
  Defaults to ``ProcessingConfig(dropna=True, ordinal_encoding=True)``.


.. _CarverParallelism:

Per-feature parallelism (``n_jobs``)
------------------------------------

With ``ProcessingConfig(n_jobs=k)`` and ``k > 1``, :class:`BaseCarver` dispatches
one task per feature through ``multiprocessing.Pool.imap_unordered``. Each worker
receives a pickled deep copy of the :class:`CombinationEvaluator` and a single
``(feature, xagg, xagg_dev)`` payload; mutations stay local to the worker process
and the parent reattaches the (mutated) feature on completion. Verbose per-feature
logging is silenced — a single dispatch banner is printed when ``verbose=True``.

.. tip::

    Worth it only on **a few hundred features or more**. Below that, pool startup
    and pickle overhead dominate and the single-process path is faster. The
    :ref:`DP top-K search <DPTopK>` already removes the per-feature compute
    bottleneck, so most users will not need ``n_jobs > 1``.


.. _DroppedFeatures:

Dropped features (no robust combination)
----------------------------------------

A feature for which **no** candidate combination passed the
:ref:`viability filter <Viability>` (Wilson ``min_freq`` on train and dev,
distinct target rates, train/dev rank preservation) is removed from
``carver.features`` and retained on :attr:`carver.dropped_features` so the user
can inspect *why* it was dropped without re-fitting.

The :attr:`summary` and :attr:`history` properties append rows from dropped
features with two marker columns:

* ``dropped`` (``bool``) — ``True`` for rows from a dropped feature, ``False``
  otherwise.
* ``dropped_reason`` (``str`` | ``None``) — synthesized from the dominant
  failing-test message across the feature's historized combinations (e.g.
  *"Inversion of target rates per modality"*, *"Non-representative modality for
  min_freq=2.00%"*).

A dropped feature most commonly signals that **X_dev is too small or not
representative of X** for that feature: every candidate combination viable on
train flipped its target-rate ordering on dev. Increasing the dev sample size,
relaxing ``max_n_mod``, or dropping the feature entirely are the three available
levers.


Classification tasks
--------------------

.. _BinaryCarver:

Binary Classification
^^^^^^^^^^^^^^^^^^^^^

Within :class:`BinaryCarver`, a binary target consists of a column :math:`y` that only contains :math:`0` and :math:`1` (no :class:`str`).

At the basis of :class:`BinaryCarver`'s' built-in association measures lays `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.
It is computed only once per feature :math:`x` against the binary target :math:`y`.
The crosstab between :math:`y` and each possible combination of modalities of :math:`x` is then obtained via a vectorized, `numpy.add <https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html>`_. powered, implementation of `pandas.groupby <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html>`_.

:class:`BinaryCarver` takes advantage of `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ to perform association measuring.
It gives Pearson's :math:`\chi^2` statistics computed from crosstabs.

Cramér's :math:`V` is then computed using :math:`V=\sqrt{\frac{\chi^2}{n}}` where :math:`n` is the number of observation.
This implementation has been simplified taking into account the binary target :math:`y` to improve performances.

Finally, Tschuprow's :math:`T` is computed using :math:`T=\frac{V}{\sqrt{\sqrt{n_x-1}}}` where :math:`n_x` is the per-combination number of modalities.

For two combinations of modalities of :math:`x`, a higher :math:`T` or :math:`V` value indicates a stronger relationship with the binary target :math:`y`.



.. autoclass:: AutoCarver.BinaryCarver
    :members: fit, transform, fit_transform, save, load, summary, history



.. _MulticlassCarver:

Multiclass Classification
^^^^^^^^^^^^^^^^^^^^^^^^^

Within :class:`MulticlassCarver`, a multiclass target consists of a column :math:`y` that contains several values :math:`y_0` to :math:`y_{n_y-1}` where :math:`n_y>2` is the number of modalities taken by :math:`y`. Two carvers handle this shape, with a real trade-off between them:

* :class:`MulticlassCarver` (this class, **default choice**) carves each feature **once**, against the full :math:`n_y`-class crosstab: one bucket set per feature, ``copy=False`` supported, and roughly :math:`(n_y - 1)\times` faster than the one-vs-rest alternative. Use it when a single multiclass model consumes the carved features.
* :ref:`OneVsRestCarver` fits a separate :class:`BinaryCarver` per class (:math:`n_y - 1` of them, one class held out as reference), producing :math:`n_y - 1` versions of every feature. Use it when the carved features feed :math:`n_y - 1` independent one-vs-rest scorecards — OVR buckets may score *higher per class* by construction, since each one is optimized against only that class.

.. admonition:: Migration note (breaking change)
   :class: warning

   Prior to this release, ``MulticlassCarver`` **was** the one-vs-rest carver.
   That behavior is unchanged but renamed: import :class:`OneVsRestCarver`
   instead. ``MulticlassCarver`` now refers to the joint carver described below.

:class:`MulticlassCarver` is a sibling of :ref:`OrdinalCarver` (both sit
directly on ``BaseCarver`` and aggregate a ``feature-groups × target-levels``
crosstab): the :math:`n_y` classes are **unordered** here, so qualitative
modalities are ordered by their correspondence-analysis first-axis score (the
chi²-optimal 1-D embedding — see :ref:`CAOrdering`) instead of a target-rate
mean (:ref:`TargetMeanOrdering`), and the association
measure generalizes the binary :math:`\chi^2` from a 2-column to a
:math:`(B, n_y)`-column table:

Cramér's :math:`V=\sqrt{\frac{\chi^2}{n\,(\min(B, n_y)-1)}}` and Tschuprow's
:math:`T=\sqrt{\frac{\chi^2}{n\,\sqrt{(B-1)(n_y-1)}}}`, where :math:`B` is the
number of groups in the candidate combination. At :math:`n_y=2` both reduce
exactly to :class:`BinaryCarver`'s own formulas.

For two combinations of modalities of :math:`x`, a higher :math:`T` or :math:`V`
value indicates a stronger relationship with the multiclass target :math:`y`.

.. autoclass:: AutoCarver.MulticlassCarver
    :members: fit, transform, fit_transform, save, load, summary, history


.. _OneVsRestCarver:

One-vs-Rest Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For values :math:`y_0` to :math:`y_{n_y-1}` of :math:`y`, an indicator feature is built: :math:`Y_0 = \mathbb{1}_{y=y_0}` to :math:`Y_{n_y-1} = \mathbb{1}_{y=y_{n_y-1}}`.

:class:`OneVsRestCarver` repeatedly applies :class:`BinaryCarver` for features :math:`Y_0` to :math:`Y_{n_y-1}` (:math:`n_y - 1` fits — one class is held out as the implicit reference). Thus, the same association measures are implemented: Tschuprow's :math:`T` and Cramér's :math:`V`.

For two combinations of modalities of a feature :math:`x`, a higher :math:`T` or :math:`V` value indicates a stronger relationship with the binary target :math:`Y`.


.. autoclass:: AutoCarver.OneVsRestCarver
    :members: fit, transform, fit_transform, save, load, summary, history




Regression tasks
----------------

.. _ContinuousCarver:

Continuous Regression
^^^^^^^^^^^^^^^^^^^^^

Within :class:`ContinuousCarver`, a continuous target consists of a column :math:`y` that contains values from :math:`-\inf` to :math:`+\inf` (no ``str``).

The association with a categorical/ordinal feature :math:`x` is computed using `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_.

Kruskal-Wallis' :math:`H` test statistic, as known as one-way ANOVA on ranks, allows one to check that two samples originate from the same distribution.
It is used to determine whether or not :math:`y` is distributed the same when :math:`x=0, ..., x=n_x`, where :math:`n_x` is the number of modalities taken by :math:`x`.

For two combinations of modalities of :math:`x`, a higher :math:`H` value indicates that there is a greater difference between the medians of the samples.

.. autoclass:: AutoCarver.ContinuousCarver
    :members: fit, transform, fit_transform, save, load, summary, history


Ordinal tasks
-------------

.. _OrdinalCarver:

Ordinal Classification
^^^^^^^^^^^^^^^^^^^^^^

Within :class:`OrdinalCarver`, an **ordinal** target is a column :math:`y` whose
values are **integer-encoded ordered levels** (e.g. :math:`1..K` with
:math:`K > 2`); the level order is read from the ascending integer values. A
two-level target should use :ref:`BinaryCarver` and a free, unordered target
:ref:`MulticlassCarver` — :class:`OrdinalCarver` rejects both at ``fit`` time, as
it does a non-integer (continuous) or string target.

The association with a feature :math:`x` is measured with a **rank-correlation**
statistic computed on the ordered contingency table (feature groups × ordinal
target levels). Unlike the binary :math:`\chi^2`, a rank statistic *rewards a
grouping whose order matches the target's order* — exactly what an ordinal target
calls for. The default is Kendall/Stuart's :ref:`tau-c <tau_c>`; Kendall's
:ref:`tau-b <tau_b>` and the original :ref:`Somers' D <somersd>` are also
available via ``combination_evaluator``. The symmetric Kendall taus self-balance
to a robust, parsimonious number of modalities (only adding a split when it is
genuinely discriminative), whereas Somers' D leans toward the coarsest split.
See :ref:`OrdinalCombinations` for the metric definitions and the search.

For two combinations of modalities of :math:`x`, a higher tau / Somers' D value
indicates a grouping whose ordering agrees more strongly with the ordinal
target's order.

.. autoclass:: AutoCarver.OrdinalCarver
    :members: fit, transform, fit_transform, save, load, summary, history

