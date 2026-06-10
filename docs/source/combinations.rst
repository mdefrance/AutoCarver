.. _Combinations:


Combinations
============

**Combinations** are at the core of **Carvers**. They are used to identify the best combination
from all possible combinations with up to :attr:`max_n_mod` modalities.

A pre-built :class:`CombinationEvaluator` instance can be passed to any carver via the
``combination_evaluator`` keyword. Each subclass defaults to a task-appropriate metric:
:class:`TschuprowtCombinations` for :class:`BinaryCarver` / :class:`MulticlassCarver`,
:class:`KruskalCombinations` for :class:`ContinuousCarver`.

The animation below starts from the six ordered bins a
:class:`QuantitativeDiscretizer` produces (its final state — see
:ref:`QuantitativeDiscretizer`) and shows the core step: every consecutive
grouping into ``max_n_mod`` groups is scored by its association with the binary
target (Tschuprow's T) and the table fills best-first in growing ``top_k``
batches (the :ref:`progressive top-K DP <DPTopK>` search). The highest-scoring
grouping that passes the :ref:`viability filter <Viability>` is kept
(gold row). Adjacent bins sharing a colour in a row are merged into one group.

.. image:: _static/animations/combinations.svg
   :alt: Combinations search animation — ordered bins, then consecutive groupings ranked by Tschuprow's T filling a table best-first, with the selected grouping highlighted
   :width: 100%
   :align: center

.. autoclass:: AutoCarver.combinations.CombinationEvaluator
    :members: get_best_combination, save, load

The highest-scoring grouping is not necessarily the one that is kept: each
candidate must clear the :ref:`viability filter <Viability>` (minimum frequency
via a Wilson score interval, distinct consecutive target rates, and train/dev
rank preservation). That filter is documented on its own page.


.. _DPTopK:

Search strategy — interval dynamic programming (DP) with progressive top-K
--------------------------------------------------------------------------

For fixed ``min_freq``, ``max_n_mod`` and association metric, **AutoCarver
returns the partition that maximises the metric among admissible candidates**.
The DP described below is a search-strategy optimisation; it does **not**
prune the candidate set and does **not** change the statistical claim.
Bit-exact agreement with the legacy enumerate-and-score path is pinned by
parity tests (``tests/combinations/binary/test_dp_chi2_parity.py``,
``tests/combinations/continuous/test_dp_kruskal_parity.py``).


The search problem
^^^^^^^^^^^^^^^^^^

For a feature with raw modalities :math:`m_0, \dots, m_{n-1}` already ordered
(by ordinal rank, target rate, or numeric quantile), the carver searches over
**consecutive segmentations** with at most ``max_n_mod`` groups: a partition is
fully determined by integer split positions
:math:`0 = s_0 < s_1 < \dots < s_k = n` with :math:`k \le \text{max_n_mod}`.
The chosen partition maximises the association metric subject to the
:ref:`viability filter <Viability>` (Wilson ``min_freq`` on train + dev,
distinct target rates, preserved rank between train and dev when dev is
provided).

The legacy path enumerated every admissible partition, scored each, then
walked them in metric-desc order. This is correct but wasteful — only the top
handful of candidates ever survive the viability walk.


The DP idea
^^^^^^^^^^^

The DP exploits two properties shared by both supported metrics
(Kruskal-Wallis :math:`H` for continuous targets, Pearson :math:`\chi^2` for
binary targets):

1. **Segmentation structure.** A partition is a sequence of disjoint
   consecutive intervals :math:`[s_g, s_{g+1})`. Sub-problems factorise over
   the right boundary :math:`j` and the number of groups :math:`k`.
2. **Additive decomposability of the metric over groups, given fixed**
   :math:`k`. Both :math:`H` and :math:`\chi^2` reduce — at fixed
   :math:`k` and after factoring out :math:`k`-dependent normalising
   constants — to a sum over groups of a quantity that depends **only on a
   single interval** :math:`[i, j)` of raw modalities.

We therefore run an interval DP indexed by :math:`(k, j)` whose state is the
**top-K** prefixes (by partial score) ending at split :math:`j` with
:math:`k` groups:

.. math::

    \text{dp}[k][j] = \operatorname*{top\text{-}K}_{i \in [k-1,\, j)} \big\{\,
       \text{dp}[k-1][i] \oplus \text{seg_cost}(i,\, j)\, \big\}

The final candidate list is :math:`\bigcup_k \text{dp}[k][n]`, sorted desc and
truncated to ``top_k``.

.. raw:: html

   <div style="border: 1px solid #d1d5db; border-radius: 8px; padding: 1.4em 1.5em; margin: 1.6em 0; background: #f9fafb; color: #1f2937;">
     <div style="font-size: 1.05em; font-weight: 600; margin-bottom: 0.5em; color: #111827;">
       How the DP fills its table — a worked sketch
     </div>
     <div style="font-size: 0.9em; color: #6b7280; margin-bottom: 1.1em;">
       Why it beats enumerate-and-score, in one picture.
     </div>

     <div style="margin-bottom: 0.6em;">
       <strong>1. The search space.</strong>
       Take a feature with 6 ordered raw modalities. A partition with
       <em>k</em>&nbsp;=&nbsp;3 groups is fully determined by 2 internal split positions.
       Two candidate partitions:
     </div>
     <div style="font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; line-height: 2; margin: 0.4em 0 1em 1.2em; font-size: 0.95em;">
       <div>
         <span style="color: #6b7280; display: inline-block; width: 1.6em;">A:</span>
         <span style="background: #fde68a; padding: 2px 8px; border-radius: 3px;">m₀ m₁</span>
         <span style="margin: 0 6px; color: #9ca3af;">│</span>
         <span style="background: #bfdbfe; padding: 2px 8px; border-radius: 3px;">m₂ m₃</span>
         <span style="margin: 0 6px; color: #9ca3af;">│</span>
         <span style="background: #ddd6fe; padding: 2px 8px; border-radius: 3px;">m₄ m₅</span>
       </div>
       <div>
         <span style="color: #6b7280; display: inline-block; width: 1.6em;">B:</span>
         <span style="background: #fde68a; padding: 2px 8px; border-radius: 3px;">m₀ m₁</span>
         <span style="margin: 0 6px; color: #9ca3af;">│</span>
         <span style="background: #bfdbfe; padding: 2px 8px; border-radius: 3px;">m₂ m₃ m₄</span>
         <span style="margin: 0 6px; color: #9ca3af;">│</span>
         <span style="background: #ddd6fe; padding: 2px 8px; border-radius: 3px;">m₅</span>
       </div>
     </div>

     <div style="margin-bottom: 0.6em;">
       <strong>2. The shared-prefix insight.</strong>
       A and B share the first group
       <span style="font-family: ui-monospace, monospace; background: #fde68a; padding: 1px 6px; border-radius: 3px;">m₀ m₁</span>.
       The naïve path re-scores that prefix once per candidate that contains it; for
       <em>n</em>&nbsp;=&nbsp;6,&nbsp;<em>k</em>&nbsp;=&nbsp;3 that's
       <span style="font-family: ui-monospace, monospace;">C(5, 2) = 10</span>
       partitions and a lot of wasted work. The DP scores each prefix
       <em>once</em> and reuses it.
     </div>

     <div style="margin: 1.2em 0 0.6em 0;">
       <strong>3. The DP table.</strong>
       <span style="font-family: ui-monospace, monospace;">dp[k][j]</span>
       holds the top-K best scores of partitioning the first <em>j</em> modalities
       into exactly <em>k</em> groups. We fill it row-by-row, left-to-right:
     </div>
     <div style="display: flex; justify-content: center; margin: 0.6em 0;">
       <table style="border-collapse: collapse; font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; font-size: 0.92em;">
         <thead>
           <tr>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">k \ j</th>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">1</th>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">2</th>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">3</th>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">4</th>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">5</th>
             <th style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; color: #374151;">6</th>
           </tr>
         </thead>
         <tbody>
           <tr>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; font-weight: 600; color: #374151;">k=1</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
           </tr>
           <tr>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; font-weight: 600; color: #374151;">k=2</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #f3f4f6; color: #d1d5db;">—</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
           </tr>
           <tr>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; background: #e5e7eb; font-weight: 600; color: #374151;">k=3</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #f3f4f6; color: #d1d5db;">—</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #f3f4f6; color: #d1d5db;">—</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fff;">●</td>
             <td style="border: 1px solid #d1d5db; padding: 6px 14px; text-align: center; background: #fef3c7; color: #92400e; font-weight: 700;">★</td>
           </tr>
         </tbody>
       </table>
     </div>
     <div style="font-size: 0.9em; color: #4b5563; margin-top: 0.4em; text-align: center;">
       The answer for <em>k</em>&nbsp;=&nbsp;3 lives at the ★ cell:
       <span style="font-family: ui-monospace, monospace;">dp[3][6]</span>.
       Cells marked <span style="color: #9ca3af;">—</span> are infeasible
       (can't make <em>k</em> groups out of fewer than <em>k</em> modalities).
     </div>

     <div style="margin: 1.3em 0 0.6em 0;">
       <strong>4. The recurrence — one cell at a time.</strong>
       To fill <span style="font-family: ui-monospace, monospace;">dp[k][j]</span>,
       try every possible <em>last</em> split position
       <span style="font-family: ui-monospace, monospace;">i</span> and combine:
     </div>
     <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.9em 1.2em; margin: 0.4em 0 0.8em 1.2em; font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; font-size: 0.95em; line-height: 1.6;">
       <span style="color: #374151;">dp[k][j]</span>
       &nbsp;=&nbsp;
       <span style="color: #6b7280;">top-K over</span>
       &nbsp;i&nbsp;
       <span style="color: #6b7280;">of</span>
       &nbsp;{&nbsp;
       <span style="background: #dbeafe; padding: 1px 6px; border-radius: 3px;" title="already-computed best score for the first k−1 groups, ending at split i">dp[k−1][i]</span>
       &nbsp;⊕&nbsp;
       <span style="background: #fce7f3; padding: 1px 6px; border-radius: 3px;" title="closed-form contribution of the final group [i, j) — from prefix sums, no enumeration">seg_cost(i, j)</span>
       &nbsp;}
     </div>
     <div style="font-size: 0.9em; color: #4b5563; margin: 0 0 0.4em 1.2em;">
       <span style="background: #dbeafe; padding: 1px 6px; border-radius: 3px; font-family: ui-monospace, monospace;">dp[k−1][i]</span>
       is already computed (previous row).
       <span style="background: #fce7f3; padding: 1px 6px; border-radius: 3px; font-family: ui-monospace, monospace;">seg_cost(i, j)</span>
       is the contribution of the final group
       <span style="font-family: ui-monospace, monospace;">[i, j)</span> — closed-form
       from prefix sums (no inner enumeration). So each cell costs
       <span style="font-family: ui-monospace, monospace;">O(j)</span>
       work, and the whole table is
       <span style="font-family: ui-monospace, monospace;">O(K · max_n_mod · n²)</span>
       instead of the
       <span style="font-family: ui-monospace, monospace;">O(2ⁿ)</span>
       enumerate-and-score path.
     </div>

     <div style="margin-top: 1.3em; padding: 0.8em 1em; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 0 4px 4px 0; font-size: 0.93em; color: #1e3a8a;">
       <strong>Why "progressive" top-K?</strong>
       The DP returns the top <code style="background: #dbeafe; padding: 1px 4px; border-radius: 2px;">top_k</code>
       partitions by score, then the viability filter (Wilson <code style="background: #dbeafe; padding: 1px 4px; border-radius: 2px;">min_freq</code>,
       distinct target rates, train/dev rank) walks them in order. If none pass,
       <code style="background: #dbeafe; padding: 1px 4px; border-radius: 2px;">top_k</code>
       doubles and the DP re-runs — keeping the common case (a viable winner in the
       first batch) cheap, while preserving the optimality guarantee in the worst case.
     </div>
   </div>


Progressive top-K
^^^^^^^^^^^^^^^^^

The DP returns the **top-K** scored partitions, not all of them. The viability
walk consumes that list in metric-desc order; if no candidate is viable in the
current top-K we **double** ``top_k`` and re-run the DP, walking only the
newly-appeared entries. Repeats until either:

* a viable candidate is found, or
* the DP returns fewer than ``top_k`` entries — every consecutive partition
  has been emitted; no viable exists for this feature.

Doubling guarantees the search is **exhaustive in the worst case**: the same
admissible candidate set is eventually considered as in the legacy
enumerate-and-score path. Total work is bounded by :math:`\sim 2 \times` a
single DP run at the final ``top_k``. The common case (viable found in the
initial top-K) costs :math:`O(K \cdot n^2 \cdot \text{top\_k} \cdot \log
\text{top\_k})` ops, **independent of the total combination count** — which
scales combinatorially in :math:`n` and ``max_n_mod`` and reached :math:`\sim
8\text{M}` at :math:`n=40,\,\text{max_n_mod}=7` previously.

The initial top-K is configurable via the class attribute
:attr:`CombinationEvaluator.dp_top_k_initial` (default ``1000``).


NaN fan-out path
^^^^^^^^^^^^^^^^

When ``dropna=True`` and the feature has NaNs, the DP runs on the **non-NaN
sub-index** to produce base partitions. Each base is then **fanned out** across
NaN placements:

* NaN folded into each existing group;
* NaN as its own group when ``len(base) < max_n_mod``;
* plus the degenerate ``[all_non_nan, [NaN]]`` partition.

Each variant is scored in closed form (``_kruskal_h_for_combination`` /
``_chi2_assoc_for_combination``) against the **full** per-modality stats — the
NaN row is still in the aggregated sample because ``_apply_best_combination``
rebuilt it from raw after the non-NaN DP. Variants are walked sorted desc,
dedup'd by partition key across progressive iterations so combinations carried
over from a smaller ``top_k`` are not re-tested.


What does **not** change
^^^^^^^^^^^^^^^^^^^^^^^^

* The admissible candidate set: consecutive segmentations with
  :math:`k \le \text{max_n_mod}`.
* The :ref:`viability filter <Viability>`: Wilson ``min_freq`` on train
  + dev, distinct target rates, rank preservation.
* The optimality claim: **for fixed** ``min_freq``\ **,** ``max_n_mod``\ **,
  and metric, no admissible combination scores higher than the one returned.**

The DP is a **search-strategy optimisation**, not a statistical change.


.. _TargetRates:

Target rates
------------

Every combination evaluator carries a **target rate** — the per-modality summary
of the target that the carver reports and, crucially, the scalar the
:ref:`viability filter <Viability>` orders by. It is passed via the
``target_rate`` keyword and defaults to a task-appropriate choice (the target
mean, ``TargetMean``, for every built-in evaluator).

A target rate plays **two distinct roles**:

#. **Display statistic.** One value per modality, stored on
   ``feature.statistics`` and surfaced in the carved-feature summary, so the
   grouping can be read off in interpretable units (an event rate, an odds
   ratio, a mean target, …).
#. **Ordering key for viability.** The same per-modality value is what the
   :ref:`distinct-rate test <DistinctRatesViability>` requires to differ
   between consecutive modalities, and what the
   :ref:`train/dev rank-preservation veto <RankViability>` sorts on. A
   combination whose target-rate ordering collapses or flips is rejected.

Because of this dual role, two properties decide whether a candidate rate is a
good fit:

* **It must be an orderable scalar.** The viability checks need a single value
  per modality with a meaningful monotone ordering. A symmetric measure (e.g. a
  Gini-style impurity, maximal at :math:`p = 0.5`) is fine as a display
  statistic but a poor ordering key.
* **Decomposability buys a fast path.** When a rate can be reconstructed from
  per-raw-modality sufficient statistics it can opt into a closed-form path
  (``compute_from_stats``) that costs :math:`O(k)` per combination instead of
  re-aggregating the raw sample on every candidate. The continuous mean does
  this; rates that need the full value multiset (median, quantiles) cannot and
  fall back to the general aggregation path.

.. autoclass:: AutoCarver.combinations.utils.TargetRate
    :members: compute


.. _BinaryTargetRates:

Binary target rates
^^^^^^^^^^^^^^^^^^^

For binary (and multiclass) targets the per-modality input is a two-column
crosstab :math:`(n_0, n_1)`, so every rate below is closed-form. The default is
the event rate :math:`p = n_1 / (n_0 + n_1)` (``TargetMean``).

.. autoclass:: AutoCarver.combinations.binary.binary_target_rates.TargetMean

.. _ContinuousTargetRates:

Continuous target rates
^^^^^^^^^^^^^^^^^^^^^^^

For continuous targets the per-modality input is the multiset of target values.
The default ``TargetMean`` is decomposable from per-modality :math:`(n, \sum y)`
aggregates and therefore implements the closed-form ``compute_from_stats`` fast
path; ``TargetMedian`` is **not** decomposable from sums and uses the general
aggregation path.

.. autoclass:: AutoCarver.combinations.continuous.continuous_target_rates.TargetMean

.. autoclass:: AutoCarver.combinations.continuous.continuous_target_rates.TargetMedian


Custom target rates
^^^^^^^^^^^^^^^^^^^

A custom rate subclasses ``BinaryTargetRate`` or ``ContinuousTargetRate`` and
implements ``_compute`` (one value per modality). To make it serialisable
through ``save`` / ``load``, add it to the evaluator's ``_target_rate_classes``
registry. When the rate is additively decomposable from per-modality sums,
override ``compute_from_stats`` so the search uses the closed-form path.

Candidate extensions that fit this contract (not yet implemented):

* **Binary, closed-form:** logit / log-odds :math:`\log(n_1 / n_0)`, and a
  column-normalised weight-of-evidence
  :math:`\log\!\big((n_1/\textstyle\sum n_1)\,/\,(n_0/\textstyle\sum n_0)\big)`.
* **Continuous, decomposable** (extend the carried stats with
  :math:`\sum y^2`): variance and standard deviation.
* **Continuous, non-decomposable** (general path only): inter-quartile range,
  arbitrary quantiles, and trimmed/robust location — useful for heavy-tailed
  targets where the mean ordering is unstable.


Classification tasks
--------------------

.. _DPChi2:

Pearson :math:`\chi^2` (binary targets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a 2-column contingency table (binary target), each group :math:`g`
contributes counts :math:`(n_{0,g},\, n_{1,g})`. With row marginals
:math:`R_g = n_{0,g} + n_{1,g}`, column marginals :math:`C_c = \sum_g n_{c,g}`,
and grand total :math:`N = \sum_g R_g`, Pearson's statistic is

.. math::

    \chi^2 = \sum_{g, c} \frac{(O_{g, c} - E_{g, c})^2}{E_{g, c}},
    \quad E_{g, c} = \frac{R_g \cdot C_c}{N}.

Two key observations:

* **Given a fixed number of groups** :math:`k`, the column marginals
  :math:`C_c` and total :math:`N` depend **only on** :math:`k` (and a constant
  ``tol`` shift applied to every cell — matching the legacy
  ``chi2_contingency(xagg + tol)`` call):
  :math:`C_c = N_c + k\cdot\text{tol}`,
  :math:`N = N_0 + N_1 + 2k\cdot\text{tol}`. They are **invariant under
  re-partitioning at fixed** :math:`k`.
* Therefore, at fixed :math:`k`, :math:`\chi^2` is **additive over groups**:
  each group contributes :math:`(O - E)^2 / E` summed over its two cells,
  with :math:`E` derivable from :math:`(n_{0,g},\, n_{1,g})` and the constants
  :math:`(C_0, C_1, N)`. The **Yates correction** (subtract :math:`0.5` from
  :math:`|O - E|` iff the table is exactly :math:`2 \times 2`, matching scipy's
  default) is applied **only when** :math:`k = 2`, which is again known at the
  DP level.

The DP is therefore run **once per** :math:`k \in [2,\, \text{max_n_mod}]`
with the constants :math:`(C_0, C_1, N, \text{yates_flag})` fixed; per-:math:`k`
top-K lists are merged and re-truncated:

.. math::

    \text{seg_cost}_k(i,\, j) =
    \chi^2\text{ contribution of }[i,\, j)\text{ under }
    (C_0,\, C_1,\, N,\, \text{yates_flag} = (k = 2)).

Cramér's :math:`V = \sqrt{\chi^2 / N_{obs}}` and Tschuprow's
:math:`T = V / \sqrt[4]{k - 1}` are monotone transforms of :math:`\chi^2` at
fixed :math:`k`, so sorting by either is equivalent to sorting by :math:`\chi^2`
**within each** :math:`k` **slice**. The cross-:math:`k` merge re-applies the
configured ``sort_by`` so the global top-K is correct under either metric.
Statistical equivalence to :func:`scipy.stats.chi2_contingency` is **bit-exact**
(parity tests pin the :math:`+\text{tol}` shift, the Yates handling, and the
:math:`\text{round}(x / \text{tol}) \cdot \text{tol}` quantisation).


.. _CramervCombinations:

Cramér's V Combinations
^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`Cramerv` for more details on the metric.

.. autoclass:: AutoCarver.combinations.CramervCombinations
    :members: save, load

.. _TschuprowtCombinations:

Tschuprow's T Combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`Tschuprowt` for more details on the metric.

.. autoclass:: AutoCarver.combinations.TschuprowtCombinations
    :members: save, load

Regression tasks
----------------

.. _DPKruskal:

Kruskal-Wallis H (continuous targets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a partition with :math:`n_g` observations per group, rank sum
:math:`R_g`, total :math:`N = \sum_g n_g`, and **tie correction**
:math:`T = 1 - \sum_i (t_i^3 - t_i) / (N^3 - N)` (depends only on the pooled
:math:`y` multiset), the Kruskal-Wallis statistic is

.. math::

    H = \frac{1}{T}\left[\,\frac{12}{N(N+1)} \sum_g \frac{R_g^2}{n_g} - 3(N+1)\,\right].

Two key observations:

* **Per-modality** :math:`(R_i, n_i)`, the total :math:`N`, and the tie
  correction :math:`T` depend only on the raw feature ranking — *not* on the
  partition. They are computed **once** by ranking :math:`y` once over the
  pooled sample (see ``_modality_rank_stats``).
* :math:`\sum_g R_g^2 / n_g` is **additive over groups**. With prefix sums
  ``R_prefix`` and ``n_prefix`` over the raw modalities, a single interval's
  contribution is closed-form:

  .. math::

      \text{seg_cost}(i, j) =
      \frac{\big(\text{R_prefix}[j] - \text{R_prefix}[i]\big)^2}
           {\text{n_prefix}[j] - \text{n_prefix}[i]}.

The DP maximises :math:`\sum_g \text{seg_cost}` over partitions; :math:`H`
is recovered at the end by applying the constant prefactor
:math:`12 / (N(N+1))`, the constant offset :math:`-3(N+1)`, and dividing by
:math:`T`. Statistical equivalence to :func:`scipy.stats.kruskal` is
**bit-exact** — the DP only re-orders the search.


.. _KruskalCombinations:

Kruskal's H Combinations
^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`kruskal` for more details on the metric.

.. autoclass:: AutoCarver.combinations.KruskalCombinations
    :members: save, load
