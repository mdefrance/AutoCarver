.. _Combinations:


Combinations
============

**Combinations** are at the core of **Carvers**. They are used to identify the best combination
from all possible combinations with up to :attr:`max_n_mod` modalities.

A pre-built :class:`CombinationEvaluator` instance can be passed to any carver via the
``combination_evaluator`` keyword. Each subclass defaults to a task-appropriate metric:
:class:`TschuprowtCombinations` for :class:`BinaryCarver` / :class:`OneVsRestCarver`,
:class:`TschuprowtMulticlassCombinations` for the joint :class:`MulticlassCarver`,
:class:`KruskalCombinations` for :class:`ContinuousCarver`, and
:class:`KendallTauCCombinations` for :class:`OrdinalCarver`.

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

The DP exploits two properties shared by the supported metrics
(Kruskal-Wallis :math:`H` for continuous targets, Pearson :math:`\chi^2` for
binary targets, the :ref:`additive C − D numerator <OrdinalCombinations>` of
the rank statistics for ordinal targets):

1. **Segmentation structure.** A partition is a sequence of disjoint
   consecutive intervals :math:`[s_g, s_{g+1})`. Sub-problems factorise over
   the right boundary :math:`j` and the number of groups :math:`k`.
2. **Additive decomposability of the metric over groups, given fixed**
   :math:`k`. Each metric reduces — at fixed
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
       distinct target rates, train/dev rank) walks them in order. If none pass and
       escalation is enabled,
       <code style="background: #dbeafe; padding: 1px 4px; border-radius: 2px;">top_k</code>
       grows ×4 and the DP re-runs — keeping the common case (a viable winner in the
       first batch) cheap, while preserving the optimality guarantee in the worst case.
     </div>
   </div>


Progressive top-K
^^^^^^^^^^^^^^^^^

The DP returns the **top-K** scored partitions, not all of them. The viability
walk consumes that list in metric-desc order; if no candidate is viable in the
current top-K and escalation is enabled we grow ``top_k`` **×4** and re-run the
DP, walking only the newly-appeared entries. Repeats until either:

* a viable candidate is found, or
* the DP returns fewer than ``top_k`` entries — every consecutive partition
  has been emitted; no viable exists for this feature.

When escalation is enabled the search is **exhaustive in the worst case**: the
same admissible candidate set is eventually considered as in the legacy
enumerate-and-score path. Total work is bounded by :math:`\sim 1.33 \times` a
single DP run at the final ``top_k``. The common case (viable found in the
initial top-K) costs :math:`O(K \cdot n^2 \cdot \text{top\_k} \cdot \log
\text{top\_k})` ops, **independent of the total combination count** — which
scales combinatorially in :math:`n` and ``max_n_mod`` and reached :math:`\sim
8\text{M}` at :math:`n=40,\,\text{max_n_mod}=7` previously.

The initial top-K is configurable via the class attribute
:attr:`CombinationEvaluator.dp_top_k_initial` (default ``2000``). The ×4
escalation is opt-in via ``ProcessingConfig.dp_escalate`` (default ``False``):
when off, the search stops at the initial top-K and warns on the first fail
instead of growing ``top_k``.


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


.. _ModalityOrdering:
.. _TargetRates:

Modality ordering and target rates
----------------------------------

The DP above only ever scores **consecutive** segmentations: once the raw
modalities are laid out on a line, every candidate group is an interval of
that line. For quantitative features that line is the numeric order, and for
ordinal features the user-declared ranking. A **categorical** feature has no
intrinsic order, so one must be built from the data — and because only
consecutive groups are ever considered, this pre-search ordering decides
which groupings are *reachable at all*. It is computed once, on the train
sample, by the :ref:`CategoricalDiscretizer` step embedded in every carver
pipeline.

The scalar built for that pre-sort does not disappear once the search
starts. Every combination evaluator carries a **target rate** — the
per-modality summary of the target that the carver reports and, crucially,
the scalar the :ref:`viability filter <Viability>` orders by — re-computed
for every candidate grouping the search proposes. It is passed via the
``target_rate`` keyword and defaults to a task-appropriate choice,
documented per target type below. A target rate plays **two distinct
roles**:

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

Each target type below tells the two-stage story in one place: how the
pre-search order is built, and which target rates carry that ordering
through the viability filter.

.. _TargetMeanOrdering:

Target-mean ordering (binary, continuous and ordinal targets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the target is numeric — binary :math:`\{0, 1\}`, continuous, or
integer-encoded ordinal levels — each modality :math:`m` is scored by its
**mean target value** over the train sample,

.. math::

    \bar{y}_m = \frac{1}{n_m} \sum_{i:\, x_i = m} y_i

(a ``y.groupby(x).mean()``), and modalities are sorted by ascending
:math:`\bar{y}_m`. For a binary target this is the per-modality event rate
:math:`P(y = 1 \mid x = m)`.

This choice is what makes the consecutive-only restriction statistically
harmless rather than limiting: sorting by target mean puts modalities of
similar risk next to each other, so the intervals the DP can form are exactly
the merges of *comparable-risk* modalities — and the resulting carved feature
has monotone target rates across its bins, the layout scorecard-style models
expect.

For an **ordinal** target a raw mean over the integer *encoding* of the levels
would depend on the levels' spacing, not only their order: encoding the same
four levels as :math:`\{1, 2, 3, 4\}` or :math:`\{1, 2, 3, 10\}` could produce
different modality orderings, even though the :ref:`tau statistics
<OrdinalCombinations>` that score the combinations are rank-based and
encoding-invariant. :class:`OrdinalCarver` therefore maps the levels through
the scale resolved from its ``target_scale`` mode before taking the mean —
train ridits by default (see :ref:`RiditOrdering` below), the raw encoding
with ``target_scale="level"`` (count targets), or user-declared representative
values with a ``{level: value}`` dict.

.. _BinaryTargetRates:

Binary target rates
"""""""""""""""""""

For binary targets (and :ref:`OneVsRestCarver`'s per-class binary sub-fits) the
per-modality input is a two-column crosstab :math:`(n_0, n_1)`, so every rate
below is closed-form. The default is the event rate
:math:`p = n_1 / (n_0 + n_1)` (``TargetMean``) — the same scalar the pre-sort
ordered the raw modalities by.

.. autoclass:: AutoCarver.combinations.binary.binary_target_rates.TargetMean

.. _ContinuousTargetRates:

Continuous target rates
"""""""""""""""""""""""

For continuous targets the per-modality input is the multiset of target values.
The default ``TargetMean`` is decomposable from per-modality :math:`(n, \sum y)`
aggregates and therefore implements the closed-form ``compute_from_stats`` fast
path; ``TargetMedian`` is **not** decomposable from sums and uses the general
aggregation path.

.. autoclass:: AutoCarver.combinations.continuous.continuous_target_rates.TargetMean

.. autoclass:: AutoCarver.combinations.continuous.continuous_target_rates.TargetMedian

.. _OrdinalTargetRates:

Ordinal target rates
""""""""""""""""""""

For ordinal targets the per-modality input is the ordered contingency table
(feature groups × ordinal target levels) — the binary crosstab generalised from
two columns to one column per ordinal level. The default ``TargetMeanRidit`` is
the per-group count-weighted **mean train-ridit** of the levels (see
:ref:`RiditOrdering` below); ``TargetMeanLevel`` is the per-group **mean
ordinal level** :math:`\sum_j \text{level}_j \cdot n_{gj} / n_{g+}`, read from
the (integer) crosstab columns or from user-declared ``level_values``. Both
are monotone in the target's order, so they drive the ``min_freq`` viability
test and the train/dev rank-preservation veto exactly like the
binary/continuous target means. Rather than instantiating these directly,
declare the scale through :class:`OrdinalCarver`'s ``target_scale`` — the
carver derives the modality pre-sort from the same resolved rate, so the two
stages can never disagree.

.. autoclass:: AutoCarver.combinations.ordinal.ordinal_target_rates.TargetMeanRidit

.. autoclass:: AutoCarver.combinations.ordinal.ordinal_target_rates.TargetMeanLevel

.. _RiditOrdering:

Ridit scoring
"""""""""""""

As noted :ref:`above <TargetMeanOrdering>`, a raw-encoding ordinal pre-sort
(and the ``TargetMeanLevel`` viability rate) consume the integer *encoding* of
the levels numerically, while the tau statistics scoring the combinations are
purely rank-based. The default ``target_scale="ridit"`` instead scores each
level by its **ridit** against the train marginal:

.. math::

    r_j = F(j - 1) + \tfrac{1}{2} f_j,

where :math:`f_j` is the train frequency of level :math:`j` and
:math:`F(j-1)` the cumulative frequency of all lower levels — i.e. the
level's mean midrank rescaled to :math:`[0, 1]`. A group's mean ridit is
exactly the per-group quantity concordance statistics (Kendall's taus,
Somers' D) respond to, and it is invariant under any strictly increasing
re-encoding of the levels: :math:`\{1, 2, 3, 4\}` and
:math:`\{1, 2, 3, 10\}` carve identically.

Not every integer-encoded ordered target is order-only, though, so the scale
is a user-declared ``target_scale`` mode driving both the pre-sort and the
viability rate from a single resolved scale:

* ``"ridit"`` (**default**) — order-only levels (*Poor* / *Fair* / *Good*):
  encoding-invariant, semantically honest for "ordinal".
* ``"level"`` — count targets (e.g. 0–5 claims), where the encoding *is* the
  scale and the mean level (expected count) is the right summary.
* ``{level: value}`` — known representative values per level (e.g. a
  calibrated default probability per rating grade), strictly increasing.

When individual continuous target values are available, use
:class:`ContinuousCarver` instead.

.. _CAOrdering:
.. _MulticlassTargetRates:

Correspondence-analysis ordering (multiclass targets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A multiclass (unordered) target has no numeric mean: with classes
``{"car", "bike", "train"}`` there is no per-modality scalar to sort by. The
joint :class:`MulticlassCarver` instead orders modalities along the **first
axis of a correspondence analysis (CA)** of the raw ``modalities × classes``
crosstab — the 1-D embedding of the table that captures the largest share of
its :math:`\chi^2` inertia
(:mod:`AutoCarver.stats.correspondence_analysis`).

Fitting the axis:

#. Normalise the crosstab to proportions :math:`P`, with row masses
   :math:`r_m` (modality frequencies) and column masses :math:`c_k` (class
   frequencies).
#. Form the standardized residuals
   :math:`S_{mk} = (P_{mk} - r_m c_k) / \sqrt{r_m c_k}` — the signed
   cell-wise contributions to :math:`\chi^2 / n`, zero everywhere iff
   feature and target are independent.
#. Take the SVD of :math:`S`; the first right singular vector :math:`v_1`
   is the class-side axis.

Each modality is then scored by projecting its **row profile**
:math:`p_{m\cdot}` (its own distribution across classes) onto that fixed
axis, via the CA transition formula

.. math::

    \text{score}(m) = \sum_k \frac{p_{mk} - c_k}{\sqrt{c_k}} \; v_{1k},

and modalities are sorted by ascending score. Four properties matter for
carving:

* **Chi²-optimal.** The first CA axis is the 1-D layout that captures the
  most of the table's :math:`\chi^2` association — the natural companion to
  the :math:`\chi^2`-derived Cramér's V / Tschuprow's T the multiclass
  evaluators maximise, playing the role the target-mean sort plays for a
  binary target.
* **Train-only, fit once.** The axis is fit on the feature's raw train
  crosstab and never refit: because the transition formula only needs a
  row's own profile plus the fixed column masses and axis, the same axis
  scores candidate grouped tables *and* the dev-sample projection used by
  the rank-preservation veto.
* **Label-independent and deterministic.** Scores depend only on the counts,
  never on the modalities' or classes' text, and the axis sign is anchored
  on the largest-mass row (content-based tie-breaks), so any row permutation
  of the same table yields the same ordering.
* **Degenerate fallback.** With :math:`\le 2` modalities, fewer than 2
  classes, or no :math:`\chi^2` structure (near-zero first singular value),
  the CA axis is meaningless and modalities fall back to a deterministic
  frequency-descending order.

The same fitted axis doubles as the multiclass **target rate**: the default
``CAScoreRate`` projects each row of a candidate grouping's crosstab
(feature groups × classes) onto it, giving the scalar the
:ref:`viability filter <Viability>` orders by — monotone along the axis by
construction, so it drives the ``min_freq`` distinct-rate test and the
train/dev rank-preservation veto exactly like the binary/ordinal target
means. Because the axis is never refit on dev, the veto compares train and
dev against one shared yardstick rather than two independently-defined axes —
and the search-space ordering and the viability ordering can never disagree.

.. autoclass:: AutoCarver.combinations.multiclass.multiclass_target_rates.CAScoreRate



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


.. _MulticlassChi2:

Pearson :math:`\chi^2` generalised to K classes (multiclass targets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The joint :class:`MulticlassCarver` generalises :ref:`DPChi2` from a 2-column
table to a :math:`(B, K)` one, :math:`K` the number of target classes. With row
marginals :math:`R_g`, column marginals :math:`C_c` and total :math:`N` defined
exactly as before,

.. math::

    \chi^2 = N \left(\sum_{g, c} \frac{O_{g,c}^2}{R_g\, C_c} - 1\right),
    \qquad
    V = \sqrt{\frac{\chi^2}{N\,(\min(B, K) - 1)}},
    \qquad
    T = \sqrt{\frac{\chi^2}{N\,\sqrt{(B-1)(K-1)}}}.

The same two observations that make the binary DP possible still hold — at
fixed :math:`k` groups, the column marginals and total depend only on
:math:`k` (plus the constant ``tol`` shift), so :math:`\chi^2` is additive over
groups and the DP in :ref:`DPChi2` carries over unchanged, generalised from a
``(2,)`` to a ``(K,)`` per-segment observed vector. Yates' correction still
applies only when the table is exactly :math:`2\times 2` (:math:`k=2` **and**
:math:`K=2`). At :math:`K=2` both :math:`V` and :math:`T` — and the DP itself —
are numerically identical to the binary evaluator, pinned bit-for-bit by a
parity test.

Unlike binary/ordinal features, a multiclass target gives qualitative
modalities no numeric rate to order by before the DP walks them. They are
instead ordered by their **correspondence-analysis (CA) first-axis score** —
the 1-D embedding that maximises chi² association — computed once from the raw
(un-grouped) crosstab and fixed for the rest of the search. How the axis is
fit, why its ordering is deterministic and label-independent, and how the
same axis doubles as the multiclass target rate (``CAScoreRate``) is
detailed in :ref:`CAOrdering`.

.. _CramervMulticlassCombinations:

Cramér's V Multiclass Combinations
""""""""""""""""""""""""""""""""""

.. autoclass:: AutoCarver.combinations.CramervMulticlassCombinations
    :members: save, load

.. _TschuprowtMulticlassCombinations:

Tschuprow's T Multiclass Combinations (default)
""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: AutoCarver.combinations.TschuprowtMulticlassCombinations
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


.. _OrdinalCombinations:

Ordinal tasks
-------------

For an **ordinal** target (integer-encoded ordered levels), a combination is
scored by a **rank-association** statistic on the ordered contingency table
:math:`(r \times c)` — :math:`r` feature groups (rows, target-rate order) ×
:math:`c` ordinal target levels (cols, ascending). All three statistics below are
built from the same pair counts:

* :math:`C` — **concordant** pairs (both members order the same way on the
  feature and on the target);
* :math:`D` — **discordant** pairs (members order oppositely);
* :math:`P_0 = n(n-1)/2` — all pairs, with :math:`n` the number of observations;
* :math:`T_X`, :math:`T_Y` — pairs **tied** on the feature / on the target
  (equal row / equal column); :math:`P_0 - T_X` and :math:`P_0 - T_Y` are the
  pairs untied on each margin;
* :math:`m = \min(r', c')` — the smaller of the number of **non-empty** grouped
  rows :math:`r'` and target levels :math:`c'`.

The concordant-minus-discordant count :math:`C - D` is computed in closed form
from the table's cumulative cell sums (``_concordant_minus_discordant``); the
three measures are monotone-comparable transforms of it. Each measure is
``None`` for a degenerate table (its denominator vanishes), mirroring the
continuous evaluator's ``None`` convention. Parity against
:func:`scipy.stats.kendalltau` (tau-b) and :func:`scipy.stats.somersd` is pinned
by ``tests/combinations/ordinal/test_ordinal_associations.py`` and the property
suite ``tests/properties/combinations/test_ordinal_combinations_properties.py``.


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


Search strategy — additive :math:`C - D` interval DP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ordinal evaluators reuse the :ref:`progressive top-K interval DP <DPTopK>`
pattern. The key fact is that the numerator :math:`C - D` of a consecutive
grouping decomposes **additively** over groups:

.. math::

    (C - D)(\text{grouping}) = \text{TotalBetween} - \sum_g
    \text{WithinSegment}(g),

where ``TotalBetween`` (the :math:`C - D` of the fully-split table) is constant
and ``WithinSegment`` — the concordant−discordant pairs *removed* by merging the
rows of a segment — is prefix-summable. An interval DP that keeps, per number of
groups :math:`k`, the partitions with the largest numerator therefore enumerates
the best candidates without materialising every consecutive partition.

* For :math:`\tau_c` the per-:math:`k` denominator :math:`n^2 (m-1) / (2m)` is
  **constant**, so numerator-optimal :math:`=` metric-optimal: the DP is
  **exact** for tau-c even at the smallest ``top_k``.
* For :math:`\tau_b` and Somers' D the denominator depends on the group sizes
  (through :math:`T_X`), so the kept top-K candidates are **re-scored** with
  their true denominators and ranked — exact once ``top_k`` is exhaustive, a
  top-K approximation otherwise (progressive doubling makes it exhaustive in the
  worst case, exactly as in the binary/continuous DPs).

The :ref:`NaN fan-out <DPTopK>` is **not** re-implemented: it runs after this DP
has applied the best non-NaN grouping, over the already-small grouped label set,
so the inherited enumerate-and-score path is cheap there.


.. _KendallTauCCombinations:

Kendall's tau-c Combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`tau_c` for more details on the metric.

.. autoclass:: AutoCarver.combinations.KendallTauCCombinations
    :members: save, load

.. _KendallTauBCombinations:

Kendall's tau-b Combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`tau_b` for more details on the metric.

.. autoclass:: AutoCarver.combinations.KendallTauBCombinations
    :members: save, load

.. _SomersDCombinations:

Somers' D Combinations
^^^^^^^^^^^^^^^^^^^^^^

See :ref:`somersd` for more details on the metric.

.. autoclass:: AutoCarver.combinations.SomersDCombinations
    :members: save, load
