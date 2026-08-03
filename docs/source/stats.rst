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

.. _stats_correspondence_analysis:

Correspondence analysis
------------------------

.. autofunction:: AutoCarver.stats.fit_ca_axis

.. autofunction:: AutoCarver.stats.ca_row_scores

.. autoclass:: AutoCarver.stats.CAAxis
    :members:
