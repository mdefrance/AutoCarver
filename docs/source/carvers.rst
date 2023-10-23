AutoCarver
==========

The core of **AutoCarver** consists of the following Data Optimization steps: 

   1. Identifying the most associated combination from all ordered combinations of modalities.
   2. Testing all combinations of NaNs grouped to one of those modalities.

**AutoCarver** can optimize association for all target types: binary, multiclass or continuous.

.. _BinaryCarver:

Binary Classification
---------------------

In **AutoCarver** a binary target consists of a column :math:`y` that only contains :math:`0` and :math:`1` (no ``str``).

At the basis of **AutoCarver**'s' built-in association measures lays `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.
It is computed only once per feature :math:`x` against the binary target :math:`y`.
The crosstab between :math:`y` and each possible combination of modalities of :math:`x` is then obtained via a vectorized, `numpy.add <https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html>`_. powered, implementation of `pandas.groupby <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html>`_.

**AutoCarver** takes advantage of `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ to perform association measuring.
It gives Pearson's :math:`\chi^2` statistics computed from crosstabs.

Cramér's :math:`V` can then be computed using :math:`V=\sqrt{\frac{\chi^2}{n}}` where :math:`n` is the number of observation.
This implementation has been simplified taking into account the binary target :math:`y` to improve performances.

Finally, Tschuprow's :math:`T` is computed using :math:`T=\frac{V}{\sqrt{\sqrt{n_x-1}}}` where :math:`n_x` is the per-combination number of modalities.

For two combinations of modalities of :math:`x`, a higher :math:`T` or :math:`V` value indicates a stronger relationship with the binary target :math:`y`.


.. note::

	* For more details, see :ref:`Chi2`, :ref:`Cramerv` and :ref:`Tschuprowt`.



.. autoclass:: AutoCarver.BinaryCarver
    :members: fit, transform, fit_transform

.. autofunction:: AutoCarver.BinaryCarver.summary


.. _ContinuousCarver:

Continuous Regression
---------------------

In **AutoCarver** a continuous target consists of a column :math:`y` that contains values from :math:`-\inf` to :math:`+\inf` (no ``str``).

The association with a categorical/ordinal feature :math:`x` is computed using `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_.

Kruskal-Wallis' :math:`H` test statistic, as known as one-way ANOVA on ranks, allows one to check that two samples originate from the same distribution.
It is used to determine whether or not :math:`y` is distributed the same when :math:`x=0, ..., x=n_x`, where :math:`n_x` is the number of modalities taken by :math:`x`.

For two combinations of modalities of :math:`x`, a higher :math:`H` value indicates that there is a greater difference between the medians of the samples.


.. note::

	* Make sure to set ``AutoCarver.sort_by="kruskal"`` to use **AutoCarver** for continuous targets.
	* For more details, see :ref:`Kruskal`.


.. autoclass:: AutoCarver.ContinuousCarver
    :members: fit, transform, fit_transform

.. autofunction:: AutoCarver.ContinuousCarver.summary




.. _MulticlassCarver:

Multilclass Classification
--------------------------

.. autoclass:: AutoCarver.MulticlassCarver
    :members: fit, transform, fit_transform

.. autofunction:: AutoCarver.MulticlassCarver.summary



.. _BaseCarver:

AutoCarver, the automated, fast-paced data processing pipeline
--------------------------------------------------------------

.. autoclass:: AutoCarver.BaseCarver
    :members: fit, transform, fit_transform

.. autofunction:: AutoCarver.BaseCarver.summary


AutoCarver saving and loading
-----------------------------

.. autofunction:: AutoCarver.BinaryCarver.to_json

.. autofunction:: AutoCarver.load_carver
