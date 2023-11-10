.. _Carvers:

Carvers
=======

The core of **AutoCarver** resides in its **Carvers**, they provide the following Data Optimization steps: 

   1. Identifying the most associated combination from all ordered combinations of modalities.
   2. Testing all combinations of NaNs grouped to one of those modalities.

Target-specific tools allow for association optimization per desired task:
 * :ref:`BinaryCarver` 
 * :ref:`MulticlassCarver`
 * :ref:`ContinuousCarver`




Classification tasks
--------------------

.. _BinaryCarver:

Binary Classification
^^^^^^^^^^^^^^^^^^^^^

Within :class:`BinaryCarver`, a binary target consists of a column :math:`y` that only contains :math:`0` and :math:`1` (no ``str``).

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
    :members: fit, transform, fit_transform, to_json, summary, get_history


.. _MulticlassCarver:

Multilclass Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Within :class:`MulticlassCarver`, a multiclass target consists of a column :math:`y` that contains several values :math:`y_0` to :math:`y_{n_y}` where :math:`n_y>2` is the number of modalities taken by :math:`y`.

For values :math:`y_0` to :math:`y_{n_y-1}` of :math:`y`, an indicator feature is built: :math:`Y_0 = \mathbb{1}_{y=y_0}` to :math:`Y_{n_y-1} = \mathbb{1}_{y=y_{n_y-1}}`.

:class:`MulticlassCarver` repeatedly applies :class:`BinaryCarver` for features :math:`Y_0` to :math:`Y_{n_y-1}`. Thus, the same association measure are implemented: Tschuprow's :math:`T` and Cramér's :math:`V`.

For two combinations of modalities of a feature :math:`x`, a higher :math:`T` or :math:`V` value indicates a stronger relationship with the binary target :math:`Y`.


.. autoclass:: AutoCarver.MulticlassCarver
    :members: fit, transform, fit_transform, to_json, summary, get_history




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
    :members: fit, transform, fit_transform, to_json, summary, get_history




Saving and loading
------------------

.. autofunction:: AutoCarver.BaseCarver.to_json

.. autofunction:: AutoCarver.load_carver
