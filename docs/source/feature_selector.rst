FeatureSelector
===============

**AutoCarver** implements ``FeatureSelector``, an association-centric feature selection tool.
It consists of the following Data Selection steps: 

   1. Measuring association with a binary target and ranking features accordingly.
   2. Filtering out features too asociated to a better ranked feature.

``FeatureSelector`` allows one to select features ase on there type: quantitative or qualitative.

By default, quantitative features are:

 * Ranked according to Kurskal-Wallis' test statistic.
 * Filtered according to Spearman correlation coefficient

By default, qualitative features are:

 * Ranked according to Cramer's V
 * Filtered according to Cramer's V

In general, associations are computed according to the provided data types of :math:`x` and :math:`y`:

+-----------------------+---------------------------------------------------------------------+-------------------------------+
| :math:`x` \\ :math:`y`| Qualitatitve                                                        | Quantitative                  |
+-----------------------+---------------------------------------------------------------------+-------------------------------+
| Qualitative           | Pearson's :math:`\chi^2`, Cramér's :math:`V`, Tschuprow's :math:`T` | Kruskal-Wallis, R coefficient |
+-----------------------+---------------------------------------------------------------------+-------------------------------+
| Quantitative          | Kruskal-Wallis, R coefficient                                       | Spearman, Pearson             |
+-----------------------+---------------------------------------------------------------------+-------------------------------+

See :ref:`Measures` and :ref:`Filters`, for details on measures and filters' implementation.

.. note::

    Additionnal measure/filter specific parameters can be added as keayword arguments.

.. _FeatureSelector:

FeatureSelector, an association centric tool for feature pre-selection
----------------------------------------------------------------------


.. autoclass:: AutoCarver.feature_selection.FeatureSelector
    :members: select

.. _Measures:

Association measures, X by y 
----------------------------

Quantitative measures
.....................

Kruskal-Wallis :math:`H` test statistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a quantititative feature :math:`x`, the association with a binary target :math:`y` is computed using `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_.

Kruskal-Wallis :math:`H` test statistic, as known as one-way ANOVA on ranks, allows one to check that two features originate from the same disctribution.
It is used to determine whether or not :math:`x` is distributed the same when :math:`y=1` compared to when :math:`y=0`.
It is computed using the following formula:


.. math::

    H= (n-1) \frac{ \sum_{i=1}^{n_y}{ n_{y=i} (\bar{x_r^{i.}} - \bar{x_r}) } } { \sum_{i=1}^{n_y}{ \sum_{j=1}^{n_{y=i}}{ (x_r^{ij} - \bar{x_r}) } } }



where:

 * :math:`n` is the number of observations
 * :math:`n_y` is the number of modalities of :math:`y`
 * :math:`n_{y=i}` is the number of observations taking :math:`y`'s :math:`i` th modality
 * :math:`x_r` is the ranked version of :math:`x`
 * :math:`x_r^{ij}` is the :math:`j` th observation of :math:`x_r` when :math:`y` takes its :math:`i` th modality
 * :math:`\bar{x_r^{i.}}=\sum_{j=1}^{n_{y=i}}` is the sample mean of :math:`x_r` when :math:`y` takes its :math:`i` th modality
 * :math:`\bar{x_r}=\sum_{i=1}^{n_y}{\sum_{j=1}^{n_{y=i}}}` is the sample mean of :math:`x_r`





.. autofunction:: AutoCarver.feature_selection.measures.kruskal_measure

.. note::
    ``kruskal_measure`` is the default measure for quantitative features (i.e. when ``FeatureSelector.measures=[]`` and ``FeatureSelector.quantitative_features`` is provided).





Coefficient of determination :math:`R`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. autofunction:: AutoCarver.feature_selection.measures.R_measure





Quantitative outlier Detection
..............................

.. autofunction:: AutoCarver.feature_selection.measures.zscore_measure





.. autofunction:: AutoCarver.feature_selection.measures.iqr_measure






Qualitative measures
....................

Pearson's :math:`\chi^2` test statistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a qualititative feature :math:`x`, the association with a qualitative binary target :math:`y` is computed based on the `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.

Pearson's :math:`\chi^2` test statistic is then computed using `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ to perform association measuring.
The formula is the following:

.. math::

    \chi^2=\sum_{i=1}^{n_x}{\sum_{j=1}^{n_y}{\frac{(n_{ij} - \frac{n_{i.}n_{.j}}{n})^2}{\frac{n_{i.}n_{.j}}{n}}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_x` is the number of modalities of :math:`x`
 * :math:`n_y` is the number of modalities of :math:`y`
 * :math:`n_{ij}` is the number of observations that take modality :math:`i` of :math:`x` and modality :math:`j` of :math:`y`
 * :math:`n_{i.}=\sum_{i=1}^{n_x}` is the total number of observations that take modality :math:`i` of :math:`x`
 * :math:`n_{.j}=\sum_{j=1}^{n_y}` is the total number of observations that take modality :math:`j` of :math:`y`

.. autofunction:: AutoCarver.feature_selection.measures.chi2_measure



Cramér's :math:`V`
^^^^^^^^^^^^^^^^^^

Based on Pearson's :math:`\chi^2`, Cramér's :math:`V` is computed using the following formula:

.. math::
    
    V=\sqrt{\frac{\chi^2}{n\min(n_x-1, n_y-1)}}

where:

 * :math:`n` is the number of observations
 * :math:`n_x` is the number of modalities of :math:`x`
 * :math:`n_y` is the number of modalities of :math:`y`
 
.. autofunction:: AutoCarver.feature_selection.measures.cramerv_measure

.. note::
    ``cramerv_measure`` is the default measure for qualitative features (i.e. when ``FeatureSelector.measures=[]`` and ``FeatureSelector.qualititative_features`` is provided).



Tschuprow's :math:`T`
^^^^^^^^^^^^^^^^^^^^^



Based on Pearson's :math:`\chi^2`, Tschuprow's :math:`T` is computed using the following formula:

.. math::
    
    T=\sqrt{\frac{\chi^2}{n\sqrt{(n_x-1)(n_y-1)}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_x` is the number of modalities of :math:`x`
 * :math:`n_y` is the number of modalities of :math:`y`
 
.. autofunction:: AutoCarver.feature_selection.measures.tschuprowt_measure






Base data information
.....................

.. autofunction:: AutoCarver.feature_selection.measures.nans_measure

.. note::
    ``nans_measure`` is evaluated by default by ``FeatureSelector``. If threshold is reached, feature will automatically be dropped by ``filters.thresh_filter()``.

.. autofunction:: AutoCarver.feature_selection.measures.dtype_measure
    
.. note::
    ``dtype_measure`` is evaluated by default by ``FeatureSelector``. If threshold is reached, feature will automatically be dropped by ``filters.thresh_filter()``.

.. autofunction:: AutoCarver.feature_selection.measures.mode_measure

.. note::
    ``mode_measure`` is evaluated by default by ``FeatureSelector``. If threshold is reached, feature will automatically be dropped by ``filters.thresh_filter()``.







.. _Filters:

Association filters, X by X 
---------------------------

Quantitative filters
....................



Pearson's :math:`r`
^^^^^^^^^^^^^^^^^^^

For a quantititative feature :math:`x_1`, the association with a quantitative feature :math:`x_2` is computed using `pandas.DataFrame.corr <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html>`_.

Pearson's :math:`r`, as known as the bivariate correlation, is a measure of linear correlation between quantitative features.
It is computed using the following formula:

.. math::

    r_{x_1x_2}= \frac{\sum_{i=1}^{n}{(x_1^i-\bar{x_1})(x_2^i-\bar{x_2})}}{\sqrt{\sum_{i=1}^{n}{(x_1^i-\bar{x_1})^2}}  \sqrt{\sum_{i=1}^{n}{(x_2^i-\bar{x_2})^2}}}

where:

 * :math:`n` is the number of observations
 * :math:`x_1^i` is the :math:`i` th observation of :math:`x_1`
 * :math:`x_2^i` is the :math:`i` th observation of :math:`x_2`
 * :math:`\bar{x_1}=\frac{1}{n}\sum_{i=1}^n{x_1^i}` is the sample mean of :math:`x_1`
 * :math:`\bar{x_2}=\frac{1}{n}\sum_{i=1}^n{x_2^i}` is the sample mean of :math:`x_2`


.. autofunction:: AutoCarver.feature_selection.filters.pearson_filter




Spearman's :math:`\rho`
^^^^^^^^^^^^^^^^^^^^^^^


For a quantitative feature :math:`x`, the corresponding rank feature :math:`x_r` is the sorted sample of :math:`x` such that any :math:`i` in :math:`(1, n-1)` verifies :math:`x_r^i \leq x_r^{i+1}`, where :math:`n` is the number of observations.

Spearman's :math:`\rho` is Pearson's :math:`r` computed on the rank features. As so, Spearman's :math:`\rho` is computed with the following formula:

.. math::

    \rho=r_{x_{1_{r}}x_{2_{r}}}

where:

 * :math:`x_{1_{r}}` is the ranked version of :math:`x_1`
 * :math:`x_{2_{r}}` is the ranked version of :math:`x_2`
 * :math:`r_{x_{1_{r}}x_{2_{r}}}` is Pearson's :math:`r` linear correlation coefficient between :math:`x_{1_{r}}` and :math:`x_{2_{r}}`

.. autofunction:: AutoCarver.feature_selection.filters.spearman_filter
    
.. note::
    ``spearman_filter`` is the default measure for quantitative features (i.e. when ``FeatureSelector.filters=[]`` and ``FeatureSelector.quantititative_features`` is provided).


    






Qualitative filters
...................



Cramér's :math:`V`
^^^^^^^^^^^^^^^^^^

For a qualititative feature :math:`x_1`, the association with a qualitative feature :math:`x_2` is computed based on the `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.

Pearson's :math:`\chi^2` statistics is then computed using `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ to perform association measuring.
The formula is the following:

.. math::

    \chi^2=\sum_{i=1}^{n_{x_1}}{\sum_{j=1}^{n_{x_2}}{\frac{(n_{ij} - \frac{n_{i.}n_{.j}}{n})^2}{\frac{n_{i.}n_{.j}}{n}}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_{x_1}` is the number of modalities of :math:`x_1`
 * :math:`n_{x_2}` is the number of modalities of :math:`x_2`
 * :math:`n_{ij}` is the number of observations that take modality :math:`i` of :math:`x_1` and modality :math:`j` of :math:`x_2`
 * :math:`n_{i.}=\sum_{i=1}^{n_{x_1}}` is the total number of observations that take modality :math:`i` of :math:`x_1`
 * :math:`n_{.j}=\sum_{j=1}^{n_{x_2}}` is the total number of observations that take modality :math:`j` of :math:`x_2`



Based on Pearson's :math:`\chi^2`, Cramér's :math:`V` is computed using the following formula:

.. math::
    
    V=\sqrt{ \frac{ \chi^2 }{ n\min(n_{x_1}-1, n_{x_2}-1) } }

where:

 * :math:`n` is the number of observations
 * :math:`n_{x_1}` is the number of modalities of :math:`x_1`
 * :math:`n_{x_2}` is the number of modalities of :math:`x_2`
 

.. autofunction:: AutoCarver.feature_selection.filters.cramerv_filter
    
.. note::
    ``cramerv_filter`` is the default filter for qualitative features (i.e. when ``FeatureSelector.filters=[]`` and ``FeatureSelector.qualititative_features`` is provided).




Tschuprow's :math:`T`
^^^^^^^^^^^^^^^^^^^^^


Based on Pearson's :math:`\chi^2`, Tschuprow's :math:`T` is computed using the following formula:

.. math::
    
    T=\sqrt{\frac{\chi^2}{n\sqrt{(n_{x_1}-1)(n_{x_2}-1)}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_{x_1}` is the number of modalities of :math:`x_1`
 * :math:`n_{x_2}` is the number of modalities of :math:`x_2`
 

.. autofunction:: AutoCarver.feature_selection.filters.tschuprowt_filter






Other filters
.............

.. autofunction:: AutoCarver.feature_selection.filters.thresh_filter

.. note::
    ``thresh_filter`` is used by default by ``FeatureSelector``. Automatically features that did not pas :ref:`Measures`. 

.. autofunction:: AutoCarver.feature_selection.filters.measure_filter
