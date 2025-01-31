.. _Selectors:

Selectors
=========

**AutoCarver** implements **Selectors**, they provide the following, association-centric, Data Selection steps: 

   1. Measuring association with a target and ranking features accordingly.
   2. Filtering out features too asociated to a better ranked feature.

It allows one to select features:

    * Whatever there type: quantitative or qualitative
    * Whatever the optimization task: :ref:`ClassificationSelector` or :ref:`RegressionSelector`

By default, quantitative features are:

 * Ranked according to :ref:`kruskal`
 * Filtered according to :ref:`Spearman's rho correlation coefficient <spearman_filter>`

By default, qualitative features are:

 * Ranked according to :ref:`tschuprowt`
 * Filtered according to :ref:`tschuprowt_filter`

In general, associations are computed according to the provided data types of :math:`x` and :math:`y`:

+-----------------------+---------------------------------------------------------------------+--------------------------------------------------+
| :math:`x` \\ :math:`y`| Qualitatitve                                                        | Quantitative                                     |
+-----------------------+---------------------------------------------------------------------+--------------------------------------------------+
| Qualitative           | Pearson's :math:`\chi^2`, Cramér's :math:`V`, Tschuprow's :math:`T` | Kruskal-Wallis' :math:`H`, :math:`R` coefficient |
+-----------------------+---------------------------------------------------------------------+--------------------------------------------------+
| Quantitative          | Kruskal-Wallis' :math:`H`, :math:`R` coefficient                    | Pearson's :math:`r`, Spearman's :math:`\rho`     |
+-----------------------+---------------------------------------------------------------------+--------------------------------------------------+

See :ref:`Measures` and :ref:`Filters`, for details on measures and filters' implementation.


.. _ClassificationSelector:

Classification tasks
--------------------


.. autoclass:: AutoCarver.selectors.ClassificationSelector
    :members: select


.. _RegressionSelector:

Regression tasks
----------------


.. autoclass:: AutoCarver.selectors.RegressionSelector
    :members: select






.. _Measures:

Association measures, X by y 
----------------------------



Quantitative measures
.....................


.. _distance:

Distance Correlation
^^^^^^^^^^^^^^^^^^^^


For two **quantitative** features :math:`x` and :math:`y`, the Distance Correlation can be computed using the following formula:

.. math::

    1 - \frac{ (x - \bar{x}) (y - \bar{y}) } { ||x - \bar{x}||_2 ||y - \bar{y}||_2  }


where:

 * :math:`n_x` is the number of observations of :math:`x`
 * :math:`n_y` is the number of observations of :math:`y`
 * :math:`\bar{y}=\sum_{i=1}^{n_y}{y_{i}}` is the sample mean of :math:`y`
 * :math:`\bar{x}=\sum_{i=1}^{n_x}{x_{i}}` is the sample mean of :math:`x`
 * :math:`||x - \bar{x}||_2 = \sqrt{ \sum_{i=1}^{n_x}{  (x_i - \bar{x})^2 } }` is the euclidean norm of :math:`x`
 * :math:`||y - \bar{y}||_2 = \sqrt{ \sum_{i=1}^{n_x}{  (y_i - \bar{y})^2 } }` is the euclidean norm of :math:`y`




The Distance Correlation is computed using `scipy.spatial.distance.correlation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html>`_.



.. autoclass:: AutoCarver.selectors.measures.DistanceMeasure
    :members: compute_association, validate



.. note::

    * ``distance_measure`` is the default measure for quantitative features in regression tasks (i.e. when ``RegressionSelector.quantitative_filters=None`` and ``RegressionSelector.quantitative_features`` is provided).
    * If ``thresh_distance`` is reached, feature will automatically be dropped.



.. _kruskal:

Kruskal-Wallis' :math:`H` test statistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


For a **quantitative** feature :math:`x`, the corresponding order feature :math:`x_o` is the sorted sample of :math:`x` such that any :math:`i` in :math:`(1, n-1)` verifies :math:`x_o^i \leq x_o^{i+1}`, where :math:`n` is the number of observations. For the same feature :math:`x`, the corresponding rank :math:`x_r` is the index of :math:`x`'s values in :math:`x_o`.


The association with a **qualitative** target :math:`y` is computed using `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_.

Kruskal-Wallis' :math:`H` test statistic, as known as one-way ANOVA on ranks, allows one to check that two samples originate from the same distribution.
It is used to determine whether or not :math:`x` is distributed the same when :math:`y=y_0` to :math:`y=y_{n_y-1}` where :math:`n_y` is the number of modalities taken by :math:`y`.
It is computed using the following formula:


.. math::

    H = (n-1) \frac{ \sum_{i=1}^{n_y}{ n_{y=i} (\bar{x_r^{i.}} - \bar{x_r})^2 } } { \sum_{i=1}^{n_y}{ \sum_{j=1}^{n_{y=i}}{ (x_r^{ij} - \bar{x_r})^2 } } }



where:

 * :math:`n` is the number of observations
 * :math:`n_y` is the number of modalities of :math:`y`
 * :math:`n_{y=i}` is the number of observations taking :math:`y`'s :math:`i` th modality
 * :math:`x_r` is the ranked version of :math:`x`
 * :math:`x_r^{ij}` is the :math:`j` th observation of :math:`x_r` when :math:`y` takes its :math:`i` th modality
 * :math:`\bar{x_r^{i.}}=\sum_{j=1}^{n_{y=i}}x_r^{ij}` is the sample mean of :math:`x_r` when :math:`y` takes its :math:`i` th modality
 * :math:`\bar{x_r}=\sum_{i=1}^{n_y}{\sum_{j=1}^{n_{y=i}}}x_r^{ij}` is the sample mean of :math:`x_r`


.. autoclass:: AutoCarver.selectors.measures.KruskalMeasure
    :members: compute_association, validate


.. note::

    * ``kruskal_measure`` is the default measure for quantitative features in classification tasks (i.e. when ``ClassificationSelector.quantitative_filters=None`` and ``ClassificationSelector.quantitative_features`` is provided).
    * ``kruskal_measure`` is the default measure for qualitative features in regression tasks (i.e. when ``RegressionSelector.qualitative_filters=None`` and ``RegressionSelector.qualitative_features`` is provided).
    * If ``thresh_kruskal`` is reached, feature will automatically be dropped.


    

.. _R:


Coefficient of determination :math:`R`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
For a **binary** feature :math:`y` and a **quantitative** feature :math:`x` the following linear regression model is fitted using `statsmodels.formula.api.ols <https://www.statsmodels.org/dev/generated/statsmodels.formula.api.ols.html>`_:

.. math::

    x = \alpha + \beta y + \epsilon

where:
 * :math:`\alpha` and :math:`\beta` are the coefficient of the linear regression model
 * :math:`\epsilon` is the residual of the linear regression model 

The determination coefficient, often denoted as :math:`R^2`, is a statistical measure that quantifies the goodness of fit of a linear regression model.
In this specific case, it is equal to the square of Pearson's :math:`r` correlation coefficient between :math:`x` and :math:`y`.
It is computed with the following formula:

.. math::

    R = \sqrt{ 1 - \frac{ SS_{res} }{ SS_{tot} } }


where:

 * :math:`n` is the number of observations
 * :math:`SS_{res} = \sum_{i=1}^n{(x_i - \alpha - \beta y_i)^2} = \sum_{i=1}^n{\epsilon_i^2}` is the residual sum of squares
 * :math:`SS_{tot} = \sum_{i=1}^n{(x_i - \bar{x})^2}` is the total sum of squares
 * :math:`\bar{x}=\sum_{i=1}^{n}{x_i}` is the sample mean of :math:`x`




.. autoclass:: AutoCarver.selectors.measures.RMeasure
    :members: compute_association, validate


.. note::

    If ``thresh_R`` is reached, feature will automatically be dropped.




.. _zscore:

Outlier Detection: Standard score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard score can be applied as a measure of deviation to determine outlier for **quantitative** features.
For a feature :math:`x` it is computed for any oservation :math:`x_i` as follows:

.. math::

    z_i = \frac{x_i - \bar{x}}{S}

where:

 * :math:`n` is the number of observations
 * :math:`\bar{x}=\frac{1}{n}\sum_{j=1}^n{x_j}` is the sample mean of :math:`x`
 * :math:`S=\sqrt{\frac{1}{n-1}\sum_{j=1}^n{(x_j - \bar{x})^2}}` is the sample standard deviation of :math:`x`


.. autoclass:: AutoCarver.selectors.measures.ZscoreOutlierMeasure
    :members: compute_association, validate


.. note::

    If ``thresh_zscore`` is reached, feature will automatically be dropped.



.. _iqr:

Outlier Detection: Interquartile range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interquartile range is widely used as an outlier detection metric for **quantitative** features.
For a feature :math:`x` it is computed as follows:

.. math::

    IQR = Q_3 - Q_1

where:

 * :math:`Q_1` is the 25th percentile of the :math:`x`
 * :math:`Q_3` is the 75th percentile of the :math:`x`


Any observation :math:`x_i` of feature :math:`x`, can be considered an outlier if it does not verify:

.. math::

    Q1 - 1.5 IQR \leq x_i \leq Q3 + 1.5 IQR


.. autoclass:: AutoCarver.selectors.measures.IqrOutlierMeasure
    :members: compute_association, validate


.. note::

    If ``thresh_iqr`` is reached, feature will automatically be dropped.






Qualitative measures
....................


.. _chi2:

Pearson's :math:`\chi^2` test statistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a **qualititative** feature :math:`x`, the association with a **qualitative** target :math:`y` is computed based on the `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.

Pearson's :math:`\chi^2` test statistic is then computed using `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ to perform association measuring.
The formula is the following:

.. math::

    \chi^2=\sum_{i=1}^{n_x}{\sum_{j=1}^{n_y}{\frac{(n_{ij} - \frac{n_{i.}n_{.j}}{n})^2}{\frac{n_{i.}n_{.j}}{n}}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_x` is the number of modalities of :math:`x`
 * :math:`n_y` is the number of modalities of :math:`y`
 * :math:`n_{ij}` is the number of observations that take modality :math:`i` of :math:`x` and modality :math:`j` of :math:`y`
 * :math:`n_{i.}=\sum_{i=1}^{n_x}n_{ij}` is the total number of observations that take modality :math:`i` of :math:`x`
 * :math:`n_{.j}=\sum_{j=1}^{n_y}n_{ij}` is the total number of observations that take modality :math:`j` of :math:`y`

.. autoclass:: AutoCarver.selectors.measures.Chi2Measure
    :members: compute_association, validate

.. note::

    If ``thresh_chi2`` is reached, feature will automatically be dropped.


.. _Cramerv:

Cramér's :math:`V`
^^^^^^^^^^^^^^^^^^

Based on Pearson's :math:`\chi^2`, Cramér's :math:`V` is computed using the following formula:

.. math::
    
    V=\sqrt{\frac{\chi^2}{n\min(n_x-1, n_y-1)}}

where:

 * :math:`n` is the number of observations
 * :math:`n_x` is the number of modalities of :math:`x`
 * :math:`n_y` is the number of modalities of :math:`y`
 
.. autoclass:: AutoCarver.selectors.measures.CramervMeasure
    :members: compute_association, validate

.. note::

    If ``thresh_cramerv`` is reached, feature will automatically be dropped.


.. _Tschuprowt:

Tschuprow's :math:`T`
^^^^^^^^^^^^^^^^^^^^^



Based on Pearson's :math:`\chi^2`, Tschuprow's :math:`T` is computed using the following formula:

.. math::
    
    T=\sqrt{\frac{\chi^2}{n\sqrt{(n_x-1)(n_y-1)}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_x` is the number of modalities of :math:`x`
 * :math:`n_y` is the number of modalities of :math:`y`
 
.. autoclass:: AutoCarver.selectors.measures.TschuprowtMeasure
    :members: compute_association, validate

   
.. note::

    * :class:`TschuprowtMeasure` is the default measure for qualitative features in classification tasks (i.e. when ``ClassificationSelector.qualitative_filters=None`` and ``ClassificationSelector.qualitative_features`` is provided).
    * If :attr:`TschuprowtMeasure.threshold` is reached, feature will automatically be dropped.







.. _Filters:

Association filters, X by X 
---------------------------

Quantitative filters
....................


.. _pearson_filter:

Pearson's :math:`r`
^^^^^^^^^^^^^^^^^^^

For a **quantititative** feature :math:`x_1`, the association with a **quantitative** feature :math:`x_2` is computed using `pandas.DataFrame.corr <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html>`_.

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


.. autofunction:: AutoCarver.selectors.filters.pearson_filter


.. autoclass:: AutoCarver.selectors.filters.PearsonFilter
    :members: filter



.. note::

    If ``thresh_corr`` is reached, feature will automatically be dropped.



.. _spearman_filter:

Spearman's :math:`\rho`
^^^^^^^^^^^^^^^^^^^^^^^


For a **quantitative** feature :math:`x`, the corresponding order feature :math:`x_o` is the sorted sample of :math:`x` such that any :math:`i` in :math:`(1, n-1)` verifies :math:`x_o^i \leq x_o^{i+1}`, where :math:`n` is the number of observations. For the same feature :math:`x`, the corresponding rank :math:`x_r` is the index of :math:`x`'s values in :math:`x_o`.

Spearman's :math:`\rho` is Pearson's :math:`r` computed on the rank features. As so, Spearman's :math:`\rho` is computed with the following formula:

.. math::

    \rho=r_{x_{1_{r}}x_{2_{r}}}

where:

 * :math:`x_{1_{r}}` is the ranked version of :math:`x_1`
 * :math:`x_{2_{r}}` is the ranked version of :math:`x_2`
 * :math:`r_{x_{1_{r}}x_{2_{r}}}` is Pearson's :math:`r` linear correlation coefficient between :math:`x_{1_{r}}` and :math:`x_{2_{r}}`


.. autoclass:: AutoCarver.selectors.filters.SpearmanFilter
    :members: filter

.. note::

    * ``spearman_filter`` is the default filter for quantitative features (i.e. when ``quantitative_filters=None`` and ``quantitative_features`` is provided).
    * If ``thresh_corr`` is reached, feature will automatically be dropped.








Qualitative filters
...................


.. _cramerv_filter:

Cramér's :math:`V`
^^^^^^^^^^^^^^^^^^

For a **qualititative** feature :math:`x_1`, the association with a **qualitative** feature :math:`x_2` is computed based on the `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.

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
 

.. autoclass:: AutoCarver.selectors.filters.CramervFilter
    :members: filter

.. note::
    
    If ``thresh_corr`` is reached, feature will automatically be dropped.



.. _tschuprowt_filter:

Tschuprow's :math:`T`
^^^^^^^^^^^^^^^^^^^^^


Based on Pearson's :math:`\chi^2`, Tschuprow's :math:`T` is computed using the following formula:

.. math::
    
    T=\sqrt{\frac{\chi^2}{n\sqrt{(n_{x_1}-1)(n_{x_2}-1)}}}

where:

 * :math:`n` is the number of observations
 * :math:`n_{x_1}` is the number of modalities of :math:`x_1`
 * :math:`n_{x_2}` is the number of modalities of :math:`x_2`
 


.. autoclass:: AutoCarver.selectors.filters.TschuprowtFilter
    :members: filter

.. note::

    * ``tschuprowt_filter`` is the default filter for qualitative features (i.e. when ``qualitative_filters=None`` and ``qualititative_features`` is provided).
    * If ``thresh_corr`` is reached, feature will automatically be dropped.

