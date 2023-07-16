FeatureSelector
===============

.. _FeatureSelector:

FeatureSelector, an association centric tool for feature pre-selection
----------------------------------------------------------------------


.. autoclass:: AutoCarver.feature_selection.FeatureSelector
    :members:

.. _Measures:

Association measures, X by y 
----------------------------

Quantitative measures
.....................

.. autofunction:: AutoCarver.feature_selection.measures.kruskal_measure

.. note::
    ``kruskal_measure`` is the default measure for quantitative features (i.e. when ``FeatureSelector.measures=[]`` and ``FeatureSelector.quantitative_features`` is provided).

.. autofunction:: AutoCarver.feature_selection.measures.R_measure

Quantitative outlier Detection
..............................

.. autofunction:: AutoCarver.feature_selection.measures.zscore_measure

.. autofunction:: AutoCarver.feature_selection.measures.iqr_measure


Qualitative measures
....................

.. autofunction:: AutoCarver.feature_selection.measures.chi2_measure

.. autofunction:: AutoCarver.feature_selection.measures.cramerv_measure

.. note::
    ``cramerv_measure`` is the default measure for qualitative features (i.e. when ``FeatureSelector.measures=[]`` and ``FeatureSelector.qualititative_features`` is provided).

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

.. autofunction:: AutoCarver.feature_selection.filters.spearman_filter
    
.. note::
    ``spearman_filter`` is the default measure for quantitative features (i.e. when ``FeatureSelector.filters=[]`` and ``FeatureSelector.quantititative_features`` is provided).

.. autofunction:: AutoCarver.feature_selection.filters.pearson_filter
.. autofunction:: AutoCarver.feature_selection.filters.vif_filter
    
Qualitative filters
...................

.. autofunction:: AutoCarver.feature_selection.filters.cramerv_filter
    
.. note::
    ``cramerv_filter`` is the default filter for qualitative features (i.e. when ``FeatureSelector.filters=[]`` and ``FeatureSelector.qualititative_features`` is provided).

.. autofunction:: AutoCarver.feature_selection.filters.tschuprowt_filter

Other filters
.............

.. autofunction:: AutoCarver.feature_selection.filters.thresh_filter

.. note::
    ``thresh_filter`` is used by default by ``FeatureSelector``. Automatically features that did not pas :ref:`Measures`. 

.. autofunction:: AutoCarver.feature_selection.filters.measure_filter
