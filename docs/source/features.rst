.. _features:

Features
========

The `AutoCarver.features` module defines a set of features used in the AutoCarver project. This module includes classes and functions to handle different types of features, such as qualitative and quantitative features.

Features
--------

.. autoclass:: AutoCarver.features.Features
    :members: qualitatives, categoricals, ordinals, quantitatives, names, versions, summary, history, load, to_json

Qualitatitve features
---------------------

.. autoclass:: AutoCarver.features.CategoricalFeature
    :members: is_qualitative, is_categorical, is_ordinal, has_nan, has_default, summary, history

.. autoclass:: AutoCarver.features.OrdinalFeature
    :members: is_qualitative, is_categorical, is_ordinal, has_nan, has_default, summary, history


Quantitative features
---------------------

.. autoclass:: AutoCarver.features.QuantitativeFeature
    :members: is_quantitative, has_nan, has_default, summary, history
