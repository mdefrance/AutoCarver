.. _features:

Features
========

The `AutoCarver.features` module defines a set of features used in the AutoCarver project. This module includes classes and functions to handle different types of features, such as qualitative and quantitative features.

Features
--------

.. autoclass:: AutoCarver.features.Features
    :members: from_list, qualitatives, categoricals, ordinals, quantitatives, names, versions, summary, history, load, to_json

.. note::

    Use the default constructor when you only have column names; use
    :meth:`Features.from_list` to wrap already-instantiated feature objects.


FeaturesConfig
^^^^^^^^^^^^^^

Collection-level state propagated to every feature in a :class:`Features`. Internal
feature attributes (``nan``, ``default``, ``ordinal_encoding``, ``has_nan``,
``has_default``, ``dropna``, ``is_fitted``) are not part of the public
:class:`BaseFeature` constructor — set them via :class:`FeaturesConfig` and pass
the instance to :class:`Features` or :meth:`Features.from_list`.

.. autoclass:: AutoCarver.features.FeaturesConfig

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