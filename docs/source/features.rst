.. _features:

Features
========

The `AutoCarver.features` module defines a set of features used in the AutoCarver project. This module includes classes and functions to handle different types of features, such as qualitative and quantitative features.

Features
--------

.. autoclass:: AutoCarver.features.Features
    :members: from_list, qualitatives, categoricals, ordinals, quantitatives, datetimes, names, versions, summary, history, load, to_json

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
------------------

A :class:`NumericalFeature` is the concrete numeric feature type. Declare them from the
:class:`Features` constructor via the ``numericals`` argument. :class:`QuantitativeFeature`
remains the abstract umbrella shared by numericals and datetimes.

.. autoclass:: AutoCarver.features.NumericalFeature
    :members: is_quantitative, has_nan, has_default, summary, history

.. autoclass:: AutoCarver.features.QuantitativeFeature
    :members: is_quantitative, has_nan, has_default, summary, history


Datetime features
-----------------

A :class:`DatetimeFeature` is a quantitative feature backed by a datetime column. It is
discretized as the number of seconds elapsed since a user-provided ``reference_date``
(see :meth:`DatetimeFeature.to_timedelta`), after which it behaves exactly like any other
quantitative feature (quantile bucketization, carving, ...).

``reference_date`` may be **either** a fixed date literal **or** the name of another
datetime column in ``X``. The two are disambiguated at fit time: if ``reference_date``
matches a column of the fitted ``X``, the elapsed seconds are computed row-wise against
that column; otherwise it is parsed as a fixed date. A row whose reference column value is
missing (``NaT``) yields ``NaN``.

Datetimes can be declared from the :class:`Features` constructor as
``(column name, reference_date)`` pairs::

    from AutoCarver.features import Features

    features = Features(
        numericals=["age"],
        datetimes=[
            ("signup_date", "2020-01-01"),   # seconds since a fixed date
            ("churn_date", "signup_date"),   # seconds since another column
        ],
    )

They are tracked under :attr:`Features.datetimes` and are also part of
:attr:`Features.quantitatives` (so the quantitative pipeline processes them transparently).
The datetime-to-seconds conversion is performed by the :ref:`TimedeltaDiscretizer`.

.. autoclass:: AutoCarver.features.DatetimeFeature
    :members: is_quantitative, is_datetime, to_timedelta, has_nan, summary, history