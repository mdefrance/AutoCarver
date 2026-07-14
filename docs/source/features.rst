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


Auditing and adjusting carved bins
-----------------------------------

Carving is transparent: inspect what was decided, then override it if domain
knowledge says otherwise. Every manual edit below is applied by ``transform``
exactly like a carved bin, and per-bin statistics are kept consistent: merged
bins get exactly aggregated counts and rates, while bins whose content can no
longer be derived from the fit (after a move or split) show ``NaN`` until refit.

.. code-block:: python

    binary_carver.fit(train_set, train_set[target], X_dev=dev_set, y_dev=dev_set[target])

    feature = features("ordinal1")
    print(feature.summary)       # per-bin frequencies and target rates
    print(feature.history)       # every combination tried, with viability verdicts

    # merge two bins you consider equivalent
    feature.group(["low", "medium"], "medium")

    # move a single modality into another bin, or give it its own bin
    feature.move("high", "low to medium")  # raw modality, target bin label
    feature.ungroup("high")

    x_discretized = binary_carver.transform(train_set)

Quantitative bins are intervals, so their editing verbs differ: split a bin at
a value of your choosing, or shift the boundary between two adjacent bins.

.. code-block:: python

    feature = features("numerical1")
    print(feature.summary)

    feature.split("(-inf, 2.50e+01]", at=10.0)       # one bin becomes two
    feature.set_boundary("(-inf, 1.00e+01]", at=15.0)  # boundary 10.0 -> 15.0

    x_discretized = binary_carver.transform(train_set)

Ordinal features enforce their declared order: a ``move``/``ungroup`` that would
leave a bin non-contiguous in the original ordering raises a ``ValueError``
(labels always render as honest ``"first to last"`` ranges).

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