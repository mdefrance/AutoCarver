Discretizers
============



.. _BaseDiscretizer:

Base Discretizer
----------------

.. autoclass:: AutoCarver.discretizers.BaseDiscretizer
    :members:


.. note::
    All implemented Discretizers (and even ``AutoCarver``) inherit from :ref:`BaseDiscretizer`, and thus inherit its methods ``transform()``, ``fit_transform()``, ``to_json()`` and ``summary()``.

.. _Discretizer:

Discretizer, a complete discretization pipeline
-----------------------------------------------

.. autoclass:: AutoCarver.discretizers.Discretizer
    :members:


Quantitative Discretizers
-------------------------


.. _QuantitativeDiscretizer:

Complete pipeline for continuous and discrete features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.QuantitativeDiscretizer
    :members:

.. _QuantileDiscretizer:

Quantile Discretizer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.QuantileDiscretizer
    :members:

Qualitative Discretizers
------------------------

.. _QualitativeDiscretizer:

Complete pipeline for categorical and ordinal features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.QualitativeDiscretizer
    :members:


.. _DefaultDiscretizer:

Default Discretizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.DefaultDiscretizer
    :members:


.. _OrdinalDiscretizer:

Ordinal Discretizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.OrdinalDiscretizer
    :members:


.. _ChainedDiscretizer:

Chained Discretizer
^^^^^^^^^^^^^^^^^^^

This Discretizer can be used prior to using ``AutoCarver`` or ``Discretizer`` to simplify modalities more intelligently.
The defined ordering can then be passed into the ``values_orders`` parameter for further discretization. 

.. autoclass:: AutoCarver.discretizers.ChainedDiscretizer
    :members:

.. _StringDiscretizer:

String Discretizer
------------------

.. autoclass:: AutoCarver.discretizers.StringDiscretizer
    :members:


.. _GroupedList:

GroupedList
-----------

.. note::
    :ref:`AutoCarver` would not exist if it was not for ``GroupedList``. It allows for a complete historization of the data processing steps, thanks to its ``GroupedList.content`` dictionnary.
    All modalities are stored inside the ``GroupedList`` and can safely be linked to there respective group label. 

.. autoclass:: AutoCarver.discretizers.GroupedList
    :members:
