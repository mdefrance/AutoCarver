Discretizers
============

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


.. _BaseDiscretizer:

Base Discretizer
----------------

All ``Discretizer`` objects (and even ``AutoCarver``) inherit from the ``BaseDiscretizer``.

.. autoclass:: AutoCarver.discretizers.BaseDiscretizer
    :members:


.. _StringDiscretizer:

String Discretizer
------------------

.. autoclass:: AutoCarver.discretizers.StringDiscretizer
    :members:


.. _GroupedList:

GroupedList
-----------

.. autoclass:: AutoCarver.discretizers.GroupedList
    :members:
