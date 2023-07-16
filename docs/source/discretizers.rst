Discretizers
============

Discretizer, a complete discretization pipeline
-----------------------------------------------

.. autoclass:: AutoCarver.discretizers.Discretizer
    :members:


Quantitative Discretizers
-------------------------

Complete pipeline for continuous and discrete features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.QuantitativeDiscretizer
    :members:

Quantile Discretizer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.QuantileDiscretizer
    :members:

Qualitative Discretizers
------------------------

Complete pipeline for categorical and ordinal features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.QualitativeDiscretizer
    :members:

Default Discretizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.DefaultDiscretizer
    :members:

Ordinal Discretizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: AutoCarver.discretizers.OrdinalDiscretizer
    :members:

Chained Discretizer
^^^^^^^^^^^^^^^^^^^

This Discretizer can be used prior to using ``AutoCarver`` or ``Discretizer`` to simplify modalities more intelligently.
The defined ordering can then be passed into the ``values_orders`` parameter for further discretization. 

.. autoclass:: AutoCarver.discretizers.ChainedDiscretizer
    :members:

Base Discretizer
----------------

All ``Discretizer`` objects (and even ``AutoCarver``) inherit from the ``BaseDiscretizer``.

.. autoclass:: AutoCarver.discretizers.BaseDiscretizer
    :members:


String Discretizer
------------------

.. autoclass:: AutoCarver.discretizers.StringDiscretizer
    :members:


GroupedList
-----------

.. autoclass:: AutoCarver.discretizers.GroupedList
    :members:
