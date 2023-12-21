.. _Discretizers:

Discretizers
============

**AutoCarver** implements **Discretizers**. It provides the following Data Preparation tools: 

+------------------------------------+-------------------------------------------------------------------------+
| Discretizer / Data Type            | Data Preparation                                                        |
+====================================+=========================================================================+
| :ref:`ContinuousDiscretizer`:      | Over-represented values are set as there own modality                   |
|                                    |                                                                         |
| Continuous Data                    | Automatic quantile bucketization of under-represented values            |
|                                    |                                                                         |
| Discrete Data                      | Modalities are ordered by default real number ordering                  |
|                                    |                                                                         |
+------------------------------------+-------------------------------------------------------------------------+
| :ref:`OrdinalDiscretizer`:         | Under-represented modalities are grouped with the closest modality      |
|                                    |                                                                         |
| Ordinal Data                       | Modalities are ordered according to provided modality ranking           |
|                                    |                                                                         |
+------------------------------------+-------------------------------------------------------------------------+
| :ref:`CategoricalDiscretizer`:     | Under-represented modalities are grouped into a default value           |
|                                    |                                                                         |
| Categorical Data                   | Modalities are ordered by target rate                                   |
|                                    |                                                                         |
+------------------------------------+-------------------------------------------------------------------------+

.. note::

   * Representativity threshold of modalities is user selected (``min_freq`` attribute).
   * At this step, if any, ``numpy.nan`` are set as there own modality (no given order).
   * Helps improve modality relevancy and reduces the set of possible combinations to test from.
   * Included in all carving pipelines: :class:`BinaryCarver`, :class:`MulticlassCarver`, :class:`ContinuousCarver`.

.. _Discretizer:

Discretizer, a complete discretization pipeline
-----------------------------------------------

.. autoclass:: auto_carver.discretizers.Discretizer
    :members: fit, transform, fit_transform, to_json, summary




Quantitative Data
-----------------


.. _QuantitativeDiscretizer:

Complete pipeline for continuous and discrete features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: auto_carver.discretizers.QuantitativeDiscretizer
    :members: fit, transform, fit_transform, to_json, summary

.. _ContinuousDiscretizer:

Continuous Discretizer
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: auto_carver.discretizers.ContinuousDiscretizer
    :members: fit, transform, fit_transform, to_json, summary



Qualitative Data
----------------

.. _QualitativeDiscretizer:

Complete pipeline for categorical and ordinal features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: auto_carver.discretizers.QualitativeDiscretizer
    :members: fit, transform, fit_transform, to_json, summary


.. _CategoricalDiscretizer:

Categorical Discretizer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: auto_carver.discretizers.CategoricalDiscretizer
    :members: fit, transform, fit_transform, to_json, summary


.. _OrdinalDiscretizer:

Ordinal Discretizer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: auto_carver.discretizers.OrdinalDiscretizer
    :members: fit, transform, fit_transform, to_json, summary


.. _ChainedDiscretizer:

Chained Discretizer
^^^^^^^^^^^^^^^^^^^

:class:`ChainedDiscretizer` can be used prior to using any carving pipeline or any other discretizer to group categorical modalities more intelligently.
By providing a set of modality groups, the user can introduce use case specific knowledge into the discretization process.
The fitted ordering can then be passed as ``values_orders`` parameter for further discretization. 

.. autoclass:: auto_carver.discretizers.ChainedDiscretizer
    :members: fit, transform, fit_transform, to_json, summary


.. _StringDiscretizer:

String Discretizer
^^^^^^^^^^^^^^^^^^

:class:`StringDiscretizer` is used as a data preparation tool to convert qualitative data to ``str`` type.

.. autoclass:: auto_carver.discretizers.StringDiscretizer
    :members: fit, transform, fit_transform, to_json, summary


.. _GroupedList:

GroupedList
-----------

.. note::
    **AutoCarver** would not exist without :class:`GroupedList`. It allows for a complete historization of the data processing steps, thanks to its ``content`` dictionnary attribute.
    All modalities are stored inside the :class:`GroupedList` and can safely be linked to there respective group label. 

.. autoclass:: auto_carver.discretizers.GroupedList
    :members:



Saving and Loading
------------------

.. autofunction:: auto_carver.discretizers.BaseDiscretizer.to_json

.. autofunction:: auto_carver.discretizers.load_discretizer
