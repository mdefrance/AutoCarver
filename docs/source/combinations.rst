.. _Combinations:


Combinations
============

**Combinations** are at the core of **Carvers**. They are used to identify the best combination
from all possible combinations with up to :attr:`max_n_mod` modalities.

A pre-built :class:`CombinationEvaluator` instance can be passed to any carver via the
``combination_evaluator`` keyword. Each subclass defaults to a task-appropriate metric:
:class:`TschuprowtCombinations` for :class:`BinaryCarver` / :class:`MulticlassCarver`,
:class:`KruskalCombinations` for :class:`ContinuousCarver`.

.. autoclass:: AutoCarver.combinations.CombinationEvaluator
    :members: get_best_combination, save, load


Classification tasks
--------------------

.. _CramervCombinations:

Cramér's V Combinations
^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`Cramerv` for more details on the metric.

.. autoclass:: AutoCarver.combinations.CramervCombinations
    :members: save, load

.. _TschuprowtCombinations:

Tschuprow's T Combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`Tschuprowt` for more details on the metric.

.. autoclass:: AutoCarver.combinations.TschuprowtCombinations
    :members: save, load

Regression tasks
----------------

.. _KruskalCombinations:

Kruskal's H Combinations
^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`kruskal` for more details on the metric.

.. autoclass:: AutoCarver.combinations.KruskalCombinations
    :members: save, load
