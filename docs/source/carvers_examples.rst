Carvers Examples
================

These examples explain how to use task specific **Carvers**.

Binary Classification Example
-----------------------------

Example of the Titanic survival classification.

.. toctree::
    :glob:
    :maxdepth: 1

    examples/Carvers/BinaryClassification/binary_classification_example

One-vs-Rest Classification Example
-----------------------------------

Example of the Iris type classification using :class:`OneVsRestCarver` (a
separate binning per class). For the joint :class:`MulticlassCarver` — one
binning per feature against the full K-class target — see :ref:`MulticlassCarver`.

.. toctree::
    :glob:
    :maxdepth: 1

    examples/Carvers/MulticlassClassification/multiclass_classification_example

Continuous Regression Example
-----------------------------

Example of the California House Prices regression.

.. toctree::
    :glob:
    :maxdepth: 1

    examples/Carvers/ContinuousRegression/continuous_regression_example
