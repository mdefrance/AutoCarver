About
=====

.. warning::
   Work in progress! Sorry for the potential errors and lack of documentation.

Why AutoCarver?
---------------

**AutoCarver** is a powerful Python package designed to address the fundamental question of *What's the best processing for my model's features?*

It offers an automated and optimized approach to processing and engineering your data, resulting in improved model performance, enhanced explainability, and reduced feature dimensionality.
As of today, this set of tools is available for binary classification problems only.

Key Features:

1. **Data Processing and Engineering**: **AutoCarver** performs automatic bucketization and carving of a DataFrame's columns to maximize their correlation with a binary target variable. By leveraging advanced techniques, it optimizes the preprocessing steps for your data, leading to enhanced predictive accuracy.

2. **Improved Model Explainability**: **AutoCarver** aids in understanding the relationship between the processed features and the target variable. By uncovering meaningful patterns and interactions, it provides valuable insights into the underlying data dynamics, enhancing the interpretability of your binary classification models.

3. **Reduced Feature Dimensionality**: **AutoCarver** excels at reducing feature dimensionality, especially in scenarios involving one-hot encoding. It identifies and preserves only the most statistically relevant modalities, ensuring that your models focus on the most informative aspects of the data while eliminating noise and redundancy.

4. **Statistical Accuracy and Relevance**: **AutoCarver** incorporates statistical techniques to ensure that the selected modalities have a sufficient number of observations, minimizing the risk of drawing conclusions based on insufficient data. This helps maintain the reliability and validity of your binary classification models.

5. **Robustness Testing**: **AutoCarver** goes beyond feature processing by assessing the robustness of the selected modalities. It performs tests to evaluate the stability and consistency of the chosen features across different datasets or subsets, ensuring their reliability in various scenarios.

**AutoCarver** is a valuable tool for data scientists and practitioners involved in binary classification problems, such as credit scoring, fraud detection, and risk assessment. By leveraging its automated feature processing capabilities, you can unlock the full potential of your data, leading to more accurate predictions, improved model explainability, and better decision-making in your classification tasks.

Under the hood feature overview
-------------------------------

**AutoCarver** is a two step pipeline. 

I. Data Preparation: conversion to ordinal data buckets
.......................................................

**AutoCarver** implements :ref:`Discretizer`. It provides the following Data Preparation tools: 

+------------------------------------+-------------------------------------------------------------------------+
| Discretizer                        | Data Processing                                                         |
+====================================+=========================================================================+
| :ref:`QuantileDiscretizer`:        | Over-represented values are set as there own modality                   |
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
| :ref:`DefaultDiscretizer`:         | Under-represented modalities are grouped into a default value           |
|                                    |                                                                         |
| Categorical Data                   | Modalities are ordered by target rate                                   |
|                                    |                                                                         |
+------------------------------------+-------------------------------------------------------------------------+

.. note::

   * Representativity threshold of modalities is user selected (``min_freq`` attribute).
   * At this step, if any, NaNs are set as there own modality (no given order).
   * Helps improve modality relevancy and reduces the set of possible combinations to test from.
   * These steps are all included in the ``AutoCarver`` pipeline.

II. Data Optimization: maximization of bucket association
.........................................................

The core of **AutoCarver** is :ref:`AutoCarver`. It consists of the following Data Optimization steps: 

   1. Identifying the most associated combination from all ordered combinations of modalities.
   2. Testing all combinations of NaNs grouped to one of those modalities.

.. note::

   * The user chooses the maximum number of modality per feature (``max_n_mod`` attribute).
   * The user chooses whether or not to group NaNs to other values (``dropna`` attribute).

III. (Optional) Data Selection: model feature pre-selection
...........................................................

**AutoCarver** implements :ref:`FeatureSelector`. It consists of the following Data Selection steps: 

   1. Measuring association with a binary target and ranking features accordingly.
   2. Filtering out features too asociated to a better ranked feature.

.. note::

   * The user defines the inter-feature correlation thresholds.
   * This step is available for both qualitative and quantitative features (discretized or not).
   * See :ref:`Measures` and :ref:`Filters`.


Performances
------------

Execution time has been measured for several values of key paramaters of **AutoCarver**


.. csv-table::
   :header: min_freq, max_n_mod, X.shape[0], len(features), Execution Time


   0.01, 5, 100000000, 15, 0.001
   0.02, 4, 100000000, 15, 0.001
