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

Under the hood feature overlook
-------------------------------

**AutoCarver** is a two step pipeline. 

I. Data Preparation: conversion to ordinal data buckets
.......................................................

**AutoCarver** implements :ref:`Discretizer`. It provides the following Data Preparation tools: 

.. csv-table::
   :header:

   Discretizer, Data type, Processing 

   :ref:`QuantileDiscretizer`, Quantitative discrete/continuous, 1. Over-represented values are set as there own modality. 2. Automatic quantile bucketization of under-represented values. 3. Modalities are ordered by default real number ordering.
   :ref:`OrdinalDiscretizer`, Qualitative ordinal, 1. Under-represented modalities are grouped to the closest modality (according to provided ranking, by target rate). 2. Modalities are ordered according to provided modality ranking.
   :ref:`DefaultDiscretizer`, Qualitative categorical, 1. Under-represented modalities are grouped into a default value 2. Modalities are ordered by target rate.

.. note::

   * Values are considered over-represented or under-represented based on the cross package, user defined, attribute ``min_freq``.
   * At this step, if any, NaNs are set as set as there own, non-ordered, modality.
   * This step greatly reduces the number of possible combinations between feature modalities and enhances modalities' relevancy.
   * These steps are all included in the ``AutoCarver`` pipeline.

II. Data Optimization: bucket association maximization
......................................................

The core of **AutoCarver** is :ref:`AutoCarver`. It consists of the following Data Optimization steps: 

   1. Identifying the most associated combination from all ordered combinations of non-NaN modalities.
   2. Testing all combinations of NaNs grouped to one of those modalities.

.. note::

   * The user chooses the maximum number of modality per feature (``max_n_mod`` attribute).
   * The user chooses whether or not to group NaNs to other values (``dropna`` attribute).

III. (Optional) Data Selection: model feature pre-selection
...........................................................

**AutoCarver** implements :ref:`FeatureSelector`. It consists of the following Data Selection steps: 

   1. Measuring association with binary target and ranking of features accordingly.
   2. Filtering out features too asociated to a better ranked feature.

.. note::

   * The user defines the inter-feature correlation thresholds.
   * This step is available for both qualitative and quantitative features (discretized or not).
   * See :ref:`Measures` and :ref:`Filters`.


Performances
------------

Execution time has been measured for several values of the key paramaters of **AutoCarver**


.. csv-table::
   :header: min_freq, max_n_mod, X.shape[0], len(features), Execution Time


   0.01, 5, 100000000, 15, 0.001
   0.02, 4, 100000000, 15, 0.001
