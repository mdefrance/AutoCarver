About
=====

.. warning::
   Work in progress! Sorry for the potential errors and lack of documentation.

Why AutoCarver?
---------------

**AutoCarver** is a powerful Python package designed to address the fundamental question  What's the best processing for my model's features?

It offers an automated and optimized approach to processing and engineering your data, resulting in improved model performance, enhanced explainability, and reduced feature dimensionality.
As of today, this set of tools is available for binary classification problems only.

Key Features:

1. **Data Processing and Engineering**: ``AutoCarver`` performs automatic bucketization and carving of a DataFrame's columns to maximize their correlation with a binary target variable. By leveraging advanced techniques, it optimizes the preprocessing steps for your data, leading to enhanced predictive accuracy.

2. **Improved Model Explainability**: ``AutoCarver`` aids in understanding the relationship between the processed features and the target variable. By uncovering meaningful patterns and interactions, it provides valuable insights into the underlying data dynamics, enhancing the interpretability of your binary classification models.

3. **Reduced Feature Dimensionality**: ``AutoCarver`` excels at reducing feature dimensionality, especially in scenarios involving one-hot encoding. It identifies and preserves only the most statistically relevant modalities, ensuring that your models focus on the most informative aspects of the data while eliminating noise and redundancy.

4. **Statistical Accuracy and Relevance**: ``AutoCarver`` incorporates statistical techniques to ensure that the selected modalities have a sufficient number of observations, minimizing the risk of drawing conclusions based on insufficient data. This helps maintain the reliability and validity of your binary classification models.

5. **Robustness Testing**: ``AutoCarver`` goes beyond feature processing by assessing the robustness of the selected modalities. It performs tests to evaluate the stability and consistency of the chosen features across different datasets or subsets, ensuring their reliability in various scenarios.

``AutoCarver`` is a valuable tool for data scientists and practitioners involved in binary classification problems, such as credit scoring, fraud detection, and risk assessment. By leveraging its automated feature processing capabilities, you can unlock the full potential of your data, leading to more accurate predictions, improved model explainability, and better decision-making in your classification tasks.


It can be used to process and select features in binary classification problems:

* Credit Scoring

* Fraud Detection

* Drift Detection

* Churn Detection and Prevention

* Palability Score



Feature Discretization
......................

Discretizers are the base of AutoCarver (it is one itself).


Quantitative feature discretization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

works for continuous and discrete quantitative features

A two step process:

 1. QuantileDiscretizer:
 
 * Cuts by quantiles

 * All values from -inf to inf covered

 * Over-represented values are left by themselves

 * outputs an ordinal qualitative feature

2.

 * OrdinalDiscretizer: 



Qualitative Discretizer
^^^^^^^^^^^^^^^^^^^^^^^

AutoCarver
..........

 1. **AutoCarver**: Bucketization of qualitative, ordinal, discrete and quantitative features that maximizes association with a binary target

 * vectorized groupby sum
 * on crosstabs
 * all consecutive combinations


FeatureSelector
...............

 2. **FeatureSelector**: Feature selection that maximizes association with binary target that offers control over inter-feature association


Performances
------------

Execution time has been measured for several values of the key paramaters of `AutoCarver`.


.. csv-table::
   :header: min_freq, max_n_mod, X.shape[0], len(features), Execution Time


   0.01, 5, 100000000, 15, 0.001
   0.02, 4, 100000000, 15, 0.001
