</p>
<p align="left">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/autocarver">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autocarver">
    <img alt="GitHub" src="https://img.shields.io/github/license/mdefrance/autocarver">
    <img src="https://github.com/mdefrance/AutoCarver/actions/workflows/pytest.yml/badge.svg">
    <img src='https://readthedocs.org/projects/autocarver/badge/?version=latest' alt='Documentation Status' />
</p>


# ReadTheDocs

Go check out the brand new package documentation at [ReadTheDocs](https://autocarver.readthedocs.io/en/latest/index.html)!

# Install

**AutoCarver** can be installed from [PyPI](https://pypi.org/project/AutoCarver):

## Minimal install

To install the base features (without vizualization tools), use the following:

<pre>
pip install autocarver
</pre>

## Unabling pretty printing

To enable ``pretty_print`` features (nice colorful html tables in jupyter), use the following:

<pre>
pip install autocarver[jupyter]
</pre>

# AutoCarver

**AutoCarver** is a powerful Python package designed to address the fundamental question of *What's the best processing for my model's features?*

It offers an automated and optimized approach to processing and engineering your data, resulting in improved model performance, enhanced explainability, and reduced feature dimensionality.
As of today, this set of tools is available for binary classification and regression problems only.

Key Features:

1. **Data Processing and Engineering**: **AutoCarver** performs automatic bucketization and carving of a DataFrame's columns to maximize their correlation with a target variable. By leveraging advanced techniques, it optimizes the preprocessing steps for your data, leading to enhanced predictive accuracy.

2. **Improved Model Explainability**: **AutoCarver** aids in understanding the relationship between the processed features and the target variable. By uncovering meaningful patterns and interactions, it provides valuable insights into the underlying data dynamics, enhancing the interpretability of your models.

3. **Reduced Feature Dimensionality**: **AutoCarver** excels at reducing feature dimensionality, especially in scenarios involving one-hot encoding. It identifies and preserves only the most statistically relevant modalities, ensuring that your models focus on the most informative aspects of the data while eliminating noise and redundancy.

4. **Statistical Accuracy and Relevance**: **AutoCarver** incorporates statistical techniques to ensure that the selected modalities have a sufficient number of observations, minimizing the risk of drawing conclusions based on insufficient data. This helps maintain the reliability and validity of your models.

5. **Robustness Testing**: **AutoCarver** goes beyond feature processing by assessing the robustness of the selected modalities. It performs tests to evaluate the stability and consistency of the chosen features across different datasets or subsets, ensuring their reliability in various scenarios.

**AutoCarver** is a valuable tool for data scientists and practitioners involved in binary classification or regression problems, such as credit scoring, fraud detection, and risk assessment. By leveraging its automated feature processing capabilities, you can unlock the full potential of your data, leading to more accurate predictions, improved model explainability, and better decision-making in your classification tasks.

