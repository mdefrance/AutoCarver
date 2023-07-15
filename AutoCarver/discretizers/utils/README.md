</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/autocarver?style=flat-square">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autocarver?style=flat-square">
    <img alt="GitHub" src="https://img.shields.io/github/license/mdefrance/autocarver?style=flat-square">
</p>

This is a work in progress.

# AutoCarver

**AutoCarver** is a powerful set of tools designed for binary classification problems. It offers a range of functionalities to enhance the feature engineering process and improve the performance of binary classification models. It provides:
 1. **AutoCarver**: Bucketization of qualitative, ordinal and quantitative features that maximizes association with a binary target
 2. **FeatureSelector**: Feature selection that maximizes association with binary target that offers control over inter-feature association.

## Detailed Examples


### Discretizers Examples

The `AutoCarver.Discretizers` is a user-friendly tool that enables the discretization of various types of data into basic buckets. With this package, users can easily transform qualitative, qualitative ordinal, and quantitative data into discrete categories for further analysis and modeling.

#### Quickly build basic buckets with Discretizer

The `AutoCarver.Discretizers` is a user-friendly tool that enables the discretization of various types of data into basic buckets. With this package, users can easily transform qualitative, qualitative ordinal, and quantitative data into discrete categories for further analysis and modeling.

`Discretizer` is the combination of `QuantitativeDiscretizer` and `QuantitativeDiscretizer`.

Following parameters must be set for `Discretizer`:
- `quanti_features`, list of column names of quantitative data to discretize
- `quanli_features`, list of column names of qualitative and qualitative ordinal data to discretize
- `min_freq`, should be set from 0.01 (preciser, decreased stability) to 0.05 (faster, increased stability).
  - *For qualitative data:*  Minimal frequency of a modality, less frequent modalities are grouped in the `default_value='__OTHER__'` modality. Values are ordered based on `y_train` bucket mean.
  - *For qualitative ordinal data:* Less frequent modalities are grouped to the closest modality  (smallest frequency or closest target rate), between the superior and inferior values (specified in the `values_orders` dictionnary).
  - *For quantitative data:* Equivalent to the inverse of `QuantitativeDiscretizer`'s `q` parameter. Number of quantiles to initialy cut the feature in. Values more frequent than `min_freq` will be set as their own group and remaining frequency will be cut into proportionaly less quantiles (`1/min_freq:=max(round(non_frequent * 1/min_freq), 1)`). 
- `values_orders`, dict of qualitative ordinal features matched to the order of their modalities
  - *For qualitative ordinal data:* `dict` of features values and `GroupedList` of their values. Modalities less frequent than `min_freq` are automaticaly grouped to the closest modality (smallest frequency or closest target rate), between the superior and inferior values.

```python
from AutoCarver.Discretizers import Discretizer

quanti_features = ['amount', 'distance', 'length', 'height']  # quantitative features to be discretized
quali_features = ['age', 'type', 'grade', 'city']  # qualitative features to be discretized

# specifying orders of qualitative ordinal features
values_orders = {
    'age': ['0-18', '18-30', '30-50', '50+'],
    'grade': ['A', 'B', 'C', 'D', 'J', 'K', 'NN']
}

# pre-processing of features into categorical ordinal features
discretizer = Discretizer(quanti_features=quanti_features, quali_features=quali_features, min_freq=0.02, values_orders=values_orders)
discretizer.fit_transform(X_train, y_train)
discretizer.transform(X_dev)

# storing built buckets
values_orders.update(discretizer.values_orders)

# append the discretizer to the feature engineering pipeline
pipe += [('Discretizer', discretizer)]
```


Overall, the Discretizers package provides a straightforward and efficient solution for discretizing qualitative, qualitative ordinal, and quantitative data into simple buckets. By transforming data into discrete categories, it enables researchers, analysts, and data scientists to gain insights, perform statistical analyses, and build models on discretized data.

For more details and further functionnalities look into AutoCarver.Discretizers README.

For qualitative features, unknown modalities passed to `Discretizer.transform` (that where not passed to `Discretizer.fit`) are automaticaly grouped to the `default_value='__OTHER__'` modality.

By default, samples are modified and not copied (recommanded for large datasets). Use `copy=True` if you want a new `DataFrame` to be returned.


#### QualitativeDiscretizer Example

**TODO: add StringConverter**

`QualitativeDiscretizer` enables the transformation of qualitative data into statistically relevant categories, facilitating model robustness.
 - *Qualitative Data* consists of categorical variables without any inherent order
 - *Qualitative Ordinal Data* consists of categorical variables with a predefined order or hierarchy

Following parameters must be set for `QualitativeDiscretizer`:
- `features`, list of column names of qualitative and qualitative ordinal data to discretize
- `min_freq`, should be set from 0.01 (preciser, decreased stability) to 0.05 (faster, increased stability).
  - *For qualitative data:*  Minimal frequency of a modality, less frequent modalities are grouped in the `default_value='__OTHER__'` modality. Values are ordered based on `y_train` bucket mean.
  - *For qualitative ordinal data:* Less frequent modalities are grouped to the closest modality  (smallest frequency or closest target rate), between the superior and inferior values (specified in the `values_orders` dictionnary).
- `values_orders`, dict of qualitative ordinal features matched to the order of their modalities
  - *For qualitative ordinal data:* `dict` of features values and `GroupedList` of their values. Modalities less frequent than `min_freq` are automaticaly grouped to the closest modality (smallest frequency or closest target rate), between the superior and inferior values.


```python
from AutoCarver.Discretizers import QualitativeDiscretizer

quali_features = ['age', 'type', 'grade', 'city']  # qualitative features to be discretized

# specifying orders of qualitative ordinal features
values_orders = {
    'age': ['0-18', '18-30', '30-50', '50+'],
    'grade': ['A', 'B', 'C', 'D', 'J', 'K', 'NN']
}

# pre-processing of features into categorical ordinal features
quali_discretizer = QualitativeDiscretizer(features=quali_features, min_freq=0.02, values_orders=values_orders)
quali_discretizer.fit_transform(X_train, y_train)
quali_discretizer.transform(X_dev)

# storing built buckets
values_orders.update(quali_discretizer.values_orders)

# append the discretizer to the feature engineering pipeline
pipe.steps.append(['QualitativeDiscretizer', quali_discretizer])
```

`QualitativeDiscretizer` ensures that the ordinal nature of the data is preserved during the discretization process, resulting in meaningful and interpretable categories.

At this step, all `numpy.nan` are kept as their own modality. **not all of them**

#### QuantitativeDiscretizer Example

**TODO: change q for min_freq**

`QuantitativeDiscretizer` enables the transformation of quantitative data into automatically determined intervals of ranges of values, facilitating model robustness.
 - *Quantitative Data* consists of continuous and discrete numerical variables.

Following parameters must be set for `QuantitativeDiscretizer`:
- `features`, list of column names of quantitative data to discretize
- `q`, should be set from 20 (faster, increased stability) to 50 (preciser, decreased stability).
  - *For quantitative data:* Number of quantiles to initialy cut the feature in. Values more frequent than `1/q` will be set as their own group and remaining frequency will be cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`). 

```python
from AutoCarver.Discretizers import QuantitativeDiscretizer

quanti_features = ['amount', 'distance', 'length', 'height']  # quantitative features to be discretized

# pre-processing of features into categorical ordinal features
quanti_discretizer = QuantitativeDiscretizer(features=quanti_features, q=40)
quanti_discretizer.fit_transform(X_train, y_train)
quanti_discretizer.transform(X_dev)

# storing built buckets
values_orders.update(quanti_discretizer.values_orders)

# append the discretizer to the feature engineering pipeline
pipe.steps.append(['QuantitativeDiscretizer', quanti_discretizer])
```

At this step, all `numpy.nan` are kept as their own modality.

```python
from pickle import load

# restoring the pipeline
pipe = load(open('my_pipe.pkl', 'rb'))

# applying pipe to a validation set or in production
X_val = pipe.transform(X_val)
```
**TODO: add before after picture**
