</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/autocarver?style=flat-square">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/autocarver?style=flat-square">
    <img alt="GitHub" src="https://img.shields.io/github/license/mdefrance/autocarver?style=flat-square">
</p>

This is a work in progress.

# AutoCarver

**AutoCarver** is a powerful set of tools designed for binary classification problems. It offers a range of functionalities to enhance the feature engineering process and improve the performance of binary classification models. It provides:
 1. **Discretizers**: Discretization of qualitative (ordinal or not) and quantitative features
 2. **AutoCarver**: Bucketization of qualitative features that maximizes association with a binary target feature
 3. **FeatureSelector**: Feature selection that maximizes association with binary target that offers control over inter-feature association.

## Install

AutoCarver can be installed from [PyPI](https://pypi.org/project/AutoCarver):

<pre>
pip install autocarver
</pre>


## Quick-Start Examples

### Setting up Samples, initiating Pipeline

`AutoCarver` is able to test the robustness of buckets on a dev sample `X_dev`.

One of the great advantages of the `AutoCarver` package is its seamless integration with scikit-learn pipelines, making it incredibly convenient for production-level implementations. By leveraging scikit-learn's pipeline functionality, `AutoCarver` can be effortlessly incorporated into the end-to-end machine learning workflow.

```python
# defining training and testing sets
X_train, y_train = ...  # used to fit the AutoCarver and the model
X_dev, y_dev = ...  # used to validate the AutoCarver's buckets and optimize the model's parameters/hyperparameters
X_test, y_test = ...  # used to evaluate the final model's performances
```



### Maximize target association of features' buckets with AutoCarver

All features need to be discretized via a `Discretizer` so `AutoCarver` can group their modalities. Following parameters must be set for `Discretizer`:

All specified features can now automatically be carved in an association maximising grouping of their modalities while reducing their number. Following parameters must be set for `AutoCarver`:
- `values_orders`, dict of all features matched to the order of their modalities
- `sort_by`, association measure used to find the optimal group modality combination.
  - Use `sort_by='cramerv'` for more modalities, less robust.
  - Use `sort_by='tschuprowt'` for more robust modalities.
  - **Tip:** a combination of features carved with `sort_by='cramerv'` and `sort_by='tschuprowt'` can sometime prove to be better than only one of those.
- `max_n_mod`, maximum number of modalities for the carved features (excluding `numpy.nan`). All possible combinations of less than `max_n_mod` groups of modalities will be tested. Should be set from 4 (faster) to 6 (preciser).
- `keep_nans`, whether or not to try groupin missing values to non-missing values. Use `keep_nans=True` if you want `numpy.nan` to remain as a specific modality.

```python
from AutoCarver.auto_carver import AutoCarver

quantitative_features = ['Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative', 'Discrete_Quantitative_rarevalue']
qualitative_features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan", "Discrete_Qualitative_noorder", "Discrete_Qualitative_lownan_noorder", "Discrete_Qualitative_rarevalue_noorder"]
ordinal_features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan", "Discrete_Qualitative_highnan"]
values_orders = {
    "Qualitative_Ordinal": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
    "Qualitative_Ordinal_lownan": ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+'],
    "Discrete_Qualitative_highnan" : ["1", "2", "3", "4", "5", "6", "7"],
}
target = "quali_ordinal_target"

# intiating AutoCarver
auto_carver = AutoCarver(
    quantitative_features=quantitative_features,
    qualitative_features=qualitative_features,
    ordinal_features=ordinal_features,
    values_orders=values_orders,
    min_freq=0.05,  # minimum frequency per modality
    max_n_mod=4,  # maximum number of modality per feature
    sort_by='cramerv',
    output_dtype='float',
    copy=True,
    verbose=True,
)

# fitting on training sample, a dev sample can be specified to evaluate carving robustness
x_discretized = auto_carver.fit_transform(x_train, x_train[target], X_test=x_dev, y_test=x_dev[target])

# transforming dev/test sample accordingly
x_dev_discretized = auto_carver.transform(x_dev)
x_test_discretized = auto_carver.transform(x_test)

```
<p align="left">
  <img width="500" src="/docs/auto_carver_fit.PNG" />
</p>


### Cherry picking the most target-associated features with FeatureSelector

Following parameters must be set for `FeatureSelector`:
- `features`, list of candidate features by column name
- `n_best`, number of features to select
- `sample_size=1`, size of sampled list of features speeds up computation. By default, all features are used. For sample_size=0.5, FeatureSelector will search for the best features in features[:len(features)//2] and then in features[len(features)//2:]. Should be set between ]0, 1]. 
  - **Tip:** for a DataFrame of 100 000 rows, `sample_size` could be set such as `len(features)*sample_size` equals 100-200.
- `measures`, list of `FeatureSelector`'s association measures to be evaluated. Ranks features based on last measure of the list.
  - *For qualitative data* implemented association measures are `chi2_measure`, `cramerv_measure`, `tschuprowt_measure`
  - *For quantitative data* implemented association measures are `kruskal_measure`, `R_measure` and implemented outlier metrics are `zscore_measure`, `iqr_measure`
- `filters`, list of `FeatureSelector`'s filters used to put aside features.
  - *For qualitative data* implemented correlation-based filters are `cramerv_filter`, `tschuprowt_filter`
  - *For quantitative data* implemented linear filters are `spearman_filter`, `pearson_filter` and `vif_filter` for multicolinearity filtering

**TODO: add by default measures and filters + add ranking according to several measures  + say that it filters out non-selected columns**


**TODO; add pictures say that it does not make sense to use zscore_measure as last measure**

```python
from AutoCarver.feature_selector import FeatureSelector
from AutoCarver.feature_selector import tschuprowt_measure, cramerv_measure, cramerv_filter, tschuprowt_filter, measure_filter

features = auto_carver.features  # after AutoCarver, everything is qualitative
values_orders = auto_carver.values_orders

measures = [cramerv_measure, tschuprowt_measure]  # measures of interest (the last one is used for ranking)
filters = [tschuprowt_filter, measure_filter]  # filtering out by inter-feature correlation

# select the best 25 most target associated qualitative features
quali_selector = FeatureSelector(
    features=features,  # features to select from
    n_best=25,  # best 25 features
    values_orders=values_orders,
    measures=measures, filters=filters,   # selected measures and filters
    thresh_mode=0.9,  # filters out features with more than 90% of their mode
    thresh_nan=0.9,  # filters out features with more than 90% of missing values
    thresh_corr=0.5,  # filters out features with spearman greater than 0.5 with a better feature
    name_measure='cramerv_measure', thresh_measure=0.06,  # filters out features with cramerv_measure lower than 0.06
    verbose=True  # displays statistics
)
X_train = quali_selector.fit_transform(X_train, y_train)
X_dev = quali_selector.transform(X_dev)

# append the selector to the feature engineering pipeline
pipe += [('QualiFeatureSelector', quali_selector)]
```



### Storing, reusing the AutoCarver

**TODO:** The `AutoCarver` can be stored as a .json file.

```python
import json

# storing as json file
with open('my_carver.json', 'wb') as my_carver_json:
    json.dump({feature: values.contained for feature, values in auto_carver.values_orders.items()}, my_carver_json)
```

The stored .json, can then be used to initialize a new `base_discretizers.GroupedListDiscretizer`.

```python
from AutoCarver.discretizers.utils.base_discretizers import GroupedListDiscretizer

# storing as json file
with open('my_carver.json', 'rb') as my_carver_json:
    values_orders = json.load(my_carver_json)

# initiating AutoCarver
auto_carver = GroupedListDiscretizer(
    features=self.features,
    values_orders=self.values_orders,
    copy=self.copy,
    input_dtypes=self.input_dtypes,
    str_nan=self.str_nan,
    verbose=self.verbose,
    output_dtype=self.output_dtype,
)

```









## Detailed Examples

### StringConverter Example

             
```python
from AutoCarver.Converters import StringConverter

stringer = StringConverter(features=quali_features)
X_train = stringer.fit_transform(X_train)
X_dev = stringer.transform(X_dev)

# append the string converter to the feature engineering pipeline
pipe.steps.append(['StringConverter', stringer])
```


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


### FeatureSelector Examples
#### Quantitative data

             
             
```python
from AutoCarver.FeatureSelector import FeatureSelector
from AutoCarver.FeatureSelector import zscore_measure, iqr_measure, kruskal_measure, R_measure, measure_filter, spearman_filter

measures = [zscore_measure, iqr_measure, kruskal_measure, R_measure]  # measures of interest (the last one is used for ranking)
filters = [measure_filter, spearman_filter]  # filtering out by inter-feature correlation

# select the best 25 most target associated quantitative features
quanti_selector = FeatureSelector(
    features=quanti_features,  # features to select from
    n_best=25,  # best 25 features
    measures=measures, filters=filters,   # selected measures and filters
    thresh_mode=0.9,  # filters out features with more than 90% of their mode
    thresh_nan=0.9,  # filters out features with more than 90% of missing values
    thresh_corr=0.5,  # filters out features with spearman greater than 0.5 with a better feature
    name_measure='R_measure', thresh_measure=0.06,  # filters out features with R_measure lower than 0.06
    verbose=True  # displays statistics
)
X_train = quanti_selector.fit_transform(X_train, y_train)
X_dev = quanti_selector.transform(X_dev)

# append the selector to the feature engineering pipeline
pipe.steps.append(['QuantiFeatureSelector', quanti_selector])
pipe += [('QuantiFeatureSelector', quanti_selector)]
```


**FeatureSelector TODO: add how to build on measures and filters**


### Converters Examples
#### CrossConverter

             
```python
from AutoCarver.Converters import CrossConverter

# qualitative and quantitative features should be discretized (and bucketized with AutoCarver)
to_cross = quali_features + quanti_features

cross_converter = CrossConverter(to_cross)
X_train = cross_converter.fit_transform(X_train, y_train)
X_dev = cross_converter.transform(X_dev)

# append the crosser to the feature engineering pipeline
pipe += [('CrossConverter', cross_converter)]

quali_features_built = crosser.new_features  # adding to qualitative_features_built for no further feature engineering
print(f"Qualitative features built: total {len(quali_features_built)}")
```
