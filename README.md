# AutoCarver

**AutoCarver** is an approach for maximising a qualitative feature's association with a binary target feature while reducing it's number of distinct modalities.
Can also be used to discretize quantitative features, that are prealably cut in quantiles.

 Modalities/values of features are carved/regrouped according to a computed specific order defined based on their types:
  - *Qualitative features* grouped based on target rate per modality.
  - *Qualitative ordinal features* grouped based on specified modality order.
  - *Quantitative features* grouped based on the order of their values.
 
Uses Tschurpow's T or Cramer's V to find the optimal carving (regrouping) of modalities/values of features.

`AutoCarver` is an `sklearn` transformer.

Only implementend for binary classification problems.

## Install

AutoCarver can be installed from [PyPI](https://pypi.org/project/AutoCarver):

<pre>
pip install --upgrade autocarver
</pre>

## Complete Example

#### Setting up Samples

`AutoCarver` tests the robustness of carvings on a specific sample. For this purpose, the use of an out of time sample is recommended. 

```python
# defining training and testing sets
X_train, y_train = ...
X_test, y_test = ...
X_val, y_val = ...
```

#### Formatting features to be carved

All features need to be discretized via a `Discretizer` so `AutoCarver` can group their modalities. Following parameters must be set for `Discretizer`:

- `min_freq`, should be set from 0.01 (preciser) to 0.05 (faster, increased stability).
  - *For qualitative features:*  Minimal frequency of a modality, less frequent modalities are grouped in the `default_value='__OTHER__'` modality.
  - *For qualitative ordinal features:* Less frequent modalities are grouped to the closest modality  (smallest frequency or closest target rate), between the superior and inferior values (specified in the `values_orders` dictionnary).

- `q`, should be set from 10 (faster) to 20 (preciser).
  - *For quantitative features:* Number of quantiles to initialy cut the feature. Values more frequent than `1/q` will be set as their own group and remaining frequency will be cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`). 

- `values_orders`
  - *For qualitative ordinal features:* `dict` of features values and `GroupedList` of their values. Modalities less frequent than `min_freq` are automaticaly grouped to the closest modality (smallest frequency or closest target rate), between the superior and inferior values.

At this step, all `numpy.nan` are kept as their own modality.

For qualitative features, unknown modalities passed to `Discretizer.transform` (that where not passed to `Discretizer.fit`) are automaticaly grouped to the `default_value='__OTHER__'` modality.

By default, samples are modified and not copied (recommanded for large datasets). Use `copy=True` if you want a new `DataFrame` to be returned.

```python
from AutoCarver.Discretizers import Discretizer

# specifying features to be carved
quantitatives = ['amount', 'distance', 'length', 'height']
qualitatives = ['age', 'type', 'grade', 'city']

# specifying orders of categorical ordinal features
values_orders = {
    'age': ['0-18', '18-30', '30-50', '50+'],
    'grade': ['A', 'B', 'C', 'D', 'J', 'K', 'NN']
}

# pre-processing of features into categorical ordinal features
discretizer = Discretizer(quantitatives, qualitatives, min_freq=0.02, q=20, values_orders=values_orders)
discretizer.fit_transform(X_train, y_train)
discretizer.transform(X_test)

# updating features' values orders (at this step every features are qualitative ordinal)
values_orders = discretizer.values_orders
```

#### Automatic Carving of features

All specified features can now automatically be carved in an association maximising grouping of their modalities while reducing their number. Following parameters must be set for `AutoCarver`:

- `sort_by`, association measure used to find the optimal group modality combination.
  - Use `sort_by='cramerv'` for more modalities, less robust.
  - Use `sort_by='tschuprowt'` for more robust modalities.
  - **Tip:** a combination of features carved with `sort_by='cramerv'` and `sort_by='tschuprowt'` can sometime prove to be better than only one of those.

- `max_n_mod`, maximum number of modalities for the carved features (excluding `numpy.nan`). All possible combinations of less than `max_n_mod` groups of modalities will be tested. Should be set from 4 (faster) to 6 (preciser).

At this step, all `numpy.nan` are grouped to the best non-NaN value (after they were grouped). Use `keep_nans=True` if you want `numpy.nan` to remain as a specific modality.


```python
from AutoCarver.AutoCarver import AutoCarver

# intiating AutoCarver
auto_carver = AutoCarver(values_orders, sort_by='cramerv', max_n_mod=5, verbose=True)

# fitting on training sample 
# a test sample can be specified to evaluate carving robustness
auto_carver.fit_transform(X_train, y_train, X_test, y_test)

# applying transformation on test sample
auto_carver.transform(X_test)
```
<p align="left">
  <img width="500" src="/docs/auto_carver_fit.PNG" />
</p>

#### Storing, reusing an AutoCarver

The `Discretizer` and `AutoCarver` steps can be stored in a `Pipeline` and can than be stored as a `pickle` file.

```python
from pickle import dump
from sklearn.pipeline import Pipeline

# storing Discretizer
pipe = [('Discretizer', discretizer)]

# storing fitted AutoCarver in a Pipeline
pipe += [('AutoCarver', auto_carver)]
pipe = Pipeline(pipe)

# storing as pickle file
dump(pipe, open('my_pipe.pkl', 'wb'))
```

The stored `Pipeline`, can then be used to transform new datasets.

```python
from pickle import load

# restoring the pipeline
pipe = load(open('my_pipe.pkl', 'rb'))

# applying pipe to a validation set or in production
X_val = pipe.transform(X_val)
```
