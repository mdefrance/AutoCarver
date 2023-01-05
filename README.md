# AutoCarver

**AutoCarver** is an approach for maximising a categorical feature's association with a binary target feature while reducing it's number of distinct modalities.
Can alse be used to discretize quantitative features, that are prealably cut in quantiles.

 Modalities/values of features are carved/regrouped according to a computed specific order defined based on their types:
     - [Qualitative features] grouped based on target rate per modality.
     - [Qualitative ordinal features] grouped based on specified modality order.
     - [Quantitative features] grouped based on the order of their values.
 
 Uses Tschurpow's T or Cramer's V to find the optimal carving (regrouping) of modalities/values of features.

Only implementend for binary classification problems.

## Install

AutoCarver can be installed from [PyPI](https://pypi.org/project/AutoCarver):

<pre>
pip install --upgrade autocarver
</pre>

## Example

#### Setting up Samples

AutoCarver tests the robustness of carvings on specific sample. For this purpose, the use of an out of time sample is recommended. 

```python
# defining training and testing sets
X_train, y_train = ...
X_test, y_test = ...
```

#### Formatting features to de carved

All features need to be discretized via a Discretizer so AutoCarver can group their modalities. Following parameters must be set for Discretizer:

- `min_freq`, should be set from 0.01 (preciser) to 0.05 (faster, increased stability).
  - For qualitative features:  Minimal frequency of a modality, less frequent modalities are grouped in the `default_value='__OTHER__'` modality.
  - For qualitative ordinal features: Less frequent modalities are grouped to the closest modality  (smallest frequency or closest target rate), between the superior and inferior values (specified in the `values_orders` dictionnary).

- `q`, should be set from 10 (faster) to 20 (preciser).
  - For quantitative features: Number of quantiles to initialy cut the feature. Values more frequent than `1/q` will be set as their own group and remaining frequency will be cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`). 

- `values_orders`
  - For qualitative ordinal features: `dict` of features values and `GroupedList` of their values. Modalities less frequent than `min_freq` are automaticaly grouped to the closest modality (smallest frequency or closest target rate), between the superior and inferior values.

At this step, all `numpy.nan` are kept as their own modality.
For qualitative features, unknown modalities passed to `Discretizer.transform` (that where not passed to `Discretizer.fit`) are automaticaly grouped to the `default_value='__OTHER__'` modality.

```python
from AutoCarver.Discretizers import Discretizer

# specifying features to be carved
selected_quanti = ['amount', 'distance', 'length', 'height']
selected_quali = ['age', 'type', 'grade', 'city']

# specifying orders of categorical ordinal features
values_orders = {
    'age': ['0-18', '18-30', '30-50', '50+'],
    'grade': ['A', 'B', 'C', 'D', 'J', 'K', 'NN']
}

# pre-processing of features into categorical ordinal features
discretizer = Discretizer(selected_quanti, selected_quali, min_freq=0.02, q=20, values_orders=values_orders)
X_train = discretizer.fit_transform(X_train, y_train)
X_test = discretizer.transform(X_test)

# updating features' values orders (at this step every features are qualitative ordinal)
values_orders = discretizer.values_orders
```

#### Automatic Carving of features

All specified features can now automatically be carved in an association maximising grouping of their modalities while reducing their number. Following parameters must be set for AutoCarver:

- `sort_by`, association measure used to find the optimal group modality combination.
  - Use `'cramerv'` for more modalities, less robust.
  - Use `'tschuprowt'` for more robust modalities.
**Tip:** a combination of features carved with `sort_by='cramerv'` and `sort_by='tschuprowt'` can sometime prove to be better than only one of those.

- `sample_size`, sample size used for stratified sampling per feature modalities by target rate. Should be set from 0.01 (faster, use with large dataset) to 0.5 (preciser, use with small dataset).

- `max_n_mod`, maximum number of modalities for the carved features (excluding `numpy.nan`). All possible combinations of less than `max_n_mod` groups of modalities will be tested. Should be set from 4 (faster) to 6 (preciser).

```python
from AutoCarver import AutoCarver

# intiating AutoCarver
auto_carver = AutoCarver(values_orders, sort_by='cramerv', max_n_mod=5, sample_size=0.01)

# fitting on training sample 
# a test sample can be specified to evaluate carving robustness
X_train = auto_carver.fit_transform(X_train, y_train, X_test, y_test)

# applying transformation on test sample
X_test = auto_carver.transform(X_test)

# identifying non stable/robust features
print(auto_carver.non_viable_features)
```


#### Storing, reusing an AutoCarver

```python
from sklearn.pipeline import Pipeline

# storing Discretizer
pipe = [('Discretizer', discretizer)]

# storing fitted AutoCarver in a sklearn.pipeline.Pipeline
pipe += [('AutoCarver', auto_carver.discretizer)]
pipe = Pipeline(pipe)

# applying pipe to a validation set or in production
X_val = pipe.transform(X_val)
```
