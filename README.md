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

AutoCarver needs to test the robustness of carvings on specific sample. For this purpose, the use of an out of time sample is recommended. 


<pre>
from AutoCarver import AutoCarver
from AutoCarver.Discretizers import Discretizer
from sklearn.pipeline import Pipeline

# defining training and testing sets
X_train, y_train = ...
X_test, y_test = ...
</pre>

All features need to be discretized via a Discretizer so AutoCarver can group their modalities. Following parameters must be set for Discretizer:
- `min_freq`, should be set from 0.01 (preciser) to 0.05 (faster, increased stability).
  - For qualitative features:  Minimal frequency of a modality, less frequent modalities are grouped in the `'__OTHER__'` modality.
  - For qualitative ordinal features: Less frequent modalities are grouped to the closest modality  (smallest frequency or closest target rate), between the superior and inferior values (specified in the <pre>values_orders</pre> dictionnary).

- `q`, should be set from 10 (faster) to 20 (preciser).
  - For quantitative features: Number of quantiles to initialy cut the feature. Values more frequent than `1/q` will be set as their own group and remaining frequency will be cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`). 


- `values_orders`
  - For qualitative ordinal features: `dict` of features values and list of orders of their values. Modalities less frequent than `min_freq` are automaticaly grouped to the closest modality (smallest frequency or closest target rate), between the superior and inferior values.


<pre>
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
</pre>


# storing Discretizer
pipe = [('Discretizer', discretizer)]

# updating features' values orders (at this step every features are qualitative ordinal)
values_orders = discretizer.values_orders

# intiating AutoCarver
auto_carver = AutoCarver(values_orders, sort_by='cramerv', max_n_mod=5, sample_size=0.01)

# fitting on training sample 
# a test sample can be specified to evaluate carving robustess
X_train = auto_carver.fit_transform(X_train, y_train, X_test, y_test)

# applying transformation on test sample
X_test = auto_carver.transform(X_test)

# identifying non stable/robust features
print(auto_carver.non_viable_features)

# storing fitted GroupedListDiscretizer in a sklearn.pipeline.Pipeline
pipe += [('AutoCarver', auto_carver.discretizer)]
pipe = Pipeline(pipe)

# applying pipe to a validation set or in production
X_val = pipe.transform(X_val)
</pre>
