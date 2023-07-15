Quick Start
============

Setting up samples
------------------

`AutoCarver` is able to test the robustness of buckets on a dev sample `X_dev`.

.. code-block:: python

    # defining training and testing sets
    X_train, y_train = ...  # used to fit the AutoCarver and the model
    X_dev, y_dev = ...  # used to validate the AutoCarver's buckets and optimize the model's parameters/hyperparameters
    X_test, y_test = ...  # used to evaluate the final model's performances

Picking up columns to Carve
---------------------------

TODO: automatic conversion to str for qualitative features

.. code-block:: python

    quantitative_features = ['Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative', 'Discrete_Quantitative_rarevalue']
    qualitative_features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan", "Discrete_Qualitative_noorder", "Discrete_Qualitative_lownan_noorder", "Discrete_Qualitative_rarevalue_noorder"]
    target = "quali_ordinal_target"

Using AutoCarver
----------------

Fitting AutoCarver
..................

All specified features can now automatically be carved in an association maximising grouping of their modalities while reducing their number. Following parameters must be set for `AutoCarver`:

* `values_orders`, dict of all features matched to the order of their modalities

* `max_n_mod`, maximum number of modalities for the carved features (excluding `numpy.nan`). All possible combinations of less than `max_n_mod` groups of modalities will be tested. Should be set from 4 (faster) to 6 (preciser).

* `dropna`, whether or not to try grouping missing values to non-missing values. Use `keep_nans=True` if you want `numpy.nan` to remain as a specific modality.

* `sort_by`, association measure used to find the optimal group modality combination.

    * Use `sort_by='cramerv'` for more modalities, less robust.

    * Use `sort_by='tschuprowt'` for more robust modalities.

    * **Tip:** a combination of features carved with `sort_by='cramerv'` and `sort_by='tschuprowt'` can sometime prove to be better than only one of those.

.. code-block:: python

    from AutoCarver import AutoCarver

    # intiating AutoCarver
    auto_carver = AutoCarver(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        min_freq=0.05,
        max_n_mod=4,
        sort_by='cramerv',
        output_dtype='float',
        copy=True,
        verbose=True,
    )

    # fitting on training sample, a dev sample can be specified to evaluate carving robustness
    x_discretized = auto_carver.fit_transform(x_train, x_train[target], X_dev=x_dev, y_dev=x_dev[target])



Applying AutoCarver
...................

.. code-block:: python

    # transforming dev/test sample accordingly
    x_dev_discretized = auto_carver.transform(x_dev)
    x_test_discretized = auto_carver.transform(x_test)


Saving AutoCarver
.................

The `AutoCarver` can safely be stored as a .json file.

.. code-block:: python

    import json

    # storing as json file
    with open('my_carver.json', 'w') as my_carver_json:
        json.dump(auto_carver.to_json(), my_carver_json)


Loading AutoCarver
..................

The `AutoCarver` can safely be loaded from a .json file.

.. code-block:: python

    import json

    from AtuoCarver import load_carver

    # loading json file
    with open('my_carver.json', 'r') as my_carver_json:
        auto_carver = load_carver(json.load(my_carver_json))

Feature Selection
-----------------

Following parameters must be set for `FeatureSelector`:

* `features`, list of candidate features by column name

* `n_best`, number of features to select

* `sample_size=1`, size of sampled list of features speeds up computation. By default, all features are used. For sample_size=0.5, FeatureSelector will search for the best features in features[:len(features)//2] and then in features[len(features)//2:]. Should be set between ]0, 1]. 

    * **Tip:** for a DataFrame of 100 000 rows, `sample_size` could be set such as `len(features)*sample_size` equals 100-200.

* `measures`, list of `FeatureSelector`'s association measures to be evaluated. Ranks features based on last measure of the list.

    * *For qualitative data* implemented association measures are `chi2_measure`, `cramerv_measure`, `tschuprowt_measure`   
   
    * *For quantitative data* implemented association measures are `kruskal_measure`, `R_measure` and implemented outlier metrics are `zscore_measure`, `iqr_measure`

* `filters`, list of `FeatureSelector`'s filters used to put aside features.
    
    * *For qualitative data* implemented correlation-based filters are `cramerv_filter`, `tschuprowt_filter`
    * *For quantitative data* implemented linear filters are `spearman_filter`, `pearson_filter` and `vif_filter` for multicolinearity filtering

**TODO: add by default measures and filters + add ranking according to several measures  + say that it filters out non-selected columns**


**TODO; add pictures say that it does not make sense to use zscore_measure as last measure**

.. code-block:: python

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

In-depth examples
--------------------------------------

TODO: Add links to other documentations 