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

    from AutoCarver.auto_carver import AutoCarver

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

    from AtuoCarver.auto_carver import load_carver

    # loading json file
    with open('my_carver.json', 'r') as my_carver_json:
        auto_carver = load_carver(json.load(my_carver_json))

Feature Selection
.................
TODO

Further readings and in-depth examples
......................................

TODO: Add links to other documentations 