Quick Start
===========

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

.. code-block:: python

    quantitative_features = ['Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative', 'Discrete_Quantitative_rarevalue']
    qualitative_features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan", "Discrete_Qualitative_noorder", "Discrete_Qualitative_lownan_noorder", "Discrete_Qualitative_rarevalue_noorder"]
    target = "quali_ordinal_target"

Qualitative features will automatically be converted to ``str`` if necessary.
Ordinal features can be added, alongside there expected ordering. See :ref:`Examples`.

Using AutoCarver
----------------

Fitting AutoCarver
..................

.. code-block:: python

    from AutoCarver import AutoCarver

    # intiating AutoCarver
    auto_carver = AutoCarver(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        min_freq=0.02,  # minimum frequency per modality
        max_n_mod=5,  # maximum number of modality per Carved feature
        sort_by='cramerv',  # measure used to select the best combination of modalities
        pretty_print=True,  # showing statistics
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

.. code-block:: python

    from AutoCarver.feature_selector import feature_selector

    # select the best 25 most target associated qualitative features
    feature_selector = FeatureSelector(
        qualitative_features=features,  # features to select from
        n_best=25,  # number of features to select
        pretty_print=True  # displays statistics
    )
    best_features = feature_selector.select(X_train, y_train)


In-depth examples
-----------------

See :ref:`Examples`.