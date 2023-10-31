Quick Start
===========


Setting things up
-----------------

Target type and Carver selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on one's desired modelling task, several Carvers are implemented:

 * :ref:`BinaryCarver`
 * :ref:`MulticlassCarver`
 * :ref:`ContinuousCarver`

In the following quick start example, we will consider a binary classification problem:

.. code-block:: python

    target = "binary_target"

Hence the use of :class:`BinaryCarver` in following code blocks.



Data Sampling
^^^^^^^^^^^^^

**AutoCarver** unables testing for robustness of carved modalities on ``X_dev`` while maximizing the association between ``X_train`` and ``y_train``.

.. code-block:: python

    # defining training and testing sets
    train_set = ...  # used to fit the AutoCarver and the model
    dev_set = ...  # used to validate the AutoCarver's buckets and optimize the model's parameters/hyperparameters
    test_set = ...  # used to evaluate the final model's performances



Picking up columns to Carve
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    quantitative_features = ['Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative', 'Discrete_Quantitative_rarevalue']
    qualitative_features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan", "Discrete_Qualitative_noorder", "Discrete_Qualitative_lownan_noorder", "Discrete_Qualitative_rarevalue_noorder"]

Qualitative features will automatically be converted to ``str`` if necessary.
Ordinal features can be added, alongside there expected ordering. See :ref:`Examples`.




Using AutoCarver
----------------

Fitting AutoCarver
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from AutoCarver import BinaryCarver

    # intiating AutoCarver
    auto_carver = BinaryCarver(
        quantitative_features=quantitative_features,
        qualitative_features=qualitative_features,
        min_freq=0.02,  # minimum frequency per modality
        max_n_mod=5,  # maximum number of modality per Carved feature
        sort_by='tschuprowt',  # measure used to select the best combination of modalities
        verbose=True,  # showing statistics
    )

    # fitting on training sample, a dev sample can be specified to evaluate carving robustness
    x_discretized = auto_carver.fit_transform(train_set, train_set[target], X_dev=dev_set, y_dev=dev_set[target])



Applying AutoCarver
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # transforming dev/test sample accordingly
    dev_set_discretized = auto_carver.transform(dev_set)
    test_set_discretized = auto_carver.transform(tes_set)



Saving AutoCarver
^^^^^^^^^^^^^^^^^

All Carvers can safely be stored as a ``.json`` file.

.. code-block:: python

    import json

    # storing as json file
    with open('my_carver.json', 'w') as my_carver_json:
        json.dump(auto_carver.to_json(), my_carver_json)


Loading AutoCarver
^^^^^^^^^^^^^^^^^^

The `AutoCarver` can safely be loaded from a .json file.

.. code-block:: python

    import json

    from AutoCarver import load_carver

    # loading json file
    with open('my_carver.json', 'r') as my_carver_json:
        auto_carver = load_carver(json.load(my_carver_json))



Feature Selection
-----------------

.. code-block:: python

    from AutoCarver.feature_selection import FeatureSelector

    # select the best 25 most target associated qualitative features
    feature_selector = FeatureSelector(
        qualitative_features=features,  # features to select from
        n_best=25,  # number of features to select
        verbose=True  # displays statistics
    )
    best_features = feature_selector.select(train_set_discretized, train_set_discretized[target])


In-depth examples
-----------------

See :ref:`Examples`.