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

Hence the use of :class:`BinaryCarver` and :class:`ClassificationSelector` in following code blocks.



Data Sampling
^^^^^^^^^^^^^

**AutoCarver** unables testing for robustness of carved modalities on ``X_dev`` while maximizing the association between ``X_train`` and ``y_train``.

.. code-block:: python

    # defining training and testing sets
    train_set = ...  # used to fit the AutoCarver and the model
    dev_set = ...  # used to validate the AutoCarver's buckets and optimize the model's parameters/hyperparameters
    test_set = ...  # used to evaluate the final model's performances



Setting up Features to Carve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from AutoCarver import Features

    features = Features(
        quantitatives=['quantitative1', 'quantitative2', 'discrete1', 'discrete2_with_nan'],
        categoricals=['categorical1', 'categorical2', 'categorical3_with_nan'],
        ordinal={'ordinal1': ['low', 'medium', 'high'], 'ordinal2_with_nan': ['low', 'medium', 'high']},
    )

Qualitative features will automatically be converted to :class:`str` if necessary.
Ordinal features are added, alongside there expected ordering.




Using AutoCarver
----------------

Fitting AutoCarver
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from AutoCarver import BinaryCarver

    # intiating AutoCarver
    binary_carver = BinaryCarver(
        features=features,
        min_freq=0.02,  # minimum frequency per modality
        max_n_mod=5,  # maximum number of modality per Carved feature
        verbose=True,  # showing statistics
    )

    # fitting on training sample, a dev sample can be specified to evaluate carving robustness
    x_discretized = binary_carver.fit_transform(train_set, train_set[target], X_dev=dev_set, y_dev=dev_set[target])



Applying AutoCarver
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # transforming dev/test sample accordingly
    dev_set_discretized = binary_carver.transform(dev_set)
    test_set_discretized = binary_carver.transform(tes_set)



Saving AutoCarver
^^^^^^^^^^^^^^^^^

All **Carvers** can safely be serialized as a ``.json`` file.

.. code-block:: python

    binary_carver.save('my_carver.json')


Loading AutoCarver
^^^^^^^^^^^^^^^^^^

**Carvers** can safely be loaded from a ``.json`` file.

.. code-block:: python

    from AutoCarver import BinaryCarver

    binary_carver = BinaryCarver.load('my_carver.json')



Feature Selection
-----------------

.. code-block:: python

    from AutoCarver.selectors import ClassificationSelector

    # select the best 25 most target associated features
    classification_selector = ClassificationSelector(
        features=features,  # features to select from
        n_best_per_type=25,  # number of features to select per data type
        verbose=True,  # displays statistics
    )
    best_features = classification_selector.select(train_set_discretized, train_set_discretized[target])

