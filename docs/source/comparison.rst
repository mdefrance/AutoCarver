.. _comparison:

Comparison with other binning libraries
=======================================

Three Python libraries are usually considered for feature discretization:

* **AutoCarver** — supervised, target-association-driven binning with dev-set robustness validation.
* `optbinning <https://github.com/guillermo-navas-palencia/optbinning>`_ — supervised binning: CART pre-binning, then constraint / mixed-integer programming over the pre-bins.
* `sklearn.preprocessing.KBinsDiscretizer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html>`_ — unsupervised quantile / uniform / k-means binning.

This page compares them on scope, algorithm, and ergonomics so you can pick the right tool for your problem. The runnable code snippets are unit-tested in ``tests/examples/test_comparison_snippets.py``.

The question is not only what each library adds, but what you lose without it: skip ordinal handling and the declared modality order of your features is gone; skip a dev-set veto and nothing stops a bin that only exists in your training sample.


Scope at a glance
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * -
     - **AutoCarver**
     - **optbinning**
     - **KBinsDiscretizer**
   * - Supervised (uses ``y``)
     - yes
     - yes
     - no
   * - Binary classification
     - :class:`BinaryCarver`
     - ``OptimalBinning``
     - n/a
   * - Multiclass classification
     - :class:`MulticlassCarver` — **one binning per feature** (joint, against the full K-class target); :class:`OneVsRestCarver` available for a per-class binning instead
     - ``MulticlassOptimalBinning`` — one-vs-rest only (a separate binning per class)
     - n/a
   * - Regression / continuous target
     - :class:`ContinuousCarver`
     - ``ContinuousOptimalBinning``
     - n/a
   * - Quantitative features
     - yes
     - yes
     - yes
   * - Categorical features
     - yes
     - yes
     - no (must encode first)
   * - Ordinal features (with known order)
     - yes (:class:`OrdinalDiscretizer` enforces the declared order)
     - via ``user_splits`` workaround
     - no
   * - ``NaN`` as own modality
     - yes
     - yes
     - no (raises)
   * - Held-out dev-set robustness check
     - **yes (built in)**
     - no
     - no
   * - Optimality guarantee for fixed ``min_freq`` / ``max_n_mod`` / metric
     - **yes — exhaustive top-K search over admissible combinations (interval dynamic programming, DP)**
     - yes, over its CART pre-bins (CP / MIP, under its own constraints)
     - n/a (no objective)
   * - Confidence-interval-guarded ``min_freq``
     - **yes — Wilson score interval (tunable** ``min_freq_alpha`` **)**
     - no (hard threshold)
     - n/a
   * - Per-feature parallelism for hundreds-to-thousands of features
     - yes (``n_jobs`` via ``multiprocessing.Pool``)
     - yes (``BinningProcess(n_jobs=...)``)
     - yes (sklearn-native ``n_jobs`` semantics)
   * - Per-bin stats + carving history after ``fit``
     - **yes —** ``Features.summary`` **and** ``Features.history``
     - via ``binning_table``
     - no
   * - JSON round-trip persistence
     - yes
     - via pickle
     - via pickle
   * - sklearn ``Pipeline`` compatible
     - yes (``BaseEstimator`` / ``TransformerMixin``)
     - yes
     - yes
   * - Feature pre-selection helpers
     - :class:`ClassificationSelector`, :class:`RegressionSelector`
     - no
     - no


Algorithmic axis
----------------

The three libraries answer "what's a good bin?" with very different objectives:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Library
     - Objective
     - Constraint surface
   * - **AutoCarver**
     - Maximize **Tschuprow's T** (default) or **Cramér's V** between the carved feature and the binary target — generalised to a K-class :math:`\chi^2` for the joint :class:`MulticlassCarver` (see :ref:`MulticlassChi2`) — or **Kruskal-Wallis H** for continuous targets — via **exhaustive top-K interval DP** over consecutive segmentations. The DP exploits additive decomposability of :math:`H` (and of :math:`\chi^2` at fixed :math:`k`) to enumerate the top-K partitions in closed form; progressive top-K doubling keeps the worst case exhaustive while making the common case essentially free. For fixed ``min_freq``, ``max_n_mod`` and metric, no other admissible combination scores higher. NaN groupings are fanned out and re-scored in closed form. See :ref:`DPTopK` for details and parity guarantees against ``scipy.stats``.
     - ``min_freq`` (minimum bucket share, gated by a Wilson score CI at significance ``min_freq_alpha`` — see :ref:`MinFreqViability`), ``max_n_mod`` (cap on number of modalities), monotonic ordering for ordinal features (enforced by :class:`OrdinalDiscretizer`), and a dev-set veto: any candidate that flips its target-rate ordering on the dev set is rejected.
   * - **optbinning**
     - Maximize **Information Value (IV)** (binary) or split-gain analogues. A CART decision tree first produces pre-bins (``prebinning_method="cart"`` by default); the optimal merge of those pre-bins is then solved with constraint programming (CP-SAT, the default ``solver="cp"``) or a mixed-integer program (``solver="mip"``).
     - User-declarable monotonicity, minimum bin size, maximum number of bins, optional WoE smoothing, and constraint blocks (e.g. PSI-based stability).
   * - **KBinsDiscretizer**
     - **No target awareness.** Splits are placed on the marginal distribution of ``X`` only: equal-frequency (``quantile``), equal-width (``uniform``), or 1-D k-means.
     - ``n_bins`` per feature; that's it.

The takeaway: **AutoCarver and optbinning both optimize against the target**, but AutoCarver's robustness step (the dev-set veto, with a Wilson-CI-guarded ``min_freq`` check on both train and dev) is something optbinning does not do natively — you'd have to script it yourself with cross-validation. The two libraries place their stability check at opposite ends: optbinning constrains PSI *while* solving for bins, AutoCarver vetoes rank inversions *while* carving and then re-runs the whole viability filter — plus PSI and a :math:`\chi^2` drift test — against any later sample (see :ref:`Stability`). KBinsDiscretizer is a different category: it's a fast preprocessing primitive, not a supervised binner.


Side-by-side: bin a mixed feature set on the same data
-------------------------------------------------------

The same problem — discretize four numeric columns and one categorical column of the Titanic data — solved three ways. All three blocks are runnable; the optbinning and KBinsDiscretizer blocks are skipped automatically in CI when those libraries are not installed.


AutoCarver
^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from AutoCarver import BinaryCarver, Features

    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(url)
    target = "Survived"
    train, dev = train_test_split(data, test_size=0.33, random_state=42, stratify=data[target])

    features = Features(
        categoricals=["Sex"],
        numericals=["Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"],
        ordinals={"Pclass": ["1", "2", "3"]},
    )
    carver = BinaryCarver(features=features, min_freq=0.05, max_n_mod=5)
    carver.fit(train, train[target], X_dev=dev, y_dev=dev[target])
    train_binned = carver.transform(train)

* **One call** covers numeric, categorical, and ordinal columns.
* The dev set is consumed at ``fit`` time: any bin combination whose target-rate ordering doesn't survive on the dev sample is discarded.
* Persisting the fitted state is ``carver.save("titanic_carver.json")``.


optbinning
^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from optbinning import BinningProcess

    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(url)
    target = "Survived"
    train, _ = train_test_split(data, test_size=0.33, random_state=42, stratify=data[target])

    variable_names = [
        "Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard",
        "Sex", "Pclass",
    ]
    binning_process = BinningProcess(
        variable_names=variable_names,
        # Pclass is nominal here: optbinning has no first-class ordinal type
        categorical_variables=["Sex", "Pclass"],
    )
    binning_process.fit(train[variable_names], train[target])
    train_binned = pd.DataFrame(
        binning_process.transform(train[variable_names], metric="bins"),
        columns=variable_names,
        index=train.index,
    )

* ``BinningProcess`` bins all declared columns in **one** ``fit`` (a per-feature ``OptimalBinning`` API also exists).
* No held-out validation step; you'd add cross-validation yourself.
* Ordinal columns must be passed as ``categorical`` (with optional ``user_splits``), losing the known order.


KBinsDiscretizer
^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import KBinsDiscretizer

    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(url)
    target = "Survived"
    train, _ = train_test_split(data, test_size=0.33, random_state=42, stratify=data[target])

    numeric_cols = ["Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"]
    train_numeric = train[numeric_cols].fillna(train[numeric_cols].median())

    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    train_binned = pd.DataFrame(
        kbd.fit_transform(train_numeric),
        columns=numeric_cols,
        index=train.index,
    )

* **Unsupervised** — the target is never used, so the bins do not maximize anything related to ``y``.
* No support for categoricals or ``NaN`` — you must impute and encode first.
* Strong baseline when you need fast, model-agnostic binning and you accept that bins won't be target-optimal.


When to pick which
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pick
     - When
   * - **AutoCarver**
     - You want supervised binning **and** you have (or can carve out) a dev sample, you mix numeric / categorical / ordinal columns, you need a JSON-portable artifact to ship to a scorecard or production model, or you also need feature pre-selection.
   * - **optbinning**
     - You want IV-driven binning solved as a true optimization problem, you need fine-grained per-feature constraints (monotonicity, WoE smoothing, PSI-based stability), and you are comfortable managing validation yourself.
   * - **KBinsDiscretizer**
     - You need a fast, unsupervised preprocessing step inside an sklearn ``Pipeline`` — e.g. as input to a tree-free linear model — and you don't need target-aware bins.

A reasonable rule of thumb: reach for **KBinsDiscretizer** when binning is a *preprocessing* concern, **AutoCarver** when binning is a *modelling* concern with a held-out validation budget, and **optbinning** when you need to encode hard business constraints into each feature's bin definition.


Benchmark notebook
------------------

A runnable side-by-side benchmark on two public datasets — Home Credit Default Risk (binary, mixed dtypes, via Kaggle) and Allstate Claims Severity (regression, mixed dtypes, via Kaggle) — comparing the three libraries on fit time, downstream-model score, ``train`` → ``test`` score drop, and one-hot model size:

.. toctree::
    :glob:
    :maxdepth: 1

    examples/Comparison/comparison_notebook

The numbers are illustrative — single run, single machine, fixed seed — and are **not** an IV / Tschuprow's T leaderboard, since those metrics structurally favour the library whose objective they are. Re-run on your own data before drawing conclusions.


Caveats
-------

* All three libraries are actively maintained; the table reflects the public APIs as of AutoCarver |release| (2026-05). Open an issue if anything has drifted.
* The DP top-K search strategy is statistically equivalent to the previous enumerate-and-score path: parity tests pin bit-exact agreement against :func:`scipy.stats.kruskal` (continuous) and :func:`scipy.stats.chi2_contingency` (binary, including the Yates correction). Performance numbers in older issues, pre-DP, are no longer representative.
