"""Quick-start example used by the README and by tests/examples/test_quick_start.py.

The body between the ``--8<-- [start:quick_start]`` / ``--8<-- [end:quick_start]``
markers is extracted verbatim (after dedent) by ``docs/sync_readme.py`` and
injected into ``README.md`` between the ``<!-- quick-start:start -->`` /
``<!-- quick-start:end -->`` sentinels. Keep both the markers and the
sentinels in place when editing.
"""

from __future__ import annotations


def main() -> None:
    """Trains a :class:`BinaryCarver` on the Titanic dataset.

    Writes ``titanic_carver.json`` to the current working directory.
    """
    # --8<-- [start:quick_start]
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from AutoCarver import BinaryCarver, Features

    # 1. Load data
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(url)
    target = "Survived"

    # 2. Train / dev split, stratified on the target
    train, dev = train_test_split(data, test_size=0.33, random_state=42, stratify=data[target])

    # 3. Declare features by type
    features = Features(
        categoricals=["Sex"],
        quantitatives=["Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"],
        ordinals={"Pclass": ["1", "2", "3"]},
    )

    # 4. Fit the carver (dev set drives the robustness checks)
    carver = BinaryCarver(features=features, min_freq=0.05, max_n_mod=5)
    train_processed = carver.fit_transform(train, train[target], X_dev=dev, y_dev=dev[target])
    dev_processed = carver.transform(dev)

    # 5. Inspect the carved buckets, target rate, and association
    print(carver.summary)

    # 6. Persist for later use
    carver.save("titanic_carver.json")
    # carver = BinaryCarver.load("titanic_carver.json")
    # --8<-- [end:quick_start]

    # silence unused-variable warnings without altering the snippet above
    _ = (train_processed, dev_processed)


if __name__ == "__main__":
    main()
