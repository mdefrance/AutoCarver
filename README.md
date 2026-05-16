
<p align="center">
    <img alt="AutoCarver Logo" src="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/artwork/auto_carver_symbol_small.png" width="25%">
</p>


[![PyPI](https://img.shields.io/pypi/v/autocarver)](https://pypi.org/project/AutoCarver)
[![Python](https://img.shields.io/pypi/pyversions/autocarver)](https://pypi.org/project/AutoCarver/)
[![License](https://img.shields.io/github/license/mdefrance/autocarver)](LICENSE)
[![SPEC 0](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Docs](https://readthedocs.org/projects/autocarver/badge/?version=latest)](https://autocarver.readthedocs.io/en/latest/)
[![Tests](https://github.com/mdefrance/AutoCarver/actions/workflows/pytest.yml/badge.svg)](https://github.com/mdefrance/AutoCarver/actions/workflows/pytest.yml)
[![Coverage](https://codecov.io/gh/mdefrance/AutoCarver/branch/main/graph/badge.svg)](https://codecov.io/gh/mdefrance/AutoCarver)


**AutoCarver** automates supervised feature discretization (binning) to maximize statistical association with your target — using Tschuprow's T or Cramér's V — and validates the chosen bins against a held-out dev set. It supports **binary classification**, **multiclass classification**, and **regression**, and is widely used for credit scoring, fraud detection, and risk modeling.


## Install

```bash
pip install autocarver
```


## Quick Start

Binary classification on the Titanic dataset:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from AutoCarver import BinaryCarver, Features

# 1. Load data
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)
target = "Survived"

# 2. Train / dev split, stratified on the target
train, dev = train_test_split(
    data, test_size=0.33, random_state=42, stratify=data[target]
)

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
```

For multiclass classification use `MulticlassCarver`; for regression use `ContinuousCarver` — the API is identical. To pre-select features by target association and inter-feature redundancy, pipe the carved output through `ClassificationSelector` or `RegressionSelector`.


## Why AutoCarver?

- **Optimal supervised binning** — maximizes Tschuprow's T (default) or Cramér's V between each feature and the target instead of relying on hand-tuned quantiles.
- **Robust to data drift** — every candidate bin combination is validated on a dev set, rejecting any whose target rates flip or whose buckets fall below `min_freq`.
- **Interpretable buckets** — human-readable boundaries you can audit, document, and ship to a scorecard.
- **Dimensionality reduction** — groups under-represented modalities and caps bins per feature (`max_n_mod`), which is especially useful before one-hot encoding.
- **Feature pre-selection** — `ClassificationSelector` / `RegressionSelector` rank features by target association and filter on inter-feature correlation.


## Documentation

Full reference, tutorials, and end-to-end notebook examples on [ReadTheDocs](https://autocarver.readthedocs.io/en/latest/index.html).
