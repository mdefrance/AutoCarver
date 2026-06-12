
</p>
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/artwork/auto_carver_symbol_dark.svg">
        <img alt="AutoCarver Logo" src="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/artwork/auto_carver_symbol_light.svg" width="240">
    </picture>
</p>

<p align="center">
    <img alt="AutoCarver in one loop: discretize, rank groupings, carve" src="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/_static/animations/readme_full_pipeline.svg" width="100%">
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

<!-- quick-start:start -->
```python
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
```
<!-- quick-start:end -->

For multiclass classification use `MulticlassCarver`; for regression use `ContinuousCarver` — the API is identical. To pre-select features by target association and inter-feature redundancy, pipe the carved output through `ClassificationSelector` or `RegressionSelector`.


## Why AutoCarver?

- **Optimal supervised binning** — exhaustive search over admissible bin combinations maximizes Tschuprow's T (default) or Cramér's V. For fixed `min_freq`, `max_n_mod` and metric, no other combination scores higher.
- **Robust to data drift** — every candidate bin combination is validated on a dev set, rejecting any whose target rates flip or whose buckets fall below `min_freq`.
- **First-class ordinal features** — `OrdinalDiscretizer` enforces your declared modality order, so under-represented levels are merged with their nearest neighbour instead of being collapsed by frequency.
- **Inspect what was carved** — `features.summary` and `features.history` give you the bin definitions, per-bin target rate / frequency, and the full carving trace right off the fitted carver.
- **Interpretable buckets** — human-readable boundaries you can audit, document, and ship to a scorecard.
- **Dimensionality reduction** — groups under-represented modalities and caps bins per feature (`max_n_mod`), which is especially useful before one-hot encoding.
- **Feature pre-selection** — `ClassificationSelector` / `RegressionSelector` rank features by target association and filter on inter-feature correlation.


## How does it compare?

|                                                   | **AutoCarver**                                                     | [**optbinning**](https://github.com/guillermo-navas-palencia/optbinning) | [**sklearn KBinsDiscretizer**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) |
| ------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Supervised (uses `y`)                             | yes                                                                | yes                                                                      | no                                                                              |
| Algorithm                                         | **exhaustive search** over admissible combinations                 | mixed-integer program (CBC)                                              | quantile / uniform / k-means                                                    |
| Optimality for given `min_freq` / `max_n_mod` / metric | **guaranteed — best of every admissible combination**          | provably optimal under MIP constraints                                   | n/a — no target objective                                                       |
| Target types                                      | binary, multiclass, continuous                                     | binary, multiclass, continuous                                           | n/a                                                                             |
| Numeric **and** categorical **and** ordinal in one `fit` | yes                                                          | one binner per feature                                                   | numeric only                                                                    |
| Ordinal features with enforced order              | **yes — `OrdinalDiscretizer` preserves your declared order**       | via `user_splits` workaround (loses ordering)                            | no                                                                              |
| `NaN` handled as its own modality                 | yes                                                                | yes                                                                      | no (raises)                                                                     |
| Held-out dev-set robustness check                 | **yes — built into `fit`**                                         | no (script CV yourself)                                                  | no                                                                              |
| Per-bin stats + carving history after `fit`       | **`features.summary`, `features.history`**                         | `binning_table`                                                          | no                                                                              |
| JSON round-trip persistence                       | yes (`carver.save("...json")`)                                     | via `pickle`                                                             | via `pickle`                                                                    |
| sklearn `Pipeline` compatible                     | yes                                                                | yes                                                                      | yes                                                                             |
| Feature pre-selection helpers                     | `ClassificationSelector`, `RegressionSelector`                     | no                                                                       | no                                                                              |

Side-by-side runnable snippets and a "when to pick which" guide live on the [comparison page](https://autocarver.readthedocs.io/en/latest/comparison.html).


## Documentation

Full reference, tutorials, and end-to-end notebook examples on [ReadTheDocs](https://autocarver.readthedocs.io/en/latest/index.html).
