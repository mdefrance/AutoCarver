
</p>
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/artwork/auto_carver_logo_dark.svg">
        <img alt="AutoCarver Logo" src="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/artwork/auto_carver_logo_light.svg" width="80%">
    </picture>
</p>

[![PyPI](https://img.shields.io/pypi/v/autocarver)](https://pypi.org/project/AutoCarver)
[![Python](https://img.shields.io/pypi/pyversions/autocarver)](https://pypi.org/project/AutoCarver/)
[![License](https://img.shields.io/github/license/mdefrance/autocarver)](LICENSE)
[![SPEC 0](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Docs](https://readthedocs.org/projects/autocarver/badge/?version=latest)](https://autocarver.readthedocs.io/en/latest/)
[![Tests](https://github.com/mdefrance/AutoCarver/actions/workflows/pytest.yml/badge.svg)](https://github.com/mdefrance/AutoCarver/actions/workflows/pytest.yml)
[![Coverage](https://codecov.io/gh/mdefrance/AutoCarver/branch/main/graph/badge.svg)](https://codecov.io/gh/mdefrance/AutoCarver)


<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/_static/animations/readme_full_pipeline_dark.svg">
        <img alt="AutoCarver in one loop: discretize, rank groupings, carve" src="https://raw.githubusercontent.com/mdefrance/AutoCarver/main/docs/source/_static/animations/readme_full_pipeline_light.svg" width="100%">
    </picture>
</p>


**AutoCarver** turns raw numeric, categorical, and ordinal columns into optimal, drift-robust, human-readable bins in a few lines of code. Stop losing model performance to suboptimal manual binning — and stop discovering overfit bins in production monitoring.

- **Provably optimal** — exhaustive search: for a fixed `min_freq`, `max_n_mod` and metric (Tschuprow's T by default, or Cramér's V), no other admissible bin combination scores higher.
- **Robust by construction** — every candidate grouping is vetoed unless it holds on a held-out dev set (and optional CV folds), at `fit` time rather than in monitoring.
- **Define → carve → model** — declare your `Features`, `fit` a carver, `transform`: the whole feature set is carved in one supervised pass, not one notebook per feature. One carver per target type — `BinaryCarver`, `MulticlassCarver`, `OrdinalCarver`, `ContinuousCarver` (regression) — all with the identical API.
- **AI-assisted** — a local MCP server lets your LLM assistant qualify and carve columns through tool calls, fully on your machine.

Built for credit scoring, fraud detection, and risk modeling.


## 🆕 What's New

**📊 Cross-validated robustness.** `fit` now accepts a `cv` argument for extra
held-out robustness views on top of (or instead of) a dev set:
`carver.fit(X, y, cv=5)`. Accepts an int, any scikit-learn splitter, or
explicit index pairs, resolved via `sklearn.model_selection.check_cv` — folds
veto over-fit combinations but never reorder them (ranks stay anchored to the
full train set). See [Cross-validation folds](https://autocarver.readthedocs.io/en/latest/viability.html#cross-validation-folds).

**🤖 LLM & MCP integration.** AutoCarver now ships a local [Model Context Protocol](https://modelcontextprotocol.io) server: point an MCP-aware assistant (VS Code Copilot, Claude Desktop, Cursor, …) at a data file and let it *qualify* the columns and *carve* them against your target through tool calls. The server runs **fully on your machine** — your dataset is never sent to AutoCarver or any external service (only your own LLM provider sees what the assistant shares). Carving quality depends on the LLM, so have a human confirm the feature definitions before production use. See the [LLM & MCP guide](https://autocarver.readthedocs.io/en/latest/mcp.html).

```bash
pip install "autocarver[mcp]"
```


## Install

```bash
pip install autocarver
```


## Quick Start

You already have a DataFrame and a target — that's step 1 of 6 done. The remaining five lines-worth take you to carved, dev-validated bins. Binary classification on the Titanic dataset:

<!-- quick-start:start -->
```python
from pathlib import Path

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
    numericals=["Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"],
    ordinals={"Pclass": ["1", "2", "3"]},
)

# 4. Fit the carver (dev set drives the robustness checks)
carver = BinaryCarver(features=features)
train_processed = carver.fit_transform(train, train[target], X_dev=dev, y_dev=dev[target])
dev_processed = carver.transform(dev)

# 5. Inspect the carved buckets, target rate, and association
print(carver.summary)

# 6. Persist for later use
carver.save(Path("titanic_carver.json"))
# carver = BinaryCarver.load(Path("titanic_carver.json"))
```
<!-- quick-start:end -->

`min_freq` and `max_n_mod` are the only two knobs that matter to start with — the defaults (`0.02` / `5`) reflect common scoring practice, and every behavioral toggle lives in one `ProcessingConfig` object. Scan, adjust, move on.

For multiclass classification use `MulticlassCarver` (one binning per feature, against the full K-class target) — or `OneVsRestCarver` for a separate binning per class; for ordinal targets use `OrdinalCarver`; for regression use `ContinuousCarver` — the API is identical. To pre-select features by target association and inter-feature redundancy, pipe the carved output through `ClassificationSelector` or `RegressionSelector`.


## What you get

- **No performance left on the table** — exhaustive search over admissible bin combinations maximizes Tschuprow's T (default) or Cramér's V: for fixed `min_freq`, `max_n_mod` and metric, no other combination scores higher, so you never wonder whether a better grouping existed.
- **Stop silent overfitting before production** — bins that only exist in your training sample degrade quietly under drift. Every candidate combination is validated on a dev set (and optional CV folds): any whose target rates flip or whose buckets fall below `min_freq` is rejected at fit time, not discovered in monitoring.
- **First-class ordinal features** — `OrdinalDiscretizer` enforces your declared modality order, so under-represented levels are merged with their nearest neighbour instead of being collapsed by frequency.
- **You are the final auditor** — `features.summary` and `features.history` expose the bin definitions, per-bin target rate / frequency, and the full carving trace; disagree with a boundary and you can override it, and `transform` applies your fix like any carved bin:

  ```python
  feature = features("Siblings/Spouses Aboard")  # any fitted feature; labels are [0, 1, 2]
  feature.group([1], 2)  # merge two bins you consider equivalent
  ```
- **Interpretable buckets** — human-readable boundaries you can audit, document, and ship to a scorecard.
- **Dimensionality reduction** — groups under-represented modalities and caps bins per feature (`max_n_mod`), which is especially useful before one-hot encoding.
- **Feature pre-selection** — `ClassificationSelector` / `RegressionSelector` rank features by target association and filter on inter-feature correlation.


## How does it compare?

|                                                   | **Manual binning**                                  | **AutoCarver**                                                     | [**optbinning**](https://github.com/guillermo-navas-palencia/optbinning) | [**sklearn KBinsDiscretizer**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) |
| ------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Supervised (uses `y`)                             | only as far as your patience goes                    | yes                                                                | yes                                                                      | no                                                                              |
| Algorithm                                         | eyeballing distributions, notebook by notebook       | **exhaustive search** over admissible combinations                 | mixed-integer program (CBC)                                              | quantile / uniform / k-means                                                    |
| Optimality for given `min_freq` / `max_n_mod` / metric | none — first acceptable grouping wins           | **guaranteed — best of every admissible combination**              | provably optimal under MIP constraints                                   | n/a — no target objective                                                       |
| Target types                                      | any, at ~1 feature/hour                              | **binary, multiclass, ordinal, continuous**                        | binary, multiclass, continuous                                           | n/a                                                                             |
| Numeric **and** categorical **and** ordinal in one `fit` | each feature is its own project               | yes                                                                | one binner per feature                                                   | numeric only                                                                    |
| Ordinal features with enforced order              | if you remember to                                   | **yes — `OrdinalDiscretizer` preserves your declared order**       | via `user_splits` workaround (loses ordering)                            | no                                                                              |
| `NaN` handled as its own modality                 | usually forgotten                                    | yes                                                                | yes                                                                      | no (raises)                                                                     |
| Held-out dev-set robustness check                 | rarely — too tedious to script per feature           | **yes — dev set + optional k-fold CV, built into `fit`**           | no (script CV yourself)                                                  | no                                                                              |
| Per-bin stats + carving history after `fit`       | scattered notebook cells                             | **`features.summary`, `features.history`**                         | `binning_table`                                                          | no                                                                              |
| JSON round-trip persistence                       | copy-pasted bound lists                              | yes (`carver.save("...json")`)                                     | via `pickle`                                                             | via `pickle`                                                                    |
| sklearn `Pipeline` compatible                     | no                                                   | yes                                                                | yes                                                                      | yes                                                                             |
| Feature pre-selection helpers                     | no                                                   | `ClassificationSelector`, `RegressionSelector`                     | no                                                                       | no                                                                              |

Side-by-side runnable snippets and a "when to pick which" guide live on the [comparison page](https://autocarver.readthedocs.io/en/latest/comparison.html).


## Documentation

Full reference, tutorials, and end-to-end notebook examples on [ReadTheDocs](https://autocarver.readthedocs.io/en/latest/index.html).
