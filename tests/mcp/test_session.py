"""Tests for the stateful CarverSession (qualify -> carve workflow)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import fixture, raises

from AutoCarver.mcp.session import CarverSession


@fixture
def dataset(tmp_path):
    """A small csv with categorical, numerical, nested and binary-target columns."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "cat": rng.choice(["A", "B", "C"], size=n),
            "num": rng.normal(size=n),
            "child": rng.choice(["A1", "A2", "B1"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    # child rolls cleanly into parent (A* -> A, B* -> B)
    X["parent"] = X["child"].str[0]
    path = tmp_path / "data.csv"
    X.to_csv(path, index=False)
    return str(path)


def test_load_dataset_reports_shape(dataset):
    session = CarverSession()
    info = session.load_dataset(dataset, target="target")
    assert info["rows"] == 300
    assert info["target"] == "target"
    assert "cat" in info["columns"]


def test_load_dataset_unknown_target_raises(dataset):
    session = CarverSession()
    with raises(ValueError, match="target"):
        session.load_dataset(dataset, target="nope")


def test_methods_require_loaded_dataset():
    session = CarverSession()
    with raises(ValueError, match="no dataset loaded"):
        session.list_columns()


def test_suggest_features_excludes_target(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    draft = session.suggest_features()
    assert "target" not in draft
    assert draft["num"] == {"type": "numerical"}
    assert draft["cat"] == {"type": "categorical"}


def test_set_and_drop_feature(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    session.suggest_features()

    session.set_feature("child", "nested", parents=["parent"])
    assert session.draft["child"] == {"type": "nested", "parents": ["parent"]}

    draft = session.drop_feature("parent")
    assert "parent" not in draft


def test_set_feature_validates_required_options(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    with raises(ValueError, match="ordinal requires"):
        session.set_feature("cat", "ordinal")


def test_validate_nesting_through_session(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    assert session.validate_nesting("child", ["parent"])["valid"] is True


def test_run_carver_binary_auto(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    session.suggest_features()
    session.drop_feature("child")
    session.drop_feature("parent")

    result = session.run_carver(task="auto", min_freq=0.1, max_n_mod=4)
    assert result["task"] == "binary"
    assert "num" in result["kept_features"]
    assert isinstance(result["summary"], list)


def test_save_carver_roundtrips(dataset, tmp_path):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    session.suggest_features()
    session.drop_feature("child")
    session.drop_feature("parent")
    session.run_carver(task="auto", min_freq=0.1, max_n_mod=4)

    path = str(tmp_path / "carver.json")
    info = session.save_carver(path)
    assert info["saved"] == path

    from AutoCarver.carvers import BinaryCarver

    reloaded = BinaryCarver.load(Path(path))
    assert reloaded.is_fitted
    assert "num" in [feature.version for feature in reloaded.features]


def test_evaluate_stability_scores_a_new_extract(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    session.suggest_features()
    session.drop_feature("child")
    session.drop_feature("parent")
    session.run_carver(task="auto", min_freq=0.1, max_n_mod=4)

    # the very sample it was carved on: perfectly stable by construction
    report = session.evaluate_stability(dataset)
    assert report["has_target"] is True
    assert report["unstable_features"] == []
    assert all(row["psi"] < 1e-9 for row in report["per_feature"])
    assert json.loads(json.dumps(report))  # MCP transports plain JSON


def test_evaluate_stability_without_a_target(dataset, tmp_path):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    session.suggest_features()
    session.drop_feature("child")
    session.drop_feature("parent")
    session.run_carver(task="auto", min_freq=0.1, max_n_mod=4)

    unlabelled = pd.read_csv(dataset).drop(columns=["target"])
    path = str(tmp_path / "unlabelled.csv")
    unlabelled.to_csv(path, index=False)

    report = session.evaluate_stability(path)
    assert report["has_target"] is False
    assert all(row["viable"] is None for row in report["per_feature"])
    assert all(row["psi"] is not None for row in report["per_feature"])


def test_evaluate_stability_requires_run(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    with raises(ValueError, match="no fitted carver"):
        session.evaluate_stability(dataset)


def test_evaluate_stability_rejects_an_unknown_named_target(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    session.suggest_features()
    session.drop_feature("child")
    session.drop_feature("parent")
    session.run_carver(task="auto", min_freq=0.1, max_n_mod=4)

    with raises(ValueError, match="not found in columns"):
        session.evaluate_stability(dataset, target="nope")


def test_save_carver_requires_run(dataset):
    session = CarverSession()
    session.load_dataset(dataset, target="target")
    with raises(ValueError, match="no fitted carver"):
        session.save_carver("carver.json")


def test_run_carver_requires_target_and_draft(dataset):
    session = CarverSession()
    session.load_dataset(dataset)  # no target
    with raises(ValueError, match="no target"):
        session.run_carver()

    session.load_dataset(dataset, target="target")
    with raises(ValueError, match="draft is empty"):
        session.run_carver()
