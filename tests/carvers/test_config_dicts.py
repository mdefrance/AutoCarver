"""Configs are TypedDicts: a plain dict and the class call are interchangeable, unknown keys raise."""

from pathlib import Path

import pandas as pd
from pytest import raises

from AutoCarver import BinaryCarver, ClassificationSelector, Features, SelectionConfig
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import FeaturesConfig


def _data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"cat": ["a", "b", "c", "d"] * 50, "num": list(range(200))})
    y = pd.Series([0, 1, 0, 1] * 25 + [1, 1, 0, 1] * 25)
    return X, y


def _carver(config) -> BinaryCarver:
    return BinaryCarver(features=Features(categoricals=["cat"], numericals=["num"]), config=config)


def test_dict_and_class_call_give_the_same_carver():
    X, y = _data()
    from_dict = _carver({"min_freq_alpha": 0.01, "n_jobs": 1}).fit(X, y)
    from_call = _carver(ProcessingConfig(min_freq_alpha=0.01, n_jobs=1)).fit(X, y)

    assert from_dict.config == from_call.config
    assert from_dict.transform(X).equals(from_call.transform(X))


def test_partial_dict_keeps_the_carver_defaults():
    carver = _carver({"verbose": True})

    assert carver.config["verbose"] is True
    assert carver.config["dropna"] is True  # carver context default
    assert carver.config["ordinal_encoding"] is True
    assert carver.config["rescue_rare"] is True
    assert carver.config["min_freq_alpha"] == 0.05
    assert set(carver.config) == set(ProcessingConfig.__annotations__)


def test_unknown_keys_raise():
    with raises(TypeError, match="drop_na"):
        _carver({"drop_na": True})
    with raises(TypeError, match="verbos"):
        ClassificationSelector(Features(categoricals=["cat"]), 1, config={"verbos": True})
    with raises(TypeError, match="defaults"):
        Features(categoricals=["cat"], config={"defaults": "__OTHER__"})


def test_selection_and_features_configs_accept_dicts():
    given = {"verbose": True}
    selector = ClassificationSelector(Features(categoricals=["cat"]), 1, config=given)
    assert selector.config == SelectionConfig(verbose=True)  # stored as given (a copy)
    assert selector.config is not given

    features = Features(categoricals=["cat"], config=FeaturesConfig(default="__OTHER__"))
    assert features("cat").default == "__OTHER__"


def test_dict_config_survives_save_load(tmp_path: Path):
    X, y = _data()
    carver = _carver({"min_freq_alpha": 0.01, "dp_escalate": True}).fit(X, y)
    carver.save(tmp_path / "carver.json")
    loaded = BinaryCarver.load(tmp_path / "carver.json")

    assert loaded.config == carver.config
    assert loaded.transform(X).equals(carver.transform(X))


def test_progress_bar_follows_verbose(monkeypatch):
    """verbose=False shows no progress bar anywhere (tqdm.auto would render a notebook widget);
    verbose=True does, on the serial and the parallel (n_jobs > 1) paths."""
    import AutoCarver.carvers.utils.base_carver as base_carver
    from AutoCarver.combinations.utils import combination_evaluator

    disabled: list[bool] = []

    def recording_tqdm(iterable=None, *args, disable=False, **kwargs):
        disabled.append(bool(disable))
        return iterable

    monkeypatch.setattr(base_carver, "tqdm", recording_tqdm)
    monkeypatch.setattr(combination_evaluator, "tqdm", recording_tqdm)
    X, y = _data()

    for verbose in (False, True):
        for n_jobs in (1, 2):
            disabled.clear()
            _carver({"verbose": verbose, "n_jobs": n_jobs}).fit(X, y)
            assert disabled, (verbose, n_jobs)
            assert all(d is (not verbose) for d in disabled), (verbose, n_jobs, disabled)
