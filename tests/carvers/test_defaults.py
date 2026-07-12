"""Smart defaults: min_freq/max_n_mod are optional, defaulting to 0.02 / 5
(the documented recommendation) so users can construct a carver without
tuning from scratch."""

from AutoCarver import BinaryCarver, ContinuousCarver, MulticlassCarver, OneVsRestCarver, OrdinalCarver
from AutoCarver.features import Features


def test_binary_carver_defaults():
    carver = BinaryCarver(features=Features(categoricals=["feature"]))
    assert carver.min_freq == 0.02
    assert carver.max_n_mod == 5


def test_continuous_carver_defaults():
    carver = ContinuousCarver(features=Features(categoricals=["feature"]))
    assert carver.min_freq == 0.02
    assert carver.max_n_mod == 5


def test_ordinal_carver_defaults():
    carver = OrdinalCarver(features=Features(categoricals=["feature"]))
    assert carver.min_freq == 0.02
    assert carver.max_n_mod == 5


def test_multiclass_carver_defaults():
    carver = MulticlassCarver(features=Features(categoricals=["feature"]))
    assert carver.min_freq == 0.02
    assert carver.max_n_mod == 5


def test_one_vs_rest_carver_defaults():
    carver = OneVsRestCarver(features=Features(categoricals=["feature"]))
    assert carver.min_freq == 0.02
    assert carver.max_n_mod == 5
