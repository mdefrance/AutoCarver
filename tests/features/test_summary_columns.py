"""Every concrete TargetRate's column must stay a summary column, not an index level."""

import pytest

import AutoCarver.combinations  # noqa: F401  (registers every TargetRate subclass)
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.features.features import per_modality_columns


def _concrete_target_rates(cls=TargetRate):
    for sub in cls.__subclasses__():
        yield from _concrete_target_rates(sub)
        if not getattr(sub, "__abstractmethods__", None):
            yield sub


@pytest.mark.parametrize("rate_cls", list(_concrete_target_rates()), ids=lambda c: c.__name__)
def test_target_rate_column_is_per_modality(rate_cls):
    # `rate_cls.__name__` is the real Python class name (e.g. "TargetMean"); the column key
    # actually used by `TargetRate.compute` is the instance-level `__name__` override
    # (e.g. "target_mean") — instantiate to read the value `compute()` really emits.
    assert rate_cls().__name__ in per_modality_columns()
