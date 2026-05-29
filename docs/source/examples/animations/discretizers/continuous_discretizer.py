"""Animation spec for ContinuousDiscretizer on a continuous feature.

Plays Stages 0 → 2: raw distribution → over-rep value flagged → after-CD bins.
"""

from __future__ import annotations

NAME = "continuous_discretizer"
FEATURE = "continuous"
TARGET = "binary"  # CD doesn't use y; kept for the (feature, target) variant key
STOP_AFTER_STAGE = 2
