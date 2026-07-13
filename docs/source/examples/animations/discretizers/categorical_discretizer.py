"""Animation spec for CategoricalDiscretizer on a categorical feature.

Plays Stages 0 → 2: raw bars + target-rate dots → rare modalities collapsed
into __OTHER__ → bars reordered by ascending target rate (dot trace monotonic).
"""

NAME = "categorical_discretizer"
FEATURE = "categorical"
TARGET = "binary"
STOP_AFTER_STAGE = 2
