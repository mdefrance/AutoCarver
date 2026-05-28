"""Animation spec for QualitativeDiscretizer.

Two stacked strips: Port (categorical, top) and AgeGroup (ordinal, bottom).
Plays Stages 0 → 3: both strips raw → CategoricalDiscretizer applied (ordinal
dimmed) → OrdinalDiscretizer merge arrows → final merged state.
"""

from __future__ import annotations

NAME = "qualitative_discretizer"
FEATURE = "qualitative"
TARGET = "binary"
STOP_AFTER_STAGE = 4
