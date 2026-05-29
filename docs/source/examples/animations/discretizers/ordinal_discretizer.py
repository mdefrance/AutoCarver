"""Animation spec for OrdinalDiscretizer on an ordinal feature.

Plays Stages 0 → 2: raw bars in declared ordinal order with rare modalities
outlined → merge-direction arrows drawn → merged bars spanning absorbed slots.
"""

from __future__ import annotations

NAME = "ordinal_discretizer"
FEATURE = "ordinal"
TARGET = "binary"
STOP_AFTER_STAGE = 2
