"""Animation spec for QuantitativeDiscretizer on a continuous feature with
multiple class-fare spikes — the scenario that forces OrdinalDiscretizer to
fire on top of ContinuousDiscretizer's quantile cut.

Plays Stages 0 → 4: raw KDE → over-rep markers → after-CD bars with rare
outlined → merge-direction arrows → after-QD merged bars.
"""

from __future__ import annotations

NAME = "quantitative_discretizer"
FEATURE = "quantitative"
TARGET = "binary"
STOP_AFTER_STAGE = 4
