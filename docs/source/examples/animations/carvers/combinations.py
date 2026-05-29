"""Animation spec for the consecutive-combinations search — the core step
shared by every carver.

Starts from the QuantitativeDiscretizer stage-4 output (6 ordered bins) and
fills a table with consecutive groupings ranked by Tschuprow's T, best-first in
growing top-K batches (the progressive-doubling DP search). Plays Stages 0 → 3:
input strip → top-2 → top-4 → top-8 with the selected grouping highlighted.
"""

from __future__ import annotations

NAME = "combinations"
FEATURE = "combinations"
TARGET = "binary"
STOP_AFTER_STAGE = 3
