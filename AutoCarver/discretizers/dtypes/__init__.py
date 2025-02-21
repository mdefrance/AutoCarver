""" Discretizers for different data types. """

from .string_discretizer import StringDiscretizer
from .timedelta_discretizer import TimedeltaDiscretizer

__all__ = ["StringDiscretizer", "TimedeltaDiscretizer"]
