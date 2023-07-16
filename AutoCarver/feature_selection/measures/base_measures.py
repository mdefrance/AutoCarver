""" Base measures that defines Quantitative and Qualitative features.
"""
from typing import Any

from pandas import Series


def nans_measure(
    active: bool,
    association: dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> tuple[bool, dict[str, Any]]:
    """Measure of the percentage of NaNs

    Parameters
    ----------
    thresh_nan, float: default 1.
      Maximum percentage of NaNs in a feature
    """

    # whether or not tests where passed
    if active:
        nans = x.isnull()  # ckecking for nans
        pct_nan = nans.mean()  # Computing percentage of nans

        # updating association
        association.update({"pct_nan": pct_nan})

        # Excluding feature that have to many NaNs
        thresh_nan = params.get("thresh_nan", 0.999)
        active = pct_nan < thresh_nan
        if not active:
            print(
                f"Feature {x.name} will be discarded (more than {thresh_nan:2.2%} of nans). Otherwise, set a greater value for thresh_nan."
            )

    return active, association


def dtype_measure(
    active: bool,
    association: dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> tuple[bool, dict[str, Any]]:
    """Gets dtype"""

    # updating association
    association.update({"dtype": x.dtype})

    return active, association


def mode_measure(
    active: bool,
    association: dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> tuple[bool, dict[str, Any]]:
    """Measure of the percentage of the Mode

    Parameters
    ----------
    thresh_mode, float: default 1.
      Maximum percentage of the mode of a feature
    """

    # whether or not tests where passed
    if active:
        mode = x.mode(dropna=True).values[0]  # computing mode
        pct_mode = (x == mode).mean()  # Computing percentage of the mode

        # updating association
        association.update({"pct_mode": pct_mode, "mode": mode})

        # Excluding feature with too frequent modes
        thresh_mode = params.get("thresh_mode", 0.999)
        active = pct_mode < thresh_mode
        if not active:
            print(
                f"Feature {x.name} will be discarded (more than {thresh_mode:2.2%} of its mode). Otherwise, set a greater value for thresh_mode."
            )

    return active, association
