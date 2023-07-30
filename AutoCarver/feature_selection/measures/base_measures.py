""" Base measures that defines Quantitative and Qualitative features.
"""
from typing import Any, Callable

from pandas import Series


def make_measure(
    measure: Callable,
    active: bool,
    association: dict[str, Any],
    x: Series,
    y: Series,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Wrapper to make measures from base metrics

    Parameters
    ----------
    measure : Callable
        _description_
    active : bool
        _description_
    association : dict[str, Any]
        _description_
    x : Series
        Feature to measure
    y : Series
        Binary target feature

    Returns
    -------
    tuple[bool, dict[str, Any]]
        _description_
    """
    # check that previous steps where passed for computational optimization
    if active:
        # use the measure
        active, measurement = measure(x, y, **kwargs)

        # update association table
        association.update(measurement)

    return active, association


def nans_measure(
    x: Series,
    y: Series = None,
    thresh_nan: float = 0.999,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Measure of the percentage of NaNs

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``
    thresh_nan : float, optional
        Maximum percentage of NaNs in a feature, by default ``0.999``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether or not there are to many NaNs and the percentage of NaNs
    """
    nans = x.isnull()  # ckecking for nans
    pct_nan = nans.mean()  # Computing percentage of nans

    # updating association
    measurement = {"pct_nan": pct_nan}

    # Excluding feature that have to many NaNs
    active = pct_nan < thresh_nan
    if not active:
        print(
            f"Feature {x.name} will be discarded (more than {thresh_nan:2.2%} of nans). Otherwise, set a greater value for thresh_nan."
        )

    return active, measurement


def dtype_measure(
    x: Series,
    y: Series = None,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Feature's dtype

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        True and the feature's dtype
    """
    # getting dtype
    measurement = {"dtype": x.dtype}

    return True, measurement


def mode_measure(
    x: Series,
    y: Series = None,
    thresh_mode: float = 0.999,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Measure of the percentage of the Mode

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``
    thresh_mode : float, optional
        Maximum percentage of a feature's mode, by default ``0.999``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether or not the mode is overrepresented and the percentage of mode
    """
    mode = x.mode(dropna=True).values[0]  # computing mode
    pct_mode = (x == mode).mean()  # Computing percentage of the mode

    # updating association
    measurement = {"pct_mode": pct_mode, "mode": mode}

    # Excluding feature with too frequent modes
    active = pct_mode < thresh_mode
    if not active:
        print(
            f"Feature {x.name} will be discarded (more than {thresh_mode:2.2%} of its mode). Otherwise, set a greater value for thresh_mode."
        )

    return active, measurement
