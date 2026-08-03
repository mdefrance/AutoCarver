"""Pretty print functions for selectors"""

import pandas as pd

from AutoCarver.features import BaseFeature
from AutoCarver.selectors.filters import BaseFilter
from AutoCarver.selectors.measures import BaseMeasure

RANKED_COLUMNS = ["measure", "association", "rank", "filter", "redundancy", "filtered_with"]


def format_default_measures(feature: BaseFeature) -> dict:
    """Gate measures (Nan, Mode, outliers) keep one column each: they apply to every feature."""
    return {
        name: payload.get("value")
        for name, payload in feature.measures.items()
        if payload.get("info", {}).get("is_default")
    }


def format_ranking_measure(feature: BaseFeature, measure: BaseMeasure) -> dict:
    """Ranking measure: its *name* goes in a column so both types share ``association``."""
    name = measure.__name__.replace("Measure", "")
    return {
        "measure": name,
        "association": feature.measures.get(measure.__name__, {}).get("value"),
        "rank": feature.measures.get(f"{name}Rank", {}).get("value"),
    }


def format_redundancy_filter(feature: BaseFeature, filters: list[BaseFilter]) -> dict:
    """First applied redundancy filter, same name-in-a-column treatment."""
    for filter_ in filters:
        payload = feature.filters.get(filter_.__name__)
        if payload is not None:
            return {
                "filter": filter_.__name__.replace("Filter", ""),
                "redundancy": payload.get("value"),
                "filtered_with": payload.get("info", {}).get("correlation_with"),
            }
    return {"filter": None, "redundancy": None, "filtered_with": None}


def format_ranked_features(
    features: list[BaseFeature], measures: list[BaseMeasure], filters: list[BaseFilter]
) -> pd.DataFrame:
    """Builds one uniform per-feature association table for a single type branch.

    One row per feature, or per (feature, ranking measure) pair when several
    measures apply to the type. The ranking measure and redundancy filter are
    named in the ``measure``/``filter`` columns rather than getting a column
    each, so the qualitative and quantitative branches concatenate into a
    single non-ragged frame.
    """
    rows = []
    for feature in features:
        defaults = format_default_measures(feature)
        if len(measures) == 0:
            rows.append({"feature": feature, **defaults})
        for measure in measures:
            rows.append(
                {
                    "feature": feature,
                    **defaults,
                    **format_ranking_measure(feature, measure),
                    **format_redundancy_filter(feature, filters),
                }
            )

    if len(rows) == 0:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    default_columns = [column for column in frame.columns if column not in RANKED_COLUMNS + ["feature"]]
    frame = frame.reindex(columns=["feature"] + default_columns + [c for c in RANKED_COLUMNS if c in frame])
    if "rank" in frame:
        frame = frame.sort_values(by="rank", ascending=True, na_position="last")
    return frame
