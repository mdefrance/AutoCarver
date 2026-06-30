"""Pretty print functions for selectors"""

import pandas as pd

from AutoCarver.features import BaseFeature


def format_measure(feature: BaseFeature, measure: dict) -> dict:
    """Format measure for display"""
    # formatting measure
    formatted = {}
    for measure_name, measure_value in measure.items():
        # adding feature name and measure value
        formatted.update({"feature": feature, measure_name: measure_value.get("value")})
        # adding correlation with
        if not measure_value.get("info", {}).get("is_default"):
            correlation_with = measure_value.get("info", {}).get("correlation_with")
            # formatting target correlation
            if correlation_with != "target" and correlation_with is not None:
                formatted.update({measure_name.replace("Filter", "With"): correlation_with})

    return formatted


def format_ranked_features(features: list[BaseFeature]) -> pd.DataFrame:
    """formats ranked features for display"""

    # adding measures and filters
    measures = []
    for feature in features:
        measures.append(format_measure(feature, {**feature.measures, **feature.filters}))

    # finding a sorting measure
    if len(measures) > 0:
        ranks = [col for measure in measures for col in measure.keys() if col.endswith("Rank")]
        if len(ranks) > 0:
            sort_by = ranks[0]
            return pd.DataFrame(measures).sort_values(by=sort_by, ascending=True)
    return pd.DataFrame(measures)
