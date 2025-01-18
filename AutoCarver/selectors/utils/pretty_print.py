""" Pretty print functions for selectors """

from pandas import DataFrame

from ...features import BaseFeature


def format_measure(feature: BaseFeature, measure: dict) -> dict:
    """ Format measure for display """
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
                formatted.update({f"{measure_name}_with": correlation_with})

    return formatted


def format_ranked_features(features: list[BaseFeature]) -> DataFrame:
    """  formats ranked features for display """

    # adding measures and filters
    measures = []
    for feature in features:
        measures.append(format_measure(feature, {**feature.measures, **feature.filters}))

    # finding a sorting measure
    ranks = [col for col in measures[0].keys() if col.endswith("_Rank")]
    if len(ranks) > 0:
        sort_by = ranks[0]
        return DataFrame(measures).sort_values(by=sort_by, ascending=False)
    return DataFrame(measures)
