""" Pretty print functions for selectors """

from pandas import DataFrame

from ...features import BaseFeature


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


def format_ranked_features(features: list[BaseFeature]) -> DataFrame:
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
            return DataFrame(measures).sort_values(by=sort_by, ascending=True)
    return DataFrame(measures)


def prettier_measures(association: DataFrame) -> str:
    """Pretty display of measures and filters per feature on the same line"""

    # finding columns with indicators to colorize
    subset = [
        column
        for column in association.columns
        # checking for an association indicator
        if any(column.endswith(indic) for indic in ["Measure", "Filter"])
    ]

    # adding coolwarm color gradient
    nicer_association = association.style.background_gradient(cmap="coolwarm", subset=subset)
    # printing inline notebook
    nicer_association = nicer_association.set_table_attributes("style='display:inline'")

    # finding columns with indicators to colorize
    subset = [
        column
        for column in association.columns
        # checking for an association indicator
        if any(column.endswith(indic) for indic in ["Measure", "Filter"])
    ]

    # adding numerical columns
    subset += association.select_dtypes(include=["number"]).columns.tolist()

    # lower precision for specific columns
    nicer_association = nicer_association.format({measure: "{:.4f}" for measure in subset})

    # conversion to html
    return nicer_association._repr_html_()
