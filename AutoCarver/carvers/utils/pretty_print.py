""" Set of tools used for pretty printing"""

from pandas import DataFrame

from ...features import BaseFeature


def prettier_xagg(
    nice_xagg: DataFrame = None,
    caption: str = None,
    hide_index: bool = False,
) -> str:
    """Pretty display of frequency and target rate per modality on the same line

    Parameters
    ----------
    nice_xagg : DataFrame, optional
        Target rate and frequency per modality, by default None
    caption : str, optional
        Title of the HTML table, by default None
    hide_index : bool, optional
        Whether or not to hide the index (for dev distribution)

    Returns
    -------
    str
        HTML format of the crosstab
    """
    # checking for a provided xtab
    nicer_xagg = ""
    if nice_xagg is not None:
        # checking for non unique indices
        if any(nice_xagg.index.duplicated()):
            nice_xagg.reset_index(inplace=True)

        # adding coolwarm color gradient
        nicer_xagg = nice_xagg.style.background_gradient(cmap="coolwarm")

        # printing inline notebook
        nicer_xagg = nicer_xagg.set_table_attributes("style='display:inline'")

        # lower precision
        nicer_xagg = nicer_xagg.format(precision=4)

        # adding custom caption/title
        if caption is not None:
            nicer_xagg = nicer_xagg.set_caption(caption)

        # hiding index for dev
        if hide_index:
            nicer_xagg.hide(axis="index")

        # converting to html
        nicer_xagg = nicer_xagg._repr_html_()  # pylint: disable=W0212

    return nicer_xagg


def index_mapper(feature: BaseFeature, xtab: DataFrame = None) -> DataFrame:
    """Prints a binary xtab's statistics

    Parameters
    ----------
    order_get : Callable
        Ordering of modalities used to map indices
    xtab : Dataframe
        A crosstab, by default None

    Returns
    -------
    DataFrame
        Target rate and frequency per modality
    """
    # checking for an xtab
    mapped_xtab = None
    if xtab is not None:
        # copying initial xtab
        mapped_xtab = xtab.copy()

        # for qualitative features -> mapping with values_orders.content
        if feature.is_qualitative:
            mapped_index = [feature.values.get(idx, idx) for idx in mapped_xtab.index]
            # removing str_default and deduplicating for features converted to str
            mapped_index = [
                list(set(str(idx) for n, idx in enumerate(mapped_idx) if idx != feature.default))
                for mapped_idx in mapped_index
            ]
            mapped_index = [
                (
                    mapped_idx[-1] + " to " + mapped_idx[0]
                    if len(mapped_idx) > 2
                    else mapped_idx[0] if len(mapped_idx) == 0 else ", ".join(mapped_idx)
                )
                for mapped_idx in mapped_index
            ]
        # for quantitative features -> mapping with labels_orders.keys
        else:
            mapped_index = feature.labels[:]

        # modifying indices based on provided order
        mapped_xtab.index = mapped_index

    return mapped_xtab
