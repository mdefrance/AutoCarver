"""Tool to build optimized buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from warnings import warn

from .discretizers import GroupedList
from .utils import BinaryCarver, ContinuousCarver, MulticlassCarver


class AutoCarver(BinaryCarver, ContinuousCarver, MulticlassCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary or continuous target.

    First fits a :ref:`Discretizer`. Raw data should be provided as input (not a result of ``Discretizer.transform()``).
    """

    def __init__(
        self,
        min_freq: float,
        sort_by: str,
        *,
        target_type: str = None,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        output_dtype: str = "float",
        dropna: bool = True,
        copy: bool = False,
        verbose: bool = False,
        pretty_print: bool = False,
        **kwargs,
        # min_carved_freq: float = 0,  # TODO: update this parameter so that it is set according to frequency rather than number of groups
        # unknown_handling: str = "raises",  # TODO: add parameter to remove unknown values whatsoever
        # str_nan: str = "__NAN__",
        # str_default: str = "__OTHER__",
    ) -> None:
        """Initiates an ``AutoCarver``.

        Parameters
        ----------
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        sort_by : str
            To be choosen amongst ``["tschuprowt", "cramerv", "kruskal"]``
            Metric to be used to perform association measure between features and target.

            * Binary target: use ``"tschuprowt"``, for Tschuprow's T.
            * Binary target: use ``"cramerv"``, for Cramér's V.
            * Continuous target: use ``"kruskal"``, for Kruskal-Wallis' H test statistic.

            **Tip**: use ``"tschuprowt"`` for more robust, or less output modalities,
            use ``"cramerv"`` for more output modalities.

        quantitative_features : list[str], optional
            List of column names of quantitative features (continuous and discrete) to be carved, by default ``None``

        qualitative_features : list[str], optional
            List of column names of qualitative features (non-ordinal) to be carved, by default ``None``

        ordinal_features : list[str], optional
            List of column names of ordinal features to be carved. For those features a list of
            values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        max_n_mod : int, optional
            Maximum number of modality per feature, by default ``5``

            All combinations of modalities for groups of modalities of sizes from 1 to ``max_n_mod`` will be tested.
            The combination with the greatest association (as defined by ``sort_by``) will be the selected one.

            **Tip**: should be set between 4 (faster, more robust) and 7 (slower, preciser, less robust)

        output_dtype : str, optional
            To be choosen amongst ``["float", "str"]``, by default ``"float"``

            * ``"float"``, grouped modalities will be converted to there corresponding floating rank.
            * ``"str"``, a per-group modality will be set for all the modalities of a group.

        dropna : bool, optional
            * ``True``, ``AutoCarver`` will try to group ``numpy.nan`` with other modalities.
            * ``False``, ``AutoCarver`` all non-``numpy.nan`` will be grouped, by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, as long as
            information on AutoCarver's processing and tables of target rates and frequencies for
            X, by default ``False``

        pretty_print : bool, optional
            If ``True``, adds to the verbose some HTML tables of target rates and frequencies for X and, if provided, X_dev.
            Overrides the value of ``verbose``, by default ``False``

        **kwargs
            Pass values for ``str_default``and ``str_nan`` of ``Discretizer`` (default string values).

        Examples
        --------
        See `AutoCarver examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # checking for provided target_type
        if target_type is None:
            target_type = "binary"
            warn(
                " - [AutoCarver] target_type was not provided, defaulting to 'binary'. "
                "Attribute target_type will be mandatory starting from v6.0.0.",
                FutureWarning
            )

        # gathering attributes
        args = (min_freq, sort_by)
        kwargs.update({
            "quantitative_features": quantitative_features,
            "qualitative_features": qualitative_features,
            "ordinal_features": ordinal_features,
            "values_orders": values_orders,
            "max_n_mod": max_n_mod,
            "output_dtype": output_dtype,
            "dropna": dropna,
            "copy": copy,
            "verbose": verbose,
            "pretty_print": pretty_print,
        })
        
        # Initiating BinaryCarver
        if target_type == "binary":
            BinaryCarver.__init__(self, *args, **kwargs)

        # Initiating BinaryCarver
        elif target_type == "continuous":
            ContinuousCarver.__init__(self, *args, **kwargs)

        # Initiating BinaryCarver
        elif target_type == "multiclass":
            MulticlassCarver.__init__(self, *args, **kwargs)

        # Raising error
        else:
            implemented_target_types = ['binary', 'continuous', 'multiclass']
            assert target_type in implemented_target_types, (
                f" - [AutoCarver] {target_type} is not a valid target_type, choose from "
                f"{implemented_target_types}."
            )
