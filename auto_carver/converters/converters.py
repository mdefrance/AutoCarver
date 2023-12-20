"""Tools for FeatureEngineering."""

from typing import Any, Dict, List

from numpy import nan, tanh
from pandas import DataFrame, Series, notna, to_datetime
from sklearn.base import BaseEstimator, TransformerMixin


class TimeDeltaConverter(BaseEstimator, TransformerMixin):
    """Converts specified DateTime columns into TimeDeltas between themselves

     - Str Columns are converted to pandas datetime columns
     - New TimeDelta column names are stored in TimeDeltaConverter.delta_features

    Parameters
    ----------
    features: List[str]
        List of DateTime columns to be converted to string.
    nans: Any, default numpy.nan
        Date value to be considered as missing data.
    drop: bool, default True
        Whether or not to drop initial DateTime columns (specified in features).
    """

    def __init__(
        self,
        features: List[str],
        nans: Any = nan,
        copy: bool = False,
        drop: bool = True,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        nans : Any, optional
            _description_, by default nan
        copy : bool, optional
            _description_, by default False
        drop : bool, optional
            _description_, by default True
        """
        print("Warning: not tested for package version greater than 4.")
        self.features = features[:]
        self.copy = copy
        self.new_features: List[str] = []
        self.nans = nans
        self.drop = drop

    def fit(self, X: DataFrame, y=None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : _type_, optional
            _description_, by default None
        """
        # creating list of names of delta featuers
        for i, date1 in enumerate(self.features):
            for date2 in self.features[i + 1 :]:
                self.new_features += [f"delta_{date1}_{date2}"]

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # copying dataset
        Xc = X
        if self.copy:
            Xc = X.copy()

        # converting back nans
        if notna(self.nans):
            Xc[self.features] = Xc[self.features].where(Xc[self.features] != self.nans, nan)

        # converting to datetime
        Xc[self.features] = to_datetime(Xc[self.features].stack()).unstack()

        # iterating over each combination of dates
        for i, date1 in enumerate(self.features):
            for date2 in self.features[i + 1 :]:
                # computing timedelta of days
                Xc[f"delta_{date1}_{date2}"] = (Xc[date1] - Xc[date2]).dt.days

        # dropping date columns
        Xc = Xc.drop(self.features, axis=1)

        return Xc


class GroupNormalizer(BaseEstimator, TransformerMixin):
    """Normalizes a feature's values based on specified means and stds per groups' values

    Parameters
    ----------
    groups: List[str]
        List of qualitative features used to compute feature's mean/std per there modalities.
    features: List[str]
        List of quantitative features to be normalized.
    """

    def __init__(self, groups: List[str], features: List[str], copy: bool = True) -> None:
        """_summary_

        Parameters
        ----------
        groups : List[str]
            _description_
        features : List[str]
            _description_
        copy : bool, optional
            _description_, by default True
        """
        print("Warning: not tested for package version greater than 4.")
        self.features = features[:]
        self.groups = groups[:]

        self.copy = copy

        self.group_means = {}
        self.group_stds = {}

        self.new_features = []

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None
        """
        # iterating over each grouping column
        for group in self.groups:
            # computing group mean
            self.group_means.update({group: X.groupby(group)[self.features].mean().to_dict()})

            # computing group std
            self.group_stds.update({group: X.groupby(group)[self.features].std().to_dict()})

            # adding built features
            self.new_features += [f"{f}_norm_{group}" for f in self.features]

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # coppying dataframe
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each group
        for group in self.groups:
            # computing observation level mean
            means = Xc[self.features].apply(
                lambda u: Xc[group].apply(self.group_means[group][u.name].get)
            )

            # computing observation level std
            stds = Xc[self.features].apply(
                lambda u: Xc[group].apply(self.group_stds[group][u.name].get)
            )

            # applying normalization to the feature
            Xc = Xc.join(
                Xc[self.features]
                .sub(means)
                .replace(0, nan)
                .divide(stds)
                .rename({f: f"{f}_norm_{group}" for f in self.features}, axis=1)
            )

        del means
        del stds

        return Xc


class TanhNormalizer(BaseEstimator, TransformerMixin):
    """Tanh Normalization that keeps data distribution and borns between 0 and 1

    Parameters
    ----------
    features: List[str]
        List of quantitative features to be normalized.
    """

    def __init__(self, features: List[str], copy: bool = False) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        copy : bool, optional
            _description_, by default False
        """
        print("Warning: not tested for package versions greater than 4.")
        self.features = features[:]
        self.copy = copy

        self.distribs = None

    def fit(self, X: DataFrame, y=None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : _type_, optional
            _description_, by default None
        """
        self.distribs = X[self.features].agg(["mean", "std"]).to_dict(orient="index")

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # copying dataset
        Xc = X
        if self.copy:
            Xc = X.copy()

        # applying tanh normalization
        Xc[self.features] = (
            tanh(Xc[self.features].sub(self.distribs["mean"]).divide(self.distribs["std"]) * 0.01)
            + 1
        ) / 2

        return Xc


class CrossConverter(BaseEstimator, TransformerMixin):
    """Normalizes a feature's values based on specified means and stds per groups' values

    Parameters
    ----------
    features: List[str]
        List of qualitative features to be crossed should be passed through AutoCarver early on.
    """

    def __init__(self, features: List[str], copy: bool = True) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        copy : bool, optional
            _description_, by default True
        """
        print("Warning: not tested for package version greater than 4.")
        self.features = features[:]
        self.copy = copy
        self.new_features: List[str] = []
        self.values: Dict[str, List[Any]] = {}

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None
        """
        # iterating over each feature
        for i, feature1 in enumerate(self.features):
            # unique values of feature1
            unq1 = X[feature1].astype(str).unique()

            for feature2 in self.features[i + 1 :]:
                # adding features names to the list of built features
                self.new_features += [f"{feature2}_x_{feature1}"]

                # unique values of feature2
                unq2 = X[feature2].astype(str).unique()

                self.values.update(
                    {f"{feature2}_x_{feature1}": [u2 + "_x_" + u1 for u2 in unq2 for u1 in unq1]}
                )

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # coppying dataframe
        Xc = X
        if self.copy:
            Xc = X.copy()

        # converting features to strings
        Xc_features = Xc[self.features].astype(str)

        # iterating over each group
        for i, feature in enumerate(self.features):
            # features to cross with
            Xc_tocross = Xc_features[self.features[i + 1 :]]

            # feature to be crossed
            Xc_crosser = Xc_features[feature]

            # crossing features
            Xc_crossed = Xc_tocross.apply(lambda u: u + "_x_" + Xc_crosser).rename(
                {f: f"{f}_x_{feature}" for f in self.features[i + 1 :]}, axis=1
            )

            # applying normalization to the feature
            Xc = Xc.join(Xc_crossed)

        del Xc_features
        del Xc_crossed
        del Xc_tocross
        del Xc_crosser

        return Xc
