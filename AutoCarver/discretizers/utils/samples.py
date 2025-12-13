from dataclasses import dataclass, field

from pandas import DataFrame, Series

from AutoCarver.features import Features


@dataclass
class Sample:
    """sample class to store X and y"""

    X: DataFrame
    y: Series = None

    def __getitem__(self, key):
        """Returns the DataFrame or the Series"""
        if key == "X":
            return self.X
        if key == "y":
            return self.y

        raise KeyError(key)

    def __iter__(self):
        """Returns an iterator over the DataFrame"""
        return iter(["X", "y"])

    def keys(self):
        """Returns the keys of the DataFrame"""
        return ["X", "y"]

    @property
    def shape(self):
        """Returns the shape of the DataFrame"""
        return self.X.shape

    @property
    def index(self):
        """Returns the index of the DataFrame"""
        return self.X.index

    @property
    def columns(self):
        """Returns the columns of the DataFrame"""
        return self.X.columns

    def __len__(self):
        return len(self.X)

    def fillna(self, features: Features) -> None:
        """fills up nans for features that have some"""
        self.X = features.fillna(self.X)

    def unfillna(self, features: Features) -> DataFrame:
        """reinstating nans when not supposed to group them"""
        return features.unfillna(self.X)


@dataclass
class Samples:
    """
    A container for storing training and development samples.

    Attributes:
        train (Sample): The training sample, containing features (X) and target (y).
        dev (Sample): The development sample, containing features (X) and target (y).

    Example:
        >>> import pandas as pd
        >>> from base_carver import Sample, Samples
        >>> X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> y_train = pd.Series([0, 1, 0])
        >>> X_dev = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [10, 11, 12]})
        >>> y_dev = pd.Series([1, 0, 1])
        >>> train_sample = Sample(X=X_train, y=y_train)
        >>> dev_sample = Sample(X=X_dev, y=y_dev)
        >>> samples = Samples(train=train_sample, dev=dev_sample)
        >>> print(samples.train.X)
           feature1  feature2
        0         1         4
        1         2         5
        2         3         6
        >>> print(samples.dev.y)
        0    1
        1    0
        2    1
        dtype: int64
    """

    train: Sample = field(default_factory=lambda: Sample(X=None))
    dev: Sample = field(default_factory=lambda: Sample(X=None))

    def fillna(self, features: Features) -> None:
        """fills up nans in X and X_dev"""
        self.train.X = features.fillna(self.train.X)
        if self.dev.X is not None:
            self.dev.X = features.fillna(self.dev.X)
