""" Defines fixtures for all pytests"""

from numpy import arange, nan, random
from pandas import DataFrame
from pytest import fixture


def init_df(seed: int, size: int = 10000) -> DataFrame:
    """Initializes a DataFrame used in tests

    Parameters
    ----------
    seed : int
        Seed for the random samples
    size : int
        Generated sample size

    Returns
    -------
    DataFrame
        A DataFrame to perform Discretizers tests
    """

    # Set random seed for reproducibility
    random.seed(seed)

    # Generate random qualitative ordinal features
    qual_ord_features = (
        ["Low-"] * int(1 * 100)
        + ["Low"] * int(0 * 100)
        + ["Low+"] * int(12 * 100)
        + ["Medium-"] * int(10 * 100)  # 13%
        + ["Medium"] * int(24 * 100)
        + ["Medium+"] * int(6 * 100)
        + ["High-"] * int(0 * 100)  # 40%
        + ["High"] * int(7 * 100)
        + ["High+"] * int(40 * 100)  # 47 %
    )
    # qual_ord_features = ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+']
    ordinal_data = random.choice(qual_ord_features, size=size)

    # adding binary target associated to qualitative ordinal feature
    binary = [
        1 - (qual_ord_features.index(val) / (len(qual_ord_features) - 1)) for val in ordinal_data
    ]

    # Generate random qualitative features
    qual_features = (
        ["Category A"] * int(1 * 100)
        + ["Category B"] * int(0 * 100)
        + ["Category C"] * int(25 * 100)
        + ["Category D"] * int(3 * 100)  # 26%
        + ["Category E"] * int(65 * 100)
        + ["Category F"] * int(6 * 100)  # 74%
    )
    qualitative_data = random.choice(qual_features, size=size)

    # Generate random quantitative features
    quantitative_data = random.rand(size) * 1000

    # Generate random discrete quantitative feature
    discrete_quantitative_data = random.choice(
        arange(1, 8), size=size, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02]
    )

    # Create DataFrame
    data = {
        "Qualitative_Ordinal": ordinal_data,
        "Qualitative": qualitative_data,
        "Quantitative": quantitative_data,
        "Binary": binary,
        "Discrete_Quantitative": discrete_quantitative_data,
    }
    df = DataFrame(data)

    df["quali_ordinal_target"] = df["Binary"].apply(
        lambda u: random.choice([0, 1], p=[1 - (u * 1 / 3), (u * 1 / 3)])
    )

    # building specific cases for base_discretizer
    df["Qualitative_Ordinal_lownan"] = df["Qualitative_Ordinal"].replace("Low-", nan)
    df["Qualitative_Ordinal_highnan"] = df["Qualitative_Ordinal"].replace("High+", nan)

    # building specific cases for qualitative_discretizer
    df["Qualitative_grouped"] = df["Qualitative"].replace("Category A", "Category D")
    df["Qualitative_highnan"] = df["Qualitative"].replace("Category F", nan)
    df["Qualitative_lownan"] = df["Qualitative"].replace("Category A", nan)

    # building specific cases for quantitative_discretizer
    df["Discrete_Quantitative_highnan"] = df["Discrete_Quantitative"].replace(1, nan)
    df["Discrete_Quantitative_lownan"] = df["Discrete_Quantitative"].replace(7, nan)
    df["Discrete_Quantitative_rarevalue"] = df["Discrete_Quantitative_lownan"].fillna(0.5)

    # building specific cases for qualitative_discretizer
    df["Discrete_Qualitative_noorder"] = df["Discrete_Quantitative"]
    df["Discrete_Qualitative_highnan"] = df["Discrete_Quantitative_highnan"]
    df["Discrete_Qualitative_lownan_noorder"] = df["Discrete_Quantitative_lownan"]
    df["Discrete_Qualitative_rarevalue_noorder"] = df["Discrete_Quantitative_rarevalue"]

    # wrong data
    df["nan"] = nan
    df["ones"] = "1"
    df["one"] = 1
    df["one_nan"] = df["nan"]
    df.loc[0, "one_nan"] = 1
    df["ones_nan"] = df["nan"]
    df.loc[size - 1, "ones_nan"] = "1"

    return df


@fixture
def x_train() -> DataFrame:
    """Simulates a Train sample

    Returns
    -------
    DataFrame
        Train sample
    """
    # initiating dev sample
    return init_df(123)


@fixture
def x_dev_1() -> DataFrame:
    """Simulates a Dev sample

    Returns
    -------
    DataFrame
        Dev sample
    """
    # initiating dev sample
    return init_df(1234)


@fixture
def x_dev_2() -> DataFrame:
    """Simulates a Dev sample

    Returns
    -------
    DataFrame
        Dev sample
    """
    # initiating dev sample
    return init_df(12345)


@fixture
def x_dev_wrong_1(x_dev_1: DataFrame) -> DataFrame:
    """Simulates a wrong Dev sample (unexpected modality)

    Parameters
    ----------
    x_dev_1 : DataFrame
        Simulated Dev sample

    Returns
    -------
    DataFrame
        Wrong Dev sample with unexpected modality
    """
    # initiating dev sample
    x_dev = x_dev_1.copy()

    # replacing a value for a unknown value
    x_dev["Qualitative"] = x_dev["Qualitative"].replace("Category C", "Category Y")

    return x_dev


@fixture
def x_dev_wrong_2(x_dev_1: DataFrame) -> DataFrame:
    """Simulates a wrong Dev sample (introduced nans)

    Parameters
    ----------
    x_dev_1 : DataFrame
        Simulated Dev sample

    Returns
    -------
    DataFrame
        Wrong Dev sample with nans
    """
    # initiating dev sample
    x_dev = x_dev_1.copy()

    # replacing a value for a unknown value
    x_dev["Qualitative"] = x_dev["Qualitative"].replace("Category C", nan)

    return x_dev
