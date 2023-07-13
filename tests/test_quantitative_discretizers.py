"""Set of tests for quantitative_discretizers module."""

from AutoCarver.discretizers.utils.quantitative_discretizers import *
from numpy import inf

def test_quantile_discretizer(x_train: DataFrame):
    """Tests QuantileDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """
    
    features = ['Quantitative', 'Discrete_Quantitative', 'Discrete_Quantitative_highnan', 'Discrete_Quantitative_lownan', 'Discrete_Quantitative_rarevalue']
    min_freq = 0.1

    discretizer = QuantileDiscretizer(features, min_freq, copy=True)
    x_discretized = discretizer.fit_transform(x_train)
    
    assert all(x_discretized.Quantitative.value_counts(normalize=True) == min_freq), "Wrong quantiles"
    assert discretizer.values_orders["Discrete_Quantitative_highnan"] == [2.0, 3.0, 4.0, 7.0, inf, '__NAN__'], "NaNs should be added to the order"
    assert '__NAN__' in x_discretized["Discrete_Quantitative_highnan"].unique(), "NaNs should be filled with the str_nan value"
    assert discretizer.values_orders["Discrete_Quantitative_lownan"] == [1.0, 2.0, 3.0, 4.0, 6.0, inf, '__NAN__'], "NaNs should not be grouped whatsoever"
    assert discretizer.values_orders["Discrete_Quantitative_rarevalue"] == [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, inf], "Wrongly associated rare values"
