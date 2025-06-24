# Utility functions for Normality Utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import scipy.stats as stats
import datetime as dt
from itertools import product
from scipy.stats import normaltest
import os

def compute_log_returns(prices):
    """
    Computes daily log returns from a price series.

    Parameters:
        prices (pd.Series or pd.DataFrame): Price data

    Returns:
        pd.Series: Daily log returns
    """
    return np.log(prices / prices.shift(1)).dropna()

def normality_test_results(log_returns):
    """
    Applies D’Agostino and Pearson’s normality test.

    Parameters:
        log_returns (pd.Series): The series of log returns

    Returns:
        tuple: (p-value, 'Possibly Normal' or 'Not Normal')
    """
    p = stats.normaltest(log_returns)[1]
    conclusion = "Not Normal" if p < 0.05 else "Possibly Normal"
    return p, conclusion

def remove_outliers(data, z_thresh=2.5):
    """
    Removes data points that are Z-score outliers.

    Parameters:
        data (pd.Series): Data to filter
        z_thresh (float): Z-score threshold (default=2.5)

    Returns:
        np.ndarray: Filtered values
    """
    data = data.values.flatten()
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return data[z_scores < z_thresh]

def test_normality_over_time(log_returns, window=126):
    """
    Performs rolling window normality test.

    Parameters:
        log_returns (pd.Series): Log return data
        window (int): Window size in days (default=126)

    Returns:
        list of tuples: (window index, start date, p-value)
    """
    results = []
    for i in range(0, len(log_returns) - window + 1, window):
        segment = log_returns[i:i + window]
        p_val = stats.normaltest(segment)[1]
        results.append((i, segment.index[0], p_val))
    return results

def generate_weight_combinations(n_assets, step=0.05):
    """
    Generates combinations of asset weights summing to 1.

    Parameters:
        n_assets (int): Number of assets
        step (float): Increment size for weights (default=0.05)

    Yields:
        tuple: Weight combination
    """
    from itertools import product
    grid = np.arange(0, 1 + step, step)
    for combo in product(grid, repeat=n_assets):
        if np.isclose(sum(combo), 1.0, atol=1e-4):
            yield combo

def compute_weighted_portfolio(log_return_dict, symbols, weights):
    """
    Computes portfolio log returns using weighted average.

    Parameters:
        log_return_dict (dict): Dictionary of log return Series
        symbols (list): List of stock tickers
        weights (list or array): Corresponding weights

    Returns:
        pd.Series: Combined portfolio log returns
    """
    weighted_returns = [
        weights[i] * log_return_dict[symbols[i]]
        for i in range(len(symbols))
    ]
    return pd.concat(weighted_returns, axis=1).dropna().sum(axis=1)

# def load_weights(name):
#     """Load weights from saved CSV."""
#     df = pd.read_csv(f"{output_dir}/{name}_weights.csv")
#     return df['Weight'].values

# low_risk_weights = load_weights("low_risk")


# def run_normality_test(name, returns):
#     p = stats.normaltest(returns)[1]
#     result = "Possibly Normal" if p > 0.05 else "Not Normal"
#     print(f"{name} Portfolio: p = {p:.4f} → {result}")
#     return p, result

# p_low, _ = run_normality_test("Low-Risk", returns_low)
