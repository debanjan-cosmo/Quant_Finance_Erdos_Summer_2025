# Utility functions for Portfolio Utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime as dt
import os
from scipy.optimize import minimize

# Function to compute annualized portfolio return



def portfolio_return(weights, mean_returns):
    """
    Computes annualized portfolio return from daily mean returns and weights.
    
    Parameters:
    weights (array): Portfolio weights
    mean_returns (Series): Daily mean returns of assets
    
    Returns:
    float: Annualized portfolio return
    """
    return np.dot(weights, mean_returns) * 252

# Define constraints: weights sum to 1 and bounded within min/max


def get_constraints(min_w=0.05, max_w=0.35):
    """
    Creates constraints for portfolio optimization.
    
    Parameters:
    min_w (float): Minimum allowed weight per asset
    max_w (float): Maximum allowed weight per asset
    
    Returns:
    tuple: Constraint dictionaries
    """
    return (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},                  # weights sum to 1
        {'type': 'ineq', 'fun': lambda x: np.min(x) - min_w},           # each weight >= min_w
        {'type': 'ineq', 'fun': lambda x: max_w - np.max(x)}            # each weight <= max_w
    )

# Looser constraints allow more concentrated bets


# def neg_return(weights, mean_returns=mean_daily_returns):
#     return -np.dot(weights, mean_returns) * 252

# Same relaxed constraints as high-risk case


# def save_weights(weights, name):
#     """Save weights as CSV with ticker names."""
#     df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
#     df.to_csv(f"{output_dir}/{name}_weights.csv", index=False)

# def load_weights(name):
#     """Load weights from saved CSV."""
#     df = pd.read_csv(f"{output_dir}/{name}_weights.csv")
#     return df['Weight'].values