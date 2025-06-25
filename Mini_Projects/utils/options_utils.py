# Utility functions for Options Utils

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call_price(S, K, sigma, T, r=0.0):
    """
    Computes the Black-Scholes price for a European call option.

    Parameters:
        S     : Spot price of the asset
        K     : Strike price of the option
        sigma : Volatility (annualized)
        T     : Time to maturity (in years)
        r     : Risk-free interest rate

    Returns:
        float: Call option price
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, sigma, T, r=0.0):
    """
    Computes the Black-Scholes price for a European put option.

    Parameters:
        S     : Spot price of the asset
        K     : Strike price of the option
        sigma : Volatility (annualized)
        T     : Time to maturity (in years)
        r     : Risk-free interest rate

    Returns:
        float: Put option price
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)