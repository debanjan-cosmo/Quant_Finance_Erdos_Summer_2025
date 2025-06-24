
# Quantitative Finance Mini Projects – Erdos Institute Summer 2025

This repository contains a collection of four mini projects completed as part of the **"Introduction to Quantitative Methods in Finance"** course offered by the Erdős Institute in Summer 2025. Each project explores a core topic in quantitative finance using real financial data, simulation models, and statistical analysis. The projects are implemented in Python with modular, reusable code, and are presented through well-documented Jupyter notebooks.

## Contents

### 1. Portfolio Optimization and Risk-Return Tradeoff  
**Notebook**: `01_Portfolio_Optimization_Risk_Return_Analysis.ipynb`  
**Module**: `portfolio_utils.py`  
- Implements mean-variance optimization (Markowitz framework).  
- Constructs low-risk, high-risk, and high-return portfolios.  
- Analyzes return and volatility trade-offs using historical data.  
- Saves and compares portfolio weights and performance.

### 2. Normality Assumptions in Log Returns  
**Notebook**: `02_Log_Return_Normality_Investigation.ipynb`  
**Module**: `normality_utils.py`  
- Examines if daily log returns of stocks follow a normal distribution.  
- Uses Q-Q plots, histograms, and statistical tests (Shapiro-Wilk, Jarque-Bera).  
- Discusses implications of non-normality for risk modeling and VaR estimation.

### 3. Sensitivity Analysis of Black-Scholes Option Pricing  
**Notebook**: `03_Black_Scholes_Sensitivity_Analysis.ipynb`  
**Module**: `options_utils.py`  
- Analyzes how call and put option prices vary with spot price, time to maturity, and volatility.  
- Computes option Greeks and visualizes sensitivities.  
- Builds intuition behind option pricing behavior under the Black-Scholes model.

### 4. Delta Hedging under Stochastic Volatility  
**Notebook**: `04_Delta_Hedging_Stochastic_Volatility.ipynb`  
**Module**: `hedging_utils.py`  
- Simulates stock paths using Heston and GARCH(1,1) models.  
- Compares delta hedging performance under stochastic vs. constant volatility.  
- Analyzes profit & loss (P&L) distributions and hedging errors.

## Directory Structure

```
Mini_Projects/
│
├── 01_Portfolio_Optimization_Risk_Return_Analysis.ipynb
├── 02_Log_Return_Normality_Investigation.ipynb
├── 03_Black_Scholes_Sensitivity_Analysis.ipynb
├── 04_Delta_Hedging_Stochastic_Volatility.ipynb
│
├── portfolio_utils.py
├── normality_utils.py
├── options_utils.py
├── hedging_utils.py
│
├── data/                      # Input datasets (CSV or downloaded data)
└── portfolio_weights/         # Saved optimized portfolio weights
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/debanjan-cosmo/Quant_Finance_Erdos_Summer_2025.git
cd Quant_Finance_Erdos_Summer_2025/Mini_Projects
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

The project uses Python 3.8+ and the following key libraries:
- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn, statsmodels
- yfinance, arch, plotly, tqdm, mplfinance

## Usage

1. Launch the notebook interface:
   ```bash
   jupyter lab
   ```
2. Open any of the four `.ipynb` notebooks.
3. The notebooks are modular—functions are imported from `.py` files using:
   ```python
   from portfolio_utils import *
   ```
4. You may edit or run them directly. Data is either downloaded using APIs (like `yfinance`) or provided in the `data/` folder.

## Highlights

- Realistic simulations of market behavior (GARCH, Heston)  
- Clean separation of logic using utility scripts  
- Emphasis on interpretability and visualization  
- Compatible with both local and cloud-based Jupyter environments

## Acknowledgments

These projects were completed as part of the Summer 2025 offering of the **Introduction to Quantitative Methods in Finance** course at the **Erdős Institute**.  
Special thanks to the instructors and mentors for their guidance.
