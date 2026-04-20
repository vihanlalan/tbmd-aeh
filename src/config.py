"""
================================================================================
TBMD Framework — Configuration & Constants
================================================================================

All hyperparameters are defined here in a single location.
Nothing is hidden in the code.

Author: Vihan Lalan
Date:   March 2026
================================================================================
"""

import numpy as np

# ── Global random seed ────────────────────────────────────────────────────────
SEED = 42


# ── Simulation parameters ────────────────────────────────────────────────────
# Calibrated to match S&P 100 empirical properties:
#   daily vol ~1%, autocorr 0.02–0.22, herding 0.05–0.60
#   regime persistence 88–97%  (Hamilton 1989)
REGIME_PARAMS = {
    0: {
        'name':     'Efficient_Bull',
        'mu':       0.0005,   # daily drift
        'vol':      0.010,    # daily volatility
        'autocorr': 0.02,     # return autocorrelation
        'herding':  0.05,     # cross-asset co-movement intensity
        'p_stay':   0.97,     # regime persistence probability
    },
    1: {
        'name':     'Trending_Momentum',
        'mu':       0.0008,
        'vol':      0.015,
        'autocorr': 0.12,
        'herding':  0.25,
        'p_stay':   0.92,
    },
    2: {
        'name':     'Behavioral_Crash',
        'mu':      -0.0020,
        'vol':      0.030,
        'autocorr': 0.22,
        'herding':  0.60,
        'p_stay':   0.88,
    },
}


# ── Walk-forward parameters ──────────────────────────────────────────────────
WF_CONFIG = {
    'n_assets':              30,     # number of simulated assets
    'n_days':              1500,     # simulation length (~6 years)
    'train_window':          252,    # rolling training window (1 year)
    'step_size':              21,    # refit frequency (monthly)
    'warmup_days':           252,    # days before first prediction
    'transaction_cost_bps':   10,    # one-way cost in basis points
    'signal_confidence_hi':   0.60,  # long threshold on model probability
    'signal_confidence_lo':   0.40,  # short threshold on model probability
    'momentum_lookback':      20,    # cross-sectional momentum window
    'top_n_assets':            5,    # assets in long/short leg
}


# ── BPC construction parameters ──────────────────────────────────────────────
BPC_CONFIG = {
    'cs_dispersion_window':   20,    # cross-sectional herding window
    'vol_asym_window':        20,    # vol asymmetry window
    'autocorr_window':        60,    # return autocorrelation window
    'sentiment_ar_coef':       0.7,  # AR(1) coefficient for simulated sentiment
    'sentiment_mkt_loading':   0.3,  # market factor loading for sentiment
    'rolling_z_window':       252,   # normalization window for Z-scores
}


# ── Efficiency score parameters ───────────────────────────────────────────────
RES_CONFIG = {
    'window':                 60,    # rolling window for Hurst / VR / LB tests
    'vr_lag':                  5,    # variance ratio lag q (Lo-MacKinlay)
    'lb_lags':                10,    # number of lags for Ljung-Box test
    'hurst_max_lag_frac':     0.5,   # max lag as fraction of window for R/S
}


# ── Real-data configuration ──────────────────────────────────────────────────
# Date range: covers 2008 GFC, 2011 European crisis, 2018 drawdown,
#             2020 COVID crash, 2022 rate shock — five distinct stress episodes
START_DATE = '2005-01-01'
END_DATE   = '2024-12-31'

# S&P 100 current constituents (as of 2024)
# Survivorship bias note: this excludes historical members that were removed.
SP100_TICKERS = [
    'AAPL', 'ABBV', 'ABT',  'ACN',  'ADBE', 'AIG',  'AMD',  'AMGN',
    'AMT',  'AMZN', 'AVGO', 'AXP',  'BA',   'BAC',  'BK',   'BKNG',
    'BLK',  'BMY',  'BRK-B','C',    'CAT',  'CHTR', 'CL',   'CMCSA',
    'COF',  'COP',  'COST', 'CRM',  'CSCO', 'CVS',  'CVX',  'DE',
    'DHR',  'DIS',  'DUK',  'EMR',  'EXC',  'F',    'FDX',  'GD',
    'GE',   'GILD', 'GM',   'GOOGL','GS',   'HD',   'HON',  'IBM',
    'INTC', 'JNJ',  'JPM',  'KHC',  'KO',   'LIN',  'LLY',  'LMT',
    'LOW',  'MA',   'MCD',  'MDLZ', 'MDT',  'MET',  'META', 'MMM',
    'MO',   'MRK',  'MS',   'MSFT', 'NEE',  'NFLX', 'NKE',  'NOW',
    'NVDA', 'ORCL', 'OXY',  'PEP',  'PFE',  'PG',   'PM',   'PYPL',
    'QCOM', 'RTX',  'SBUX', 'SCHW', 'SO',   'SPG',  'T',    'TGT',
    'TMO',  'TMUS', 'TSLA', 'TXN',  'UNH',  'UNP',  'UPS',  'USB',
    'V',    'VZ',   'WBA',  'WFC',  'WMT',  'XOM',
]

# ── Synthetic benchmark results (from synthetic validation) ───────────────────
# These are the numbers from the paper's synthetic validation.
# We compare real-data results against these to validate directional claims.
SYNTHETIC_RESULTS = {
    'RCF Strategy':       {'Sharpe': 0.43,  'DSR': 0.71,  'Ann_Return_%':  1.8},
    'Unconditional BPC':  {'Sharpe': -2.47, 'DSR': 0.00,  'Ann_Return_%': -14.4},
    'Momentum Baseline':  {'Sharpe': -8.60, 'DSR': 0.00,  'Ann_Return_%': -2.8},
    'Mean Reversion':     {'Sharpe':  0.85, 'DSR': 0.95,  'Ann_Return_%':  0.3},
    'Buy & Hold':         {'Sharpe':  0.24, 'DSR': 0.70,  'Ann_Return_%':  4.1},
}

# ── Five directional claims to validate ───────────────────────────────────────
DIRECTIONAL_CLAIMS = [
    "RCF_Sharpe > Unconditional_Sharpe",
    "RCF_DSR > Unconditional_DSR",
    "RES_corr_with_VIX > 0.20",
    "At_least_3_BPC_components_BH_significant",
    "Behavioral_stress_periods_show_higher_RCF_returns",
]

# ── File paths ────────────────────────────────────────────────────────────────
SENTIMENT_CACHE  = 'sentiment_cache.csv'
PRICE_CACHE      = 'price_cache.csv'
SPY_CACHE        = 'spy_cache.csv'
OUTPUT_DIR       = './'
