"""
================================================================================
TBMD Framework — Market Simulation
Hamilton (1989) Markov Regime-Switching Model
================================================================================

Generates synthetic multi-asset return data with known ground-truth regime
labels for validating the RES detector. Real market data cannot provide this
because regimes must be inferred from the same data used for validation,
creating circularity.

Key References:
    Hamilton (1989) — Markov Regime-Switching Model

Author: Vihan Lalan
================================================================================
"""

import numpy as np
import pandas as pd

from .config import SEED, REGIME_PARAMS, WF_CONFIG


def simulate_regime_switching_market(
    n_assets: int = WF_CONFIG['n_assets'],
    n_days:   int = WF_CONFIG['n_days'],
    regimes:  dict = REGIME_PARAMS,
    seed:     int = SEED,
) -> tuple:
    """
    Hamilton (1989) Markov Regime-Switching Market Simulator.

    Generates synthetic multi-asset return data with:
      - Latent regime path (ground-truth labels for validation)
      - Regime-dependent drift, volatility, and autocorrelation
      - Cross-asset correlation via common market factor
      - Behavioural herding component (assets cluster in crash regime)

    Parameters
    ----------
    n_assets : int
        Number of simulated assets.
    n_days : int
        Simulation length in trading days.
    regimes : dict
        Regime parameter dictionary (see REGIME_PARAMS).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    returns_df   : pd.DataFrame  shape (n_days, n_assets)  — daily log returns
    prices_df    : pd.DataFrame  shape (n_days, n_assets)  — price levels
    regime_series: pd.Series     shape (n_days,)           — true regime labels
    regimes      : dict          — regime parameter dictionary (unchanged)
    market_factor: np.ndarray    shape (n_days,)           — common market factor
    """
    np.random.seed(seed)
    n_regimes = len(regimes)

    # ── Build transition matrix from p_stay parameters ───────────────────────
    trans_matrix = np.zeros((n_regimes, n_regimes))
    for i, r in regimes.items():
        p_stay  = r['p_stay']
        p_leave = (1.0 - p_stay) / (n_regimes - 1)
        for j in range(n_regimes):
            trans_matrix[i, j] = p_stay if i == j else p_leave

    # ── Simulate latent regime path ───────────────────────────────────────────
    regime_path    = np.zeros(n_days, dtype=int)
    regime_path[0] = 0
    for t in range(1, n_days):
        regime_path[t] = np.random.choice(n_regimes, p=trans_matrix[regime_path[t-1]])

    # ── Simulate returns: market factor + herding component + idiosyncratic ──
    returns       = np.zeros((n_days, n_assets))
    market_factor = np.zeros(n_days)

    for t in range(n_days):
        r = regimes[regime_path[t]]

        # AR(1) market factor with regime-dependent autocorrelation
        if t == 0:
            market_factor[t] = np.random.normal(r['mu'], r['vol'])
        else:
            innovation        = np.random.normal(0, r['vol'] * np.sqrt(1.0 - r['autocorr']**2))
            market_factor[t]  = r['mu'] + r['autocorr'] * market_factor[t-1] + innovation

        # Herding component: all assets move together in behavioural regimes
        herding_shock = np.random.normal(0, r['vol'] * r['herding'])

        for a in range(n_assets):
            idiosyncratic = np.random.normal(0, r['vol'] * 0.40)
            returns[t, a] = (
                0.60 * market_factor[t]
                + 0.30 * herding_shock
                + 0.10 * idiosyncratic
            )

    # ── Build DataFrames ──────────────────────────────────────────────────────
    dates       = pd.date_range(start='2015-01-01', periods=n_days, freq='B')
    asset_names = [f'Asset_{i:02d}' for i in range(n_assets)]

    returns_df    = pd.DataFrame(returns, index=dates, columns=asset_names)
    prices_df     = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates, columns=asset_names,
    )
    regime_series = pd.Series(regime_path, index=dates, name='regime')

    return returns_df, prices_df, regime_series, regimes, market_factor
