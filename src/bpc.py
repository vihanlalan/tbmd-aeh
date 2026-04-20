"""
================================================================================
TBMD Framework — Behavioral Proxy Composite (BPC)
================================================================================

Replaces the unobservable BAM equation MB(b,t) = Σ_i B(i,b,t)·W(i,t)
with five observable, computable proxies, each grounded in the published
behavioral finance and market microstructure literature.

┌─────────────────────────────────────────────────────────────────────┐
│  Proxy         │ Behavioral Construct │ Literature                  │
├─────────────────────────────────────────────────────────────────────┤
│  herding_z     │ Herding / conformity │ Christie & Huang (1995)     │
│  vol_asym      │ Loss aversion        │ Ang et al. (2006)           │
│  return_autocorr│ Momentum bias       │ Lo & MacKinlay (1988)       │
│  sentiment     │ Investor sentiment   │ Baker & Wurgler (2007)      │
│  vol_ratio     │ Volatility regime    │ Schwert (1989)              │
│  BPC_composite │ All above combined   │ This paper                  │
└─────────────────────────────────────────────────────────────────────┘

Author: Vihan Lalan
================================================================================
"""

import numpy as np
import pandas as pd

from .config import SEED, BPC_CONFIG


def build_behavioral_proxy_composite(
    returns_df:    pd.DataFrame,
    market_factor: np.ndarray,
    cfg:           dict = BPC_CONFIG,
) -> pd.DataFrame:
    """
    Construct the Behavioral Proxy Composite (BPC) — Synthetic Version.

    Uses an AR(1) sentiment proxy calibrated to match empirical FinBERT
    sentiment autocorrelation and market correlation (Baker & Wurgler 2007).

    Parameters
    ----------
    returns_df    : pd.DataFrame  shape (n_days, n_assets)
    market_factor : np.ndarray    shape (n_days,)
    cfg           : dict          BPC_CONFIG parameter dictionary

    Returns
    -------
    features : pd.DataFrame  — all proxies + BPC_composite, indexed by date
    """
    n_days, n_assets = returns_df.shape
    dates    = returns_df.index
    mkt      = pd.Series(market_factor, index=dates)
    features = pd.DataFrame(index=dates)
    w        = cfg['cs_dispersion_window']
    norm_w   = cfg['rolling_z_window']

    # ── 1. Herding Proxy ─────────────────────────────────────────────────────
    # Logic: low cross-sectional return dispersion → assets moving together → herding
    # Sign: inverted so that HIGH herding_z = high behavioural activity
    cs_dispersion          = returns_df.std(axis=1)
    raw_herd               = -cs_dispersion.rolling(w).mean()
    roll_mean              = raw_herd.rolling(norm_w).mean()
    roll_std               = raw_herd.rolling(norm_w).std().clip(lower=1e-8)
    features['herding_z']  = (raw_herd - roll_mean) / roll_std

    # ── 2. Loss Aversion Proxy ────────────────────────────────────────────────
    # Logic: loss aversion implies downside moves exceed upside moves in magnitude
    # Measure: ratio of downside realized vol to upside realized vol
    features['downside_vol'] = mkt.rolling(w).apply(
        lambda x: x[x < 0].std() if (x < 0).sum() > 2 else np.nan, raw=True
    )
    features['upside_vol'] = mkt.rolling(w).apply(
        lambda x: x[x > 0].std() if (x > 0).sum() > 2 else np.nan, raw=True
    )
    features['vol_asym'] = (
        features['downside_vol'] / features['upside_vol'].clip(lower=1e-8) - 1.0
    )

    # ── 3. Return Autocorrelation Proxy ───────────────────────────────────────
    # Logic: significant positive autocorrelation = momentum bias / information lag
    # Negative autocorrelation = overreaction / mean-reversion pattern
    features['return_autocorr'] = mkt.rolling(cfg['autocorr_window']).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 10 else np.nan,
        raw=True,
    )

    # ── 4. Sentiment Proxy ────────────────────────────────────────────────────
    # In production: replace with FinBERT scores on daily news headlines
    # Here: AR(1) process calibrated to match empirical sentiment autocorrelation
    # and market factor loading from Baker & Wurgler (2007)
    noise     = np.random.normal(0, 1, n_days)
    sentiment = np.zeros(n_days)
    mkt_std   = np.std(market_factor)
    ar_coef   = cfg['sentiment_ar_coef']
    mkt_load  = cfg['sentiment_mkt_loading']
    for t in range(1, n_days):
        running_std = np.std(market_factor[:t+1]) + 1e-8
        sentiment[t] = (
            ar_coef * sentiment[t-1]
            + mkt_load * market_factor[t] / running_std
            + 0.20 * noise[t]
        )
    features['sentiment']        = sentiment
    features['sentiment_change'] = pd.Series(sentiment, index=dates).diff(5)

    # ── 5. Volatility Ratio Proxy ─────────────────────────────────────────────
    # Logic: elevated vol relative to long-run average signals stress / behavioural regimes
    features['vol_short'] = mkt.rolling(20).std()
    features['vol_long']  = mkt.rolling(252).std().clip(lower=1e-8)
    features['vol_ratio'] = features['vol_short'] / features['vol_long']

    # ── BPC Composite: rolling Z-score average of core proxies ───────────────
    core_cols = ['herding_z', 'return_autocorr', 'vol_ratio', 'sentiment']
    bpc_df    = features[core_cols].copy()

    for col in core_cols:
        rm = bpc_df[col].rolling(norm_w, min_periods=60).mean()
        rs = bpc_df[col].rolling(norm_w, min_periods=60).std().clip(lower=1e-8)
        bpc_df[col] = (bpc_df[col] - rm) / rs

    # Equal weights as baseline — elastic net weights would be estimated
    # on the training fold in walk-forward; equal weights here for simplicity
    features['BPC_composite'] = bpc_df.mean(axis=1)

    return features.ffill().dropna()


def build_bpc_real(
    returns_df:    pd.DataFrame,
    market_factor: np.ndarray,
    sentiment_z:   pd.Series = None,
    cfg:           dict = BPC_CONFIG,
) -> pd.DataFrame:
    """
    Behavioral Proxy Composite — Real Market Data Version.

    All five proxies are identical to the synthetic version.
    The only change is the sentiment component:
      - Synthetic version: AR(1) process correlated with market factor
      - Real version: UMCSENT (U. of Michigan Consumer Sentiment) from FRED,
                      lagged 1 month, normalised via expanding z-score.
                      Falls back to AR(1) if FRED data unavailable.

    Parameters
    ----------
    returns_df    : pd.DataFrame  — real daily log returns
    market_factor : np.ndarray   — SPY log returns
    sentiment_z   : pd.Series    — FRED UMCSENT z-scores (or None for fallback)
    cfg           : dict         — BPC_CONFIG

    Returns
    -------
    features : pd.DataFrame — all BPC proxies + BPC_composite
    """
    n_days, n_assets = returns_df.shape
    dates    = returns_df.index
    mkt      = pd.Series(market_factor, index=dates)
    features = pd.DataFrame(index=dates)
    w        = cfg['cs_dispersion_window']
    norm_w   = cfg['rolling_z_window']

    # ── 1. Herding Proxy ─────────────────────────────────────────────────────
    # Christie & Huang (1995): low cross-sectional dispersion = herding
    cs_dispersion         = returns_df.std(axis=1)
    raw_herd              = -cs_dispersion.rolling(w).mean()
    roll_mean             = raw_herd.rolling(norm_w).mean()
    roll_std              = raw_herd.rolling(norm_w).std().clip(lower=1e-8)
    features['herding_z'] = (raw_herd - roll_mean) / roll_std

    # ── 2. Loss Aversion Proxy ────────────────────────────────────────────────
    # Ang et al. (2006): downside vol > upside vol under loss aversion
    features['downside_vol'] = mkt.rolling(w).apply(
        lambda x: x[x < 0].std() if (x < 0).sum() > 2 else np.nan, raw=True
    )
    features['upside_vol'] = mkt.rolling(w).apply(
        lambda x: x[x > 0].std() if (x > 0).sum() > 2 else np.nan, raw=True
    )
    features['vol_asym'] = (
        features['downside_vol'] / features['upside_vol'].clip(lower=1e-8) - 1.0
    )

    # ── 3. Return Autocorrelation Proxy ───────────────────────────────────────
    # Lo & MacKinlay (1988): autocorrelation = departure from random walk
    features['return_autocorr'] = mkt.rolling(cfg['autocorr_window']).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 10 else np.nan,
        raw=True,
    )

    # ── 4. Illiquidity Proxy ─────────────────────────────────────────────────
    # Amihud (2002) illiquidity ratio: |r_t| / volume_t
    # Approximated as |mkt_return| / rolling_vol (price-based proxy)
    features['illiq_proxy'] = (
        mkt.abs() / mkt.rolling(w).std().clip(lower=1e-8)
    ).rolling(w).mean()

    # ── 5. Sentiment ─────────────────────────────────────────────────────────
    if sentiment_z is not None:
        # Real FRED UMCSENT — align to trading dates
        sent_aligned = sentiment_z.reindex(dates).ffill().bfill()
        if sent_aligned.isnull().mean() > 0.5:
            print("  WARNING: Sentiment alignment >50% missing. Using AR(1) fallback.")
            sent_aligned = None

    if sentiment_z is None or 'sent_aligned' not in dir() or sent_aligned is None:
        # AR(1) fallback (identical to synthetic version)
        print("  Using AR(1) sentiment proxy (fallback)")
        noise    = np.random.normal(0, 1, n_days)
        sentiment_arr = np.zeros(n_days)
        for t in range(1, n_days):
            running_std = np.std(market_factor[:t+1]) + 1e-8
            sentiment_arr[t] = (
                0.7 * sentiment_arr[t-1]
                + 0.3 * market_factor[t] / running_std
                + 0.2 * noise[t]
            )
        sent_aligned = pd.Series(sentiment_arr, index=dates)

    features['sentiment'] = sent_aligned

    # ── Volatility ratio proxy ────────────────────────────────────────────────
    features['vol_short'] = mkt.rolling(20).std()
    features['vol_long']  = mkt.rolling(norm_w).std().clip(lower=1e-8)
    features['vol_ratio'] = features['vol_short'] / features['vol_long']

    # ── BPC Composite ─────────────────────────────────────────────────────────
    core_cols = ['herding_z', 'return_autocorr', 'vol_ratio', 'sentiment']
    bpc_df    = features[core_cols].copy()

    for col in core_cols:
        rm = bpc_df[col].rolling(norm_w, min_periods=60).mean()
        rs = bpc_df[col].rolling(norm_w, min_periods=60).std().clip(lower=1e-8)
        bpc_df[col] = (bpc_df[col] - rm) / rs

    features['BPC_composite'] = bpc_df.mean(axis=1)

    return features.ffill().dropna()
