"""
================================================================================
TBMD Framework — Real-Time Efficiency Score (RES)
================================================================================

Combines three published statistical tests into a single composite measure
of market efficiency:

    Hurst_eff(t)  = 1 - 2|H(t) - 0.5|          (0=trending/mean-rev, 1=efficient)
    VR_eff(t)     = 1 / (1 + |VR(t) - 1|)       (0=autocorrelated, 1=random walk)
    LB_eff(t)     = p-value of Ljung-Box test    (0=correlated, 1=no autocorr)

    RES(t)        = (1/3) * [Hurst_eff + VR_eff + LB_eff]

RES close to 1.0 → market near random walk → signals unlikely to be predictive
RES close to 0.0 → behavioural regime → BPC signals potentially informative

Key References:
    Hurst (1951); Mandelbrot (1971)  — Hurst Exponent / R/S Analysis
    Lo & MacKinlay (1988)            — Variance Ratio Test
    Ljung & Box (1978)               — Autocorrelation Test

Author: Vihan Lalan
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

from .config import RES_CONFIG


def _hurst_exponent(ts: np.ndarray) -> float:
    """
    Hurst exponent via R/S (rescaled range) analysis.

    Input ts is already log-returns — no log transform applied.
    Partitions ts into sub-periods, computes R/S for each,
    and fits slope of log(R/S) vs log(n) to estimate H.

    H = 0.5  → random walk (efficient)
    H > 0.5  → long-range dependence / trending
    H < 0.5  → mean-reverting

    Reference: Hurst (1951); Mandelbrot (1971)
    """
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    if n < 20 or np.std(ts) < 1e-10:
        return 0.5

    # Sub-period sizes: n/16, n/8, n/4, n/2 (at least 8 each)
    sub_sizes = [max(8, n // k) for k in [16, 8, 4, 2]]
    rs_means, size_log = [], []

    for sz in sub_sizes:
        if sz > n:
            continue
        chunks = [ts[i:i + sz] for i in range(0, n - sz + 1, sz)]
        rs_vals = []
        for c in chunks:
            s = np.std(c)
            if s > 1e-10:
                cumdev = np.cumsum(c - c.mean())
                r = cumdev.max() - cumdev.min()
                rs_vals.append(r / s)
        if rs_vals:
            rs_means.append(np.log(np.mean(rs_vals)))
            size_log.append(np.log(sz))

    if len(rs_means) < 2:
        return 0.5

    try:
        slope = np.polyfit(size_log, rs_means, 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def _variance_ratio(ts: np.ndarray, q: int = RES_CONFIG['vr_lag']) -> float:
    """
    Lo-MacKinlay (1988) Variance Ratio statistic at lag q.

    VR(q) = 1.0  → random walk
    VR(q) > 1.0  → positive autocorrelation (momentum)
    VR(q) < 1.0  → negative autocorrelation (mean reversion)
    """
    n = len(ts)
    if n < q * 4:
        return 1.0

    mu       = np.mean(ts)
    sigma_1  = np.sum((ts[1:] - ts[:-1] - mu) ** 2) / (n - 1)

    # Overlapping q-period returns
    returns_q = np.array([np.sum(ts[i:i+q]) for i in range(n - q + 1)])
    sigma_q   = np.sum((returns_q[1:] - returns_q[:-1] - q * mu) ** 2) \
                / ((len(returns_q) - 1) * q)

    return float(sigma_q / sigma_1) if sigma_1 > 1e-12 else 1.0


def _ljung_box_pval(ts: np.ndarray, lags: int = RES_CONFIG['lb_lags']) -> float:
    """
    Ljung-Box (1978) test p-value for return autocorrelation.

    High p-value → no significant autocorrelation → efficient market
    Low p-value  → significant autocorrelation    → inefficient market
    """
    n = len(ts)
    if n < lags + 10:
        return 0.5

    acf_vals = [
        np.corrcoef(ts[:-k], ts[k:])[0, 1] if k > 0 else 1.0
        for k in range(1, lags + 1)
    ]

    q_stat = n * (n + 2) * sum(
        rho ** 2 / (n - k) for k, rho in enumerate(acf_vals, 1)
    )

    return float(1.0 - stats.chi2.cdf(q_stat, df=lags))


def compute_real_time_efficiency_score(
    market_factor: np.ndarray,
    dates:         pd.DatetimeIndex,
    cfg:           dict = RES_CONFIG,
) -> pd.DataFrame:
    """
    Real-Time Efficiency Score (RES) — Synthetic Version.

    Parameters
    ----------
    market_factor : np.ndarray    shape (n_days,)  — common market return factor
    dates         : DatetimeIndex shape (n_days,)  — corresponding dates
    cfg           : dict          RES_CONFIG parameter dictionary

    Returns
    -------
    scores : pd.DataFrame  — hurst, variance_ratio, lb_pval, efficiency_score,
                             inefficiency_score; indexed by date
    """
    window = cfg['window']
    mkt    = pd.Series(market_factor, index=dates)
    scores = pd.DataFrame(index=dates)

    # ── Rolling statistical tests ─────────────────────────────────────────────
    scores['hurst']          = mkt.rolling(window).apply(_hurst_exponent,     raw=True)
    scores['variance_ratio'] = mkt.rolling(window).apply(_variance_ratio,     raw=True)
    scores['lb_pval']        = mkt.rolling(window).apply(_ljung_box_pval,     raw=True)

    # ── Normalize each test to [0, 1] efficiency scale ───────────────────────
    hurst_eff = 1.0 - 2.0 * np.abs(scores['hurst'] - 0.5)
    vr_eff    = 1.0 / (1.0 + np.abs(scores['variance_ratio'] - 1.0))
    lb_eff    = scores['lb_pval']

    # ── Composite ─────────────────────────────────────────────────────────────
    scores['efficiency_score']   = (hurst_eff + vr_eff + lb_eff) / 3.0
    scores['inefficiency_score'] = 1.0 - scores['efficiency_score']

    return scores


# Alias for real-data usage (identical formula)
compute_res = compute_real_time_efficiency_score
