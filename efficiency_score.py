"""
efficiency_score.py
-------------------
Computes the Real-Time Efficiency Score (RES), a composite of three
established tests that together measure how close the market is to a
random walk on any given day.

Components:
  H(t)  — Hurst exponent via R/S analysis (Hurst 1951; Mandelbrot 1971)
  VR(t) — Lo-MacKinlay variance ratio at lag q=5 (Lo & MacKinlay 1988)
  LB(t) — Ljung-Box autocorrelation p-value (Ljung & Box 1978)

Normalization:
  H_eff  = 1 - 2|H - 0.5|       (1.0 at H=0.5, decays toward 0 as H deviates)
  VR_eff = 1 / (1 + |VR - 1|)   (1.0 at VR=1, decays as VR deviates)
  LB_eff = p-value               (1.0 means no significant autocorrelation)
  RES    = (H_eff + VR_eff + LB_eff) / 3

RES close to 1.0 = efficient (random walk).
RES below threshold tau = potentially exploitable inefficiency regime.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# 1. Hurst Exponent — R/S Analysis
# ------------------------------------------------------------------

def hurst_rs(series: np.ndarray) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.
    Regression of log(R/S) on log(n) across multiple sub-period lengths.

    H = 0.5 — random walk (efficient)
    H > 0.5 — trending / long memory (inefficient, persistent)
    H < 0.5 — mean reverting (inefficient, anti-persistent)

    Based on Mandelbrot (1971) and Hurst (1951).
    """
    n = len(series)
    if n < 20:
        return np.nan

    series = np.asarray(series, dtype=float)

    lags = range(10, min(n // 2, 100), max(1, n // 40))
    rs_vals = []
    lag_vals = []

    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, n - lag, lag)]
        if len(chunks) < 2:
            continue

        rs_chunk = []
        for chunk in chunks:
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
            lag_vals.append(lag)

    if len(rs_vals) < 4:
        return np.nan

    log_lags = np.log(lag_vals)
    log_rs = np.log(rs_vals)

    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return float(slope)


def rolling_hurst(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling Hurst exponent computed over a sliding window.
    """
    h_vals = []
    idx = []
    for i in range(window, len(returns) + 1):
        h = hurst_rs(returns.iloc[i - window:i].values)
        h_vals.append(h)
        idx.append(returns.index[i - 1])
    return pd.Series(h_vals, index=idx, name='hurst')


# ------------------------------------------------------------------
# 2. Ljung-Box Autocorrelation Test
# ------------------------------------------------------------------

def ljung_box_pvalue(series: np.ndarray, lags: int = 10) -> float:
    """
    Ljung-Box Q test for autocorrelation up to specified lags.
    Returns the p-value. Low p-value (< 0.05) = significant autocorrelation.
    High p-value (close to 1.0) = consistent with white noise (efficient).
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < lags + 5:
        return np.nan
    try:
        # Pure numpy/scipy implementation of Ljung-Box test
        acf_vals = [
            np.corrcoef(s[:-k], s[k:])[0, 1] if k > 0 else 1.0
            for k in range(1, lags + 1)
        ]
        q_stat = n * (n + 2) * sum(
            rho ** 2 / (n - k) for k, rho in enumerate(acf_vals, 1)
        )
        return float(1.0 - stats.chi2.cdf(q_stat, df=lags))
    except Exception:
        return np.nan


def rolling_ljungbox(returns: pd.Series, window: int = 60, lags: int = 10) -> pd.Series:
    """
    Rolling Ljung-Box p-value.
    """
    pvals = []
    idx = []
    for i in range(window, len(returns) + 1):
        p = ljung_box_pvalue(returns.iloc[i - window:i].values, lags=lags)
        pvals.append(p)
        idx.append(returns.index[i - 1])
    return pd.Series(pvals, index=idx, name='lb_pval')


# ------------------------------------------------------------------
# 3. RES Computation
# ------------------------------------------------------------------

def compute_res(returns: pd.Series,
                hurst_window: int = 252,
                vr_window: int = 60,
                lb_window: int = 60,
                vr_lag: int = 5) -> pd.DataFrame:
    """
    Compute the Real-Time Efficiency Score and its three components.

    Parameters
    ----------
    returns      : pd.Series  daily return series (market or aggregate)
    hurst_window : int        rolling window for Hurst computation (default: 252)
    vr_window    : int        rolling window for variance ratio (default: 60)
    lb_window    : int        rolling window for Ljung-Box test (default: 60)
    vr_lag       : int        lag parameter for variance ratio (default: 5)

    Returns
    -------
    pd.DataFrame with columns:
        hurst, vr, lb_pval,
        hurst_eff, vr_eff, lb_eff,
        RES
    """
    from behavioral_proxies import rolling_variance_ratio

    print("Computing RES components...")

    # Raw components
    hurst_s = rolling_hurst(returns, window=hurst_window)
    vr_s = rolling_variance_ratio(returns, window=vr_window, q=vr_lag)
    lb_s = rolling_ljungbox(returns, window=lb_window)

    # Align to common index
    df = pd.DataFrame({
        'hurst' : hurst_s,
        'vr'    : vr_s,
        'lb_pval': lb_s,
    }).dropna(how='any')

    # Normalize to [0, 1] efficiency scale
    df['hurst_eff'] = 1.0 - 2.0 * (df['hurst'] - 0.5).abs()
    df['vr_eff']    = 1.0 / (1.0 + (df['vr'] - 1.0).abs())
    df['lb_eff']    = df['lb_pval'].clip(0, 1)

    # Composite (equal-weighted)
    df['RES'] = (df['hurst_eff'] + df['vr_eff'] + df['lb_eff']) / 3.0

    # Ensure RES is in [0, 1]
    df['RES'] = df['RES'].clip(0, 1)

    print(f"RES computed: {len(df)} valid observations")
    print(f"RES statistics:")
    print(f"  Mean: {df['RES'].mean():.3f}, Std: {df['RES'].std():.3f}")
    print(f"  Min:  {df['RES'].min():.3f}, Max: {df['RES'].max():.3f}")

    return df


# ------------------------------------------------------------------
# 4. RES Validation against VIX-defined regimes
# ------------------------------------------------------------------

def validate_res(res_df: pd.DataFrame,
                 vix: pd.Series,
                 tau_percentile: float = 40.0) -> dict:
    """
    Validate RES against externally-defined VIX regimes.

    VIX thresholds (widely-used in literature):
        VIX < 15  : Efficient Bull
        15 <= VIX < 25 : Transitional
        VIX >= 25 : Behavioral Crash

    Classification task: can the RES below threshold tau
    correctly identify high-VIX (behaviorally active) days?

    Parameters
    ----------
    res_df          : pd.DataFrame  RES output from compute_res()
    vix             : pd.Series     real VIX daily close
    tau_percentile  : float         percentile of RES to use as threshold tau

    Returns
    -------
    dict with classification accuracy, AUC, precision, recall
    """
    from sklearn.metrics import (roc_auc_score, precision_score,
                                  recall_score, accuracy_score)

    # Align RES and VIX
    common = res_df.index.intersection(vix.index)
    res_aligned = res_df.loc[common, 'RES']
    vix_aligned = vix.loc[common]

    # Binary label: 1 = behavioral/inefficient (VIX >= 20)
    # Using VIX 20 as the behavioral regime threshold (Ang & Bekaert 2002)
    y_true = (vix_aligned >= 20).astype(int).values

    # RES-based prediction: low RES = inefficient = 1
    tau = np.percentile(res_aligned.dropna(), tau_percentile)
    y_pred = (res_aligned <= tau).astype(int).values

    # Remove NaN rows
    mask = ~np.isnan(res_aligned.values)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    res_vals = res_aligned.values[mask]

    # AUC: note RES is INVERTED (low = inefficient), so we use 1 - RES
    auc = roc_auc_score(y_true, 1.0 - res_vals)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    # Correlation of RES with VIX (should be negative)
    corr = np.corrcoef(res_vals, vix_aligned.values[mask])[0, 1]

    results = {
        'tau'               : round(tau, 3),
        'accuracy'          : round(acc, 3),
        'auc'               : round(auc, 3),
        'precision'         : round(prec, 3),
        'recall'            : round(rec, 3),
        'vix_res_corr'      : round(corr, 3),
        'n_observations'    : int(mask.sum()),
        'n_behavioral_days' : int(y_true.sum()),
        'pct_behavioral'    : round(float(y_true.mean()) * 100, 1),
    }

    return results
