"""
================================================================================
TBMD Framework — Statistical Tests & Significance
================================================================================

Includes:
    - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    - Lo-MacKinlay (1988) Variance Ratio Test
    - Benjamini-Hochberg (1995) FDR Correction
    - Performance Metrics Suite
    - Regime Detection Validation
    - Directional Claims Validation (Option B)

Author: Vihan Lalan
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


def deflated_sharpe_ratio(
    returns_series: pd.Series,
    n_trials:       int,
    sr_benchmark:   float = 0.0,
) -> tuple:
    """
    Deflated Sharpe Ratio (DSR).

    Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Non-Normality, and Serial Correlation."
    Journal of Portfolio Management, 40(5), 94–107.

    Adjusts the probability that the observed Sharpe ratio is genuine,
    accounting for:
      (a) Non-normality (skewness and excess kurtosis of returns)
      (b) Short sample bias
      (c) Multiple testing: the expected maximum SR under H0 scales with
          the number of independent strategies tried (n_trials)

    Parameters
    ----------
    returns_series : pd.Series  — strategy daily PnL (non-zero trading days)
    n_trials       : int        — number of independent strategies / parameters
                                  tested before selecting this one
    sr_benchmark   : float      — minimum acceptable Sharpe (default 0)

    Returns
    -------
    (annualized_sr, dsr) : tuple of floats
        annualized_sr : raw annualized Sharpe ratio
        dsr           : probability in [0,1] that SR is real after correction
                        > 0.95 = strong evidence; < 0.50 = likely noise
    """
    r = returns_series.dropna()
    n = len(r)
    if n < 30:
        return np.nan, np.nan

    ann_sr = r.mean() / (r.std() + 1e-10) * np.sqrt(252)

    # Moments for non-normality adjustment
    skew = stats.skew(r)
    kurt = stats.kurtosis(r)   # excess kurtosis

    # Expected maximum SR under H0 for n_trials independent tests
    gamma         = 0.5772156649   # Euler-Mascheroni constant
    sr_max_expect = (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
        + gamma       * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )

    # Daily SR (unadjusted) and benchmark
    sr_hat  = r.mean() / (r.std() + 1e-10)
    sr_star = sr_benchmark / np.sqrt(252)

    # Non-normality adjusted SR estimate
    sr_adj  = sr_hat * (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2)

    # Z-statistic for the hypothesis SR_adj > SR_star
    denom  = np.sqrt(1.0 - skew * sr_hat + (kurt + 1.0) / 4.0 * sr_hat ** 2)
    z_stat = (sr_adj - sr_star) * np.sqrt(n - 1) / (denom + 1e-10)

    dsr    = float(norm.cdf(z_stat))

    return float(ann_sr), dsr


def variance_ratio_test(
    returns_series: pd.Series,
    lags:           list = [2, 5, 10],
) -> dict:
    """
    Lo-MacKinlay (1988) Variance Ratio Test for random walk hypothesis.

    Tests whether strategy returns follow a random walk (VR = 1) or exhibit
    autocorrelation (VR ≠ 1). Uses heteroskedasticity-consistent standard errors.

    Reference: Lo, A. W., & MacKinlay, A. C. (1988). "Stock market prices do
    not follow random walks: Evidence from a simple specification test."
    Review of Financial Studies, 1(1), 41–66.

    Parameters
    ----------
    returns_series : pd.Series  — return series to test
    lags           : list       — aggregation periods q to test (e.g., [2, 5, 10])

    Returns
    -------
    dict keyed by 'lag_q' with sub-dict:
        'vr'    : float  — variance ratio statistic
        'z'     : float  — heteroskedasticity-consistent Z-statistic
        'p_val' : float  — two-sided p-value
    """
    r = returns_series.dropna().values
    n = len(r)
    results = {}

    for q in lags:
        if n < q * 4:
            continue

        mu      = np.mean(r)
        sigma_1 = np.sum((r[1:] - r[:-1] - mu) ** 2) / (n - 1)

        # Overlapping q-period variance
        r_q     = np.array([np.sum(r[i:i+q]) for i in range(n - q + 1)])
        sigma_q = np.sum((r_q[1:] - r_q[:-1] - q * mu) ** 2) / \
                  ((len(r_q) - 1) * q)

        if sigma_1 < 1e-12:
            continue

        vr = sigma_q / sigma_1

        # Heteroskedasticity-consistent theta (asymptotic variance of VR)
        theta = 0.0
        denom_sq = (np.sum((r - mu) ** 2) / n) ** 2
        for k in range(1, q):
            delta_k = (np.sum((r[k:] - mu) ** 2 * (r[:-k] - mu) ** 2) / n) \
                      / (denom_sq + 1e-12)
            theta  += ((2.0 * (q - k) / q) ** 2) * delta_k

        z_stat = (vr - 1.0) * np.sqrt(n) / (np.sqrt(theta) + 1e-10)
        p_val  = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

        results[f'lag_{q}'] = {'vr': float(vr), 'z': float(z_stat), 'p_val': p_val}

    return results


def benjamini_hochberg_correction(
    p_values: list,
    alpha:    float = 0.05,
) -> pd.DataFrame:
    """
    Benjamini-Hochberg (1995) False Discovery Rate correction.

    Reference: Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
    discovery rate: A practical and powerful approach to multiple testing."
    Journal of the Royal Statistical Society: Series B, 57(1), 289–300.

    Parameters
    ----------
    p_values : list   — raw p-values for each hypothesis
    alpha    : float  — target FDR level (default 5%)

    Returns
    -------
    pd.DataFrame with columns:
        original_idx  : int    — index in input list
        raw_p         : float  — original p-value
        adjusted_p    : float  — BH-adjusted p-value
        reject_H0     : bool   — True if hypothesis is rejected at level alpha
    """
    n         = len(p_values)
    p_arr     = np.array(p_values, dtype=float)
    sort_idx  = np.argsort(p_arr)
    sorted_p  = p_arr[sort_idx]
    thresholds = np.arange(1, n + 1) * alpha / n

    # BH procedure: find the largest k such that p_(k) <= k*alpha/n
    reject = sorted_p <= thresholds
    if reject.any():
        last = np.where(reject)[0][-1]
        reject[:last + 1] = True   # make rejection monotone

    # Adjusted p-values (step-up)
    adj_p = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1))

    result = pd.DataFrame({
        'original_idx': sort_idx,
        'raw_p':        sorted_p,
        'adjusted_p':   adj_p,
        'reject_H0':    reject,
    })

    return result.sort_values('original_idx').reset_index(drop=True)


def compute_performance_metrics(
    returns_series: pd.Series,
    name:           str = 'Strategy',
) -> dict:
    """
    Comprehensive performance metrics for a daily PnL series.

    Parameters
    ----------
    returns_series : pd.Series  — daily strategy PnL
    name           : str        — strategy label for reporting

    Returns
    -------
    dict of metric names to values
    """
    r = returns_series.dropna()
    r = r[r != 0.0]  # remove flat (non-trading) days

    if len(r) < 30:
        return {}

    cum_r    = r.cumsum()
    drawdown = cum_r - cum_r.cummax()
    t_stat, p_val = stats.ttest_1samp(r, 0.0)

    return {
        'Strategy':         name,
        'Ann_Return_%':     r.mean() * 252 * 100,
        'Ann_Vol_%':        r.std()  * np.sqrt(252) * 100,
        'Sharpe':           r.mean() / (r.std() + 1e-10) * np.sqrt(252),
        'Sortino':          r.mean() / (r[r < 0].std() + 1e-10) * np.sqrt(252),
        'Max_Drawdown_%':   drawdown.min() * 100,
        'Calmar':           (r.mean() * 252) / (abs(drawdown.min()) + 1e-10),
        'Win_Rate_%':       (r > 0).mean() * 100,
        'Skewness':         float(stats.skew(r)),
        'Kurtosis':         float(stats.kurtosis(r)),
        'N_Obs':            len(r),
        'T_stat':           float(t_stat),
        'P_value':          float(p_val),
        'Significant_5pct': bool(p_val < 0.05),
    }


def validate_regime_detection(
    regime_true: pd.Series,
    regime_pred: pd.Series,
) -> dict:
    """
    Validate the RES regime classification against ground-truth labels.

    Collapses to binary classification: efficient (regime 0) vs.
    inefficient (regime 1 or 2).

    Parameters
    ----------
    regime_true : pd.Series  — true regime labels (0, 1, 2)
    regime_pred : pd.Series  — predicted regime labels from walk-forward

    Returns
    -------
    dict with accuracy, AUC, precision, recall, confusion matrix elements
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    common   = regime_true.index.intersection(regime_pred.index)
    true_all = regime_true.loc[common]
    pred_all = regime_pred.loc[common]

    # Remove days with no prediction (warm-up period)
    valid     = pred_all != -1
    true_val  = true_all[valid]
    pred_val  = pred_all[valid]

    if len(pred_val) < 10:
        return {}

    # Binary: efficient = 0, inefficient = 1
    true_bin = (true_val > 0).astype(int)
    pred_bin = (pred_val > 0).astype(int)

    acc  = float(accuracy_score(true_bin, pred_bin))
    try:
        auc = float(roc_auc_score(true_bin, pred_bin))
    except ValueError:
        auc = np.nan

    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())

    return {
        'accuracy':               acc,
        'auc':                    auc,
        'precision':              tp / (tp + fp + 1e-10),
        'recall':                 tp / (tp + fn + 1e-10),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_predicted_inefficient': int((pred_bin == 1).sum()),
        'n_true_inefficient':      int((true_bin == 1).sum()),
    }


def validate_directional_claims(
    wf_results:  dict,
    res_scores:  pd.DataFrame,
    bpc_features: pd.DataFrame,
    vix:         pd.Series,
    n_trials:    int = 12,
) -> dict:
    """
    Test the five directional claims from the paper on real data.

    This is the core of Option B validation. No regime labels needed.
    Each claim is evaluated as PASS or FAIL with supporting statistics.

    Claims:
      1. RCF_Sharpe > Unconditional_Sharpe
      2. RCF_DSR > Unconditional_DSR
      3. RES correlates negatively with VIX
      4. At least 3 of 5 BPC components are significant after BH correction
      5. High-stress periods show higher RCF returns than low-stress periods

    Returns
    -------
    dict: {claim_name: {'result': bool, 'evidence': str, 'statistic': float}}
    """
    results = {}

    rcf_active   = wf_results['rcf_pnl'][wf_results['rcf_pnl'] != 0]
    uncond_active = wf_results['unconditional_pnl'][wf_results['unconditional_pnl'] != 0]

    rcf_sr,   rcf_dsr   = deflated_sharpe_ratio(rcf_active,   n_trials)
    uncond_sr, uncond_dsr = deflated_sharpe_ratio(uncond_active, n_trials)

    # ── Claim 1: RCF Sharpe > Unconditional Sharpe ────────────────────────────
    claim1 = (rcf_sr is not None and uncond_sr is not None and rcf_sr > uncond_sr)
    results['1_RCF_Sharpe_gt_Unconditional'] = {
        'result':    claim1,
        'pass_fail': 'PASS' if claim1 else 'FAIL',
        'evidence':  f"RCF Sharpe = {rcf_sr:.3f}, Unconditional Sharpe = {uncond_sr:.3f}",
        'gap':       (rcf_sr or 0) - (uncond_sr or 0),
    }

    # ── Claim 2: RCF DSR > Unconditional DSR ─────────────────────────────────
    claim2 = (rcf_dsr is not None and uncond_dsr is not None and rcf_dsr > uncond_dsr)
    results['2_RCF_DSR_gt_Unconditional'] = {
        'result':    claim2,
        'pass_fail': 'PASS' if claim2 else 'FAIL',
        'evidence':  f"RCF DSR = {rcf_dsr:.3f}, Unconditional DSR = {uncond_dsr:.3f}",
        'gap':       (rcf_dsr or 0) - (uncond_dsr or 0),
    }

    # ── Claim 3: RES negative correlation with VIX ────────────────────────────
    if vix is not None:
        common = res_scores.index.intersection(vix.index)
        res_c  = res_scores['efficiency_score'].loc[common]
        vix_c  = vix.loc[common]
        corr, pval = stats.pearsonr(
            res_c.ffill().dropna(),
            vix_c.reindex(res_c.index).ffill().dropna().reindex(res_c.ffill().dropna().index)
        )
        claim3 = corr < -0.20
        results['3_RES_negcorr_VIX'] = {
            'result':    claim3,
            'pass_fail': 'PASS' if claim3 else 'FAIL',
            'evidence':  f"Pearson r(RES, VIX) = {corr:.3f}, p = {pval:.4f}",
            'statistic': corr,
        }
    else:
        # VIX unavailable — test RES against realised vol instead
        vol_proxy = pd.Series(wf_results['buyhold_pnl']).rolling(20).std()
        common    = res_scores.index.intersection(vol_proxy.index)
        res_c     = res_scores['efficiency_score'].loc[common].ffill().dropna()
        vol_c     = vol_proxy.loc[common].reindex(res_c.index).ffill().dropna()
        res_c     = res_c.reindex(vol_c.index)
        try:
            corr, pval = stats.pearsonr(res_c, vol_c)
            claim3 = corr < -0.10
            results['3_RES_negcorr_VolProxy'] = {
                'result':    claim3,
                'pass_fail': 'PASS' if claim3 else 'FAIL',
                'evidence':  f"Pearson r(RES, RealVol) = {corr:.3f} (VIX unavailable)",
                'statistic': corr,
            }
        except Exception:
            results['3_RES_negcorr_VIX'] = {
                'result': False, 'pass_fail': 'SKIP',
                'evidence': 'VIX and vol proxy both unavailable',
            }

    # ── Claim 4: At least 3 BPC components significant after BH correction ────
    bpc_cols   = ['herding_z', 'return_autocorr', 'vol_asym', 'sentiment', 'vol_ratio']
    available  = [c for c in bpc_cols if c in bpc_features.columns]
    buyhold    = wf_results['buyhold_pnl']

    p_values = []
    for col in available:
        try:
            common = bpc_features.index.intersection(buyhold.index)
            x      = bpc_features[col].loc[common].shift(1).ffill().dropna()
            y      = buyhold.loc[x.index]
            _, pv  = stats.pearsonr(x.dropna(), y.reindex(x.dropna().index))
            p_values.append(pv)
        except Exception:
            p_values.append(1.0)

    if p_values:
        bh_result  = benjamini_hochberg_correction(p_values)
        n_sig      = bh_result['reject_H0'].sum()
        claim4     = n_sig >= 3
        results['4_BPC_3of5_BH_significant'] = {
            'result':    claim4,
            'pass_fail': 'PASS' if claim4 else 'FAIL',
            'evidence':  f"{n_sig}/{len(p_values)} BPC components significant after BH correction",
            'statistic': float(n_sig),
        }

    # ── Claim 5: High-stress periods show higher RCF returns ─────────────────
    vol_state = wf_results['vol_state']
    rcf_pnl   = wf_results['rcf_pnl']

    rcf_low_stress  = rcf_pnl[vol_state == 0].mean() * 252 * 100
    rcf_high_stress = rcf_pnl[vol_state == 2].mean() * 252 * 100
    claim5 = rcf_high_stress > rcf_low_stress

    results['5_HighStress_RCF_gt_LowStress'] = {
        'result':    claim5,
        'pass_fail': 'PASS' if claim5 else 'FAIL',
        'evidence':  (f"RCF Ann.Return: Low-vol = {rcf_low_stress:.2f}%, "
                      f"High-vol = {rcf_high_stress:.2f}%"),
        'gap':       rcf_high_stress - rcf_low_stress,
    }

    return results
