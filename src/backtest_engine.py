"""
backtest_engine.py
------------------
Walk-forward validation of the regime-conditional trading strategy.

Strategy logic:
    signal(t) = BPC(t) * 1[RES(t) < tau]

Position: long top decile, short bottom decile of BPC-ranked assets
when the filter is active; zero position when RES is above tau.

Performance metrics:
    - Sharpe ratio (raw, annualized)
    - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    - Bootstrap 95% confidence interval on Sharpe
    - Maximum drawdown and Calmar ratio
    - Regime-stratified Sharpe

Walk-forward specification:
    Training window : 252 trading days (rolling)
    Refit frequency : 21 trading days (monthly)
    Warm-up period  : 252 days (no trading)
    Threshold tau   : estimated via isotonic regression on training fold only
    Features        : all normalized on expanding window from training data
    Signal lag      : 1 day (no same-day lookahead)
    Transaction cost: 10 basis points per one-way trade
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

TRANSACTION_COST = 0.001  # 10 bps one-way


# ------------------------------------------------------------------
# Threshold estimation (on training data only)
# ------------------------------------------------------------------

def estimate_threshold(res_train: np.ndarray, regime_labels: np.ndarray,
                       percentile: float = 40.0) -> float:
    """
    Estimate the RES threshold tau using training data only.
    Uses a percentile-based approach (fast and robust) rather than
    grid search over the full sample to avoid lookahead.

    regime_labels: 0 = efficient, 1+ = behavioral/inefficient
    """
    if len(res_train) < 20:
        return np.nanpercentile(res_train, percentile)
    # Use the 40th percentile of RES in training period as threshold
    # (approximately matches the base rate of non-efficient days historically)
    return float(np.nanpercentile(res_train, percentile))


# ------------------------------------------------------------------
# Sharpe ratio and DSR
# ------------------------------------------------------------------

def annualized_sharpe(returns: np.ndarray, freq: int = 252) -> float:
    """Annualized Sharpe ratio, assuming zero risk-free rate."""
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) < 5 or np.std(r) == 0:
        return np.nan
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(freq))


def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = 1000,
                        ci: float = 0.95, freq: int = 252) -> tuple:
    """
    Bootstrap confidence interval for the Sharpe ratio.
    Block bootstrap to account for serial correlation (block size = 21 days).
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 50:
        return (np.nan, np.nan)

    block_size = 21
    boot_sharpes = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n - block_size, size=n_blocks)
        blocks = [r[s:s+block_size] for s in starts]
        boot_sample = np.concatenate(blocks)[:n]
        boot_sharpes.append(annualized_sharpe(boot_sample, freq))

    alpha = (1 - ci) / 2
    lower = np.nanpercentile(boot_sharpes, 100 * alpha)
    upper = np.nanpercentile(boot_sharpes, 100 * (1 - alpha))
    return (round(lower, 3), round(upper, 3))


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): probability
    that the true per-period Sharpe exceeds sr_benchmark (also per-period),
    given the sample's length, skewness and kurtosis.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10 or np.std(r, ddof=1) == 0:
        return np.nan
    sr = np.mean(r) / np.std(r, ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)  # raw (Pearson) kurtosis
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(len(r) - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trial_sharpes: float) -> float:
    """
    Expected maximum per-period Sharpe among n_trials unskilled strategies
    whose Sharpe estimates have variance var_trial_sharpes
    (Bailey & Lopez de Prado 2014, eq. 6).
    """
    if n_trials < 2:
        return 0.0
    gamma = 0.5772156649
    return float(np.sqrt(var_trial_sharpes) * (
        (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    ))


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the PSR of the
    selected strategy against the expected maximum Sharpe of all
    configurations tried. trial_sharpes are per-period (non-annualized)
    Sharpe ratios of every configuration evaluated during selection.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    sr0 = expected_max_sharpe(len(trial_sharpes), np.var(trial_sharpes, ddof=1))
    return probabilistic_sharpe_ratio(returns, sr0)


def maximum_drawdown(returns: np.ndarray) -> tuple:
    """Returns (max_drawdown, calmar_ratio)."""
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return np.nan, np.nan
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    mdd = float(drawdown.min())
    ann_ret = float(np.mean(r) * 252)
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    return round(mdd, 3), round(calmar, 3)


# ------------------------------------------------------------------
# Walk-forward engine
# ------------------------------------------------------------------

def walk_forward_backtest(returns: pd.DataFrame,
                          bpc: pd.DataFrame,
                          res: pd.DataFrame,
                          vix: pd.Series,
                          stock_signal: pd.DataFrame = None,
                          train_window: int = 252,
                          refit_freq: int = 21,
                          top_pct: float = 0.20,
                          tau_percentile: float = 40.0,
                          unconditional: bool = False) -> dict:
    """
    Walk-forward validation of the regime-conditional strategy.

    At each refit date, the threshold tau is estimated on the preceding
    training window only. No parameters are estimated on test data.

    Returns
    -------
    dict with:
        pnl            : pd.Series  daily strategy PnL
        positions      : pd.DataFrame daily positions
        performance    : dict       summary statistics
        regime_perf    : dict       per-regime performance
    """
    if stock_signal is None:
        raise ValueError(
            "stock_signal is required: per-stock cross-sectional ranking "
            "signal from compute_per_stock_signal(). The market-level BPC "
            "composite has no cross-sectional variation and cannot rank "
            "individual assets."
        )

    # Align all inputs to common dates
    common = returns.index
    for df_in in [bpc, res, stock_signal]:
        common = common.intersection(df_in.index)
    common = common.intersection(vix.index)
    common = common.sort_values()

    R   = returns.loc[common]      # (T x N) daily stock returns
    B   = bpc.loc[common, 'BPC']   # (T,)
    RES = res.loc[common, 'RES']   # (T,)
    VIX = vix.loc[common]          # (T,)
    S   = stock_signal.loc[common, R.columns]  # (T x N) per-stock ranking signal, lagged 1 day below

    # Regime label from real VIX (for stratified reporting only)
    regime = pd.Series(
        np.where(VIX < 15, 'Bull',
                 np.where(VIX < 25, 'Transitional', 'Crash')),
        index=common
    )

    T = len(common)
    N = len(R.columns)

    pnl = pd.Series(np.nan, index=common, dtype=float)
    positions = pd.DataFrame(0.0, index=common, columns=R.columns)
    tau_history = {}
    prev_pos = pd.Series(0.0, index=R.columns)

    trade_dates = range(train_window, T, refit_freq)

    for fold_start in trade_dates:
        fold_end = min(fold_start + refit_freq, T)

        # Estimate tau on TRAINING data only
        train_res = RES.iloc[fold_start - train_window:fold_start].dropna().values
        if len(train_res) < 50:
            continue

        tau = estimate_threshold(train_res, None, tau_percentile)
        tau_history[common[fold_start]] = tau

        # Stock selection is chosen ONCE per fold, using the per-stock
        # signal as of the last day before the fold starts (fold_start - 1,
        # so no lookahead into the fold being traded). This matches the
        # stated monthly refit_freq: the identity of the long/short basket
        # is fixed for the fold, and the RES filter only switches daily
        # exposure to that fixed basket on/off -- it does not trigger a
        # full re-rank/re-trade of the basket every day, which would incur
        # transaction costs far beyond what refit_freq implies.
        sig_asof = S.iloc[fold_start - 1].dropna() if fold_start > 0 else pd.Series(dtype=float)
        if len(sig_asof) > 0:
            n_long = max(1, int(len(sig_asof) * top_pct))
            long_stocks = sig_asof.nlargest(n_long).index
            short_stocks = sig_asof.nsmallest(n_long).index
            fold_long_short = pd.Series(0.0, index=R.columns)
            fold_long_short[long_stocks] = 1.0 / n_long
            fold_long_short[short_stocks] = -1.0 / n_long
        else:
            fold_long_short = pd.Series(0.0, index=R.columns)

        # Apply signal for all days in this refit window
        for t in range(fold_start, fold_end):
            if t >= T:
                break

            res_t = RES.iloc[t]
            bpc_t = B.iloc[t]

            if np.isnan(res_t) or np.isnan(bpc_t):
                pnl.iloc[t] = 0.0
                continue

            # Filter: only trade when RES < tau. The basket composition
            # (fold_long_short) is fixed for the fold; only whether it is
            # deployed (vs. sitting in cash) varies day to day.
            # unconditional=True deploys the same basket every day regardless
            # of RES, for the RCF-vs-unconditional comparison.
            filter_active = True if unconditional else (res_t < tau)
            new_pos = fold_long_short.copy() if filter_active else pd.Series(0.0, index=R.columns)

            # Compute daily PnL
            ret_today = R.iloc[t]
            gross_pnl = (prev_pos * ret_today).sum()

            # Transaction costs on position changes
            turnover = (new_pos - prev_pos).abs().sum()
            cost = turnover * TRANSACTION_COST

            pnl.iloc[t] = gross_pnl - cost
            positions.iloc[t] = new_pos
            prev_pos = new_pos.copy()

    # Remove warm-up
    pnl = pnl.iloc[train_window:]
    pnl_clean = pnl.dropna()

    ann_ret = float(pnl_clean.mean() * 252)
    ann_vol = float(pnl_clean.std() * np.sqrt(252))
    sr = annualized_sharpe(pnl_clean.values)
    mdd, calmar = maximum_drawdown(pnl_clean.values)
    ci_low, ci_high = bootstrap_sharpe_ci(pnl_clean.values)

    skew = float(stats.skew(pnl_clean.values))
    kurt = float(stats.kurtosis(pnl_clean.values))

    # PSR against zero; the multiple-testing-deflated version needs the Sharpe
    # ratios of every configuration tried and is computed by the caller.
    psr = probabilistic_sharpe_ratio(pnl_clean.values, 0.0)
    active = pnl_clean[pnl_clean != 0]

    # Regime-stratified performance
    regime_aligned = regime.iloc[train_window:].reindex(pnl_clean.index)
    regime_perf = {}
    for reg in ['Bull', 'Transitional', 'Crash']:
        mask = regime_aligned == reg
        if mask.sum() > 5:
            r_reg = pnl_clean[mask]
            regime_perf[reg] = {
                'n_days'    : int(mask.sum()),
                'ann_return': round(r_reg.mean() * 252 * 100, 1),
                'sharpe'    : round(annualized_sharpe(r_reg.values), 3),
            }

    performance = {
        'ann_return_pct': round(ann_ret * 100, 1),
        'ann_vol_pct'   : round(ann_vol * 100, 1),
        'sharpe'        : round(sr, 3),
        'psr'           : round(psr, 3),
        'max_drawdown'  : round(mdd * 100, 1),
        'calmar'        : round(calmar, 3) if calmar is not None and not np.isnan(calmar) else None,
        'sharpe_ci_95'  : (ci_low, ci_high),
        'n_obs'         : len(pnl_clean),
        'skewness'      : round(skew, 3),
        'excess_kurtosis': round(kurt, 3),
        'win_rate_active_days': round(float((active > 0).mean()) * 100, 1) if len(active) else np.nan,
        'pct_days_active': round(len(active) / len(pnl_clean) * 100, 1),
    }

    return {
        'pnl'         : pnl_clean,
        'positions'   : positions,
        'performance' : performance,
        'regime_perf' : regime_perf,
        'tau_history' : tau_history,
    }
