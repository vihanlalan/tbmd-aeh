"""
run_paper_analysis.py
---------------------
Produces every number reported in the paper from the cached real data
(build_dataset.py) and the Hamilton (1989) simulation.

    python src/build_dataset.py        # once, downloads data/cache/
    python src/run_paper_analysis.py   # writes outputs/paper/

Design (fixed before any lockbox result was seen):
  * Development period: walk-forward PnL dated 2005-01-01 .. 2020-12-31.
  * Lockbox period:     walk-forward PnL dated 2021-01-01 .. 2025-12-31.
  * Configuration grid (N = 12): RES threshold percentile {30, 40} x
    stock-basket refit frequency {21, 63} days x per-stock momentum
    window {20, 60, 120} days.
  * Selection rule: the configuration with the highest development-period
    RCF Sharpe ratio. That single configuration is then evaluated once on
    the lockbox, and applied unchanged to the German (DAX) sample.
"""

import os
import sys
import json
import contextlib
import io
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import backtest_engine
from behavioral_proxies import compute_bpc, compute_per_stock_signal
from efficiency_score import compute_res
from backtest_engine import (walk_forward_backtest, annualized_sharpe, maximum_drawdown,
                             bootstrap_sharpe_ci, probabilistic_sharpe_ratio,
                             deflated_sharpe_ratio)

ROOT = os.path.join(os.path.dirname(__file__), '..')
CACHE = os.path.join(ROOT, 'data', 'cache')
OUT = os.path.join(ROOT, 'outputs', 'paper')

DEV_END = '2020-12-31'
LOCK_START, LOCK_END = '2021-01-01', '2025-12-31'
TAU_PCTS, REFITS, MOM_WINDOWS = (30, 40), (21, 63), (20, 60, 120)
GRID = [(t, r, m) for t in TAU_PCTS for r in REFITS for m in MOM_WINDOWS]
N_BOOT = 2000
BLOCK = 21


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_universe(tag):
    close = pd.read_csv(os.path.join(CACHE, f'{tag}_close.csv'), index_col=0, parse_dates=True)
    volume = pd.read_csv(os.path.join(CACHE, f'{tag}_volume.csv'), index_col=0, parse_dates=True)
    close = close.loc[:LOCK_END]
    volume = volume.reindex_like(close)
    returns = close.pct_change(fill_method=None).iloc[1:]
    returns = returns.mask(unadjusted_split_mask(returns))
    dollar_volume = (close * volume).iloc[1:]
    return returns, dollar_volume


SPLIT_RATIOS = np.array([1/4, 1/3, 1/2, 2/3, 3/2, 2, 3, 4])


def unadjusted_split_mask(returns, min_abs=0.40, tol=0.02):
    """Flag daily moves that match a standard split ratio: Yahoo occasionally
    leaves a split unadjusted, producing a spurious -50% / +100% return."""
    ratio = 1.0 + returns
    near = np.zeros(returns.shape, dtype=bool)
    for s in SPLIT_RATIOS:
        near |= (np.abs(ratio.values / s - 1.0) < tol)
    return pd.DataFrame(near & (returns.abs().values > min_abs),
                        index=returns.index, columns=returns.columns)


def load_indices():
    idx = pd.read_csv(os.path.join(CACHE, 'index_close.csv'), index_col=0, parse_dates=True)
    rec = pd.read_csv(os.path.join(CACHE, 'usrec.csv'), index_col=0, parse_dates=True)['USREC']
    return idx.loc[:LOCK_END], rec


def realized_vol(px, window=21):
    return px.pct_change(fill_method=None).rolling(window).std() * np.sqrt(252) * 100


def drawdown(px):
    return px / px.cummax() - 1.0


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def block_indices(n, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
    return idx[:n]


def classification_metrics(res_series, label, tau_pct=40, train=252, seed=0):
    """RES-based classification of label==1 days with a walk-forward threshold."""
    tau = res_series.rolling(train, min_periods=train).quantile(tau_pct / 100).shift(1)
    df = pd.concat({'res': res_series, 'tau': tau, 'y': label.astype(float)}, axis=1).dropna()
    y = df['y'].values.astype(int)
    pred = (df['res'] <= df['tau']).values.astype(int)
    score = 1.0 - df['res'].values
    out = {
        'n_days': len(df),
        'base_rate_pct': round(y.mean() * 100, 1),
        'accuracy_pct': round((pred == y).mean() * 100, 1),
        'majority_class_accuracy_pct': round(max(y.mean(), 1 - y.mean()) * 100, 1),
        'precision_pct': round(y[pred == 1].mean() * 100, 1) if pred.sum() else np.nan,
        'recall_pct': round(pred[y == 1].mean() * 100, 1) if y.sum() else np.nan,
        'auc': np.nan, 'auc_ci_low': np.nan, 'auc_ci_high': np.nan,
    }
    if 0 < y.sum() < len(y):
        out['auc'] = round(roc_auc_score(y, score), 3)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(500):
            ix = block_indices(len(y), rng)
            if 0 < y[ix].sum() < len(ix):
                boots.append(roc_auc_score(y[ix], score[ix]))
        out['auc_ci_low'] = round(np.percentile(boots, 2.5), 3)
        out['auc_ci_high'] = round(np.percentile(boots, 97.5), 3)
    return out


def perf_stats(pnl, market=None):
    r = pnl.dropna()
    sr = annualized_sharpe(r.values)
    mdd, _ = maximum_drawdown(r.values)
    ci = bootstrap_sharpe_ci(r.values)
    active = r[r != 0]
    out = {
        'start': str(r.index[0].date()), 'end': str(r.index[-1].date()), 'n_days': len(r),
        'ann_return_pct': round(r.mean() * 252 * 100, 2),
        'ann_vol_pct': round(r.std() * np.sqrt(252) * 100, 2),
        'sharpe': round(sr, 3),
        'sharpe_ci_low': ci[0], 'sharpe_ci_high': ci[1],
        'psr_vs_0': round(probabilistic_sharpe_ratio(r.values, 0.0), 3),
        't_stat_mean': round(stats.ttest_1samp(r.values, 0.0).statistic, 2),
        'p_value_mean': round(stats.ttest_1samp(r.values, 0.0).pvalue, 4),
        'max_drawdown_pct': round(mdd * 100, 1),
        'pct_days_active': round(len(active) / len(r) * 100, 1),
        'win_rate_active_pct': round((active > 0).mean() * 100, 1) if len(active) else np.nan,
    }
    if market is not None:
        m = market.reindex(r.index).dropna()
        y = r.reindex(m.index)
        fit = sm.OLS(y.values, sm.add_constant(m.values)).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
        out['alpha_ann_pct'] = round(fit.params[0] * 252 * 100, 2)
        out['alpha_t_nw'] = round(fit.tvalues[0], 2)
        out['beta'] = round(fit.params[1], 3)
    return out


def sharpe_diff_test(a, b, seed=0):
    """Paired circular block bootstrap of annualized Sharpe(a) - Sharpe(b)."""
    df = pd.concat({'a': a, 'b': b}, axis=1).dropna()
    x, y = df['a'].values, df['b'].values
    obs = annualized_sharpe(x) - annualized_sharpe(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        ix = block_indices(len(x), rng)
        diffs.append(annualized_sharpe(x[ix]) - annualized_sharpe(y[ix]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {'sharpe_diff': round(obs, 3),
            'ci_low': round(np.percentile(diffs, 2.5), 3),
            'ci_high': round(np.percentile(diffs, 97.5), 3),
            'p_two_sided': round(min(p, 1.0), 4)}


def daily_sharpe(pnl):
    r = pnl.dropna().values
    return np.mean(r) / np.std(r, ddof=1)


# ------------------------------------------------------------------
# Pipeline pieces
# ------------------------------------------------------------------

def build_signals(returns, dvol, sent_series):
    bpc = quiet(compute_bpc, returns, dvol, sent_series, vr_window=60, herd_window=20,
                asym_window=20, illiq_window=20, sent_window=5)
    res = quiet(compute_res, returns.mean(axis=1), hurst_window=252, vr_window=60, lb_window=60)
    stock_signals = {m: compute_per_stock_signal(returns, dvol, mom_window=m,
                                                 asym_window=20, illiq_window=20)
                     for m in MOM_WINDOWS}
    return bpc, res, stock_signals


def backtest(returns, bpc, res, strat_series, stock_signal, tau, refit, uncond):
    return walk_forward_backtest(returns=returns, bpc=bpc, res=res, vix=strat_series,
                                 stock_signal=stock_signal, train_window=252,
                                 refit_freq=refit, top_pct=0.10, tau_percentile=tau,
                                 unconditional=uncond)['pnl']


def regime_breakdown(pnls, strat_series):
    bins = pd.cut(strat_series, [-np.inf, 15, 25, np.inf],
                  labels=['Calm (<15)', 'Transitional (15-25)', 'Stressed (>=25)'], right=False)
    rows = []
    for reg in bins.cat.categories:
        row = {'regime': reg}
        for name, p in pnls.items():
            p = p.dropna()
            mask = bins.reindex(p.index) == reg
            row['n_days'] = int(mask.sum())
            row[f'{name}_ann_return_pct'] = round(p[mask].mean() * 252 * 100, 1)
            row[f'{name}_sharpe'] = round(annualized_sharpe(p[mask].values), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['pct_time'] = (df['n_days'] / df['n_days'].sum() * 100).round(1)
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def synthetic_section():
    from tbmd_framework import simulate_regime_switching_market
    rows = []
    for seed in range(20):
        rets, _, regime, _, _ = quiet(simulate_regime_switching_market, seed=seed)
        res = quiet(compute_res, rets.mean(axis=1), hurst_window=252, vr_window=60, lb_window=60)
        label = (regime > 0).astype(int)
        m = classification_metrics(res['RES'], label.reindex(res.index), seed=seed)
        m['seed'] = seed
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'synthetic_res_detection_by_seed.csv'), index=False)
    summary = df[['accuracy_pct', 'majority_class_accuracy_pct', 'precision_pct',
                  'recall_pct', 'auc', 'base_rate_pct']].agg(['mean', 'std', 'min', 'max']).round(3)
    summary.to_csv(os.path.join(OUT, 'synthetic_res_detection_summary.csv'))
    print('\nSYNTHETIC RES detection (20 seeds):\n', summary.to_string())
    return summary


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    report['synthetic'] = synthetic_section().to_dict()

    idx, usrec = load_indices()

    # ---------------- United States ----------------
    us_ret, us_dvol = load_universe('us')
    vix = idx['^VIX'].reindex(us_ret.index).ffill()
    us_bpc, us_res, us_sig = build_signals(us_ret, us_dvol, vix)
    print(f'\nUS universe: {us_ret.shape[1]} stocks, {us_ret.index[0].date()}..{us_ret.index[-1].date()}')

    labels = pd.DataFrame({
        'VIX >= 20': vix >= 20,
        'SPY 21d realized vol >= 20%': realized_vol(idx['SPY']).reindex(us_ret.index) >= 20,
        'S&P 500 drawdown >= 10%': drawdown(idx['^GSPC']).reindex(us_ret.index) <= -0.10,
        'NBER recession': usrec.reindex(us_ret.index, method='ffill') == 1,
    })
    det = {name: classification_metrics(us_res['RES'], labels[name]) for name in labels}
    det_df = pd.DataFrame(det).T
    det_df.to_csv(os.path.join(OUT, 'us_res_detection.csv'))
    print('\nUS RES detection by regime proxy:\n', det_df.to_string())

    comp_rows = []
    for col in ['Z_herd', 'Z_vr', 'Z_asym', 'Z_sent', 'Z_illiq', 'BPC']:
        s = us_bpc[col]
        row = {'component': col}
        for name in labels:
            d = pd.concat({'s': s, 'y': labels[name].astype(float)}, axis=1).dropna()
            row[f'auc | {name}'] = round(roc_auc_score(d['y'], d['s']), 3)
        d = pd.concat({'s': s, 'y': labels['VIX >= 20'].astype(float)}, axis=1).dropna()
        row['corr | VIX >= 20'] = round(d['s'].corr(d['y']), 3)
        row['mean | VIX >= 20'] = round(d.loc[d['y'] == 1, 's'].mean(), 3)
        row['mean | VIX < 20'] = round(d.loc[d['y'] == 0, 's'].mean(), 3)
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(os.path.join(OUT, 'us_bpc_components.csv'), index=False)
    print('\nBPC components:\n', comp_df.to_string())

    # Grid on development period
    print('\nRunning 12-configuration grid...')
    grid_rows, rcf_pnls, unc_pnls = [], {}, {}
    for tau, refit, mom in GRID:
        key = (tau, refit, mom)
        rcf = backtest(us_ret, us_bpc, us_res, vix, us_sig[mom], tau, refit, False)
        ukey = (refit, mom)
        if ukey not in unc_pnls:
            unc_pnls[ukey] = backtest(us_ret, us_bpc, us_res, vix, us_sig[mom], tau, refit, True)
        rcf_pnls[key] = rcf
        dev = rcf.loc[:DEV_END]
        grid_rows.append({'tau_pct': tau, 'refit_days': refit, 'mom_window': mom,
                          'dev_rcf_sharpe': round(annualized_sharpe(dev.values), 3),
                          'dev_unc_sharpe': round(annualized_sharpe(unc_pnls[ukey].loc[:DEV_END].values), 3)})
        print(f'  {key}: dev RCF Sharpe {grid_rows[-1]["dev_rcf_sharpe"]:+.3f}')
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(OUT, 'us_dev_grid.csv'), index=False)

    best = grid_df.loc[grid_df['dev_rcf_sharpe'].idxmax()]
    sel = (int(best.tau_pct), int(best.refit_days), int(best.mom_window))
    report['selected_config'] = {'tau_pct': sel[0], 'refit_days': sel[1], 'mom_window': sel[2]}
    print(f'\nSelected on development period: tau={sel[0]} refit={sel[1]} mom={sel[2]}')

    trial_sr = [daily_sharpe(rcf_pnls[k].loc[:DEV_END]) for k in GRID]
    rcf, unc = rcf_pnls[sel], unc_pnls[(sel[1], sel[2])]
    ew = us_ret.mean(axis=1).reindex(rcf.index)
    spy = idx['SPY'].pct_change(fill_method=None).reindex(rcf.index)

    tables = {}
    for period, sl in [('development', slice(None, DEV_END)), ('lockbox', slice(LOCK_START, LOCK_END)),
                       ('full', slice(None, LOCK_END))]:
        t = {
            'RCF': perf_stats(rcf.loc[sl], spy),
            'Unconditional': perf_stats(unc.loc[sl], spy),
            'Buy & Hold (equal weight)': perf_stats(ew.loc[sl], spy),
            'Buy & Hold (SPY)': perf_stats(spy.loc[sl]),
        }
        if period == 'development':
            t['RCF']['dsr_n12'] = round(deflated_sharpe_ratio(rcf.loc[sl].dropna().values, trial_sr), 3)
        tables[period] = pd.DataFrame(t).T
        tables[period].to_csv(os.path.join(OUT, f'us_performance_{period}.csv'))
        print(f'\nUS performance, {period}:\n', tables[period].to_string())

    diff = {p: sharpe_diff_test(rcf.loc[sl], unc.loc[sl])
            for p, sl in [('development', slice(None, DEV_END)),
                          ('lockbox', slice(LOCK_START, LOCK_END)),
                          ('full', slice(None, LOCK_END))]}
    pd.DataFrame(diff).T.to_csv(os.path.join(OUT, 'us_rcf_minus_unconditional.csv'))
    print('\nRCF minus Unconditional Sharpe:\n', pd.DataFrame(diff).T.to_string())
    report['rcf_minus_unconditional'] = diff

    rb = regime_breakdown({'rcf': rcf, 'unconditional': unc, 'buyhold_ew': ew}, vix)
    rb.to_csv(os.path.join(OUT, 'us_regime_breakdown_full.csv'), index=False)
    print('\nRegime breakdown (full sample):\n', rb.to_string())

    cost_rows = []
    for bps in (0, 5, 10, 20):
        backtest_engine.TRANSACTION_COST = bps / 10000
        for name, uncond in [('RCF', False), ('Unconditional', True)]:
            p = backtest(us_ret, us_bpc, us_res, vix, us_sig[sel[2]], sel[0], sel[1], uncond)
            cost_rows.append({'cost_bps': bps, 'strategy': name,
                              'full_sharpe': round(annualized_sharpe(p.dropna().values), 3),
                              'lockbox_sharpe': round(annualized_sharpe(p.loc[LOCK_START:].dropna().values), 3)})
    backtest_engine.TRANSACTION_COST = 0.001
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(os.path.join(OUT, 'us_cost_sensitivity.csv'), index=False)
    print('\nCost sensitivity:\n', cost_df.to_string())

    pd.DataFrame({'rcf': rcf, 'unconditional': unc, 'buyhold_ew': ew, 'spy': spy}).to_csv(
        os.path.join(OUT, 'us_selected_daily_pnl.csv'))

    # ---------------- Germany (out-of-U.S., no re-tuning) ----------------
    de_ret, de_dvol = load_universe('de')
    dax = idx['^GDAXI'].reindex(de_ret.index).ffill()
    dax_rv = realized_vol(dax).reindex(de_ret.index)
    de_bpc, de_res, de_sig = build_signals(de_ret, de_dvol, dax_rv.ffill())
    print(f'\nDE universe: {de_ret.shape[1]} stocks')
    de_labels = pd.DataFrame({
        'DAX 21d realized vol >= 20%': dax_rv >= 20,
        'DAX drawdown >= 10%': drawdown(dax) <= -0.10,
    })
    de_det = pd.DataFrame({n: classification_metrics(de_res['RES'], de_labels[n]) for n in de_labels}).T
    de_det.to_csv(os.path.join(OUT, 'de_res_detection.csv'))
    print('\nDE RES detection:\n', de_det.to_string())

    de_rcf = backtest(de_ret, de_bpc, de_res, dax_rv.ffill(), de_sig[sel[2]], sel[0], sel[1], False)
    de_unc = backtest(de_ret, de_bpc, de_res, dax_rv.ffill(), de_sig[sel[2]], sel[0], sel[1], True)
    de_ew = de_ret.mean(axis=1).reindex(de_rcf.index)
    dax_ret = dax.pct_change(fill_method=None).reindex(de_rcf.index)
    de_tab = pd.DataFrame({
        'RCF': perf_stats(de_rcf, dax_ret),
        'Unconditional': perf_stats(de_unc, dax_ret),
        'Buy & Hold (equal weight)': perf_stats(de_ew, dax_ret),
    }).T
    de_tab.to_csv(os.path.join(OUT, 'de_performance_full.csv'))
    de_diff = sharpe_diff_test(de_rcf, de_unc)
    report['de_rcf_minus_unconditional'] = de_diff
    print('\nDE performance:\n', de_tab.to_string(), '\nDE RCF-Unc:', de_diff)
    de_rb = regime_breakdown({'rcf': de_rcf, 'unconditional': de_unc, 'buyhold_ew': de_ew}, dax_rv)
    de_rb.to_csv(os.path.join(OUT, 'de_regime_breakdown_full.csv'), index=False)
    print('\nDE regime breakdown:\n', de_rb.to_string())

    report['universe'] = {'us_n_stocks': int(us_ret.shape[1]), 'de_n_stocks': int(de_ret.shape[1])}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('\nDone. Outputs in outputs/paper/')


if __name__ == '__main__':
    main()
