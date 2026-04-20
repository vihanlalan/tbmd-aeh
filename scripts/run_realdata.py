"""
TBMD Framework — Real Market Data Validation Pipeline
Validates the paper's directional claims on S&P 100 data (2005–2024).

Usage:
    python scripts/run_realdata.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import SEED, SYNTHETIC_RESULTS, START_DATE, END_DATE
from src.data_ingestion import (
    fetch_price_data, fetch_fred_sentiment, fetch_vix,
    compute_vol_market_state,
)
from src.bpc import build_bpc_real
from src.res import compute_res
from src.validation import walk_forward_validation_real
from src.statistics import (
    deflated_sharpe_ratio,
    variance_ratio_test,
    compute_performance_metrics,
    validate_directional_claims,
)
from src.visualization import plot_real_data_results, plot_synthetic_vs_real

np.random.seed(SEED)


def main():
    DIVIDER = '=' * 68

    print(DIVIDER)
    print('TBMD Framework — Real Market Data Validation (Option B)')
    print('Operationalizing the Adaptive Markets Hypothesis')
    print(DIVIDER)
    print()
    print('  ⚠  SURVIVORSHIP BIAS WARNING')
    print('     Using current S&P 100 constituents. Historical members')
    print('     removed for poor performance are excluded.')
    print('     Performance metrics are biased upward by ~1-3% annualised.')
    print('     Directional claims (RCF vs Unconditional) are unaffected.')
    print()

    # ── Step 1: Price data ────────────────────────────────────────────────────
    print(f'[1/8]  Fetching price data ({START_DATE} to {END_DATE})...')
    returns_df, market_factor, dates, spy_prices = fetch_price_data()

    # ── Step 2: VIX ───────────────────────────────────────────────────────────
    print('\n[2/8]  Fetching VIX (CBOE via Yahoo Finance)...')
    vix = fetch_vix()

    # ── Step 3: Sentiment ─────────────────────────────────────────────────────
    print('\n[3/8]  Fetching UMCSENT sentiment (FRED)...')
    sentiment_z = fetch_fred_sentiment()
    sentiment_src = 'FRED UMCSENT' if sentiment_z is not None else 'AR(1) proxy (FRED unavailable)'
    print(f'       Sentiment source: {sentiment_src}')

    # ── Step 4: Volatility state (visualisation only) ─────────────────────────
    print('\n[4/8]  Computing volatility state (visualisation only)...')
    vol_state = compute_vol_market_state(market_factor, dates, vix)
    dist      = vol_state.value_counts().sort_index()
    for sid, cnt in dist.items():
        print(f'       State {sid} ({["Calm","Medium","Stress"][sid]}): '
              f'{cnt} days ({cnt/len(vol_state)*100:.1f}%)')

    # ── Step 5: BPC ───────────────────────────────────────────────────────────
    print('\n[5/8]  Building Behavioral Proxy Composite (BPC)...')
    bpc_features = build_bpc_real(returns_df, market_factor, sentiment_z)
    print(f'       Proxies built: {list(bpc_features.columns)}')

    # ── Step 6: RES ───────────────────────────────────────────────────────────
    print('\n[6/8]  Computing Real-Time Efficiency Score (RES)...')
    res_scores = compute_res(market_factor, dates)
    mean_hurst = res_scores['hurst'].mean()
    mean_res   = res_scores['efficiency_score'].mean()
    print(f'       Mean Hurst exponent:   {mean_hurst:.4f}')
    print(f'       Mean Efficiency Score: {mean_res:.4f}')
    if vix is not None:
        try:
            common = res_scores.index.intersection(vix.index)
            corr, pv = stats.pearsonr(
                res_scores['efficiency_score'].loc[common].ffill().dropna(),
                vix.loc[common].ffill().reindex(
                    res_scores['efficiency_score'].loc[common].ffill().dropna().index
                )
            )
            print(f'       r(RES, VIX) = {corr:.3f} (p={pv:.4f}) — expected negative')
        except Exception:
            pass

    # ── Step 7: Walk-forward validation ───────────────────────────────────────
    print('\n[7/8]  Running walk-forward validation...')
    print('       (This will take 15-25 minutes for a 20-year dataset)')
    all_features = pd.concat([
        bpc_features,
        res_scores[['hurst', 'variance_ratio', 'lb_pval',
                    'efficiency_score', 'inefficiency_score']],
    ], axis=1).dropna()

    wf_results = walk_forward_validation_real(all_features, returns_df, vol_state)

    # ── Step 8: Results ───────────────────────────────────────────────────────
    print('\n[8/8]  Performance Statistics — Real Data')
    print('       ' + '-' * 58)

    strategies = {
        'RCF Strategy':      wf_results['rcf_pnl'],
        'Unconditional BPC': wf_results['unconditional_pnl'],
        'Momentum Baseline': wf_results['momentum_pnl'],
        'Mean Reversion':    wf_results['mean_rev_pnl'],
        'Buy & Hold':        wf_results['buyhold_pnl'],
    }

    real_metrics = {}
    for sn, pnl in strategies.items():
        active = pnl[pnl != 0]
        m      = compute_performance_metrics(active, sn)
        if not m:
            continue
        sr, dsr = deflated_sharpe_ratio(active, n_trials=12)
        real_metrics[sn] = {**m, 'Sharpe': sr or m['Sharpe'], 'DSR': dsr}
        syn = SYNTHETIC_RESULTS.get(sn, {})
        print(f'\n       {sn}:')
        print(f'         Ann Return:  {m["Ann_Return_%"]:>7.2f}%'
              f'   [Synthetic: {syn.get("Ann_Return_%", "N/A")}%]')
        print(f'         Sharpe:      {sr:>7.3f}'
              f'      [Synthetic: {syn.get("Sharpe", "N/A")}]')
        print(f'         DSR:         {dsr:>7.3f}'
              f'      [Synthetic: {syn.get("DSR", "N/A")}]')
        print(f'         P-value:     {m["P_value"]:>8.4f}'
              f'  {"*** SIGNIFICANT" if m["P_value"] < 0.05 else "(not significant)"}')

    # ── Directional claims ────────────────────────────────────────────────────
    print('\n       DIRECTIONAL CLAIMS VALIDATION')
    print('       ' + '-' * 58)
    claim_results = validate_directional_claims(
        wf_results, res_scores, bpc_features, vix
    )
    n_pass = 0
    for claim_name, cr in claim_results.items():
        status = cr['pass_fail']
        if status == 'PASS':
            n_pass += 1
        marker = '✓' if status == 'PASS' else ('✗' if status == 'FAIL' else '—')
        print(f'       {marker} [{status}] {claim_name}')
        print(f'               {cr["evidence"]}')

    print(f'\n       OVERALL: {n_pass}/{len(claim_results)} directional claims PASSED')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for sn, m in real_metrics.items():
        syn = SYNTHETIC_RESULTS.get(sn, {})
        rows.append({
            'Strategy':          sn,
            'Real_Ann_Return_%': m.get('Ann_Return_%'),
            'Real_Sharpe':       m.get('Sharpe'),
            'Real_DSR':          m.get('DSR'),
            'Real_MaxDD_%':      m.get('Max_Drawdown_%'),
            'Real_WinRate_%':    m.get('Win_Rate_%'),
            'Real_Pvalue':       m.get('P_value'),
            'Synthetic_Sharpe':  syn.get('Sharpe'),
            'Synthetic_DSR':     syn.get('DSR'),
            'Direction_Match':   (np.sign(m.get('Sharpe', 0)) ==
                                  np.sign(syn.get('Sharpe', 0))),
        })
    pd.DataFrame(rows).to_csv('tbmd_realdata_results.csv', index=False)
    print(f'\n       Results saved to tbmd_realdata_results.csv')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\n       Generating figures...')
    plot_real_data_results(
        spy_prices, market_factor, dates, vol_state,
        bpc_features, res_scores, wf_results, vix, sentiment_src,
    )
    plot_synthetic_vs_real(real_metrics)

    print('\n' + DIVIDER)
    print('Real-data validation complete.')
    print(DIVIDER)


if __name__ == '__main__':
    main()
