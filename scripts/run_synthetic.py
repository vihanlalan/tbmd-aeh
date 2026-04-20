"""
TBMD Framework — Synthetic Validation Pipeline
Run the full empirical pipeline on Hamilton (1989) regime-switching simulation.

Usage:
    python scripts/run_synthetic.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from src.config import SEED, WF_CONFIG
from src.simulation import simulate_regime_switching_market
from src.bpc import build_behavioral_proxy_composite
from src.res import compute_real_time_efficiency_score
from src.validation import walk_forward_validation
from src.statistics import (
    deflated_sharpe_ratio,
    variance_ratio_test,
    compute_performance_metrics,
    validate_regime_detection,
)
from src.visualization import plot_framework_results, plot_multiple_testing_analysis

np.random.seed(SEED)


def main() -> None:
    DIVIDER = "=" * 64

    print(DIVIDER)
    print("TBMD Framework — Full Empirical Pipeline")
    print("Operationalizing the Adaptive Markets Hypothesis")
    print(DIVIDER)

    # ── Step 1: Simulate market data ─────────────────────────────────────────
    print("\n[1/7]  Simulating regime-switching market data...")
    returns_df, prices_df, regime_series, regimes, market_factor = \
        simulate_regime_switching_market()

    print(f"       {len(returns_df)} days × {len(returns_df.columns)} assets generated")
    print("       Regime distribution:")
    for rid, cnt in regime_series.value_counts().sort_index().items():
        pct = cnt / len(regime_series) * 100
        print(f"         Regime {rid} ({regimes[rid]['name']}): {cnt} days ({pct:.1f}%)")

    # ── Step 2: Build BPC ─────────────────────────────────────────────────────
    print("\n[2/7]  Building Behavioral Proxy Composite (BPC)...")
    bpc_features = build_behavioral_proxy_composite(returns_df, market_factor)
    print(f"       Proxies: {list(bpc_features.columns)}")

    # ── Step 3: Compute RES ───────────────────────────────────────────────────
    print("\n[3/7]  Computing Real-Time Efficiency Score (RES)...")
    res_scores = compute_real_time_efficiency_score(market_factor, returns_df.index)
    mean_hurst = res_scores['hurst'].mean()
    mean_res   = res_scores['efficiency_score'].mean()
    print(f"       Mean Hurst exponent:    {mean_hurst:.4f}")
    print(f"       Mean Efficiency Score:  {mean_res:.4f}")

    eff_corr = res_scores['efficiency_score'].corr(
        regime_series.map({0: 1.0, 1: 0.5, 2: 0.0}).reindex(res_scores.index)
    )
    print(f"       RES vs True Regime corr: {eff_corr:.4f}  (higher = better detection)")

    # ── Step 4: Walk-forward validation ──────────────────────────────────────
    print("\n[4/7]  Running walk-forward validation...")
    all_features = pd.concat([
        bpc_features,
        res_scores[['hurst', 'variance_ratio', 'lb_pval',
                    'efficiency_score', 'inefficiency_score']],
    ], axis=1).dropna()

    wf_results = walk_forward_validation(all_features, returns_df, regime_series)
    print(f"       Walk-forward complete. {len(wf_results['fold_metrics'])} folds.")

    # ── Step 5: Performance statistics ───────────────────────────────────────
    print("\n[5/7]  Performance Statistics (Walk-Forward, Net of Costs):")
    print("       " + "-" * 56)

    strategies = {
        'RCF Strategy':       wf_results['rcf_pnl'],
        'Unconditional BPC':  wf_results['unconditional_pnl'],
        'Momentum Baseline':  wf_results['momentum_pnl'],
        'Mean Reversion':     wf_results['mean_rev_pnl'],
        'Buy & Hold':         wf_results['buyhold_pnl'],
    }

    for sname, pnl in strategies.items():
        active = pnl[pnl != 0]
        m      = compute_performance_metrics(active, sname)
        if not m:
            continue
        sr, dsr = deflated_sharpe_ratio(active, n_trials=12)
        sig     = '*** SIGNIFICANT' if m['P_value'] < 0.05 else '(not significant)'
        print(f"\n       {sname}:")
        print(f"         Ann Return:  {m['Ann_Return_%']:>7.2f}%")
        print(f"         Ann Vol:     {m['Ann_Vol_%']:>7.2f}%")
        print(f"         Sharpe:      {m['Sharpe']:>7.3f}   ← raw; treat with caution")
        print(f"         DSR:         {dsr:>7.3f}   ← primary metric (Bailey & LdP 2014)")
        print(f"         Max DD:      {m['Max_Drawdown_%']:>7.2f}%")
        print(f"         Win Rate:    {m['Win_Rate_%']:>7.1f}%")
        print(f"         P-value:     {m['P_value']:>8.4f}  {sig}")

    # ── Step 6: Statistical tests ─────────────────────────────────────────────
    print("\n[6/7]  Variance Ratio Tests on RCF strategy PnL:")
    rcf_active = wf_results['rcf_pnl'][wf_results['rcf_pnl'] != 0]
    vr_results = variance_ratio_test(rcf_active)
    for lag_key, res in vr_results.items():
        stars = '***' if res['p_val'] < 0.01 else '**' if res['p_val'] < 0.05 else 'ns'
        print(f"       {lag_key}: VR={res['vr']:.4f}, Z={res['z']:.3f}, "
              f"p={res['p_val']:.4f} {stars}")

    print("\n       Regime Detection Validation:")
    regime_val = validate_regime_detection(
        wf_results['regime_true'],
        wf_results['regime_predictions'],
    )
    if regime_val:
        print(f"         Classification Accuracy: {regime_val['accuracy']:.3f}")
        print(f"         AUC:                     {regime_val['auc']:.3f}")
        print(f"         Precision:               {regime_val['precision']:.3f}")
        print(f"         Recall:                  {regime_val['recall']:.3f}")

    # ── Step 7: Figures ───────────────────────────────────────────────────────
    print("\n[7/7]  Generating figures...")
    plot_framework_results(
        returns_df, regime_series, regimes,
        bpc_features, res_scores, wf_results,
    )
    plot_multiple_testing_analysis()

    print("\n" + DIVIDER)
    print("Pipeline complete.")
    print(DIVIDER)


if __name__ == '__main__':
    main()
