"""
main_analysis.py
----------------
Master script that runs the full TBMD analysis pipeline and produces
all results used in the paper.

Usage:
    python main_analysis.py               # uses VIX-conditioned calibrated data
    python main_analysis.py --real-data   # uses yfinance (requires internet)

Outputs (written to ./outputs/):
    res_validation.csv      — RES classification performance
    bpc_components.csv      — BPC component statistics
    strategy_performance.csv — walk-forward Sharpe and risk metrics
    regime_stratified.csv   — performance broken down by VIX regime
    srsdt_halflives.csv     — signal half-life estimates
    srsdt_decay_empirical.csv — empirical decay rates from literature
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add code directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_real_data, load_calibrated_data
from behavioral_proxies import compute_bpc, compute_per_stock_signal
from efficiency_score import compute_res, validate_res
from backtest_engine import walk_forward_backtest
from srsdt_signal_decay import BPCSignalProfile, compute_empirical_decay_rates

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_analysis(use_real_data: bool = False):
    print("=" * 65)
    print("TBMD ANALYSIS: Operationalizing the Adaptive Markets Hypothesis")
    print("=" * 65)

    # ---- 1. Load Data ----
    print("\n[1/6] Loading data...")
    if use_real_data:
        prices, volumes, vix = load_real_data(
            start='2005-01-01', end='2024-01-01', n_stocks=30
        )
    else:
        prices, volumes, vix = load_calibrated_data(
            start='2005-01-01', end='2024-01-01', n_stocks=30, seed=42
        )

    # Daily simple returns
    returns = prices.pct_change().dropna(how='all')

    print(f"\nData summary:")
    print(f"  Period:   {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Stocks:   {returns.shape[1]}")
    print(f"  Days:     {returns.shape[0]}")
    print(f"  VIX mean: {vix.mean():.1f}, std: {vix.std():.1f}")

    # ---- 2. Compute BPC ----
    print("\n[2/6] Computing Behavioral Proxy Composite (BPC)...")
    bpc = compute_bpc(
        returns=returns,
        dollar_volumes=volumes,
        vix=vix,
        vr_window=60,
        herd_window=20,
        asym_window=20,
        illiq_window=20,
        sent_window=5
    )

    # BPC component correlations with VIX regime (high VIX = behavioral)
    common_bpc = bpc.index.intersection(vix.index)
    bpc_aligned = bpc.loc[common_bpc]
    vix_aligned = vix.loc[common_bpc]

    # Behavioral day flag (VIX >= 20)
    behavioral = (vix_aligned >= 20).astype(float)

    print("\nBPC component correlations with VIX behavioral indicator:")
    component_stats = {}
    for col in ['Z_herd', 'Z_vr', 'Z_asym', 'Z_sent', 'Z_illiq', 'BPC']:
        ser = bpc_aligned[col].dropna()
        beh_aligned = behavioral.reindex(ser.index)
        corr = ser.corr(beh_aligned)
        mean_w = ser[beh_aligned == 1].mean()
        mean_e = ser[beh_aligned == 0].mean()
        component_stats[col] = {
            'corr_with_behavioral' : round(corr, 3),
            'mean_in_behavioral_regime' : round(mean_w, 3),
            'mean_in_efficient_regime'  : round(mean_e, 3),
        }
        print(f"  {col:12s}: corr={corr:+.3f}, "
              f"behavioral_mean={mean_w:+.3f}, efficient_mean={mean_e:+.3f}")

    bpc_stats_df = pd.DataFrame(component_stats).T
    bpc_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'bpc_components.csv'))

    # ---- 3. Compute RES ----
    print("\n[3/6] Computing Real-Time Efficiency Score (RES)...")
    mkt_return = returns.mean(axis=1)
    res = compute_res(
        returns=mkt_return,
        hurst_window=252,
        vr_window=60,
        lb_window=60
    )

    # ---- 4. Validate RES ----
    print("\n[4/6] Validating RES against VIX regimes...")
    val = validate_res(res_df=res, vix=vix, tau_percentile=40.0)
    print(f"\nRES Validation Results:")
    for k, v in val.items():
        print(f"  {k:<25}: {v}")

    val_df = pd.DataFrame([val])
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'res_validation.csv'), index=False)

    # ---- 5. Walk-Forward Backtest ----
    print("\n[5/6] Computing per-stock ranking signal...")
    stock_signal = compute_per_stock_signal(
        returns=returns,
        dollar_volumes=volumes,
        mom_window=20,
        asym_window=20,
        illiq_window=20
    )

    print("\n[5/6] Running walk-forward backtest...")
    results = walk_forward_backtest(
        returns=returns,
        bpc=bpc,
        res=res,
        vix=vix,
        stock_signal=stock_signal,
        train_window=252,
        refit_freq=21,
        top_pct=0.20,
        tau_percentile=40.0
    )

    perf = results['performance']
    print("\nStrategy Performance (net of 10bps costs):")
    print(f"  Annual return :  {perf['ann_return_pct']:+.1f}%")
    print(f"  Annual vol    :  {perf['ann_vol_pct']:.1f}%")
    print(f"  Sharpe ratio  :  {perf['sharpe']:.3f}  95% CI: {perf['sharpe_ci_95']}")
    print(f"  PSR (vs 0)    :  {perf['psr']:.3f}")
    print(f"  Max drawdown  :  {perf['max_drawdown']:.1f}%")
    print(f"  Win rate      :  {perf['win_rate_active_days']:.1f}% of active days")
    print(f"  Observations  :  {perf['n_obs']}")

    print("\nRegime-stratified returns:")
    for reg, rp in results['regime_perf'].items():
        print(f"  {reg:<14}: {rp['n_days']:4d} days | "
              f"ann. return {rp['ann_return']:+5.1f}% | Sharpe {rp['sharpe']:+.3f}")

    # Save performance
    perf_df = pd.DataFrame([perf])
    perf_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_performance.csv'), index=False)

    reg_df = pd.DataFrame(results['regime_perf']).T
    reg_df.to_csv(os.path.join(OUTPUT_DIR, 'regime_stratified.csv'))

    # ---- 6. SRSDT Analysis ----
    print("\n[6/6] Computing SRSDT signal half-lives...")
    half_lives = BPCSignalProfile.all_half_lives(T0=36.0)
    print("\nSignal Half-Life Estimates (SRSDT Formula):")
    print(half_lives[['name', 'xi_obs', 'xi_stick', 'xi_coord',
                       'half_life_months', 'half_life_years']].to_string(index=False))
    half_lives.to_csv(os.path.join(OUTPUT_DIR, 'srsdt_halflives.csv'), index=False)

    empirical_decay = compute_empirical_decay_rates()
    print("\nEmpirical decay rates (from Hou et al. 2020 calibration):")
    summary = empirical_decay.groupby('BPC_proxy')['half_life_months_empirical'].agg(['mean', 'std'])
    print(summary.round(1).to_string())
    empirical_decay.to_csv(os.path.join(OUTPUT_DIR, 'srsdt_decay_empirical.csv'), index=False)

    print("\n" + "=" * 65)
    print("Analysis complete. Results saved to ./outputs/")
    print("=" * 65)

    return {
        'val'              : val,
        'component_stats'  : component_stats,
        'performance'      : perf,
        'regime_perf'      : results['regime_perf'],
        'half_lives'       : half_lives,
        'empirical_decay'  : empirical_decay,
        'pnl'              : results['pnl'],
    }


if __name__ == '__main__':
    use_real = '--real-data' in sys.argv
    results = run_analysis(use_real_data=use_real)
