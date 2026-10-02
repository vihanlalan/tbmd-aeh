
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from behavioral_proxies import compute_bpc, compute_per_stock_signal
from efficiency_score import compute_res
from backtest_engine import walk_forward_backtest

def load_custom_data(tickers, start='2015-01-01', end='2024-01-01'):
    print(f"Downloading custom dataset: {len(tickers)} tickers from {start} to {end}...")

    # Download everything in one call
    all_tickers = tickers + ['^VIX']
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)

    prices = raw['Close'][tickers].dropna(how='all')
    volumes_raw = raw['Volume'][tickers]

    # Dollar volume = price * share volume
    volumes = (prices * volumes_raw).dropna(how='all')
    vix = raw['Close']['^VIX'].dropna()
    vix.name = 'VIX'

    # Align all to common date index
    common_idx = prices.index.intersection(vix.index)
    prices = prices.loc[common_idx].dropna(axis=1, thresh=int(len(common_idx) * 0.8))
    volumes = volumes.loc[common_idx, prices.columns]
    vix = vix.loc[common_idx]

    # Compute log returns
    returns_df = np.log(prices / prices.shift(1)).dropna()
    vix_aligned = vix.loc[returns_df.index]

    print(f"Clean data: {len(returns_df)} days, {len(returns_df.columns)} stocks")
    return returns_df, volumes.loc[returns_df.index], vix_aligned

def run_test():
    # Using a set of Nasdaq-100 tech stocks (known for high behavioral signals)
    nasdaq_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'PEP', 'COST',
        'ADBE', 'AMD', 'NFLX', 'CMCSA', 'TMUS', 'INTC', 'AMGN', 'QCOM', 'ISRG', 'AMAT',
        'TXN', 'BKNG', 'GILD', 'VRTX', 'ADP', 'REGN', 'PANW', 'ADI', 'MU', 'LRCX'
    ]

    # 1. Load Data
    returns, volumes, vix = load_custom_data(nasdaq_tickers)

    # 2. Compute BPC
    print("\nComputing Behavioral Proxy Composite (BPC)...")
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

    # 3. Compute RES
    print("\nComputing Real-Time Efficiency Score (RES)...")
    mkt_return = returns.mean(axis=1)
    res = compute_res(returns=mkt_return)

    # 4. Compute Stock-level signal for ranking
    print("\nComputing per-stock ranking signal...")
    stock_signal = compute_per_stock_signal(
        returns=returns,
        dollar_volumes=volumes,
        mom_window=20,
        asym_window=20,
        illiq_window=20
    )

    # 5. Run Backtest
    print("\nRunning Walk-Forward Backtest...")
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
    print("\n" + "="*40)
    print(" RESULTS: NASDAQ-100 TECH SUBSET")
    print("="*40)
    print(f"Annual Return : {perf['ann_return_pct']:.2f}%")
    print(f"Annual Vol    : {perf['ann_vol_pct']:.2f}%")
    print(f"Sharpe Ratio  : {perf['sharpe']:.3f}")
    print(f"Max Drawdown  : {perf['max_drawdown']:.2f}%")
    print(f"Win Rate      : {perf['win_rate_active_days']:.1f}%")
    print(f"Days Active   : {perf['pct_days_active']:.1f}%")
    print("="*40)

if __name__ == '__main__':
    run_test()
