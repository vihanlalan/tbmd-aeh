"""
================================================================================
TBMD Framework — Real Data Ingestion
================================================================================

Academic-grade data sources:
  1. Price Data  : Yahoo Finance via yfinance — daily adjusted close
  2. Market Factor: SPY ETF via yfinance
  3. Sentiment   : FRED UMCSENT (University of Michigan Consumer Sentiment)
  4. VIX         : CBOE via yfinance (^VIX) — visualisation only

SURVIVORSHIP BIAS WARNING:
  This script downloads current S&P 100 constituents. Companies removed
  from the index due to poor performance are NOT included. This biases all
  performance metrics upward by approximately 1-3% annualised.

Author: Vihan Lalan
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd

from .config import (
    START_DATE, END_DATE, SP100_TICKERS,
    PRICE_CACHE, SPY_CACHE, SENTIMENT_CACHE,
)

# ── Third-party data libraries ───────────────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pandas_datareader.data as web
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False


def fetch_price_data(
    tickers:    list  = SP100_TICKERS,
    start:      str   = START_DATE,
    end:        str   = END_DATE,
    cache_path: str   = PRICE_CACHE,
    spy_cache:  str   = SPY_CACHE,
) -> tuple:
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Returns log returns for individual tickers and SPY separately.
    SPY is used as the market factor to avoid survivorship bias in
    the equal-weighted average.

    Returns
    -------
    returns_df    : pd.DataFrame  — daily log returns
    market_factor : np.ndarray   — SPY daily log returns
    dates         : DatetimeIndex
    spy_prices    : pd.Series    — SPY price level for visualisation
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required. Run: pip install yfinance")

    # ── Load from cache if available ──────────────────────────────────────────
    if os.path.exists(cache_path) and os.path.exists(spy_cache):
        print("  Loading prices from cache...")
        prices_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        spy_df    = pd.read_csv(spy_cache,  index_col=0, parse_dates=True)
        print(f"  Cache loaded: {len(prices_df)} days, {len(prices_df.columns)} tickers")
    else:
        print(f"  Downloading {len(tickers)} tickers from Yahoo Finance ({start} to {end})...")
        print("  This may take 3-5 minutes on first run. Data is cached after.")

        batch_size = 20
        all_prices = {}

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            print(f"    Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: "
                  f"{batch[0]}...{batch[-1]}")
            try:
                raw = yf.download(
                    batch, start=start, end=end,
                    auto_adjust=True, progress=False, threads=True,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw['Close']
                else:
                    close = raw[['Close']]
                    close.columns = batch

                for ticker in batch:
                    if ticker in close.columns:
                        series = close[ticker].dropna()
                        if len(series) > 252:
                            all_prices[ticker] = series

                time.sleep(0.5)
            except Exception as e:
                print(f"    WARNING: Batch download failed: {e}")
                continue

        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.loc[start:end]

        print("  Downloading SPY (market factor)...")
        try:
            spy_raw = yf.download('SPY', start=start, end=end,
                                  auto_adjust=True, progress=False)
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_df = spy_raw['Close']
            else:
                spy_df = spy_raw[['Close']]
        except Exception as e:
            print(f"  WARNING: SPY download failed: {e}")
            spy_df = prices_df.mean(axis=1).to_frame(name='SPY')

        prices_df.to_csv(cache_path)
        spy_df.to_csv(spy_cache)
        print(f"  Cached to {cache_path} and {spy_cache}")

    # ── Data quality checks ───────────────────────────────────────────────────
    n_raw = len(prices_df.columns)
    missing_pct = prices_df.isnull().mean()
    prices_df = prices_df.loc[:, missing_pct < 0.10]
    prices_df = prices_df.ffill().bfill()

    n_kept = len(prices_df.columns)
    print(f"  Data quality: {n_kept}/{n_raw} tickers kept "
          f"(dropped {n_raw - n_kept} with >10% missing data)")

    # ── Align dates ───────────────────────────────────────────────────────────
    if isinstance(spy_df, pd.DataFrame):
        spy_series = spy_df.iloc[:, 0]
    else:
        spy_series = spy_df

    common_dates = prices_df.index.intersection(spy_series.index)
    prices_df    = prices_df.loc[common_dates]
    spy_series   = spy_series.loc[common_dates]

    # ── Compute log returns ───────────────────────────────────────────────────
    returns_df    = np.log(prices_df / prices_df.shift(1)).dropna()
    spy_returns   = np.log(spy_series / spy_series.shift(1)).dropna()

    common_dates  = returns_df.index.intersection(spy_returns.index)
    returns_df    = returns_df.loc[common_dates]
    spy_returns   = spy_returns.loc[common_dates]
    market_factor = spy_returns.values

    print(f"  Final dataset: {len(returns_df)} trading days × {len(returns_df.columns)} assets")
    print(f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"  Market factor: SPY ETF daily log returns")

    return returns_df, market_factor, returns_df.index, spy_series.loc[common_dates]


def fetch_fred_sentiment(
    start:      str = START_DATE,
    end:        str = END_DATE,
    cache_path: str = SENTIMENT_CACHE,
) -> pd.Series:
    """
    Download University of Michigan Consumer Sentiment Index from FRED.

    Source: Federal Reserve Bank of St. Louis (FRED)
    Series: UMCSENT

    LOOKAHEAD PROTECTION: The series is lagged by 1 month before use.

    Returns
    -------
    pd.Series — daily sentiment z-scores, or None if unavailable
    """
    if os.path.exists(cache_path):
        print("  Loading sentiment from cache...")
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return cached.iloc[:, 0]

    if not DATAREADER_AVAILABLE:
        print("  WARNING: pandas_datareader unavailable. Using AR(1) sentiment proxy.")
        return None

    print("  Downloading UMCSENT from FRED...")
    try:
        umcsent = web.DataReader(
            'UMCSENT', 'fred',
            start=pd.Timestamp(start) - pd.DateOffset(months=2),
            end=end,
        )
        umcsent = umcsent['UMCSENT'].dropna()

        # Lag by 1 month to prevent lookahead bias
        umcsent_lagged = umcsent.shift(1).dropna()

        # Resample to daily business frequency via forward-fill
        daily_idx = pd.date_range(start=start, end=end, freq='B')
        sentiment_daily = umcsent_lagged.reindex(daily_idx).ffill()

        # Normalise to z-score using expanding window (no lookahead)
        expanding_mean = sentiment_daily.expanding(min_periods=12).mean()
        expanding_std  = sentiment_daily.expanding(min_periods=12).std().clip(lower=1e-8)
        sentiment_z    = (sentiment_daily - expanding_mean) / expanding_std

        sentiment_z.to_csv(cache_path, header=['UMCSENT_z'])
        print(f"  FRED sentiment downloaded: {len(sentiment_z)} days")
        return sentiment_z

    except Exception as e:
        print(f"  WARNING: FRED download failed ({e}). Using AR(1) sentiment proxy.")
        return None


def fetch_vix(
    start: str = START_DATE,
    end:   str = END_DATE,
) -> pd.Series:
    """
    Download CBOE VIX from Yahoo Finance.

    Used ONLY for visualisation. NOT used in any performance calculation.

    Returns
    -------
    pd.Series — daily VIX levels, or None if download fails
    """
    try:
        vix = yf.download('^VIX', start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close'].iloc[:, 0]
        else:
            vix = vix['Close']
        print(f"  VIX downloaded: {len(vix)} days")
        return vix
    except Exception as e:
        print(f"  WARNING: VIX download failed ({e}).")
        return None


def compute_vol_market_state(
    market_factor: np.ndarray,
    dates:         pd.DatetimeIndex,
    vix:           pd.Series = None,
) -> pd.Series:
    """
    Compute a volatility-based market state indicator for VISUALISATION ONLY.

    This is NOT a regime classifier. It is NOT used in any performance
    calculation.

    Three states from SPY realised volatility terciles:
      0 = Low vol  (calm)
      1 = Medium vol
      2 = High vol (stressed)

    Parameters
    ----------
    market_factor : np.ndarray   — SPY log returns
    dates         : DatetimeIndex
    vix           : pd.Series    — CBOE VIX (optional, preferred if available)

    Returns
    -------
    pd.Series — integer {0, 1, 2} indexed by date
    """
    mkt = pd.Series(market_factor, index=dates)

    if vix is not None:
        vol_measure = vix.reindex(dates).ffill().bfill()
    else:
        vol_measure = mkt.rolling(20).std() * np.sqrt(252)

    # Classify into terciles using expanding-window percentiles (no lookahead)
    state = pd.Series(1, index=dates)

    for t in range(60, len(dates)):
        historical_vol = vol_measure.iloc[:t]
        current_vol    = vol_measure.iloc[t]
        p33 = historical_vol.quantile(0.33)
        p67 = historical_vol.quantile(0.67)

        if current_vol < p33:
            state.iloc[t] = 0   # calm
        elif current_vol > p67:
            state.iloc[t] = 2   # stressed
        else:
            state.iloc[t] = 1   # medium

    return state
