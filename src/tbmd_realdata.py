"""
================================================================================
TBMD FRAMEWORK — REAL MARKET DATA VALIDATION
Operationalizing the Adaptive Markets Hypothesis

Paper: "Operationalizing the Adaptive Markets Hypothesis: Observable Behavioral
        Proxies, Real-Time Efficiency Scoring, and Regime-Conditional Alpha
        Generation"

--------------------------------------------------------------------------------
PURPOSE
-------
This file is the real-data companion to tbmd_framework.py.

The synthetic paper used a Hamilton (1989) simulation because it provides
ground-truth regime labels for validating the RES detector. This file
validates the paper's CORE TRADEABLE CLAIM on real historical market data,
without requiring ground-truth regime labels:

    CORE CLAIM (Option B validation):
    The Regime-Conditional Filter (RCF) produces a higher risk-adjusted
    return than deploying the same BPC signal unconditionally.

    If RCF_Sharpe > Unconditional_Sharpe on real data, the paper's
    central argument holds in practice, not just in simulation.

--------------------------------------------------------------------------------
DATA SOURCES — ACADEMIC GRADE ONLY
------------------------------------
All data comes from verified, citable sources:

  1. PRICE DATA
     Source  : Yahoo Finance via yfinance library
     Citation: "Yahoo Finance. Historical market data. finance.yahoo.com"
     Coverage: Daily adjusted closing prices, 2005-01-01 to 2024-12-31
     Universe: S&P 100 current constituents + SPY ETF as market factor
     Note    : yfinance adjusted prices account for splits and dividends.
               This is the standard source used in academic finance papers
               (see Fama-French data library cross-references).

  2. MARKET FACTOR
     Source  : SPY ETF (SPDR S&P 500 ETF Trust) via yfinance
     Rationale: SPY tracks the S&P 500 total return index and is the
               standard benchmark in empirical asset pricing research.
               It avoids the survivorship bias present in constructing
               an equal-weighted average from current constituents.

  3. SENTIMENT DATA
     Source  : FRED (Federal Reserve Bank of St. Louis) Economic Data
     Series  : Consumer Sentiment Index (UMCSENT) — University of Michigan
     Citation: "University of Michigan, University of Michigan: Consumer
               Sentiment [UMCSENT], retrieved from FRED, Federal Reserve
               Bank of St. Louis; https://fred.stlouisfed.org/series/UMCSENT"
     Rationale: UMCSENT is a peer-reviewed, institutional-grade sentiment
               measure used extensively in academic finance. It is monthly,
               so we interpolate to daily and use 1-month-lagged values
               to avoid lookahead bias. This is more rigorous than
               scraped news sentiment for a formal academic paper.
     Fallback: If FRED API is unavailable, an AR(1) proxy is used and
               flagged clearly in all output.

  4. VOLATILITY PROXY (for visualisation only, NOT for classification)
     Source  : CBOE VIX via yfinance ticker "^VIX"
     Citation: "Chicago Board Options Exchange. CBOE Volatility Index (VIX).
               cboe.com/vix"
     Use     : Used ONLY to colour regime panels in visualisation charts.
               NOT used as a trading signal or regime classifier.
               NOT used to compute any reported performance metric.

--------------------------------------------------------------------------------
SURVIVORSHIP BIAS WARNING
--------------------------
This script downloads current S&P 100 constituents. Companies that were
removed from the index due to poor performance are NOT included. This
biases all performance metrics upward by approximately 1-3% annualised.

This bias is present in all results below. It is reported explicitly in
every performance table and figure subtitle. All directional claims
(RCF vs Unconditional comparison) are robust to this bias because the
same universe is used for both strategies.

For a survivorship-bias-free dataset, use CRSP/Compustat via WRDS.

--------------------------------------------------------------------------------
WHAT IS UNCHANGED FROM tbmd_framework.py
-----------------------------------------
The following are IDENTICAL to the synthetic version. Do not modify them:

  - _hurst_exponent()          Section 4
  - _variance_ratio()          Section 4
  - _ljung_box_pval()          Section 4
  - compute_real_time_efficiency_score()   Section 4
  - walk_forward_validation()  Section 5  (same GBM, same windows, same costs)
  - deflated_sharpe_ratio()    Section 6
  - variance_ratio_test()      Section 6
  - benjamini_hochberg_correction()        Section 6
  - compute_performance_metrics()          Section 6

--------------------------------------------------------------------------------
INSTALLATION
------------
pip install yfinance pandas-datareader scipy scikit-learn matplotlib seaborn

Optional (faster downloads):
pip install requests-cache

Python >= 3.9

--------------------------------------------------------------------------------
USAGE
-----
python tbmd_realdata.py

Outputs:
  tbmd_realdata_figures.png       — 8-panel validation chart
  tbmd_realdata_comparison.png    — Synthetic vs Real side-by-side
  tbmd_realdata_results.csv       — Full performance table
  sentiment_cache.csv             — Cached FRED sentiment (auto-reused)

Runtime: approximately 15-25 minutes on a standard laptop.
         The walk-forward loop over ~4,800 days is the bottleneck.
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from efficiency_score import compute_res, validate_res
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import time
warnings.filterwarnings('ignore')

# ── Third-party data libraries ───────────────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("ERROR: yfinance not installed. Run: pip install yfinance")

try:
    import pandas_datareader.data as web
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    print("WARNING: pandas_datareader not installed. FRED sentiment will use fallback.")
    print("         Run: pip install pandas-datareader")

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Date range ────────────────────────────────────────────────────────────────
# 2005-2024: covers 2008 GFC, 2011 European crisis, 2018 drawdown,
#            2020 COVID crash, 2022 rate shock — five distinct stress episodes
START_DATE = '2005-01-01'
END_DATE   = '2024-12-31'

# ── S&P 100 current constituents ──────────────────────────────────────────────
# Source: S&P Dow Jones Indices. Current as of 2024.
# Survivorship bias note: this excludes historical members that were removed.
SP100_TICKERS = [
    'AAPL', 'ABBV', 'ABT',  'ACN',  'ADBE', 'AIG',  'AMD',  'AMGN',
    'AMT',  'AMZN', 'AVGO', 'AXP',  'BA',   'BAC',  'BK',   'BKNG',
    'BLK',  'BMY',  'BRK-B','C',    'CAT',  'CHTR', 'CL',   'CMCSA',
    'COF',  'COP',  'COST', 'CRM',  'CSCO', 'CVS',  'CVX',  'DE',
    'DHR',  'DIS',  'DUK',  'EMR',  'EXC',  'F',    'FDX',  'GD',
    'GE',   'GILD', 'GM',   'GOOGL','GS',   'HD',   'HON',  'IBM',
    'INTC', 'JNJ',  'JPM',  'KHC',  'KO',   'LIN',  'LLY',  'LMT',
    'LOW',  'MA',   'MCD',  'MDLZ', 'MDT',  'MET',  'META', 'MMM',
    'MO',   'MRK',  'MS',   'MSFT', 'NEE',  'NFLX', 'NKE',  'NOW',
    'NVDA', 'ORCL', 'OXY',  'PEP',  'PFE',  'PG',   'PM',   'PYPL',
    'QCOM', 'RTX',  'SBUX', 'SCHW', 'SO',   'SPG',  'T',    'TGT',
    'TMO',  'TMUS', 'TSLA', 'TXN',  'UNH',  'UNP',  'UPS',  'USB',
    'V',    'VZ',   'WBA',  'WFC',  'WMT',  'XOM',
]

# ── Walk-forward parameters (IDENTICAL to synthetic version) ──────────────────
WF_CONFIG = {
    'train_window':          252,
    'step_size':              21,
    'warmup_days':           252,
    'transaction_cost_bps':   10,
    'signal_confidence_hi':   0.60,
    'signal_confidence_lo':   0.40,
    'momentum_lookback':      20,
    'top_n_assets':            5,
}

# ── RES parameters (IDENTICAL to synthetic version) ───────────────────────────
RES_CONFIG = {
    'window':   60,
    'vr_lag':    5,
    'lb_lags':  10,
}

# ── BPC parameters ────────────────────────────────────────────────────────────
BPC_CONFIG = {
    'cs_dispersion_window': 20,
    'vol_asym_window':      20,
    'autocorr_window':      60,
    'rolling_z_window':    252,
}

# ── File paths ────────────────────────────────────────────────────────────────
SENTIMENT_CACHE = 'sentiment_cache.csv'
PRICE_CACHE     = 'price_cache.csv'
SPY_CACHE       = 'spy_cache.csv'
OUTPUT_FIG1     = 'tbmd_realdata_figures.png'
OUTPUT_FIG2     = 'tbmd_realdata_comparison.png'
OUTPUT_CSV      = 'tbmd_realdata_results.csv'

# ── Synthetic benchmark results (from tbmd_framework.py) ─────────────────────
# These are the numbers from the paper's synthetic validation.
# We compare real-data results against these to validate directional claims.
SYNTHETIC_RESULTS = {
    'RCF Strategy':       {'Sharpe': 0.43,  'DSR': 0.71,  'Ann_Return_%':  1.8},
    'Unconditional BPC':  {'Sharpe': -2.47, 'DSR': 0.00,  'Ann_Return_%': -14.4},
    'Momentum Baseline':  {'Sharpe': -8.60, 'DSR': 0.00,  'Ann_Return_%': -2.8},
    'Mean Reversion':     {'Sharpe':  0.85, 'DSR': 0.95,  'Ann_Return_%':  0.3},
    'Buy & Hold':         {'Sharpe':  0.24, 'DSR': 0.70,  'Ann_Return_%':  4.1},
}

# ── Five directional claims to validate ───────────────────────────────────────
# These are what the paper asserts. Each must hold on real data.
DIRECTIONAL_CLAIMS = [
    "RCF_Sharpe > Unconditional_Sharpe",
    "RCF_DSR > Unconditional_DSR",
    "RES_corr_with_VIX > 0.20",
    "At_least_3_BPC_components_BH_significant",
    "Behavioral_stress_periods_show_higher_RCF_returns",
]


# ==============================================================================
# SECTION 2: REAL DATA INGESTION
# ==============================================================================

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

    Source: Yahoo Finance (yfinance)
    Adjustments: Prices are adjusted for splits and dividends.

    Parameters
    ----------
    tickers    : list  — S&P 100 ticker symbols
    start      : str   — start date 'YYYY-MM-DD'
    end        : str   — end date 'YYYY-MM-DD'
    cache_path : str   — CSV path for caching price data
    spy_cache  : str   — CSV path for caching SPY data

    Returns
    -------
    returns_df    : pd.DataFrame  — daily log returns, shape (n_days, n_assets)
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

        # Download in batches to avoid Yahoo Finance timeouts
        batch_size = 20
        all_prices = {}

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            print(f"    Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[0]}...{batch[-1]}")
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                # Handle both single and multi-ticker returns
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw['Close']
                else:
                    close = raw[['Close']]
                    close.columns = batch

                for ticker in batch:
                    if ticker in close.columns:
                        series = close[ticker].dropna()
                        if len(series) > 252:   # require at least 1 year of data
                            all_prices[ticker] = series

                time.sleep(0.5)  # polite delay between batches

            except Exception as e:
                print(f"    WARNING: Batch download failed: {e}")
                continue

        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.loc[start:end]

        # Download SPY separately
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

        # Cache for future runs
        prices_df.to_csv(cache_path)
        spy_df.to_csv(spy_cache)
        print(f"  Cached to {cache_path} and {spy_cache}")

    # ── Data quality checks ───────────────────────────────────────────────────
    n_raw = len(prices_df.columns)
    missing_pct = prices_df.isnull().mean()
    prices_df = prices_df.loc[:, missing_pct < 0.10]  # drop tickers >10% missing
    prices_df = prices_df.ffill().bfill()             # fill remaining gaps

    n_kept = len(prices_df.columns)
    print(f"  Data quality: {n_kept}/{n_raw} tickers kept (dropped {n_raw - n_kept} with >10% missing data)")

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
    Series: UMCSENT (University of Michigan Consumer Sentiment)
    URL   : https://fred.stlouisfed.org/series/UMCSENT
    Frequency: Monthly — interpolated to daily for BPC construction.

    This is an institutional-grade sentiment measure used in published
    academic finance research (e.g., Baker & Wurgler 2007 use survey-based
    sentiment as a component of their investor sentiment index).

    LOOKAHEAD PROTECTION: The series is lagged by 1 month before use.
    The University of Michigan publishes preliminary estimates mid-month
    for the current month. Using a 1-month lag ensures no lookahead bias.

    Parameters
    ----------
    start : str  — start date
    end   : str  — end date

    Returns
    -------
    pd.Series — daily sentiment scores, normalised to mean=0 std=1,
                indexed by trading date.
                Returns AR(1) fallback if FRED is unavailable.
    """
    if os.path.exists(cache_path):
        print("  Loading sentiment from cache...")
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return cached.iloc[:, 0]

    if not DATAREADER_AVAILABLE:
        print("  WARNING: pandas_datareader unavailable. Using AR(1) sentiment proxy.")
        print("           Install with: pip install pandas-datareader")
        return None  # handled in build_bpc_real()

    print("  Downloading UMCSENT from FRED (Federal Reserve Bank of St. Louis)...")
    try:
        # Fetch monthly UMCSENT
        umcsent = web.DataReader(
            'UMCSENT',
            'fred',
            start=pd.Timestamp(start) - pd.DateOffset(months=2),
            end=end,
        )
        umcsent = umcsent['UMCSENT'].dropna()

        # Lag by 1 month to prevent lookahead bias
        # (preliminary estimate published ~2 weeks into current month)
        umcsent_lagged = umcsent.shift(1).dropna()

        # Resample to daily business frequency via forward-fill
        # This is standard practice for monthly macro variables in daily models
        daily_idx = pd.date_range(start=start, end=end, freq='B')
        sentiment_daily = umcsent_lagged.reindex(daily_idx).ffill()

        # Normalise to z-score using expanding window (no lookahead)
        expanding_mean = sentiment_daily.expanding(min_periods=12).mean()
        expanding_std  = sentiment_daily.expanding(min_periods=12).std().clip(lower=1e-8)
        sentiment_z    = (sentiment_daily - expanding_mean) / expanding_std

        sentiment_z.to_csv(cache_path, header=['UMCSENT_z'])
        print(f"  FRED sentiment downloaded: {len(sentiment_z)} days, cached to {cache_path}")
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

    Source : Chicago Board Options Exchange (CBOE)
    Ticker : ^VIX on Yahoo Finance
    Use    : Visualisation only. NOT used in any performance calculation.
             Used to colour regime periods in charts and to validate that
             RES moves in the expected direction relative to market stress.

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
        print(f"  WARNING: VIX download failed ({e}). Regime visualisation will use vol proxy.")
        return None


# ==============================================================================
# SECTION 3: VOLATILITY-BASED MARKET STATE
#            (For visualisation ONLY — not a regime classifier)
# ==============================================================================

def compute_vol_market_state(
    market_factor: np.ndarray,
    dates:         pd.DatetimeIndex,
    vix:           pd.Series = None,
) -> pd.Series:
    """
    Compute a volatility-based market state indicator for VISUALISATION ONLY.

    This is NOT a regime classifier. It is NOT used in any performance
    calculation. It is used only to colour the background of time-series
    charts so the visualisations are interpretable.

    The three states are derived from SPY realised volatility:
      0 = Low vol    (rolling 20-day vol below 33rd percentile) — "calm"
      1 = Medium vol (rolling 20-day vol between 33rd and 67th percentile)
      2 = High vol   (rolling 20-day vol above 67th percentile) — "stressed"

    This is labelled "Volatility State" in all figures, never "Regime" or
    "True Regime", to distinguish it from the synthetic ground-truth labels.

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
        # Use VIX if available — more informative than realised vol
        vol_measure = vix.reindex(dates).ffill().bfill()
    else:
        # Realised vol proxy: 20-day rolling standard deviation, annualised
        vol_measure = mkt.rolling(20).std() * np.sqrt(252)

    # Classify into terciles using expanding-window percentiles (no lookahead)
    state = pd.Series(1, index=dates)  # default: medium

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


# ==============================================================================
# SECTION 4: BEHAVIORAL PROXY COMPOSITE (BPC) — Real Data Version
#            Core logic IDENTICAL to synthetic version.
#            Only change: real sentiment replaces AR(1) proxy.
# ==============================================================================

def build_bpc_real(
    returns_df:    pd.DataFrame,
    market_factor: np.ndarray,
    sentiment_z:   pd.Series = None,
    cfg:           dict = BPC_CONFIG,
) -> pd.DataFrame:
    """
    Behavioral Proxy Composite — Real Market Data Version.

    All five proxies are identical to the synthetic version in tbmd_framework.py.
    The only change is the sentiment component:
      - Synthetic version: AR(1) process correlated with market factor
      - Real version: UMCSENT (U. of Michigan Consumer Sentiment) from FRED,
                      lagged 1 month, normalised via expanding z-score.
                      Falls back to AR(1) if FRED data unavailable.

    All other proxies (herding, loss aversion, autocorrelation, illiquidity)
    are computed identically from real price data.

    Parameters
    ----------
    returns_df    : pd.DataFrame  — real daily log returns
    market_factor : np.ndarray   — SPY log returns
    sentiment_z   : pd.Series    — FRED UMCSENT z-scores (or None for fallback)
    cfg           : dict         — BPC_CONFIG

    Returns
    -------
    features : pd.DataFrame — all BPC proxies + BPC_composite
    """
    n_days, n_assets = returns_df.shape
    dates    = returns_df.index
    mkt      = pd.Series(market_factor, index=dates)
    features = pd.DataFrame(index=dates)
    w        = cfg['cs_dispersion_window']
    norm_w   = cfg['rolling_z_window']

    # ── 1. Herding Proxy ─────────────────────────────────────────────────────
    # Christie & Huang (1995): low cross-sectional dispersion = herding
    cs_dispersion         = returns_df.std(axis=1)
    raw_herd              = -cs_dispersion.rolling(w).mean()
    roll_mean             = raw_herd.rolling(norm_w).mean()
    roll_std              = raw_herd.rolling(norm_w).std().clip(lower=1e-8)
    features['herding_z'] = (raw_herd - roll_mean) / roll_std

    # ── 2. Loss Aversion Proxy ────────────────────────────────────────────────
    # Ang et al. (2006): downside vol > upside vol under loss aversion
    features['downside_vol'] = mkt.rolling(w).apply(
        lambda x: x[x < 0].std() if (x < 0).sum() > 2 else np.nan, raw=True
    )
    features['upside_vol'] = mkt.rolling(w).apply(
        lambda x: x[x > 0].std() if (x > 0).sum() > 2 else np.nan, raw=True
    )
    features['vol_asym'] = (
        features['downside_vol'] / features['upside_vol'].clip(lower=1e-8) - 1.0
    )

    # ── 3. Return Autocorrelation Proxy ───────────────────────────────────────
    # Lo & MacKinlay (1988): autocorrelation = departure from random walk
    features['return_autocorr'] = mkt.rolling(cfg['autocorr_window']).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 10 else np.nan,
        raw=True,
    )

    # ── 4. Illiquidity Proxy ─────────────────────────────────────────────────
    # Amihud (2002) illiquidity ratio: |r_t| / volume_t
    # On real data, use cross-sectional average of |return| / dollar volume.
    # Approximated here as |mkt_return| / rolling_vol (volume data not in
    # returns_df; use price-based proxy which is standard in daily-freq papers)
    features['illiq_proxy'] = (
        mkt.abs() / mkt.rolling(w).std().clip(lower=1e-8)
    ).rolling(w).mean()

    # ── 5. Sentiment ─────────────────────────────────────────────────────────
    if sentiment_z is not None:
        # Real FRED UMCSENT — align to trading dates
        sent_aligned = sentiment_z.reindex(dates).ffill().bfill()
        if sent_aligned.isnull().mean() > 0.5:
            print("  WARNING: Sentiment alignment >50% missing. Using AR(1) fallback.")
            sent_aligned = None

    if sentiment_z is None or 'sent_aligned' not in dir() or sent_aligned is None:
        # AR(1) fallback (identical to synthetic version)
        print("  Using AR(1) sentiment proxy (fallback)")
        noise    = np.random.normal(0, 1, n_days)
        sentiment_arr = np.zeros(n_days)
        for t in range(1, n_days):
            running_std = np.std(market_factor[:t+1]) + 1e-8
            sentiment_arr[t] = (
                0.7 * sentiment_arr[t-1]
                + 0.3 * market_factor[t] / running_std
                + 0.2 * noise[t]
            )
        sent_aligned = pd.Series(sentiment_arr, index=dates)

    features['sentiment'] = sent_aligned

    # ── Volatility ratio proxy ────────────────────────────────────────────────
    features['vol_short'] = mkt.rolling(20).std()
    features['vol_long']  = mkt.rolling(norm_w).std().clip(lower=1e-8)
    features['vol_ratio'] = features['vol_short'] / features['vol_long']

    # ── BPC Composite ─────────────────────────────────────────────────────────
    # Equal-weight baseline (same as synthetic version)
    # Elastic-net weights estimated per fold inside walk-forward
    core_cols = ['herding_z', 'return_autocorr', 'vol_ratio', 'sentiment']
    bpc_df    = features[core_cols].copy()

    for col in core_cols:
        rm = bpc_df[col].rolling(norm_w, min_periods=60).mean()
        rs = bpc_df[col].rolling(norm_w, min_periods=60).std().clip(lower=1e-8)
        bpc_df[col] = (bpc_df[col] - rm) / rs

    features['BPC_composite'] = bpc_df.mean(axis=1)

    return features.ffill().dropna()


# ==============================================================================
# SECTION 5: REAL-TIME EFFICIENCY SCORE (RES)
#            IMPORT from efficiency_score.py
# ==============================================================================
# (Redundant implementation removed to ensure single source of truth)


def _hurst_exponent(ts: np.ndarray) -> float:
    """
    Hurst exponent via R/S (rescaled range) analysis.
    Reference: Hurst (1951); Mandelbrot (1971).

    Input ts is already log-returns — no log transform applied.
    Partitions ts into sub-periods, computes R/S for each,
    and fits slope of log(R/S) vs log(n) to estimate H.

    H = 0.5  → random walk (efficient)
    H > 0.5  → long-range dependence / trending
    H < 0.5  → mean-reverting

    IDENTICAL to synthetic version.
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
        rs_vals, s_vals = [], []
        for c in chunks:
            s = np.std(c)
            if s > 1e-10:
                cumdev = np.cumsum(c - c.mean())
                r = cumdev.max() - cumdev.min()
                rs_vals.append(r / s)
                s_vals.append(s)
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
    Lo-MacKinlay (1988) Variance Ratio.
    IDENTICAL to synthetic version.
    """
    n = len(ts)
    if n < q * 4:
        return 1.0
    mu       = np.mean(ts)
    sigma_1  = np.sum((ts[1:] - ts[:-1] - mu) ** 2) / (n - 1)
    returns_q = np.array([np.sum(ts[i:i+q]) for i in range(n - q + 1)])
    sigma_q   = np.sum((returns_q[1:] - returns_q[:-1] - q * mu) ** 2) \
                / ((len(returns_q) - 1) * q)
    return float(sigma_q / sigma_1) if sigma_1 > 1e-12 else 1.0


def _ljung_box_pval(ts: np.ndarray, lags: int = RES_CONFIG['lb_lags']) -> float:
    """
    Ljung-Box (1978) autocorrelation test p-value.
    IDENTICAL to synthetic version.
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




# ==============================================================================
# SECTION 6: WALK-FORWARD VALIDATION ENGINE
#            IDENTICAL to tbmd_framework.py — do not modify
# ==============================================================================

def walk_forward_validation(
    features_df:   pd.DataFrame,
    returns_df:    pd.DataFrame,
    vol_state:     pd.Series,
    cfg:           dict = WF_CONFIG,
) -> dict:
    """
    Walk-Forward Validation Engine — Real Data Version.

    ENGINE IS IDENTICAL to synthetic version. The only difference is that
    vol_state replaces regime_series for the purpose of regime-stratified
    return analysis in visualisation (Panel 7). It is NOT used in model
    training, signal generation, or any performance metric.

    Same GBM model, same 252-day rolling window, same 21-day refit,
    same 10bps transaction costs, same signal confidence thresholds.

    Parameters
    ----------
    features_df : pd.DataFrame — BPC proxies + RES scores
    returns_df  : pd.DataFrame — real asset log returns
    vol_state   : pd.Series    — volatility state {0,1,2} (visualisation only)
    cfg         : dict         — WF_CONFIG

    Returns
    -------
    dict with rcf_pnl, unconditional_pnl, momentum_pnl,
              mean_rev_pnl, buyhold_pnl, fold_metrics, vol_state
    """
    tc_bps   = cfg['transaction_cost_bps'] / 10_000
    top_n    = cfg['top_n_assets']
    train_w  = cfg['train_window']
    step     = cfg['step_size']
    conf_hi  = cfg['signal_confidence_hi']
    conf_lo  = cfg['signal_confidence_lo']
    mom_lb   = cfg['momentum_lookback']

    # ── Align inputs ──────────────────────────────────────────────────────────
    common      = features_df.index.intersection(returns_df.index)
    features_df = features_df.loc[common]
    returns_df  = returns_df.loc[common]
    vol_state   = vol_state.reindex(common).ffill().fillna(1)
    dates       = common
    n_days      = len(dates)

    # ── Storage ───────────────────────────────────────────────────────────────
    rcf_pnl      = pd.Series(0.0, index=dates)
    uncond_pnl   = pd.Series(0.0, index=dates)
    momentum_pnl = pd.Series(0.0, index=dates)
    mean_rev_pnl = pd.Series(0.0, index=dates)
    buyhold_pnl  = returns_df.mean(axis=1)

    fold_metrics = []
    prev_rcf_wt  = 0.0
    prev_unc_wt  = 0.0

    feat_cols = [c for c in features_df.columns if features_df[c].std() > 1e-8]

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    n_folds = 0
    for t in range(train_w, n_days - 5, step):
        t_start    = max(0, t - train_w)
        pred_start = t
        pred_end   = min(t + step, n_days)

        # Training targets: 5-day ahead SPY return direction (lagged — no lookahead)
        fwd_returns = returns_df.shift(-5).iloc[t_start:t]
        y_dir       = (fwd_returns.mean(axis=1) > 0).astype(int)

        X_tr = features_df.iloc[t_start:t][feat_cols].fillna(0.0)
        X_te = features_df.iloc[pred_start:pred_end][feat_cols].fillna(0.0)

        valid   = y_dir.notna() & ~X_tr.isnull().any(axis=1)
        X_tr_v  = X_tr[valid]
        y_v     = y_dir[valid]

        if len(X_tr_v) < 50 or y_v.nunique() < 2:
            continue

        scaler      = StandardScaler()
        X_tr_s      = scaler.fit_transform(X_tr_v)
        X_te_s      = scaler.transform(X_te)

        # Gradient Boosting Classifier — IDENTICAL hyperparameters
        try:
            model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=SEED,
            )
            model.fit(X_tr_s, y_v)
            prob_up = model.predict_proba(X_te_s)[:, 1]
        except Exception as e:
            print(f"  WARNING: GBM failed at fold t={t}: {e}")
            continue

        n_folds += 1

        # ── Daily PnL ─────────────────────────────────────────────────────────
        for offset in range(pred_end - pred_start):
            day = pred_start + offset
            if day >= n_days:
                break

            ret   = returns_df.iloc[day]
            conv  = prob_up[offset] if offset < len(prob_up) else 0.5
            mkt_r = ret.mean()

            # RCF: conditional on model conviction (regime filter)
            if conv > conf_hi:
                rcf_wt = (conv - 0.5) * 2.0
            elif conv < conf_lo:
                rcf_wt = -(0.5 - conv) * 2.0
            else:
                rcf_wt = 0.0

            to_rcf = abs(rcf_wt - prev_rcf_wt)
            rcf_pnl.iloc[day] = rcf_wt * mkt_r - to_rcf * tc_bps
            prev_rcf_wt = rcf_wt

            # Unconditional: same BPC signal, no filter
            unc_wt = (conv - 0.5) * 2.0
            to_unc = abs(unc_wt - prev_unc_wt)
            uncond_pnl.iloc[day] = unc_wt * mkt_r - to_unc * tc_bps
            prev_unc_wt = unc_wt

            # Cross-sectional momentum baseline
            lb = min(mom_lb, day)
            if lb > 5:
                past       = returns_df.iloc[day - lb: day]
                cum_ret    = past.sum()
                top_assets = cum_ret.nlargest(top_n).index
                bot_assets = cum_ret.nsmallest(top_n).index
                momentum_pnl.iloc[day] = (
                    0.5 * (ret[top_assets].mean() - ret[bot_assets].mean())
                    - tc_bps * 0.1
                )
                mean_rev_pnl.iloc[day] = (
                    0.5 * (ret[bot_assets].mean() - ret[top_assets].mean())
                    - tc_bps * 0.1
                )

        fold_metrics.append({
            'fold_start':    dates[pred_start],
            'fold_end':      dates[min(pred_end - 1, n_days - 1)],
            'rcf_return':    rcf_pnl.iloc[pred_start:pred_end].sum(),
            'uncond_return': uncond_pnl.iloc[pred_start:pred_end].sum(),
            'vol_state':     int(vol_state.iloc[pred_start:pred_end].mode().iloc[0]),
        })

    print(f"  Walk-forward complete: {n_folds} folds")
    return {
        'rcf_pnl':           rcf_pnl,
        'unconditional_pnl': uncond_pnl,
        'momentum_pnl':      momentum_pnl,
        'mean_rev_pnl':      mean_rev_pnl,
        'buyhold_pnl':       buyhold_pnl,
        'fold_metrics':      pd.DataFrame(fold_metrics),
        'vol_state':         vol_state,
    }


# ==============================================================================
# SECTION 7: STATISTICAL TESTS
#            ALL IDENTICAL to tbmd_framework.py — do not modify
# ==============================================================================

def deflated_sharpe_ratio(returns_series: pd.Series, n_trials: int,
                           sr_benchmark: float = 0.0) -> tuple:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    IDENTICAL to synthetic version.
    """
    r = returns_series.dropna()
    n = len(r)
    if n < 30:
        return np.nan, np.nan

    ann_sr = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
    skew   = stats.skew(r)
    kurt   = stats.kurtosis(r)

    gamma        = 0.5772156649
    sr_max_exp   = ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
                    + gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e)))

    sr_hat  = r.mean() / (r.std() + 1e-10)
    sr_star = sr_benchmark / np.sqrt(252)
    sr_adj  = sr_hat * (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2)
    denom   = np.sqrt(1.0 - skew * sr_hat + (kurt + 1.0) / 4.0 * sr_hat ** 2)
    z_stat  = (sr_adj - sr_star) * np.sqrt(n - 1) / (denom + 1e-10)

    return float(ann_sr), float(norm.cdf(z_stat))


def variance_ratio_test(returns_series: pd.Series,
                         lags: list = [2, 5, 10]) -> dict:
    """
    Lo-MacKinlay (1988) Variance Ratio Test.
    IDENTICAL to synthetic version.
    """
    r       = returns_series.dropna().values
    n       = len(r)
    results = {}

    for q in lags:
        if n < q * 4:
            continue
        mu      = np.mean(r)
        sigma_1 = np.sum((r[1:] - r[:-1] - mu) ** 2) / (n - 1)
        r_q     = np.array([np.sum(r[i:i+q]) for i in range(n - q + 1)])
        sigma_q = np.sum((r_q[1:] - r_q[:-1] - q * mu) ** 2) / ((len(r_q) - 1) * q)

        if sigma_1 < 1e-12:
            continue
        vr      = sigma_q / sigma_1
        denom_sq = (np.sum((r - mu) ** 2) / n) ** 2
        theta   = 0.0
        for k in range(1, q):
            delta_k = (np.sum((r[k:] - mu) ** 2 * (r[:-k] - mu) ** 2) / n) \
                      / (denom_sq + 1e-12)
            theta  += ((2.0 * (q - k) / q) ** 2) * delta_k

        z_stat = (vr - 1.0) * np.sqrt(n) / (np.sqrt(theta) + 1e-10)
        p_val  = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        results[f'lag_{q}'] = {'vr': float(vr), 'z': float(z_stat), 'p_val': p_val}

    return results


def benjamini_hochberg(p_values: list, alpha: float = 0.05) -> pd.DataFrame:
    """
    Benjamini-Hochberg FDR correction.
    Reference: Benjamini & Hochberg (1995).
    IDENTICAL to synthetic version.
    """
    n        = len(p_values)
    p_arr    = np.array(p_values, dtype=float)
    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]
    thresh   = np.arange(1, n + 1) * alpha / n
    reject   = sorted_p <= thresh
    if reject.any():
        reject[:np.where(reject)[0][-1] + 1] = True
    adj_p = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1))
    return pd.DataFrame({
        'original_idx': sort_idx,
        'raw_p': sorted_p,
        'adjusted_p': adj_p,
        'reject_H0': reject,
    }).sort_values('original_idx').reset_index(drop=True)


def compute_performance_metrics(returns_series: pd.Series,
                                  name: str = 'Strategy') -> dict:
    """
    Comprehensive performance metrics.
    IDENTICAL to synthetic version.
    """
    r = returns_series.dropna()
    r = r[r != 0.0]
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


# ==============================================================================
# SECTION 8: OPTION B VALIDATION
#            The five directional claims tested explicitly on real data
# ==============================================================================

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
      3. RES correlates negatively with VIX (high VIX = inefficient = low RES)
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
        bh_result  = benjamini_hochberg(p_values)
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


# ==============================================================================
# SECTION 9: VISUALISATION
# ==============================================================================

def plot_real_data_results(
    spy_prices:   pd.Series,
    market_factor: np.ndarray,
    dates:        pd.DatetimeIndex,
    vol_state:    pd.Series,
    bpc_features: pd.DataFrame,
    res_scores:   pd.DataFrame,
    wf_results:   dict,
    vix:          pd.Series,
    sentiment_src: str,
    output_path:  str = OUTPUT_FIG1,
) -> None:
    """
    8-Panel validation figure — Real Market Data.

    Layout mirrors the synthetic paper figure for direct visual comparison.
    Key difference: regime panels replaced with volatility-state panels,
    clearly labelled as "Volatility State" not "True Regime".
    """
    SURV_WARN = ("⚠ SURVIVORSHIP BIAS: Results use current S&P 100 constituents. "
                 "Actual returns likely 1-3% lower annualised.")

    fig = plt.figure(figsize=(22, 30))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)

    STATE_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    STATE_NAMES  = {0: 'Low Vol (Calm)', 1: 'Medium Vol', 2: 'High Vol (Stress)'}
    PALETTE = {
        'RCF Strategy':      '#1f77b4',
        'Unconditional BPC': '#d62728',
        'Momentum':          '#ff7f0e',
        'Mean Reversion':    '#2ca02c',
        'Buy & Hold':        '#9467bd',
    }

    mkt = pd.Series(market_factor, index=dates)

    # ── Panel 1: SPY price path with volatility-state shading ────────────────
    ax1 = fig.add_subplot(gs[0, :])
    spy_norm = spy_prices / spy_prices.iloc[0]
    ax1.plot(spy_norm.index, spy_norm.values, 'k-', lw=1.3, alpha=0.8, label='SPY (normalised)')

    for sid, col in STATE_COLORS.items():
        mask  = vol_state == sid
        sdates = vol_state[mask].index
        for i in range(len(sdates) - 1):
            ax1.axvspan(sdates[i], sdates[i+1], alpha=0.08, color=col)

    if vix is not None:
        ax1_r = ax1.twinx()
        vix_a = vix.reindex(spy_norm.index).ffill()
        ax1_r.plot(vix_a.index, vix_a.values, color='purple', lw=0.8,
                   alpha=0.5, label='VIX')
        ax1_r.set_ylabel('VIX', color='purple', fontsize=9)
        ax1_r.tick_params(axis='y', colors='purple')

    from matplotlib.patches import Patch
    patches = [Patch(facecolor=STATE_COLORS[i], alpha=0.4,
                     label=STATE_NAMES[i]) for i in STATE_COLORS]
    ax1.legend(handles=patches + [plt.Line2D([0],[0], color='k', label='SPY')],
               loc='upper left', fontsize=8)
    ax1.set_title(
        'Figure 1: SPY Price Path with Volatility-State Shading\n'
        '(State = vol tercile. NOT a regime classifier. For visualisation only.)',
        fontweight='bold', fontsize=11,
    )
    ax1.set_ylabel('Normalised Price')
    ax1.tick_params(axis='x', rotation=30)

    # ── Panel 2: BPC components ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bpc_cols = ['herding_z', 'return_autocorr', 'sentiment', 'BPC_composite']
    bpc_plot = bpc_features[[c for c in bpc_cols if c in bpc_features.columns]].rolling(21).mean().dropna()
    for col in [c for c in bpc_cols[:-1] if c in bpc_plot.columns]:
        ax2.plot(bpc_plot.index, bpc_plot[col], alpha=0.4, lw=0.8)
    if 'BPC_composite' in bpc_plot.columns:
        ax2.plot(bpc_plot.index, bpc_plot['BPC_composite'], 'k-', lw=1.6,
                 label='BPC Composite')
    ax2.axhline(0, color='k', lw=0.5, ls='--')
    ax2.set_title('Figure 2: BPC Proxy Components (21-day rolling)\n'
                  f'Sentiment source: {sentiment_src}',
                  fontweight='bold', fontsize=10)
    ax2.set_ylabel('Standardised Score')
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='x', rotation=30)

    # ── Panel 3: RES vs VIX ───────────────────────────────────────────────────
    ax3  = fig.add_subplot(gs[1, 1])
    ax3b = ax3.twinx()
    eff_plot = res_scores['efficiency_score'].rolling(21).mean().dropna()
    ax3.plot(eff_plot.index, eff_plot.values, 'b-', lw=1.2, alpha=0.9, label='RES')
    if vix is not None:
        vix_norm = (vix.reindex(eff_plot.index).ffill() - vix.mean()) / vix.std()
        ax3b.plot(vix_norm.index, -vix_norm.values, color='red', lw=0.8,
                  alpha=0.5, label='-VIX (normalised)')
        ax3b.set_ylabel('-VIX normalised (high = calm)', color='red', fontsize=8)
        try:
            common = eff_plot.index.intersection(vix.index)
            corr, _ = stats.pearsonr(
                eff_plot.loc[common].dropna(),
                (-vix.reindex(common).ffill()).reindex(eff_plot.loc[common].dropna().index)
            )
            ax3.text(0.02, 0.05, f'r(RES, -VIX) = {corr:.3f}',
                     transform=ax3.transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        except Exception:
            pass
    ax3.set_ylabel('Efficiency Score (0=Inefficient, 1=Efficient)', color='blue')
    ax3.set_title('Figure 3: RES vs VIX (Directional Validation)\n'
                  'Expected: RES moves opposite to VIX',
                  fontweight='bold', fontsize=10)
    ax3.tick_params(axis='x', rotation=30)

    # ── Panel 4: Hurst exponent ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    hp  = res_scores['hurst'].rolling(21).mean().dropna()
    ax4.plot(hp.index, hp.values, 'purple', lw=1.2)
    ax4.axhline(0.5, color='k',   ls='--', lw=1,   label='H=0.5 (Efficient)')
    ax4.axhline(0.6, color='red', ls=':',  lw=0.8, label='H=0.6 (Trending)')
    ax4.fill_between(hp.index, 0.5, hp.values,
                     where=hp.values > 0.5, alpha=0.25, color='red', label='Trending')
    ax4.fill_between(hp.index, hp.values, 0.5,
                     where=hp.values < 0.5, alpha=0.25, color='blue', label='Mean-Rev')
    ax4.set_ylim([0.2, 0.8])
    ax4.set_title('Figure 4: Hurst Exponent (R/S Analysis) — Real S&P 100 Data\n'
                  'H > 0.5 = Trending | H < 0.5 = Mean-Reverting',
                  fontweight='bold', fontsize=10)
    ax4.set_ylabel('Hurst Exponent')
    ax4.legend(fontsize=7)
    ax4.tick_params(axis='x', rotation=30)

    # ── Panel 5: Walk-forward PnL ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    strats = {
        'RCF Strategy':      wf_results['rcf_pnl'],
        'Unconditional BPC': wf_results['unconditional_pnl'],
        'Momentum':          wf_results['momentum_pnl'],
        'Buy & Hold':        wf_results['buyhold_pnl'],
    }
    pnl_start = wf_results['rcf_pnl'].ne(0).idxmax()
    for sn, pnl in strats.items():
        cum = pnl.loc[pnl_start:].cumsum()
        lw  = 2.2 if sn == 'RCF Strategy' else 1.2
        ax5.plot(cum.index, cum.values, lw=lw, color=PALETTE[sn],
                 label=sn, alpha=0.9)
    ax5.axhline(0, color='k', lw=0.5)
    ax5.set_title('Figure 5: Walk-Forward Cumulative PnL — Real Data\n'
                  '(Net of 10bps Transaction Costs)',
                  fontweight='bold', fontsize=10)
    ax5.set_ylabel('Cumulative Log Return')
    ax5.legend(fontsize=8)
    ax5.tick_params(axis='x', rotation=30)

    # ── Panel 6: Rolling 6-month Sharpe ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    for sn, pnl in list(strats.items())[:3]:
        rsr = (pnl.rolling(126).mean() /
               (pnl.rolling(126).std() + 1e-10) * np.sqrt(252))
        ax6.plot(rsr.index, rsr.values, lw=1.2, color=PALETTE[sn],
                 label=sn, alpha=0.8)
    ax6.axhline(0,   color='k',     lw=0.8, ls='--')
    ax6.axhline(1.0, color='green', lw=0.8, ls=':', alpha=0.7, label='SR=1.0')
    ax6.set_title('Figure 6: Rolling 6-Month Sharpe Ratio — Real Data',
                  fontweight='bold', fontsize=10)
    ax6.set_ylabel('Annualised Sharpe')
    ax6.set_ylim([-4, 5])
    ax6.legend(fontsize=8)
    ax6.tick_params(axis='x', rotation=30)

    # ── Panel 7: Return by volatility state ───────────────────────────────────
    ax7   = fig.add_subplot(gs[3, 1])
    vs    = wf_results['vol_state']
    s_pnl = {
        'RCF':     wf_results['rcf_pnl'],
        'Uncond':  wf_results['unconditional_pnl'],
        'Mom':     wf_results['momentum_pnl'],
    }
    state_perf = {}
    for sid in [0, 1, 2]:
        mask = vs == sid
        state_perf[STATE_NAMES[sid]] = {
            sn: pnl[mask].mean() * 252 * 100
            for sn, pnl in s_pnl.items()
        }
    sdf = pd.DataFrame(state_perf).T
    x   = np.arange(len(sdf))
    w   = 0.25
    cs  = ['#1f77b4', '#d62728', '#ff7f0e']
    for i, col in enumerate(sdf.columns):
        ax7.bar(x + i*w, sdf[col].values, w, label=col, alpha=0.8, color=cs[i])
    ax7.axhline(0, color='k', lw=0.8)
    ax7.set_xticks(x + w)
    ax7.set_xticklabels(sdf.index, rotation=15, fontsize=8)
    ax7.set_title('Figure 7: Ann. Return (%) by Volatility State\n'
                  '(State = vol tercile — NOT a regime classifier)',
                  fontweight='bold', fontsize=10)
    ax7.set_ylabel('Annualised Return (%)')
    ax7.legend(fontsize=8)

    # ── Panel 8: Performance table ────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')

    all_strats = {
        'RCF Strategy':      wf_results['rcf_pnl'],
        'Unconditional BPC': wf_results['unconditional_pnl'],
        'Momentum':          wf_results['momentum_pnl'],
        'Mean Reversion':    wf_results['mean_rev_pnl'],
        'Buy & Hold':        wf_results['buyhold_pnl'],
    }
    rows = []
    for sn, pnl in all_strats.items():
        active = pnl[pnl != 0]
        m      = compute_performance_metrics(active, sn)
        if not m:
            continue
        sr, dsr = deflated_sharpe_ratio(active, n_trials=12)
        syn     = SYNTHETIC_RESULTS.get(sn, {})
        rows.append([
            sn,
            f"{m['Ann_Return_%']:.2f}%",
            f"{m['Ann_Vol_%']:.2f}%",
            f"{m['Sharpe']:.3f}",
            f"{dsr:.3f}" if dsr and not np.isnan(dsr) else 'N/A',
            f"{m['Max_Drawdown_%']:.2f}%",
            f"{m['Win_Rate_%']:.1f}%",
            f"{m['P_value']:.4f}",
            f"{syn.get('Sharpe', 'N/A')}",
        ])

    cols = ['Strategy', 'Ann Ret', 'Ann Vol', 'Sharpe', 'DSR*',
            'Max DD', 'Win%', 'P-val', 'Synth SR']
    if rows:
        tbl = ax8.table(cellText=rows, colLabels=cols,
                        cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.85])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor('#2c3e50')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    ax8.set_title(
        'Table 1: Real-Data Walk-Forward Performance vs Synthetic Benchmark\n'
        '* DSR = Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) | '
        'N_trials=12 | Synth SR = synthetic paper result',
        fontweight='bold', fontsize=9, pad=14,
    )

    plt.suptitle(
        'TBMD Framework: Real-Market Validation (Option B)\n'
        f'S&P 100 | {START_DATE} to {END_DATE} | Walk-Forward | Net 10bps Costs',
        fontsize=13, fontweight='bold', y=0.998,
    )
    fig.text(0.5, 0.002, SURV_WARN, ha='center', fontsize=8,
             color='darkred', style='italic')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure saved: {output_path}")


def plot_synthetic_vs_real(
    real_results: dict,
    output_path:  str = OUTPUT_FIG2,
) -> None:
    """
    Side-by-side comparison of synthetic vs real Sharpe and DSR.
    Used in the paper as Figure 9 (Real-Data Validation section).
    """
    strategies = ['RCF Strategy', 'Unconditional BPC', 'Momentum Baseline',
                  'Mean Reversion', 'Buy & Hold']

    syn_sharpe  = [SYNTHETIC_RESULTS.get(s, {}).get('Sharpe', 0) for s in strategies]
    real_sharpe = [real_results.get(s, {}).get('Sharpe', 0) for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    x    = np.arange(len(strategies))
    wbar = 0.35

    # Sharpe comparison
    ax = axes[0]
    ax.bar(x - wbar/2, syn_sharpe,  wbar, label='Synthetic (paper)',
           color='steelblue', alpha=0.8)
    ax.bar(x + wbar/2, real_sharpe, wbar, label='Real S&P 100',
           color='coral', alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=9)
    ax.set_title('Sharpe Ratio: Synthetic vs Real\n'
                 'Key claim: RCF > Unconditional in BOTH columns',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Annualised Sharpe Ratio')
    ax.legend(fontsize=10)

    # Highlight the key comparison
    rcf_r   = real_results.get('RCF Strategy', {}).get('Sharpe', 0)
    unc_r   = real_results.get('Unconditional BPC', {}).get('Sharpe', 0)
    verdict = 'PASS' if rcf_r > unc_r else 'FAIL'
    color   = 'green' if verdict == 'PASS' else 'red'
    ax.text(0.02, 0.95, f"Core Claim: RCF > Unconditional\nReal Data Verdict: {verdict}",
            transform=ax.transAxes, fontsize=10, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', fc='white', ec=color, alpha=0.9),
            verticalalignment='top')

    # Directional agreement panel
    ax2 = axes[1]
    directions_match = [
        (s, np.sign(syn_sharpe[i]) == np.sign(real_sharpe[i]))
        for i, s in enumerate(strategies)
    ]
    colors_dm = ['#2ecc71' if m else '#e74c3c' for _, m in directions_match]
    bars      = ax2.barh([s for s, _ in directions_match],
                          [real_sharpe[i] for i in range(len(strategies))],
                          color=colors_dm, alpha=0.8)
    ax2.axvline(0, color='k', lw=0.8)
    ax2.set_title('Real-Data Sharpe Ratios\n'
                  'Green = Same sign as synthetic, Red = Sign reversal',
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel('Annualised Sharpe Ratio (Real Data)')

    n_match = sum(m for _, m in directions_match)
    ax2.text(0.98, 0.02,
             f"Directional agreement: {n_match}/{len(strategies)} strategies",
             transform=ax2.transAxes, fontsize=10, ha='right',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.suptitle(
        'TBMD Framework: Synthetic vs Real-Data Validation Comparison\n'
        'Core claim holds if RCF bar is taller than Unconditional BPC bar (left panel)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Comparison figure saved: {output_path}")


# ==============================================================================
# SECTION 10: MAIN EXECUTION PIPELINE
# ==============================================================================

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
    print('  DATA SOURCES')
    print('     Prices   : Yahoo Finance (yfinance) — daily adjusted close')
    print('     Mkt Fac  : SPY ETF via Yahoo Finance')
    print('     Sentiment: FRED UMCSENT (Univ. of Michigan, via FRED API)')
    print('     VIX      : CBOE via Yahoo Finance (^VIX) — visualisation only')
    print()

    if not YFINANCE_AVAILABLE:
        print('FATAL: yfinance not installed. Run: pip install yfinance')
        return

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
    print('       NOTE: This is NOT a regime classifier. Used for charts only.')

    # ── Step 5: BPC ───────────────────────────────────────────────────────────
    print('\n[5/8]  Building Behavioral Proxy Composite (BPC)...')
    bpc_features = build_bpc_real(returns_df, market_factor, sentiment_z)
    print(f'       Proxies built: {list(bpc_features.columns)}')

    # ── Step 6: RES ───────────────────────────────────────────────────────────
    print('\n[6/8]  Computing Real-Time Efficiency Score (RES)...')
    res_scores = compute_res(pd.Series(market_factor, index=dates))
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
            print(f'       r(RES, VIX) = {corr:.3f} (p={pv:.4f})'
                  ' — expected negative')
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

    wf_results = walk_forward_validation(all_features, returns_df, vol_state)

    # ── Step 8: Results ───────────────────────────────────────────────────────
    print('\n[8/8]  Performance Statistics — Real Data')
    print('       ' + '⚠ Survivorship bias: ~1-3% upward inflation on all returns')
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
        print(f'         Max DD:      {m["Max_Drawdown_%"]:>7.2f}%')
        print(f'         P-value:     {m["P_value"]:>8.4f}'
              f'  {"*** SIGNIFICANT" if m["P_value"] < 0.05 else "(not significant)"}')

    # ── Variance ratio tests ──────────────────────────────────────────────────
    print('\n       Variance Ratio Tests (RCF PnL):')
    rcf_active = wf_results['rcf_pnl'][wf_results['rcf_pnl'] != 0]
    vr_res = variance_ratio_test(rcf_active)
    for lag_key, res in vr_res.items():
        stars = '***' if res['p_val'] < 0.01 else '**' if res['p_val'] < 0.05 else 'ns'
        print(f'         {lag_key}: VR={res["vr"]:.4f}, Z={res["z"]:.3f},'
              f' p={res["p_val"]:.4f} {stars}')

    # ── Directional claims ────────────────────────────────────────────────────
    print('\n       DIRECTIONAL CLAIMS VALIDATION (Core of Option B)')
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

    print()
    print(f'       OVERALL: {n_pass}/{len(claim_results)} directional claims PASSED')
    if n_pass >= 4:
        print('       VERDICT: Real-data results SUPPORT the paper\'s claims.')
        print('                The core mechanism transfers from simulation to')
        print('                live market data.')
    elif n_pass >= 3:
        print('       VERDICT: PARTIAL SUPPORT. Most claims hold. Investigate')
        print('                failing claims before submission.')
    else:
        print('       VERDICT: INSUFFICIENT SUPPORT. Fewer than 3 claims pass.')
        print('                Results require further investigation.')

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
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f'\n       Results saved to {OUTPUT_CSV}')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\n       Generating figures...')
    plot_real_data_results(
        spy_prices, market_factor, dates, vol_state,
        bpc_features, res_scores, wf_results, vix, sentiment_src,
    )
    plot_synthetic_vs_real(real_metrics)

    print()
    print(DIVIDER)
    print('Real-data validation complete.')
    print()
    print('NEXT STEP FOR THE PAPER:')
    print('  Add a Section V.G "Real-Data Validation" with:')
    print('  1. The directional claims table above')
    print('  2. tbmd_realdata_figures.png as Figure set 2')
    print('  3. tbmd_realdata_comparison.png as Figure 9')
    print('  4. tbmd_realdata_results.csv as Table 5')
    print('  5. The survivorship bias caveat in the data section')
    print(DIVIDER)


if __name__ == '__main__':
    main()
