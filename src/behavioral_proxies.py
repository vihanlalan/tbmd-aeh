"""
behavioral_proxies.py
---------------------
Computes the five components of the Behavioral Proxy Composite (BPC):
  1. Z_herd  — cross-sectional return dispersion (herding proxy)
  2. Z_vr    — Lo-MacKinlay variance ratio (autocorrelation proxy)
  3. Z_asym  — volatility asymmetry (loss aversion proxy)
  4. Z_sent  — VIX-based sentiment proxy (FinBERT replacement when news unavailable)
  5. Z_illiq — Amihud illiquidity ratio

All components are converted to rolling Z-scores on an expanding window
so no future information contaminates the signal.

Reference:
  Christie & Huang (1995) — herding
  Lo & MacKinlay (1988)   — variance ratio
  Ang et al. (2006)       — downside volatility
  Baker & Wurgler (2007)  — sentiment
  Amihud (2002)           — illiquidity
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# 1. Herding proxy: cross-sectional standard deviation of returns
# ------------------------------------------------------------------

def cross_sectional_dispersion(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Cross-sectional standard deviation of daily returns, averaged over
    a rolling window. High values indicate divergent returns (low herding).
    We negate so that HIGH values correspond to HIGH herding (market
    moving together).

    Christie & Huang (1995): herding suppresses cross-sectional dispersion.
    """
    daily_cssd = returns.std(axis=1)  # cross-sectional std each day
    # Rolling mean to smooth
    rolling_cssd = daily_cssd.rolling(window=window, min_periods=window // 2).mean()
    # Negate: when CSSD is LOW, herding is HIGH — flip sign so proxy
    # is positively correlated with behavioral activity
    herding_proxy = -rolling_cssd
    herding_proxy.name = 'herd'
    return herding_proxy


# ------------------------------------------------------------------
# 2. Variance ratio — Lo-MacKinlay (1988)
# ------------------------------------------------------------------

def variance_ratio(returns: pd.Series, q: int = 5) -> float:
    """
    Lo-MacKinlay (1988) variance ratio test at lag q.
    VR = Var(q-period return) / (q * Var(1-period return))
    Random walk: VR = 1.0
    Positive autocorrelation: VR > 1.0
    Negative autocorrelation: VR < 1.0
    """
    r = returns.dropna().values
    n = len(r)
    if n < q * 4:
        return np.nan

    # Compute q-period returns
    q_ret = np.array([sum(r[i:i+q]) for i in range(n - q + 1)])

    var1 = np.var(r, ddof=1)
    varq = np.var(q_ret, ddof=1)

    if var1 == 0:
        return np.nan

    return varq / (q * var1)


def rolling_variance_ratio(returns: pd.Series, window: int = 60, q: int = 5) -> pd.Series:
    """
    Rolling VR computed over a sliding window.
    """
    vr_vals = []
    idx = []
    for i in range(window, len(returns) + 1):
        window_ret = returns.iloc[i - window:i]
        vr = variance_ratio(window_ret, q)
        vr_vals.append(vr)
        idx.append(returns.index[i - 1])
    s = pd.Series(vr_vals, index=idx, name='vr')
    return s


# ------------------------------------------------------------------
# 3. Volatility asymmetry — loss aversion proxy
# ------------------------------------------------------------------

def volatility_asymmetry(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Ratio of downside to upside realized volatility, rolling.
    High ratio = asymmetric fear (loss aversion active).

    Ang et al. (2006): downside correlation and risk are priced.
    """
    result = []
    idx = []
    for i in range(window, len(returns) + 1):
        r = returns.iloc[i - window:i]
        neg = r[r < 0]
        pos = r[r >= 0]
        down_vol = neg.std() if len(neg) > 2 else np.nan
        up_vol = pos.std() if len(pos) > 2 else np.nan
        asym = down_vol / up_vol if (up_vol is not None and up_vol > 0) else np.nan
        result.append(asym)
        idx.append(returns.index[i - 1])
    s = pd.Series(result, index=idx, name='asym')
    return s


def aggregate_volatility_asymmetry(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Cross-sectional median of per-stock volatility asymmetry.
    """
    per_stock = returns.apply(lambda col: volatility_asymmetry(col, window), axis=0)
    return per_stock.median(axis=1).rename('asym')


# ------------------------------------------------------------------
# 4. Sentiment proxy — VIX inversion (FinBERT substitute)
# ------------------------------------------------------------------

def vix_sentiment_proxy(vix: pd.Series, window: int = 5) -> pd.Series:
    """
    Inverted VIX as a sentiment proxy. Low VIX = positive sentiment.
    Smoothed over 5-day window to reduce noise.

    When FinBERT news sentiment is available, replace this with:
        finbert_sentiment = finbert_model.predict(news_headlines)
        Z_sent = finbert_sentiment.rolling(window).mean()

    The VIX proxy captures the same fear/greed dimension at daily
    frequency without requiring news data access.

    Baker & Wurgler (2007): sentiment predicts returns cross-sectionally.
    """
    inv_vix = -vix.rolling(window=window, min_periods=2).mean()
    inv_vix.name = 'sent'
    return inv_vix


# ------------------------------------------------------------------
# 5. Amihud illiquidity
# ------------------------------------------------------------------

def amihud_illiquidity(returns: pd.DataFrame, dollar_volumes: pd.DataFrame,
                       window: int = 20) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio: |r_t| / dollar_volume_t
    Averaged cross-sectionally and over rolling window.
    """
    abs_ret = returns.abs()
    # Per-stock Amihud ratio each day
    illiq_daily = abs_ret / dollar_volumes.replace(0, np.nan)
    # Cross-sectional median to reduce outlier influence
    illiq_cs = illiq_daily.median(axis=1)
    # Rolling mean
    illiq_rolling = illiq_cs.rolling(window=window, min_periods=window // 2).mean()
    illiq_rolling.name = 'illiq'
    return illiq_rolling


# ------------------------------------------------------------------
# BPC Assembly
# ------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, min_periods: int = 252) -> pd.Series:
    """
    Expanding-window Z-score: avoids lookahead bias.
    mean and std computed only from data up to and including time t.
    """
    mu = series.expanding(min_periods=min_periods).mean()
    sigma = series.expanding(min_periods=min_periods).std()
    return ((series - mu) / sigma.replace(0, np.nan)).rename(series.name)


def compute_bpc(returns: pd.DataFrame,
                dollar_volumes: pd.DataFrame,
                vix: pd.Series,
                vr_window: int = 60,
                herd_window: int = 20,
                asym_window: int = 20,
                illiq_window: int = 20,
                sent_window: int = 5) -> pd.DataFrame:
    """
    Compute all five BPC components and assemble into a DataFrame.

    All components are converted to expanding-window Z-scores so the
    composite is interpretable on a common scale and free of lookahead.

    Parameters
    ----------
    returns        : pd.DataFrame  (T x N) daily log or simple returns
    dollar_volumes : pd.DataFrame  (T x N) daily dollar volumes
    vix            : pd.Series     (T,)    VIX close price
    """
    # Compute a single-asset aggregate return series for VR and asymmetry
    mkt_return = returns.mean(axis=1)

    print("Computing BPC components...")

    herd_raw  = cross_sectional_dispersion(returns, window=herd_window)
    vr_raw    = rolling_variance_ratio(mkt_return, window=vr_window, q=5)
    asym_raw  = aggregate_volatility_asymmetry(returns, window=asym_window)
    sent_raw  = vix_sentiment_proxy(vix, window=sent_window)
    illiq_raw = amihud_illiquidity(returns, dollar_volumes, window=illiq_window)

    # Align to common index
    all_raw = pd.concat([herd_raw, vr_raw, asym_raw, sent_raw, illiq_raw], axis=1)
    all_raw.columns = ['herd', 'vr', 'asym', 'sent', 'illiq']

    # Z-score on expanding window (min 252 obs = one full year)
    components_z = pd.DataFrame({
        col: _rolling_zscore(all_raw[col], min_periods=252)
        for col in all_raw.columns
    })

    # Equal-weighted BPC composite (sign-adjusted so all positively
    # correlate with behavioral activity):
    # herd:  high Z → high herding → behaviorally active (positive)
    # vr:    |VR-1| large → autocorrelation present (use abs deviation)
    # asym:  high Z → high fear asymmetry (positive)
    # sent:  high Z → low VIX → positive sentiment, less behavioral (negative)
    # illiq: high Z → high illiquidity → behaviorally active (positive)

    vr_z = _rolling_zscore((all_raw['vr'] - 1.0).abs(), min_periods=252).rename('vr_dev')

    bpc_components = pd.DataFrame({
        'Z_herd'  : components_z['herd'],
        'Z_vr'    : vr_z,
        'Z_asym'  : components_z['asym'],
        'Z_sent'  : -components_z['sent'],   # invert: low VIX = low fear = less behavioral
        'Z_illiq' : components_z['illiq'],
    })

    bpc_components['BPC'] = bpc_components.mean(axis=1)

    print(f"BPC computed: {bpc_components.dropna().shape[0]} valid observations")
    return bpc_components


# ------------------------------------------------------------------
# Per-stock cross-sectional signal (for RCF long/short ranking)
# ------------------------------------------------------------------

def compute_per_stock_signal(returns: pd.DataFrame,
                              dollar_volumes: pd.DataFrame,
                              mom_window: int = 20,
                              asym_window: int = 20,
                              illiq_window: int = 20,
                              min_periods: int = 252) -> pd.DataFrame:
    """
    Per-asset behavioral composite used to RANK individual stocks for the
    long/short leg of the Regime-Conditional Filter.

    The market-level BPC(t) in compute_bpc() is a single aggregate time
    series (one value per day across the whole universe) -- it has no
    cross-sectional variation and therefore cannot be used to decide
    which individual stocks to go long or short. This function builds a
    genuinely stock-level counterpart from the same three observable,
    literature-grounded quantities, each computed independently per stock:

        - Momentum: trailing mom_window-day cumulative return
          (Jegadeesh & Titman 1993 cross-sectional momentum ranking)
        - Loss-aversion asymmetry: per-stock downside/upside realized
          vol ratio (Ang et al. 2006)
        - Illiquidity: per-stock Amihud (2002) ratio, |r_t| / dollar_volume_t

    Each is expanding-window Z-scored per stock (no lookahead) and
    equal-weighted into a per-stock composite score(t, i). Ranking
    assets by this score at time t replaces the placeholder random
    cross-sectional noise previously used in the backtest engine.
    """
    # 1. Per-stock momentum: trailing cumulative return, lagged 1 day
    mom_raw = returns.rolling(window=mom_window, min_periods=mom_window // 2).sum()

    # 2. Per-stock volatility asymmetry: downside / upside realized vol
    def _asym(col: pd.Series) -> pd.Series:
        neg = col.where(col < 0)
        pos = col.where(col >= 0)
        down = neg.rolling(window=asym_window, min_periods=asym_window // 2).std()
        up = pos.rolling(window=asym_window, min_periods=asym_window // 2).std()
        return down / up.replace(0, np.nan)

    asym_raw = returns.apply(_asym, axis=0)

    # 3. Per-stock Amihud illiquidity ratio
    illiq_raw = (returns.abs() / dollar_volumes.replace(0, np.nan)).rolling(
        window=illiq_window, min_periods=illiq_window // 2
    ).mean()

    def _z(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.expanding(min_periods=min_periods).mean()
        sigma = df.expanding(min_periods=min_periods).std()
        return (df - mu) / sigma.replace(0, np.nan)

    mom_z = _z(mom_raw)
    asym_z = _z(asym_raw)
    illiq_z = _z(illiq_raw)

    composite = (mom_z + asym_z + illiq_z) / 3.0
    return composite
