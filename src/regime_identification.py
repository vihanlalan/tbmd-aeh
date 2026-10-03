"""
regime_identification.py
------------------------
Can market regimes be identified objectively, and how accurately?

    python src/regime_identification.py --validate        # full run -> outputs/regimes/
    python src/regime_identification.py --quick           # smoke test -> outputs/regimes_quick/

Options: --device {auto,cuda,mps,cpu}  --float32  --n-sim N  --n-pelt N  --chunk N
         --out DIR  --data-dir DIR  --validate  --quick
The HMM and jump-model fits run in batches on a GPU (CUDA, or Apple silicon
through MPS) in float32, or on the CPU in float64 (see gpu_regimes.py);
PELT, GARCH and bear-market dating run on the CPU. Missing input data are
downloaded on first use (Yahoo Finance: ^GSPC from 1950, ^VIX from 1990;
FRED: USREC).

Part A  Simulation with known regimes (HMM-2, HMM-2 with t(4) shocks, HMM-3)
        and without regimes (GARCH(1,1)-t): accuracy of each method ex post
        and in real time; how often BIC finds regimes in raw returns and in
        GARCH-t residuals; how much methods agree with each other when there
        are no regimes; false-positive rate of the regime-dependent AR(1) test.
Part B  S&P 500 daily, 1950-2025: BIC and out-of-sample density forecasts of
        HMMs vs GARCH; regimes beyond GARCH (HMM on GARCH-t PIT residuals);
        cross-method agreement compared with its no-regime null; real-time
        (expanding-window) vs ex-post labels; parameter stability.
Part C  Efficiency by regime: AR(1) in high- vs low-volatility states using
        real-time labels (no look-ahead) with a White-robust difference test
        and its simulated false-positive rate; rolling variance-ratio
        rejection rates for real data and for regime / GARCH simulations.
Scorecard  Pass/fail against criteria fixed before the full run.
Outputs    CSV tables, figures (PNG), RESULTS.md and summary.json in --out.
"""

import io
import os
import sys
import time
import json
import pickle
import argparse
import datetime as dt
import contextlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
import ruptures as rpt
from arch import arch_model
from arch.univariate import ConstantMean, GARCH as GARCHVol, StudentsT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_regimes as G

warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CACHE = os.path.join(ROOT, 'data', 'cache')
OUT = os.path.join(ROOT, 'outputs', 'regimes')
N_SIM = 100          # simulation replications per DGP (--n-sim)
N_PELT = 25          # replications per DGP that also get PELT (42 s each on one CPU core)
CHUNK = 16           # series per GPU batch (--chunk); lower it if GPU memory runs out
QUICK = False
B4_FIRST_YEAR = 1970
OOS_START = '2000-01-01'
STABILITY_SPLIT = '1988-01-01'
JM_LAMBDAS = (10.0, 30.0, 100.0)
EPISODES = {'1987 crash': '1987-10-14', '2008 Lehman': '2008-09-15',
            '2011 US downgrade': '2011-08-01', '2020 Covid': '2020-02-21',
            '2022 rate shock': '2022-01-18'}
ERAS = [(1950, 1979), (1980, 1999), (2000, 2025)]
DGPS = ['HMM2 (correctly specified)', 'HMM2 with t(4) shocks (misspecified)',
        'HMM3 (three regimes)', 'GARCH(1,1)-t (no regimes)']
NO_REGIME_DGP = DGPS[3]
# ex-post methods compared with each other (Part A null and Part B real data)
AGREE_METHODS = ['HMM-2', 'Jump λ=10', 'Jump λ=30', 'Jump λ=100', 'Change-point (PELT)']


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

REQUIRED_FILES = ('gspc_long.csv', 'vix_long.csv', 'usrec.csv')


def download_regime_data(cache=None):
    """S&P 500 index from 1950, VIX from 1990 (Yahoo Finance) and NBER USREC (FRED)."""
    import urllib.request
    import yfinance as yf
    cache = cache or CACHE
    os.makedirs(cache, exist_ok=True)

    def yahoo(ticker, start):
        raw = yf.download(ticker, start=start, end='2026-01-01', auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[['Close', 'Volume']].dropna(subset=['Close'])
        df.index.name = 'Date'
        return df

    gspc = yahoo('^GSPC', '1950-01-01')
    gspc.to_csv(os.path.join(cache, 'gspc_long.csv'))
    vix = yahoo('^VIX', '1990-01-01')[['Close']]
    vix.to_csv(os.path.join(cache, 'vix_long.csv'))
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC'
    with urllib.request.urlopen(url, timeout=30) as r:
        rec = pd.read_csv(io.StringIO(r.read().decode()))
    rec.columns = ['date', 'USREC']
    rec.set_index(pd.to_datetime(rec['date'])).drop(columns='date').to_csv(os.path.join(cache, 'usrec.csv'))
    lines = [f'Downloaded {dt.datetime.now().isoformat(timespec="seconds")}',
             f'^GSPC: {gspc.index[0].date()} to {gspc.index[-1].date()}, {len(gspc)} rows (Yahoo Finance)',
             f'^VIX: {vix.index[0].date()} to {vix.index[-1].date()}, {len(vix)} rows (Yahoo Finance)',
             f'USREC: {rec["date"].iloc[0]} to {rec["date"].iloc[-1]} (FRED)']
    with open(os.path.join(cache, 'regime_data_snapshot.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def ensure_data():
    """Download the S&P 500 / VIX / NBER files on first use if the cache lacks them."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(CACHE, f))]
    if missing:
        log(f'{CACHE} is missing {missing}; downloading (Yahoo Finance, FRED)...')
        download_regime_data(CACHE)


def load_returns():
    ensure_data()
    px = pd.read_csv(os.path.join(CACHE, 'gspc_long.csv'), index_col=0, parse_dates=True)['Close']
    return (100 * np.log(px).diff()).dropna()


def load_vix():
    return pd.read_csv(os.path.join(CACHE, 'vix_long.csv'), index_col=0, parse_dates=True)['Close']


def load_usrec():
    return pd.read_csv(os.path.join(CACHE, 'usrec.csv'), index_col=0, parse_dates=True)['USREC']


def eras_for(idx):
    """Era masks restricted to the sample; labels show the years actually covered."""
    out = {}
    for a, b in ERAS:
        m = (idx.year >= a) & (idx.year <= b)
        if m.sum() > 300:
            yrs = idx[m].year
            out[f'{yrs.min()}-{yrs.max()}'] = np.asarray(m)
    return out


def span(idx):
    return f'{idx[0].year}-{idx[-1].year}'


# ------------------------------------------------------------------
# Gaussian HMM (= Hamilton Markov-switching mean/variance) helpers
# ------------------------------------------------------------------

def fit_hmm_many(xs, k, n_init=4, seed=0):
    """Fit a k-state Gaussian HMM to each series in xs, batched on the GPU.
    Returns dicts with pi, A, mu, sd (states ordered by volatility), loglik,
    smoothed and filtered state probabilities."""
    xs = [np.asarray(x, float) for x in xs]
    out = []
    for i in range(0, len(xs), CHUNK):
        out += G.fit_hmm_batch(xs[i:i + CHUNK], k, n_init=n_init, seed=seed + i)
    return out


def params_of(o):
    return {key: o[key] for key in ('pi', 'A', 'mu', 'sd')}


def fit_hmm(x, k, n_init=4, seed=0):
    o = fit_hmm_many([x], k, n_init=n_init, seed=seed)[0]
    return params_of(o), o['loglik'], o['smoothed']


def hmm_n_params(k):
    return (k - 1) + k * (k - 1) + 2 * k


def hmm_forward_many(xs, ps, priors=None):
    """Filtered state probabilities and one-step predictive log densities under
    fixed parameters, batched."""
    xs = [np.asarray(x, float) for x in xs]
    out = []
    for i in range(0, len(xs), CHUNK):
        pr = None if priors is None else priors[i:i + CHUNK]
        out += G.hmm_filter_batch(xs[i:i + CHUNK], ps[i:i + CHUNK], pr)
    return out


def hmm_forward(x, p, prior=None):
    """Filtered state probabilities and one-step predictive log density."""
    return hmm_forward_many([x], [p], None if prior is None else [prior])[0]


def simulate_hmm(p, T, rng, dist='normal', df=4):
    K = len(p['mu'])
    s = np.zeros(T, int)
    s[0] = rng.choice(K, p=p['pi'] / p['pi'].sum())
    u = rng.random(T)
    cum = np.cumsum(p['A'], axis=1)
    for t in range(1, T):
        s[t] = min(np.searchsorted(cum[s[t - 1]], u[t]), K - 1)
    if dist == 'normal':
        e = rng.standard_normal(T)
    else:
        e = rng.standard_t(df, T) / np.sqrt(df / (df - 2))
    return p['mu'][s] + p['sd'][s] * e, s


# ------------------------------------------------------------------
# Other identification methods
# ------------------------------------------------------------------

def jm_features(r, n_fit=None):
    """EWMA return / downside-deviation features (causal); standardized with the
    first n_fit rows only, so later rows do not leak into the scaling."""
    r = pd.Series(np.asarray(r, float))
    feats = {}
    for h in (5, 10, 21):
        dd = np.sqrt((r.clip(upper=0) ** 2).ewm(halflife=h).mean())
        feats[f'ret_{h}'] = r.ewm(halflife=h).mean()
        feats[f'logdd_{h}'] = np.log(dd + 0.01)
        feats[f'sortino_{h}'] = feats[f'ret_{h}'] / (dd + 0.01)
    X = pd.DataFrame(feats)
    base = X.iloc[:n_fit] if n_fit else X
    return ((X - base.mean()) / base.std()).clip(-5, 5).values


def high_vol_labels(labels, r):
    """Relabel a 2-state sequence so 1 = the higher-volatility state."""
    labels = np.asarray(labels).astype(int)
    r = np.asarray(r)
    sds = [r[labels == k].std() if (labels == k).any() else 0 for k in (0, 1)]
    return labels if sds[1] >= sds[0] else 1 - labels


def fit_jump_many(rs, lams, n_init=5):
    """Jump models for each (series, penalty) pair, batched on the GPU.
    Returns (ex_post, real_time) label pairs, 1 = higher-volatility state."""
    rs = [np.asarray(r, float) for r in rs]
    out = []
    for i in range(0, len(rs), CHUNK):
        fits = G.fit_jump_batch([jm_features(r) for r in rs[i:i + CHUNK]], lams[i:i + CHUNK], n_init=n_init)
        out += [(high_vol_labels(f['ex_post'], r), high_vol_labels(f['online'], r))
                for f, r in zip(fits, rs[i:i + CHUNK])]
    return out


def fit_jump(r, lam, online=False):
    ex_post, rt = fit_jump_many([r], [lam])[0]
    return (ex_post, rt) if online else ex_post


def changepoint_labels(r, pen_mult=3.0):
    """PELT on returns (Gaussian mean/variance cost), segments clustered by volatility."""
    x = np.asarray(r, float).reshape(-1, 1)
    algo = rpt.Pelt(model='normal', min_size=21, jump=5).fit(x)
    bkps = algo.predict(pen=pen_mult * np.log(len(x)))
    starts = [0] + bkps[:-1]
    seg_sd = np.array([x[a:b].std() for a, b in zip(starts, bkps)])
    km = KMeans(2, n_init=10, random_state=0).fit(np.log(seg_sd).reshape(-1, 1))
    hi = int(np.argmax(km.cluster_centers_.ravel()))
    lab = np.zeros(len(x), int)
    for (a, b), c in zip(zip(starts, bkps), km.labels_):
        lab[a:b] = int(c == hi)
    return lab, len(bkps) - 1


def bear_market_dating(px, thresh=0.20):
    """Lunde-Timmermann style bull/bear dating: 1 = bear (ex post)."""
    px = np.asarray(px, float)
    state = np.zeros(len(px), int)
    regime, ext, ext_i, last_switch = 0, px[0], 0, 0
    for t in range(1, len(px)):
        if regime == 0:
            if px[t] > ext:
                ext, ext_i = px[t], t
            elif px[t] <= ext * (1 - thresh):
                state[last_switch:ext_i + 1] = 0
                regime, last_switch, ext, ext_i = 1, ext_i + 1, px[t], t
        else:
            if px[t] < ext:
                ext, ext_i = px[t], t
            elif px[t] >= ext * (1 + thresh):
                state[last_switch:ext_i + 1] = 1
                regime, last_switch, ext, ext_i = 0, ext_i + 1, px[t], t
    state[last_switch:] = regime
    return state


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def switches(labels):
    return int((np.diff(labels) != 0).sum())


def detection_delay(true, est):
    """Median days from each true entry into state 1 until est is 1 (within the spell)."""
    delays, missed = [], 0
    entries = np.where((true[1:] == 1) & (true[:-1] == 0))[0] + 1
    for e in entries:
        end = e
        while end < len(true) and true[end] == 1:
            end += 1
        hit = np.where(est[e:end] == 1)[0]
        if len(hit):
            delays.append(hit[0])
        else:
            missed += 1
    return (float(np.median(delays)) if delays else np.nan,
            missed / max(len(entries), 1))


def nw_tstat(d, lags=10):
    """Mean of d and its Newey-West standard error."""
    d = np.asarray(d, float)
    m = d.mean()
    e = d - m
    n = len(e)
    v = e @ e / n + 2 * sum((1 - l / (lags + 1)) * (e[l:] @ e[:-l] / n) for l in range(1, lags + 1))
    return m, np.sqrt(v / n)


def lo_mackinlay_z(x, q=2):
    """Heteroskedasticity-robust variance-ratio z statistic (Lo & MacKinlay, 1988)."""
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    d = x - mu
    var1 = d @ d / n
    xq = np.convolve(x, np.ones(q), 'valid')
    varq = ((xq - q * mu) ** 2).sum() / (n * q)
    vr = varq / var1
    theta = 0.0
    denom = (d ** 2).sum() ** 2
    for j in range(1, q):
        delta = n * ((d[j:] ** 2) * (d[:-j] ** 2)).sum() / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    return (vr - 1) / np.sqrt(theta / n) if theta > 0 else np.nan


def ar_diff_test(r, s, mask=None):
    """r_t = a + b r_{t-1} + c s_{t-1} + d s_{t-1} r_{t-1} + e_t with White (HC0) standard errors.
    s is a 0/1 high-volatility label known at t-1 (real-time labels give a no-look-ahead test).
    Returns AR(1) in the low and high state, their difference d and t statistics."""
    r = np.asarray(r, float)
    s = np.asarray(s, float)
    y, x, d = r[1:], r[:-1], s[:-1]
    ok = ~np.isnan(d)
    if mask is not None:
        ok &= mask[1:] & mask[:-1]
    y, x, d = y[ok], x[ok], d[ok]
    nan = dict(ar1_low=np.nan, t_low=np.nan, ar1_high=np.nan, t_high=np.nan, diff=np.nan, t_diff=np.nan,
               n=int(len(y)), share_high=float(d.mean()) if len(d) else np.nan)
    if len(y) < 200 or d.sum() < 50 or (1 - d).sum() < 50:
        return nan
    X = np.column_stack([np.ones_like(x), x, d, d * x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    V = XtX_inv @ (X.T * e ** 2) @ X @ XtX_inv
    se = np.sqrt(np.diag(V))
    se_high = np.sqrt(V[1, 1] + V[3, 3] + 2 * V[1, 3])
    return dict(ar1_low=b[1], t_low=b[1] / se[1], ar1_high=b[1] + b[3], t_high=(b[1] + b[3]) / se_high,
                diff=b[3], t_diff=b[3] / se[3], n=int(len(y)), share_high=float(d.mean()))


def garch_t_fit(y):
    return quiet(arch_model(y, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit, disp='off')


def garch_pit(y):
    """GARCH(1,1)-t standardized residuals mapped to N(0,1) through the fitted t CDF.
    If GARCH-t is the true model these are i.i.d. N(0,1); leftover regimes show up as
    persistent variance shifts that a Gaussian HMM can detect."""
    res = garch_t_fit(y)
    nu = res.params['nu']
    z = np.asarray(res.std_resid) * np.sqrt(nu / (nu - 2))
    u = np.clip(stats.t.cdf(z, nu), 1e-10, 1 - 1e-10)
    return stats.norm.ppf(u)


def bic_table(xs, n_init=3, seed=0):
    """BIC of an i.i.d. Gaussian (K=1) and Gaussian HMMs K=2..4 for each series."""
    fits = {k: fit_hmm_many(xs, k, n_init=n_init, seed=seed) for k in (2, 3, 4)}
    out = []
    for i, y in enumerate(xs):
        T = len(y)
        b = {1: -2 * stats.norm.logpdf(y, y.mean(), y.std()).sum() + hmm_n_params(1) * np.log(T)}
        for k in (2, 3, 4):
            b[k] = -2 * fits[k][i]['loglik'] + hmm_n_params(k) * np.log(T)
        out.append(b)
    return out, fits


def pelt_many(ys):
    if not ys:
        return []
    from joblib import Parallel, delayed
    return [lab for lab, _ in Parallel(n_jobs=-1)(delayed(changepoint_labels)(y) for y in ys)]


def episode_lag(labels, date, horizon=126):
    """Trading days from `date` until the label is 1 (0 = already high-volatility).
    NaN if the date is outside the labelled sample or no switch within `horizon` days."""
    s = labels.dropna()
    d = pd.Timestamp(date)
    if len(s) == 0 or d < s.index[0] or d > s.index[-1]:
        return np.nan
    seg = s.loc[d:].values[:horizon]
    hit = np.where(seg == 1)[0]
    return int(hit[0]) if len(hit) else np.nan


def kappa(a, b):
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        return np.nan
    return cohen_kappa_score(a, b)


def save(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)
    return df


# ------------------------------------------------------------------
# Base fits on the real data (used by Parts A and B)
# ------------------------------------------------------------------

def base_fits(r):
    x = r.values
    split = r.index.searchsorted(pd.Timestamp(OOS_START))
    log('Fitting HMMs K=2..4 and GARCH on the full sample and the pre-2000 training sample...')
    fitted = {}
    for k in (2, 3, 4):
        of, otr = fit_hmm_many([x, x[:split]], k, n_init=5)
        fitted[k] = dict(full=of, train=otr)
    return dict(split=split, fitted=fitted, garch=garch_t_fit(x),
                p2=params_of(fitted[2]['full']), p3=params_of(fitted[3]['full']))


# ------------------------------------------------------------------
# Part A: simulation
# ------------------------------------------------------------------

def simulate_dgp(dgp, rep, T, base):
    rng = np.random.default_rng(1000 * DGPS.index(dgp) + rep)
    if dgp == DGPS[0]:
        y, s = simulate_hmm(base['p2'], T, rng)
        return y, s, None
    if dgp == DGPS[1]:
        y, s = simulate_hmm(base['p2'], T, rng, dist='t')
        return y, s, None
    if dgp == DGPS[2]:
        y, s3 = simulate_hmm(base['p3'], T, rng)
        return y, (s3 == 2).astype(int), s3
    am = ConstantMean(None, volatility=GARCHVol(p=1, q=1), distribution=StudentsT(seed=rng))
    sim = am.simulate(base['garch'].params.values, T, burn=500)
    return sim['data'].values, None, None


def part_a(r, base):
    log(f'=== Part A: simulation ({N_SIM} replications per DGP, PELT on {min(N_PELT, N_SIM)}) ===')
    T = len(r)
    ck_dir = os.path.join(OUT, '_checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    acc, ksel, agree, placebo = [], [], [], []
    for di, dgp in enumerate(DGPS):
        ck = os.path.join(ck_dir, f'partA_{di}_n{N_SIM}_p{N_PELT}_T{T}.pkl')
        if os.path.exists(ck):
            with open(ck, 'rb') as f:
                part = pickle.load(f)
            log(f'  {dgp}: loaded from checkpoint')
        else:
            t0 = time.time()
            sims = [simulate_dgp(dgp, rep, T, base) for rep in range(N_SIM)]
            ys = [y for y, _, _ in sims]
            bics, fits = bic_table(ys, seed=1000 * di)
            log(f'  {dgp}: HMM fits done ({time.time() - t0:.0f}s)')
            jm = fit_jump_many([y for y in ys for _ in JM_LAMBDAS], [lam for _ in ys for lam in JM_LAMBDAS])
            log(f'  {dgp}: jump models done ({time.time() - t0:.0f}s)')
            pit_bics, _ = bic_table([garch_pit(y) for y in ys], seed=1000 * di + 500)
            log(f'  {dgp}: GARCH-residual BIC done ({time.time() - t0:.0f}s)')
            cps = pelt_many(ys[:min(N_PELT, N_SIM)])
            log(f'  {dgp}: PELT done ({time.time() - t0:.0f}s)')
            part = dict(acc=[], ksel=[], agree=[], placebo=[])
            for rep, (y, s, s3) in enumerate(sims):
                part['ksel'].append({'dgp': dgp, 'rep': rep,
                                     'bic_k_raw_returns': min(bics[rep], key=bics[rep].get),
                                     'bic_k_garch_residuals': min(pit_bics[rep], key=pit_bics[rep].get)})
                f2 = fits[2][rep]
                ex_post = (f2['smoothed'][:, 1] > 0.5).astype(int)
                real_time = (f2['filtered'][:, 1] > 0.5).astype(int)
                jl = {lam: jm[rep * len(JM_LAMBDAS) + j] for j, lam in enumerate(JM_LAMBDAS)}
                ex = {'HMM-2': ex_post, **{f'Jump λ={lam:g}': jl[lam][0] for lam in JM_LAMBDAS}}
                if rep < len(cps):
                    ex['Change-point (PELT)'] = cps[rep]
                for i, a in enumerate(AGREE_METHODS):
                    for b in AGREE_METHODS[i + 1:]:
                        if a in ex and b in ex:
                            part['agree'].append({'dgp': dgp, 'rep': rep, 'pair': f'{a} vs {b}', 'kappa': kappa(ex[a], ex[b])})
                for name, lab in [('HMM-2 real time', real_time), ('Jump λ=30 real time', jl[30.0][1])]:
                    t = ar_diff_test(y, lab)
                    part['placebo'].append({'dgp': dgp, 'rep': rep, 'labels': name, 't_diff': t['t_diff']})
                if s is None:
                    continue
                methods = {'HMM-2 ex post (smoothed)': ex_post, 'HMM-2 real time (filtered)': real_time}
                for lam in JM_LAMBDAS:
                    methods[f'Jump λ={lam:g} ex post'] = jl[lam][0]
                    methods[f'Jump λ={lam:g} real time'] = jl[lam][1]
                if 'Change-point (PELT)' in ex:
                    methods['Change-point (PELT) ex post'] = ex['Change-point (PELT)']
                if s3 is not None:
                    sm3 = fits[3][rep]['smoothed'].argmax(1)
                    part['acc'].append({'dgp': dgp, 'rep': rep, 'method': 'HMM-3 ex post, 3-class',
                                        'balanced_accuracy': balanced_accuracy_score(s3, sm3),
                                        'kappa': cohen_kappa_score(s3, sm3), 'median_delay_days': np.nan,
                                        'missed_episodes': np.nan,
                                        'switch_ratio': switches(sm3) / max(switches(s3), 1)})
                for name, est in methods.items():
                    dl, miss = detection_delay(s, est)
                    part['acc'].append({'dgp': dgp, 'rep': rep, 'method': name,
                                        'balanced_accuracy': balanced_accuracy_score(s, est),
                                        'kappa': cohen_kappa_score(s, est), 'median_delay_days': dl,
                                        'missed_episodes': miss, 'switch_ratio': switches(est) / max(switches(s), 1)})
            with open(ck, 'wb') as f:
                pickle.dump(part, f)
            log(f'  {dgp}: done in {time.time() - t0:.0f}s')
        acc += part['acc']
        ksel += part['ksel']
        agree += part['agree']
        placebo += part['placebo']

    acc = save(pd.DataFrame(acc), 'A1_sim_accuracy_by_rep.csv')
    summ = acc.groupby(['dgp', 'method']).agg(
        n_reps=('balanced_accuracy', 'size'),
        balanced_accuracy=('balanced_accuracy', 'mean'), ba_sd=('balanced_accuracy', 'std'),
        kappa=('kappa', 'mean'), median_delay_days=('median_delay_days', 'median'),
        missed_episodes=('missed_episodes', 'mean'), switch_ratio=('switch_ratio', 'mean')).round(3).reset_index()
    save(summ, 'A1_sim_accuracy_summary.csv')

    ksel = save(pd.DataFrame(ksel), 'A2_bic_selected_k_by_rep.csv')
    k_raw = pd.crosstab(ksel.dgp, ksel.bic_k_raw_returns).reindex(columns=[1, 2, 3, 4], fill_value=0)
    k_pit = pd.crosstab(ksel.dgp, ksel.bic_k_garch_residuals).reindex(columns=[1, 2, 3, 4], fill_value=0)
    k_raw.columns = [f'K={c}' for c in k_raw.columns]
    k_pit.columns = [f'K={c}' for c in k_pit.columns]
    rates = pd.DataFrame({
        'share_K>=2_raw_returns': (ksel.bic_k_raw_returns >= 2).groupby(ksel.dgp).mean(),
        'share_K>=2_garch_residuals': (ksel.bic_k_garch_residuals >= 2).groupby(ksel.dgp).mean()}).round(3)
    save(k_raw.reset_index(), 'A2_bic_k_raw_returns.csv')
    save(k_pit.reset_index(), 'A2_bic_k_garch_residuals.csv')
    save(rates.reset_index(), 'A2_regime_detection_rates.csv')

    agree = save(pd.DataFrame(agree), 'A3_method_agreement_by_rep.csv')
    agree_null = agree.groupby(['dgp', 'pair']).kappa.agg(
        n_reps='size', mean='mean', p05=lambda v: v.quantile(0.05), p95=lambda v: v.quantile(0.95)).round(3).reset_index()
    save(agree_null, 'A3_method_agreement_null.csv')

    placebo = save(pd.DataFrame(placebo), 'A4_ar_difference_placebo_by_rep.csv')
    plac = placebo.groupby(['dgp', 'labels']).t_diff.agg(
        n_reps='size', rejection_rate_5pct=lambda v: (v.abs() > 1.96).mean(),
        mean_t='mean').round(3).reset_index()
    save(plac, 'A4_ar_difference_placebo.csv')

    print('\nSimulation accuracy (mean over replications):\n', summ.to_string(index=False))
    print('\nBIC-selected number of regimes, raw returns:\n', k_raw.to_string())
    print('\nBIC-selected number of regimes, GARCH-t PIT residuals:\n', k_pit.to_string())
    print('\nShare of replications where BIC finds regimes (K >= 2):\n', rates.to_string())
    print('\nRegime-dependent AR(1) test, false-positive rate (no true predictability in any DGP):\n', plac.to_string(index=False))
    return dict(acc=summ, k_raw=k_raw, k_pit=k_pit, rates=rates, agree=agree, agree_null=agree_null, placebo=plac)


# ------------------------------------------------------------------
# Part B: real data
# ------------------------------------------------------------------

def part_b(r, base):
    log(f'=== Part B: S&P 500, {span(r.index)} ===')
    x, idx, T = r.values, r.index, len(r)
    split = base['split']
    xtr, xte = x[:split], x[split:]
    out = {}

    # B1. Model comparison: BIC and out-of-sample density forecasts
    rows = []
    for k in (1, 2, 3, 4):
        if k == 1:
            ll_full = stats.norm.logpdf(x, x.mean(), x.std()).sum()
            lpd = stats.norm.logpdf(xte, xtr.mean(), xtr.std())
        else:
            ll_full = base['fitted'][k]['full']['loglik']
            otr = base['fitted'][k]['train']
            ptr = params_of(otr)
            prior = otr['filtered'][-1] @ ptr['A']
            _, lpd = hmm_forward(xte, ptr, prior=prior)
        rows.append({'model': f'Gaussian HMM, K={k}' if k > 1 else 'Gaussian i.i.d. (K=1)', 'n_params': hmm_n_params(k),
                     'bic_full_sample': -2 * ll_full + hmm_n_params(k) * np.log(T),
                     'oos_mean_log_score': lpd.mean(), '_lpd': lpd})
    for dist in ('normal', 't'):
        fr = quiet(arch_model(xtr, mean='Constant', vol='GARCH', p=1, q=1, dist=dist).fit, disp='off')
        full = arch_model(x, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
        sig = full.fix(fr.params.values).conditional_volatility[split:]
        mu = fr.params['mu']
        if dist == 'normal':
            lpd = stats.norm.logpdf(xte, mu, sig)
        else:
            nu = fr.params['nu']
            lpd = stats.t.logpdf(xte, nu, mu, sig * np.sqrt((nu - 2) / nu))
        rows.append({'model': f'GARCH(1,1)-{dist} (no regimes)', 'n_params': len(fr.params),
                     'bic_full_sample': quiet(full.fit, disp='off').bic, 'oos_mean_log_score': lpd.mean(), '_lpd': lpd})
    ref = rows[-1]['_lpd']
    for rw in rows:
        m_, se = nw_tstat(rw['_lpd'] - ref)
        rw['oos_diff_vs_garch_t'] = m_
        rw['dm_t_vs_garch_t'] = m_ / se if se > 0 else np.nan
    mc = pd.DataFrame(rows).drop(columns='_lpd').round(4)
    save(mc, 'B1_model_comparison.csv')
    best_hmm = mc[mc.model.str.contains('HMM')].sort_values('oos_mean_log_score').iloc[-1]
    out['model_comparison'] = mc
    out['best_hmm'] = best_hmm
    out['oos_period'] = f'{idx[split].year}-{idx[-1].year}'
    print(f'\nModel comparison (BIC on {span(idx)}; out-of-sample log score {out["oos_period"]}, '
          f'DM t > 0 means better than GARCH-t):\n', mc.to_string(index=False))

    # B2. Parameters
    par_rows = []
    for k in (2, 3):
        o = base['fitted'][k]['full']
        lab = o['smoothed'].argmax(1)
        for j in range(k):
            par_rows.append({'K': k, 'state': j, 'ann_mean_return_pct': o['mu'][j] * 252,
                             'ann_vol_pct': o['sd'][j] * np.sqrt(252), 'expected_duration_days': 1 / (1 - o['A'][j, j]),
                             'share_of_days_pct': (lab == j).mean() * 100})
    out['params'] = save(pd.DataFrame(par_rows).round(2), 'B2_hmm_parameters.csv')

    # B3. Ex-post labels from every method and their agreement
    labels = pd.DataFrame(index=idx)
    sm2 = base['fitted'][2]['full']['smoothed']
    labels['HMM-2'] = (sm2[:, 1] > 0.5).astype(int)
    labels['HMM-3 top state'] = (base['fitted'][3]['full']['smoothed'].argmax(1) == 2).astype(int)
    for lam, (ep, _) in zip(JM_LAMBDAS, fit_jump_many([x] * len(JM_LAMBDAS), list(JM_LAMBDAS))):
        labels[f'Jump λ={lam:g}'] = ep
    log('  PELT on the real series...')
    cp, n_bk = changepoint_labels(x)
    labels['Change-point (PELT)'] = cp
    labels['Bear market (20% rule)'] = bear_market_dating(np.exp(np.cumsum(x / 100)))
    labels['NBER recession'] = (load_usrec().reindex(idx, method='ffill') == 1).astype(int)
    vix = load_vix().reindex(idx)
    vix_name = f'VIX ≥ 20 ({vix.first_valid_index().year}+)'
    labels[vix_name] = (vix >= 20).astype(float).where(vix.notna())
    save(labels, 'B3_labels_ex_post.csv', index=True)
    cols = labels.columns
    kap = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            if a == b:          # labels[[a, a]] would duplicate the column
                kap.loc[a, b] = 1.0
                continue
            d = labels[[a, b]].dropna()
            kap.loc[a, b] = kappa(d[a], d[b]) if len(d) > 1 else np.nan
    out['kappa'] = save(kap.round(2), 'B3_agreement_kappa.csv', index=True)
    desc = pd.DataFrame({c: {'share_high_pct': labels[c].mean() * 100,
                             'n_switches': switches(labels[c].dropna().values.astype(int)),
                             'mean_spell_days': labels[c].notna().sum() / (switches(labels[c].dropna().values.astype(int)) + 1)}
                         for c in cols}).T.round(1)
    out['label_desc'] = save(desc, 'B3_label_descriptives.csv', index=True)
    out['pelt_breakpoints'] = n_bk
    out['labels'] = labels
    print('\nCohen kappa between methods (ex post):\n', out['kappa'].to_string())

    # B4. Real time (expanding-window refits, one per year) vs ex post
    first = max(B4_FIRST_YEAR, idx[0].year + 10)
    years = [y for y in range(first, idx[-1].year + 1)
             if ((idx >= pd.Timestamp(f'{y}-01-01')) & (idx < pd.Timestamp(f'{y + 1}-01-01'))).any()]
    log(f'  real-time refits: {len(years)} expanding windows ({years[0]}-{years[-1]})')
    n_tr = [int((idx < pd.Timestamp(f'{y}-01-01')).sum()) for y in years]
    n_end = [int((idx < pd.Timestamp(f'{y + 1}-01-01')).sum()) for y in years]
    fits = fit_hmm_many([x[:a] for a in n_tr], 2, n_init=3, seed=first)
    filt = hmm_forward_many([x[:b] for b in n_end], [params_of(f) for f in fits])
    rt_hmm = pd.Series(np.nan, index=idx)
    for a, b, (fa, _) in zip(n_tr, n_end, filt):
        rt_hmm.iloc[a:b] = (fa[a:b, 1] > 0.5).astype(int)
    rt_jm = pd.Series(np.nan, index=idx)
    for i in range(0, len(years), CHUNK):
        tr_c, end_c = n_tr[i:i + CHUNK], n_end[i:i + CHUNK]
        feats = [jm_features(x[:b], n_fit=a) for a, b in zip(tr_c, end_c)]
        for a, b, f in zip(tr_c, end_c, G.fit_jump_batch(feats, [30.0] * len(feats), n_init=3, n_fit=tr_c)):
            ref = f['ex_post']
            flip = x[:a][ref == 1].std() < x[:a][ref == 0].std()
            on = f['online'][a:b]
            rt_jm.iloc[a:b] = (1 - on) if flip else on
    rt = pd.DataFrame({'HMM-2 real time': rt_hmm, 'Jump λ=30 real time': rt_jm})
    save(rt, 'B4_labels_real_time.csv', index=True)
    pairs = [('HMM-2 real time', 'HMM-2'), ('Jump λ=30 real time', 'Jump λ=30')]
    rt_rows, ep_rows = [], []
    for c, refc in pairs:
        d = pd.concat([rt[c], labels[refc]], axis=1).dropna().astype(int)
        nyears = len(d) / 252
        rt_rows.append({'real_time_method': c, 'ex_post_reference': refc, 'period': span(d.index),
                        'agreement_pct': (d[c] == d[refc]).mean() * 100, 'kappa': kappa(d[c], d[refc]),
                        'real_time_switches_per_year': switches(d[c].values) / nyears,
                        'ex_post_switches_per_year': switches(d[refc].values) / nyears})
        for name, date in EPISODES.items():
            ep_rows.append({'method': refc, 'episode': name, 'date': date,
                            'ex_post_lag_days': episode_lag(labels[refc].where(rt[c].notna()), date),
                            'real_time_lag_days': episode_lag(rt[c], date)})
    out['rt'] = save(pd.DataFrame(rt_rows).round(3), 'B4_real_time_vs_ex_post.csv')
    out['episodes'] = save(pd.DataFrame(ep_rows), 'B4_episode_detection_lags.csv')
    out['rt_labels'] = rt
    print('\nReal time vs ex post:\n', out['rt'].to_string(index=False))
    print('\nDays from episode date until the label shows high volatility (0 = already high):\n',
          out['episodes'].to_string(index=False))

    # B5. Stability: parameters estimated on each half
    split_date = pd.Timestamp(STABILITY_SPLIT)
    mid = idx.searchsorted(split_date) if idx[0] + pd.Timedelta(days=3650) < split_date < idx[-1] - pd.Timedelta(days=3650) else T // 2
    pa, pb = [params_of(o) for o in fit_hmm_many([x[:mid], x[mid:]], 2, n_init=5)]
    filt_cross, _ = hmm_forward(x[mid:], pa)
    p2 = base['p2']
    stab = pd.DataFrame([
        {'sample': span(idx[:mid]), **_stab(pa)}, {'sample': span(idx[mid:]), **_stab(pb)},
        {'sample': f'full {span(idx)}', **_stab(p2)}]).round(2)
    out['stability'] = save(stab, 'B5_stability.csv')
    out['kappa_second_half_first_half_params_vs_full'] = round(
        kappa(labels['HMM-2'].values[mid:], (filt_cross[:, 1] > 0.5).astype(int)), 3)
    print('\nStability:\n', stab.to_string(index=False))

    # B6. Regimes beyond GARCH: HMM on GARCH-t PIT residuals of the real series
    pit = garch_pit(x)
    bics, _ = bic_table([pit], n_init=5)
    b6 = pd.DataFrame([{'K': k, 'bic_garch_t_pit_residuals': v} for k, v in bics[0].items()]).round(2)
    out['pit_bic'] = save(b6, 'B6_bic_garch_residuals.csv')
    out['pit_k'] = int(b6.sort_values('bic_garch_t_pit_residuals').K.iloc[0])
    raw_k = min({1: mc.bic_full_sample.iloc[0], **{k: mc.bic_full_sample.iloc[k - 1] for k in (2, 3, 4)}}.items(),
                key=lambda kv: kv[1])[0]
    out['raw_k'] = int(raw_k)
    print(f'\nBIC-selected K: raw returns {raw_k}; GARCH-t PIT residuals {out["pit_k"]}\n', b6.to_string(index=False))
    return out


def _stab(p):
    return {'low_vol_ann_pct': p['sd'][0] * np.sqrt(252), 'high_vol_ann_pct': p['sd'][1] * np.sqrt(252),
            'low_dur_days': 1 / (1 - p['A'][0, 0]), 'high_dur_days': 1 / (1 - p['A'][1, 1])}


# ------------------------------------------------------------------
# Part C: efficiency by regime
# ------------------------------------------------------------------

def part_c(r, base, B):
    log('=== Part C: efficiency by regime ===')
    x, idx = r.values, r.index
    rows = []
    sets = [('HMM-2 real time', 'real time', B['rt_labels']['HMM-2 real time']),
            ('Jump λ=30 real time', 'real time', B['rt_labels']['Jump λ=30 real time']),
            ('HMM-2', 'ex post (look-ahead)', B['labels']['HMM-2']),
            ('Jump λ=30', 'ex post (look-ahead)', B['labels']['Jump λ=30']),
            ('Bear market (20% rule)', 'ex post (look-ahead)', B['labels']['Bear market (20% rule)'])]
    for name, kind, lab in sets:
        lab = lab.reindex(idx)
        avail = lab.notna().values
        periods = {f'all ({span(idx[avail])})': avail}
        for v in eras_for(idx).values():
            m = v & avail
            if m.sum() > 300:     # label with the years that have labels (real-time labels start later)
                periods[span(idx[m])] = m
        for per, m in periods.items():
            rows.append({'labels': name, 'type': kind, 'period': per, **ar_diff_test(x, lab.values, m)})
    eff = save(pd.DataFrame(rows).round(3), 'C1_ar1_by_regime.csv')
    race = save(volatility_horse_race(r, B['rt_labels']), 'C3_regime_vs_volatility.csv')
    print('\nRegime label vs continuous volatility (added after the first full run):\n', race.to_string(index=False))
    print('\nAR(1) by regime, r_t on r_{t-1} interacted with the label at t-1 (White t):\n', eff.to_string(index=False))

    def rejection_rate(y, w=252, step=21):
        zs = np.array([lo_mackinlay_z(y[i:i + w], 2) for i in range(0, len(y) - w, step)])
        return (np.abs(zs) > 1.96).mean() * 100

    rng = np.random.default_rng(7)
    n_rep = 5 if QUICK else 20
    sims = {'HMM-2 (regimes, no predictability)': [rejection_rate(simulate_hmm(base['p2'], len(x), rng)[0]) for _ in range(n_rep)],
            'HMM-3 (regimes, no predictability)': [rejection_rate(simulate_hmm(base['p3'], len(x), rng)[0]) for _ in range(n_rep)],
            'GARCH(1,1)-t (no regimes, no predictability)': [rejection_rate(simulate_dgp(NO_REGIME_DGP, 10_000 + i, len(x), base)[0]) for i in range(n_rep)]}
    art = [{'series': f'Real S&P 500, {span(idx)}', 'pct_windows_rejecting': rejection_rate(x), 'sim_p95': np.nan}]
    for k, m in eras_for(idx).items():
        art.append({'series': f'Real S&P 500, {k}', 'pct_windows_rejecting': rejection_rate(x[m]), 'sim_p95': np.nan})
    for k, v in sims.items():
        art.append({'series': f'Simulated {k}, mean of {n_rep}', 'pct_windows_rejecting': np.mean(v), 'sim_p95': np.percentile(v, 95)})
    art.append({'series': 'Nominal size of the test', 'pct_windows_rejecting': 5.0, 'sim_p95': np.nan})
    art = save(pd.DataFrame(art).round(1), 'C2_rolling_vr_rejections.csv')
    print('\nRolling 1-year Lo-MacKinlay VR(2) tests, % of windows rejecting the random walk at 5%:\n', art.to_string(index=False))
    return dict(eff=eff, vr=art, race=race)


def volatility_horse_race(r, labels_rt):
    """Does the real-time regime label explain the AR(1) change beyond the volatility level?
    r_t on r_{t-1} interacted with (a) the regime label at t-1, (b) standardized log RiskMetrics
    EWMA volatility at t-1 (lambda = 0.94, known at t-1), (c) both. White (HC0) t statistics.
    Added after the first full run; not one of the pre-specified scorecard criteria."""
    sig = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean())

    def ols(y, X):
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ y
        e = y - X @ b
        V = XtXi @ (X.T * e ** 2) @ X @ XtXi
        return b, b / np.sqrt(np.diag(V))

    rows = []
    for name in labels_rt.columns:
        lab = labels_rt[name]
        start, end = lab.first_valid_index().year, r.index[-1].year
        periods = {f'{start}-{end}': (start, end)}
        if start < 2000 <= end:
            periods.update({f'{start}-1999': (start, 1999), f'2000-{end}': (2000, end)})
        for per, (a, b_) in periods.items():
            d = pd.DataFrame({'y': r, 'x': r.shift(1), 's': lab.shift(1),
                              'lv': np.log(sig).shift(1)}).loc[str(a):str(b_)].dropna()
            d['lv'] = (d.lv - d.lv.mean()) / d.lv.std()
            y, one = d.y.values, np.ones(len(d))
            b1, t1 = ols(y, np.column_stack([one, d.x, d.s, d.s * d.x]))
            b2, t2 = ols(y, np.column_stack([one, d.x, d.lv, d.lv * d.x]))
            b3, t3 = ols(y, np.column_stack([one, d.x, d.s, d.s * d.x, d.lv, d.lv * d.x]))
            rows.append({'labels': name, 'period': per, 'n': len(d),
                         'regime_only_coef': b1[3], 'regime_only_t': t1[3],
                         'vol_only_coef': b2[3], 'vol_only_t': t2[3],
                         'both_regime_coef': b3[3], 'both_regime_t': t3[3],
                         'both_vol_coef': b3[5], 'both_vol_t': t3[5]})
    return pd.DataFrame(rows).round(3)


# ------------------------------------------------------------------
# Scorecard (criteria fixed before the full run)
# ------------------------------------------------------------------

CRITERIA_NOTE = (
    'Criteria were fixed before the full run (after a 1990-2025 smoke test with 2 replications). '
    'They test (i) whether the detectors recover regimes that exist, (ii) whether the regime tests '
    'stay quiet when there are no regimes, (iii) whether the S&P 500 shows regimes beyond GARCH '
    'volatility clustering, (iv) whether real-time labels are usable, and (v) whether return '
    'autocorrelation differs by regime without look-ahead.')


def scorecard(A, B, C):
    rows = []

    def add(group, criterion, value, threshold, passed):
        rows.append({'group': group, 'criterion': criterion, 'value': value, 'threshold': threshold,
                     'result': 'PASS' if passed else 'FAIL'})

    acc = A['acc'].set_index(['dgp', 'method'])
    for m in ['HMM-2 real time (filtered)', 'Jump λ=30 real time', 'HMM-2 ex post (smoothed)', 'Jump λ=30 ex post']:
        v = acc.loc[(DGPS[0], m), 'balanced_accuracy']
        add('1 Detector accuracy (sim, true 2-regime data)', f'{m}: balanced accuracy', round(v, 3), '>= 0.80', v >= 0.80)
    rates = A['rates']
    v = rates.loc[NO_REGIME_DGP, 'share_K>=2_raw_returns']
    add('2 No false regimes', 'BIC on raw returns finds K>=2 in GARCH-t data (no regimes)', v, '<= 0.05', v <= 0.05)
    size = rates.loc[NO_REGIME_DGP, 'share_K>=2_garch_residuals']
    add('2 No false regimes', 'BIC on GARCH-t residuals finds K>=2 in GARCH-t data (size)', size, '<= 0.10', size <= 0.10)
    power = rates.loc[DGPS[0], 'share_K>=2_garch_residuals']
    add('2 No false regimes', 'BIC on GARCH-t residuals finds K>=2 in true 2-regime data (power)', power, '>= 0.80', power >= 0.80)
    valid = size <= 0.10 and power >= 0.80
    rows.append({'group': '3 Regimes in the S&P 500', 'criterion': 'BIC on GARCH-t residuals of the S&P 500 selects K>=2',
                 'value': f'K = {B["pit_k"]}', 'threshold': 'K >= 2 (valid only if size and power pass)',
                 'result': ('PASS' if B['pit_k'] >= 2 else 'FAIL') if valid else 'INCONCLUSIVE (test lacks size or power)'})
    bh = B['best_hmm']
    add('3 Regimes in the S&P 500', f'Best HMM ({bh.model}) beats GARCH-t out of sample (DM t)',
        round(bh.dm_t_vs_garch_t, 2), '>= 1.96', bh.dm_t_vs_garch_t >= 1.96)
    null = A['agree_null'][A['agree_null'].dgp == NO_REGIME_DGP].set_index('pair')
    for a, b in [('HMM-2', 'Jump λ=30'), ('HMM-2', 'Change-point (PELT)'), ('Jump λ=30', 'Change-point (PELT)')]:
        pair = f'{a} vs {b}'
        real = B['kappa'].loc[a, b]
        p95 = null.loc[pair, 'p95'] if pair in null.index else np.nan
        add('3 Regimes in the S&P 500', f'Agreement {pair}: real kappa vs no-regime (GARCH-t) 95th pct',
            f'{real:.2f} vs {p95:.2f}', 'real > null p95', real > p95)
    for _, rw in B['rt'].iterrows():
        add('4 Real-time usability', f'{rw.real_time_method} vs ex post: kappa ({rw.period})', round(rw.kappa, 3),
            '>= 0.70', rw.kappa >= 0.70)
    eff = C['eff']
    plac = A['placebo'].set_index(['dgp', 'labels'])
    for lab in ['HMM-2 real time', 'Jump λ=30 real time']:
        e = eff[(eff['labels'] == lab) & eff.period.str.startswith('all')].iloc[0]
        fp = plac.loc[(NO_REGIME_DGP, lab), 'rejection_rate_5pct']
        add('5 Efficiency differs by regime', f'{lab}: test false-positive rate in GARCH-t data', fp, '<= 0.10', fp <= 0.10)
        add('5 Efficiency differs by regime', f'{lab}: AR(1) high minus low ({e.period}), White t',
            f'{e["diff"]:+.3f} (t = {e.t_diff:.2f})', '|t| >= 1.96 (valid only if false-positive rate passes)',
            abs(e.t_diff) >= 1.96)
    sc = pd.DataFrame(rows)
    save(sc, 'S_scorecard.csv')
    print('\nScorecard:\n', sc.to_string(index=False))
    return sc


# ------------------------------------------------------------------
# Figures and report
# ------------------------------------------------------------------

def make_figures(r, A, B):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    idx = r.index
    lab = B['labels']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1.4]})
    ax1.plot(idx, np.cumsum(r.values) / 100, color='black', lw=0.7)
    hv = lab['HMM-2'].values == 1
    ax1.fill_between(idx, 0, 1, where=hv, transform=ax1.get_xaxis_transform(), color='tab:red', alpha=0.2, lw=0,
                     label='HMM-2 high-volatility state (ex post)')
    ax1.set_ylabel('cumulative log return')
    ax1.legend(loc='upper left', frameon=False)
    ax1.set_title(f'S&P 500 {span(idx)} and identified high-volatility regimes')
    bands = [('HMM-2 ex post', lab['HMM-2']), ('HMM-2 real time', B['rt_labels']['HMM-2 real time']),
             ('Jump λ=30 ex post', lab['Jump λ=30']), ('Jump λ=30 real time', B['rt_labels']['Jump λ=30 real time']),
             ('PELT', lab['Change-point (PELT)']), ('NBER recession', lab['NBER recession'])]
    for i, (name, s) in enumerate(bands):
        v = s.reindex(idx).values
        ax2.fill_between(idx, i, i + 0.8, where=v == 1, color='tab:red', lw=0)
        ax2.fill_between(idx, i, i + 0.8, where=np.isnan(v), color='lightgrey', lw=0)
    ax2.set_yticks(np.arange(len(bands)) + 0.4)
    ax2.set_yticklabels([b[0] for b in bands], fontsize=8)
    ax2.set_ylim(0, len(bands))
    fig.tight_layout()
    files.append('F1_regimes_timeline.png')
    fig.savefig(os.path.join(OUT, files[-1]), dpi=200)
    plt.close(fig)

    acc = A['acc']
    meth = ['HMM-2 ex post (smoothed)', 'HMM-2 real time (filtered)', 'Jump λ=30 ex post', 'Jump λ=30 real time',
            'Change-point (PELT) ex post']
    dg = [d for d in DGPS if d != NO_REGIME_DGP]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(meth)
    for j, m in enumerate(meth):
        sub = acc[acc.method == m].set_index('dgp').reindex(dg)
        ax.bar(np.arange(len(dg)) + j * w, sub.balanced_accuracy, w, yerr=sub.ba_sd, label=m, capsize=2)
    ax.axhline(0.8, color='grey', ls='--', lw=0.8)
    ax.set_xticks(np.arange(len(dg)) + 0.4 - w / 2)
    ax.set_xticklabels(dg, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('balanced accuracy (high-vol state)')
    ax.set_title('Simulation: recovery of known regimes (mean ± sd over replications)')
    ax.legend(fontsize=7, ncol=3, frameon=False, loc='lower left')
    fig.tight_layout()
    files.append('F2_simulation_accuracy.png')
    fig.savefig(os.path.join(OUT, files[-1]), dpi=200)
    plt.close(fig)

    ag = A['agree']
    pairs = [('HMM-2', 'Jump λ=30'), ('HMM-2', 'Change-point (PELT)'), ('Jump λ=30', 'Change-point (PELT)')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (a, b) in zip(axes, pairs):
        pair = f'{a} vs {b}'
        for d, c in [(NO_REGIME_DGP, 'tab:grey'), (DGPS[0], 'tab:blue')]:
            v = ag[(ag.dgp == d) & (ag.pair == pair)].kappa.dropna()
            if len(v):
                ax.hist(v, bins=15, alpha=0.6, color=c, label=d.split(' (')[0] + (' (no regimes)' if d == NO_REGIME_DGP else ' (regimes)'))
        ax.axvline(B['kappa'].loc[a, b], color='tab:red', lw=2, label='S&P 500')
        ax.set_title(pair, fontsize=9)
        ax.set_xlabel('Cohen kappa')
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Agreement between methods: real data vs simulated data with and without regimes', fontsize=10)
    fig.tight_layout()
    files.append('F3_agreement_vs_null.png')
    fig.savefig(os.path.join(OUT, files[-1]), dpi=200)
    plt.close(fig)
    return files


def md_table(df, index=False):
    d = df.reset_index() if index else df.copy()
    d = d.astype(object).where(pd.notna(d), '')
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    sep = '|' + '---|' * len(d.columns)
    body = ['| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in row) + ' |' for row in d.values]
    return '\n'.join([head, sep] + body)


def write_report(r, info, validation, A, B, C, sc, figs):
    L = [f'# Regime identification results', '',
         f'- Run finished: {dt.datetime.now().isoformat(timespec="seconds")} (runtime {info["runtime_min"]:.0f} min)',
         f'- Device: {info["device"]}',
         f'- Data: S&P 500 (^GSPC) daily log returns {r.index[0].date()} to {r.index[-1].date()}, n = {len(r)}; '
         f'VIX and NBER USREC for comparison. {info["snapshot"]}',
         f'- Simulation: {N_SIM} replications per DGP, T = {len(r)}; PELT on the first {min(N_PELT, N_SIM)} replications',
         f'- Mode: {"QUICK smoke test - not for the paper" if QUICK else "full run"}', '']
    if validation is not None:
        L += ['## 0. GPU implementation vs reference packages (hmmlearn, jumpmodels)', '',
              md_table(pd.DataFrame(validation)), '']
    L += ['## Scorecard', '', CRITERIA_NOTE, '', md_table(sc), '',
          '## A. Simulation', '',
          '### A1. Accuracy of each method when the true regimes are known', '',
          'Balanced accuracy and kappa for the high-volatility state; delay = median trading days from a true '
          'switch into the high state until detection; switch_ratio = detected / true number of switches.', '',
          md_table(A['acc']), '',
          '### A2. How often BIC finds regimes', '',
          'Raw returns:', '', md_table(A['k_raw'], index=True), '',
          'GARCH(1,1)-t PIT residuals (regimes beyond volatility clustering):', '', md_table(A['k_pit'], index=True), '',
          md_table(A['rates'], index=True), '',
          '### A3. Agreement between methods in simulated data (null distribution for Part B)', '',
          md_table(A['agree_null']), '',
          '### A4. Regime-dependent AR(1) test: false-positive rate (no DGP has return predictability)', '',
          md_table(A['placebo']), '',
          f'## B. S&P 500, {span(r.index)}', '',
          f'### B1. Model comparison (BIC full sample; out-of-sample mean log score {B["oos_period"]}, '
          'parameters estimated before the out-of-sample period; DM t > 0 favours the row over GARCH-t)', '',
          md_table(B['model_comparison']), '',
          '### B2. HMM parameters', '', md_table(B['params']), '',
          '### B3. Agreement between methods (Cohen kappa, ex post)', '', md_table(B['kappa'], index=True), '',
          md_table(B['label_desc'], index=True), '',
          '### B4. Real-time (expanding window, refit yearly) vs ex-post labels', '', md_table(B['rt']), '',
          md_table(B['episodes']), '',
          '### B5. Parameter stability', '', md_table(B['stability']), '',
          f'Kappa between full-sample labels and second-half labels filtered with first-half parameters: '
          f'{B["kappa_second_half_first_half_params_vs_full"]}', '',
          '### B6. Regimes beyond GARCH: BIC on GARCH(1,1)-t PIT residuals', '', md_table(B['pit_bic']), '',
          f'Selected K: raw returns {B["raw_k"]}, GARCH-t residuals {B["pit_k"]}.', '',
          '## C. Efficiency by regime', '',
          '### C1. AR(1) in low- vs high-volatility states (label at t-1; White t statistics)', '',
          md_table(C['eff']), '',
          '### C2. Rolling one-year variance-ratio tests', '', md_table(C['vr']), '',
          '### C3. Regime label vs continuous volatility (added after the first full run; not a scorecard criterion)', '',
          'Coefficient and White t of r_{t-1} x label_{t-1} and of r_{t-1} x log EWMA volatility_{t-1} '
          '(standardized), alone and together.', '', md_table(C['race']), '',
          '## Figures', ''] + [f'![{f}]({f})' for f in figs] + ['',
          '## Methods (for the paper)', '',
          '- Returns: 100 x daily log change of the S&P 500 index close.',
          '- Gaussian HMM (Hamilton 1989): K = 2-4 states, Baum-Welch EM with 3-5 random restarts, states ordered by '
          'volatility; ex-post labels = smoothed probability > 0.5; real-time labels = filtered probability > 0.5 with '
          f'parameters re-estimated each January on all data up to the previous December (from {B4_FIRST_YEAR}).',
          '- Statistical jump model (Nystrup et al. 2020; Shu, Yu & Mulvey 2024): 2 states, features = EWMA mean return, '
          'log downside deviation and Sortino ratio at half-lives 5/10/21 days, standardized on the estimation window; '
          f'jump penalty λ in {list(JM_LAMBDAS)}; online labels from the forward dynamic program.',
          '- Change points: PELT (Killick et al. 2012), Gaussian mean/variance cost, penalty 3 log T, minimum segment 21 '
          'days; segments clustered into 2 volatility groups with k-means on log segment volatility.',
          '- Bear markets: 20% peak-to-trough / trough-to-peak rule (Lunde & Timmermann 2004).',
          '- Number of regimes: BIC = -2 log L + p log T with p = (K-1) + K(K-1) + 2K.',
          '- Regimes beyond GARCH: GARCH(1,1)-t standardized residuals mapped to N(0,1) through the fitted t CDF, '
          'then the same BIC comparison.',
          f'- Out-of-sample: parameters estimated on data before {OOS_START}; one-step predictive log densities; '
          'Diebold-Mariano with Newey-West (10 lags) standard errors.',
          '- Simulation DGPs use parameters estimated on the real series: HMM-2, HMM-2 with standardized t(4) shocks, '
          'HMM-3 (truth = top state), GARCH(1,1)-t (no regimes).',
          '- AR(1) by regime: r_t on r_{t-1}, the label at t-1 and their product; White (HC0) standard errors.',
          '- Variance ratio: Lo-MacKinlay heteroskedasticity-robust VR(2) z test on 252-day windows every 21 days.',
          '- Agreement: Cohen kappa between binary high-volatility labels.', '',
          'Data: Yahoo Finance (^GSPC, ^VIX) and FRED (USREC). Yahoo Finance data may not be redistributed; '
          'cite the download date above.']
    with open(os.path.join(OUT, 'RESULTS.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def main(argv=None):
    global OUT, CACHE, N_SIM, N_PELT, CHUNK, QUICK, B4_FIRST_YEAR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                    help='auto = CUDA GPU, else Apple GPU (MPS), else CPU')
    ap.add_argument('--float32', action='store_true', help='float32 on the CPU too (reproduces GPU precision)')
    ap.add_argument('--n-sim', type=int, default=N_SIM, help='simulation replications per DGP')
    ap.add_argument('--n-pelt', type=int, default=N_PELT, help='replications per DGP that also get PELT')
    ap.add_argument('--chunk', type=int, default=CHUNK, help='series per GPU batch')
    ap.add_argument('--out', default=None, help='output directory')
    ap.add_argument('--data-dir', default=None, help='data cache directory')
    ap.add_argument('--validate', action='store_true', help='first check the GPU code against hmmlearn/jumpmodels')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 1990+ data, 2 replications, real-time refits from 2015')
    a = ap.parse_args(argv)
    t_start = time.time()
    G.set_device(None if a.device == 'auto' else a.device, 'float32' if a.float32 else None)
    N_SIM, N_PELT, CHUNK, QUICK = a.n_sim, a.n_pelt, a.chunk, a.quick
    if a.data_dir:
        CACHE = a.data_dir
    r = load_returns()
    if QUICK:
        N_SIM, N_PELT = min(N_SIM, 2), min(N_PELT, 2)
        r = r.loc['1990':]
        B4_FIRST_YEAR = 2015
    OUT = a.out or os.path.join(ROOT, 'outputs', 'regimes_quick' if QUICK else 'regimes')
    os.makedirs(OUT, exist_ok=True)
    log(f'Device: {G.device_summary()} | replications per DGP: {N_SIM} | PELT replications: {min(N_PELT, N_SIM)} | batch: {CHUNK}')
    log(f'S&P 500 daily log returns: {r.index[0].date()} to {r.index[-1].date()}, n = {len(r)} | output: {OUT}')

    validation = None
    if a.validate:
        log('=== Validation: GPU implementation vs hmmlearn / jumpmodels (last 5000 days) ===')
        import validate_gpu_regimes as V
        ok, validation = V.run_checks(r.values[-5000:])
        save(pd.DataFrame(validation), '0_validation.csv')
        if not ok:
            log('WARNING: some validation checks failed - see 0_validation.csv before using the results')

    base = base_fits(r)
    A = part_a(r, base)
    B = part_b(r, base)
    C = part_c(r, base, B)
    sc = scorecard(A, B, C)
    figs = make_figures(r, A, B)
    snap = os.path.join(CACHE, 'regime_data_snapshot.txt')
    info = {'device': G.device_summary(), 'runtime_min': (time.time() - t_start) / 60,
            'snapshot': open(snap).read().splitlines()[0] + '.' if os.path.exists(snap) else ''}
    write_report(r, info, validation, A, B, C, sc, figs)
    summary = {'device': info['device'], 'n_sim': N_SIM, 'n_pelt': min(N_PELT, N_SIM), 'quick': QUICK,
               'sample': f'{r.index[0].date()} to {r.index[-1].date()}', 'n_obs': len(r),
               'validation_passed': None if validation is None else all(v['passed'] for v in validation),
               'bic_k_raw_returns': B['raw_k'], 'bic_k_garch_residuals': B['pit_k'],
               'best_hmm_oos': B['best_hmm'].model, 'best_hmm_dm_t_vs_garch_t': float(B['best_hmm'].dm_t_vs_garch_t),
               'pelt_breakpoints': B['pelt_breakpoints'],
               'scorecard': sc[['criterion', 'value', 'result']].astype(str).to_dict('records'),
               'runtime_min': round(info['runtime_min'], 1)}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f'Done in {info["runtime_min"]:.0f} min. Results: {OUT} (start with RESULTS.md)')
    return OUT


if __name__ == '__main__':
    main()
