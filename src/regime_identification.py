"""
regime_identification.py
------------------------
Can market regimes be identified objectively, and how accurately?

    python src/build_dataset.py                    # VIX / NBER cache
    python src/regime_identification.py            # full run, writes outputs/regimes/
    python src/regime_identification.py --quick    # smoke test, writes outputs/regimes_quick/

Options: --device {auto,cuda,cpu}  --n-sim N  --chunk N
The HMM and jump-model fits run in batches on the GPU through CUDA when one
is available (see gpu_regimes.py); PELT, GARCH and bear-market dating run on
the CPU.

Part A  Simulation with known regimes: ex-post and real-time accuracy of
        each method, and whether methods find regimes when none exist.
Part B  S&P 500 daily, 1950-2025: is there evidence for discrete regimes
        (BIC, out-of-sample density forecasts vs GARCH); do independent
        methods agree; how much do real-time labels differ from ex-post ones.
Part C  Preliminary link to efficiency: return autocorrelation within
        regimes, and whether regime switching alone produces rolling-window
        "inefficiency".
"""

import io
import os
import sys
import argparse
import json
import contextlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
import ruptures as rpt
from arch import arch_model

sys.path.insert(0, os.path.dirname(__file__))
import gpu_regimes as G

warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(__file__), '..')
CACHE = os.path.join(ROOT, 'data', 'cache')
OUT = os.path.join(ROOT, 'outputs', 'regimes')
N_SIM = 100          # simulation replications per DGP (--n-sim)
CHUNK = 16           # series per GPU batch (--chunk); lower it if GPU memory runs out
QUICK = False
B4_FIRST_YEAR = 1970
JM_LAMBDAS = (10.0, 30.0, 100.0)
EPISODES = {'1987 crash': '1987-10-14', '2008 Lehman': '2008-09-15',
            '2011 US downgrade': '2011-08-01', '2020 Covid': '2020-02-21',
            '2022 rate shock': '2022-01-18'}


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def load_returns():
    px = pd.read_csv(os.path.join(CACHE, 'gspc_long.csv'), index_col=0, parse_dates=True)['Close']
    return (100 * np.log(px).diff()).dropna()


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


def ar1_within(r, labels, mask_period=None):
    """AR(1) slope using only pairs where t-1 and t are in the same state; White t-stat."""
    r = np.asarray(r)
    labels = np.asarray(labels)
    out = {}
    for k in np.unique(labels):
        sel = (labels[1:] == k) & (labels[:-1] == k)
        if mask_period is not None:
            sel &= mask_period[1:]
        y, x = r[1:][sel], r[:-1][sel]
        if len(y) < 100:
            out[int(k)] = (np.nan, np.nan, len(y))
            continue
        xc = x - x.mean()
        b = (xc @ (y - y.mean())) / (xc @ xc)
        e = (y - y.mean()) - b * xc
        se = np.sqrt((xc ** 2 * e ** 2).sum()) / (xc @ xc)
        out[int(k)] = (b, b / se, len(y))
    return out


# ------------------------------------------------------------------
# Part A: simulation with ground truth
# ------------------------------------------------------------------

def part_a(r):
    print('\n=== Part A: simulation with known regimes ===')
    x = r.values
    T = len(x)
    p2, _, _ = fit_hmm(x, 2)
    p3, _, _ = fit_hmm(x, 3)
    garch = quiet(arch_model(x, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit, disp='off')
    dgps = ['HMM2 (correctly specified)', 'HMM2 with t(4) shocks (misspecified)',
            'HMM3 (three regimes)', 'GARCH(1,1)-t (no regimes)']
    rows, ksel = [], []
    for dgp in dgps:
        sims = []
        for rep in range(N_SIM):
            rng = np.random.default_rng(1000 * dgps.index(dgp) + rep)
            s3 = None
            if dgp.startswith('HMM2 (c'):
                y, s = simulate_hmm(p2, T, rng)
            elif dgp.startswith('HMM2 with'):
                y, s = simulate_hmm(p2, T, rng, dist='t')
            elif dgp.startswith('HMM3'):
                y, s3 = simulate_hmm(p3, T, rng)
                s = (s3 == 2).astype(int)
            else:
                sim = arch_model(None, mean='Constant', vol='GARCH', p=1, q=1, dist='t').simulate(
                    garch.params.values, T, burn=500, initial_value=None)
                y, s = sim['data'].values, None
            sims.append((y, s, s3))
        ys = [y for y, _, _ in sims]

        fits = {k: fit_hmm_many(ys, k, n_init=3, seed=1000 * dgps.index(dgp)) for k in (2, 3, 4)}
        if sims[0][1] is not None:
            jm = fit_jump_many([y for y in ys for _ in JM_LAMBDAS], [lam for _ in ys for lam in JM_LAMBDAS])

        for rep, (y, s, s3) in enumerate(sims):
            bics = {1: -2 * stats.norm.logpdf(y, y.mean(), y.std()).sum() + hmm_n_params(1) * np.log(T)}
            for k in (2, 3, 4):
                bics[k] = -2 * fits[k][rep]['loglik'] + hmm_n_params(k) * np.log(T)
            kbest = min(bics, key=bics.get)
            ksel.append({'dgp': dgp, 'rep': rep, 'bic_selected_k': kbest})
            if s is None:
                continue

            f2 = fits[2][rep]
            ex_post = (f2['smoothed'][:, 1] > 0.5).astype(int)
            real_time = (f2['filtered'][:, 1] > 0.5).astype(int)
            methods = {'HMM-2 ex post (smoothed)': ex_post, 'HMM-2 real time (filtered)': real_time}
            for j, lam in enumerate(JM_LAMBDAS):
                ep, rt = jm[rep * len(JM_LAMBDAS) + j]
                methods[f'Jump model λ={lam:g} ex post'] = ep
                methods[f'Jump model λ={lam:g} real time'] = rt
            cp, _ = changepoint_labels(y)
            methods['Change-point (PELT) ex post'] = cp
            if dgp.startswith('HMM3'):
                sm3 = fits[3][rep]['smoothed']
                rows.append({'dgp': dgp, 'rep': rep, 'method': 'HMM-3 ex post, 3-class',
                             'balanced_accuracy': balanced_accuracy_score(s3, sm3.argmax(1)),
                             'kappa': cohen_kappa_score(s3, sm3.argmax(1)),
                             'median_delay_days': np.nan, 'missed_episodes': np.nan,
                             'switch_ratio': switches(sm3.argmax(1)) / max(switches(s3), 1)})
            for name, est in methods.items():
                dl, miss = detection_delay(s, est)
                rows.append({'dgp': dgp, 'rep': rep, 'method': name,
                             'balanced_accuracy': balanced_accuracy_score(s, est),
                             'kappa': cohen_kappa_score(s, est),
                             'median_delay_days': dl, 'missed_episodes': miss,
                             'switch_ratio': switches(est) / max(switches(s), 1)})
        print(f'  done: {dgp}')
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(OUT, 'sim_accuracy_by_rep.csv'), index=False)
    summ = acc.groupby(['dgp', 'method']).agg(
        balanced_accuracy=('balanced_accuracy', 'mean'), ba_sd=('balanced_accuracy', 'std'),
        kappa=('kappa', 'mean'), median_delay_days=('median_delay_days', 'median'),
        missed_episodes=('missed_episodes', 'mean'), switch_ratio=('switch_ratio', 'mean')).round(3)
    summ.to_csv(os.path.join(OUT, 'sim_accuracy_summary.csv'))
    kdf = pd.DataFrame(ksel)
    ktab = pd.crosstab(kdf.dgp, kdf.bic_selected_k)
    ktab.to_csv(os.path.join(OUT, 'sim_bic_selected_k.csv'))
    print(summ.to_string())
    print('\nBIC-selected number of regimes (count of replications):\n', ktab.to_string())
    true_dur = {'HMM2 high-vol regime expected duration (days)': 1 / (1 - p2['A'][1, 1]),
                'HMM2 low-vol regime expected duration (days)': 1 / (1 - p2['A'][0, 0])}
    return summ, ktab, true_dur


# ------------------------------------------------------------------
# Part B: real data
# ------------------------------------------------------------------

def part_b(r):
    print('\n=== Part B: S&P 500, 1950-2025 ===')
    x = r.values
    idx = r.index
    T = len(x)
    res = {}

    # B1. Number of regimes: in-sample BIC and out-of-sample density forecasts
    split = idx.searchsorted(pd.Timestamp('2000-01-01'))
    xtr, xte = x[:split], x[split:]
    rows = []
    fitted = {}
    for k in (1, 2, 3, 4):
        if k == 1:
            ll_full = stats.norm.logpdf(x, x.mean(), x.std()).sum()
            lpd = stats.norm.logpdf(xte, xtr.mean(), xtr.std())
        else:
            of, otr = fit_hmm_many([x, xtr], k, n_init=5)
            pf, ptr = params_of(of), params_of(otr)
            ll_full, smf = of['loglik'], of['smoothed']
            fitted[k] = (pf, smf)
            prior = otr['filtered'][-1] @ ptr['A']
            _, lpd = hmm_forward(xte, ptr, prior=prior)
        rows.append({'model': f'Gaussian HMM, K={k}' if k > 1 else 'Gaussian i.i.d. (K=1)',
                     'n_params': hmm_n_params(k),
                     'bic_full_sample': -2 * ll_full + hmm_n_params(k) * np.log(T),
                     'oos_mean_log_score_2000_2025': lpd.mean(), '_lpd': lpd})
    for dist in ('normal', 't'):
        am = arch_model(xtr, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
        fr = quiet(am.fit, disp='off')
        full = arch_model(x, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
        fixed = full.fix(fr.params.values)
        sig = fixed.conditional_volatility[split:]
        mu = fr.params['mu']
        if dist == 'normal':
            lpd = stats.norm.logpdf(xte, mu, sig)
        else:
            nu = fr.params['nu']
            scale = sig * np.sqrt((nu - 2) / nu)
            lpd = stats.t.logpdf(xte, nu, mu, scale)
        llf_full = quiet(full.fit, disp='off')
        rows.append({'model': f'GARCH(1,1)-{dist} (no regimes)', 'n_params': len(fr.params),
                     'bic_full_sample': llf_full.bic, 'oos_mean_log_score_2000_2025': lpd.mean(), '_lpd': lpd})
    best_hmm = max([rw for rw in rows if 'HMM' in rw['model']], key=lambda z: z['oos_mean_log_score_2000_2025'])
    for rw in rows:
        d = rw['_lpd'] - best_hmm['_lpd']
        m_, se = nw_tstat(d)
        rw['oos_diff_vs_best_hmm'] = m_
        rw['dm_t_vs_best_hmm'] = m_ / se if se > 0 else np.nan
    mc = pd.DataFrame(rows).drop(columns='_lpd').round(4)
    mc.to_csv(os.path.join(OUT, 'real_model_comparison.csv'), index=False)
    print(mc.to_string())
    res['best_hmm_oos'] = best_hmm['model']

    # B2. Fitted regimes
    par_rows = []
    for k, (pf, smf) in fitted.items():
        if k > 3:
            continue
        lab = smf.argmax(1)
        for j in range(k):
            par_rows.append({'K': k, 'state': j, 'ann_mean_return_pct': pf['mu'][j] * 252,
                             'ann_vol_pct': pf['sd'][j] * np.sqrt(252),
                             'expected_duration_days': 1 / (1 - pf['A'][j, j]),
                             'share_of_days_pct': (lab == j).mean() * 100})
    par = pd.DataFrame(par_rows).round(2)
    par.to_csv(os.path.join(OUT, 'real_hmm_parameters.csv'), index=False)
    print('\n', par.to_string())

    p2, sm2 = fitted[2]
    hmm2 = (sm2[:, 1] > 0.5).astype(int)
    hmm3 = fitted[3][1].argmax(1)

    # B3. Agreement across methods (ex post)
    labels = pd.DataFrame(index=idx)
    labels['HMM-2 (vol regime)'] = hmm2
    labels['HMM-3 top state'] = (hmm3 == 2).astype(int)
    for lam, (ep, _) in zip(JM_LAMBDAS, fit_jump_many([x] * len(JM_LAMBDAS), list(JM_LAMBDAS))):
        labels[f'Jump model λ={lam:g}'] = ep
    cp, n_bk = changepoint_labels(x)
    labels['Change-point (PELT)'] = cp
    px = np.exp(np.cumsum(x / 100))
    labels['Bear market (20% rule)'] = bear_market_dating(px)
    rec = pd.read_csv(os.path.join(CACHE, 'usrec.csv'), index_col=0, parse_dates=True)['USREC']
    labels['NBER recession'] = (rec.reindex(idx, method='ffill') == 1).astype(int)
    vix = pd.read_csv(os.path.join(CACHE, 'index_close.csv'), index_col=0, parse_dates=True)['^VIX']
    labels['VIX ≥ 20 (2005+)'] = (vix.reindex(idx) >= 20).astype(float).where(vix.reindex(idx).notna())
    labels.to_csv(os.path.join(OUT, 'real_labels_ex_post.csv'))
    cols = labels.columns
    kap = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            d = labels[[a, b]].dropna()
            kap.loc[a, b] = cohen_kappa_score(d[a].astype(int), d[b].astype(int))
    kap = kap.round(2)
    kap.to_csv(os.path.join(OUT, 'real_agreement_kappa.csv'))
    print('\nCohen kappa between methods (ex post):\n', kap.to_string())
    desc = pd.DataFrame({c: {'share_state1_pct': labels[c].mean() * 100,
                             'n_switches': switches(labels[c].dropna().values.astype(int)),
                             'mean_spell_days': len(labels[c].dropna()) / max(switches(labels[c].dropna().values.astype(int)) + 1, 1)}
                         for c in cols}).T.round(1)
    desc.to_csv(os.path.join(OUT, 'real_label_descriptives.csv'))
    print('\n', desc.to_string())
    res['pelt_breakpoints'] = n_bk

    # B4. Real time vs ex post: expanding-window refits
    years = [y for y in range(B4_FIRST_YEAR, 2026)
             if ((idx >= pd.Timestamp(f'{y}-01-01')) & (idx < pd.Timestamp(f'{y + 1}-01-01'))).any()]
    n_tr = [int((idx < pd.Timestamp(f'{y}-01-01')).sum()) for y in years]
    n_end = [int((idx < pd.Timestamp(f'{y + 1}-01-01')).sum()) for y in years]
    fits = fit_hmm_many([x[:a] for a in n_tr], 2, n_init=2, seed=B4_FIRST_YEAR)
    filt = hmm_forward_many([x[:b] for b in n_end], [params_of(f) for f in fits])
    rt_hmm = pd.Series(np.nan, index=idx)
    for a, b, (fa, _) in zip(n_tr, n_end, filt):
        rt_hmm.iloc[a:b] = (fa[a:b, 1] > 0.5).astype(int)
    lam_main = 30.0
    rt_jm = pd.Series(np.nan, index=idx)
    for i in range(0, len(years), CHUNK):
        tr_c, end_c = n_tr[i:i + CHUNK], n_end[i:i + CHUNK]
        feats = [jm_features(x[:b], n_fit=a) for a, b in zip(tr_c, end_c)]
        jfits = G.fit_jump_batch(feats, [lam_main] * len(feats), n_init=3, n_fit=tr_c)
        for a, b, f in zip(tr_c, end_c, jfits):
            ref = f['ex_post']
            flip = x[:a][ref == 1].std() < x[:a][ref == 0].std()
            on = f['online'][a:b]
            rt_jm.iloc[a:b] = (1 - on) if flip else on
    rt = pd.DataFrame({'HMM-2 real time': rt_hmm, f'Jump model λ={lam_main:g} real time': rt_jm}).dropna()
    rt.to_csv(os.path.join(OUT, 'real_labels_real_time.csv'))
    rt_rows = []
    for c, ref in [('HMM-2 real time', 'HMM-2 (vol regime)'),
                   (f'Jump model λ={lam_main:g} real time', f'Jump model λ={lam_main:g}')]:
        a, b = rt[c].astype(int), labels.loc[rt.index, ref].astype(int)
        row = {'real_time_method': c, 'ex_post_reference': ref,
               'agreement_pct': (a == b).mean() * 100, 'kappa': cohen_kappa_score(a, b),
               'real_time_switches': switches(a.values), 'ex_post_switches': switches(b.values)}
        for name, d in EPISODES.items():
            seg = a.loc[pd.Timestamp(d):]
            hit = np.where(seg.values == 1)[0]
            row[f'lag_days_{name}'] = int(hit[0]) if len(hit) else np.nan
        rt_rows.append(row)
    rtab = pd.DataFrame(rt_rows).round(3)
    rtab.to_csv(os.path.join(OUT, 'real_time_vs_ex_post.csv'), index=False)
    print('\nReal time vs ex post (1970-2025):\n', rtab.T.to_string())

    # B5. Stability: parameters estimated on each half
    mid = len(x) // 2 if QUICK else idx.searchsorted(pd.Timestamp('1988-01-01'))
    pa, pb = [params_of(o) for o in fit_hmm_many([x[:mid], x[mid:]], 2)]
    filt_cross, _ = hmm_forward(x[mid:], pa)
    stab = pd.DataFrame([
        {'sample': f'{idx[0].year}-{idx[mid - 1].year}', 'low_vol_ann_pct': pa['sd'][0] * np.sqrt(252), 'high_vol_ann_pct': pa['sd'][1] * np.sqrt(252),
         'low_dur_days': 1 / (1 - pa['A'][0, 0]), 'high_dur_days': 1 / (1 - pa['A'][1, 1])},
        {'sample': f'{idx[mid].year}-{idx[-1].year}', 'low_vol_ann_pct': pb['sd'][0] * np.sqrt(252), 'high_vol_ann_pct': pb['sd'][1] * np.sqrt(252),
         'low_dur_days': 1 / (1 - pb['A'][0, 0]), 'high_dur_days': 1 / (1 - pb['A'][1, 1])},
        {'sample': 'full 1950-2025', 'low_vol_ann_pct': p2['sd'][0] * np.sqrt(252), 'high_vol_ann_pct': p2['sd'][1] * np.sqrt(252),
         'low_dur_days': 1 / (1 - p2['A'][0, 0]), 'high_dur_days': 1 / (1 - p2['A'][1, 1])},
    ]).round(2)
    res['kappa_1988_2025_first_half_params_vs_full'] = round(
        cohen_kappa_score(hmm2[mid:], (filt_cross[:, 1] > 0.5).astype(int)), 3)
    stab.to_csv(os.path.join(OUT, 'real_stability.csv'), index=False)
    print('\nStability:\n', stab.to_string(), '\n', res)
    return labels, p2, p3, res


# ------------------------------------------------------------------
# Part C: preliminary link to efficiency
# ------------------------------------------------------------------

def part_c(r, labels, p2, p3):
    print('\n=== Part C: efficiency by regime (preliminary) ===')
    x = r.values
    idx = r.index
    eras = {'1950-1979': (idx < '1980-01-01'), '1980-1999': (idx >= '1980-01-01') & (idx < '2000-01-01'),
            '2000-2025': (idx >= '2000-01-01')}
    rows = []
    for method in ['HMM-2 (vol regime)', 'Jump model λ=30', 'Bear market (20% rule)']:
        lab = labels[method].values.astype(int)
        for era, m in eras.items():
            out = ar1_within(x, lab, m)
            for k, (b, t, n) in out.items():
                rows.append({'regime_method': method, 'era': era, 'state': k, 'ar1': b, 'white_t': t, 'n_pairs': n})
        for era, m in eras.items():
            sel = m[1:]
            y_, x_ = x[1:][sel], x[:-1][sel]
            xc = x_ - x_.mean()
            b = (xc @ (y_ - y_.mean())) / (xc @ xc)
            e = (y_ - y_.mean()) - b * xc
            se = np.sqrt((xc ** 2 * e ** 2).sum()) / (xc @ xc)
            rows.append({'regime_method': method, 'era': era, 'state': 'pooled', 'ar1': b, 'white_t': b / se, 'n_pairs': len(y_)})
    eff = pd.DataFrame(rows).round(3)
    eff.to_csv(os.path.join(OUT, 'efficiency_ar1_by_regime_era.csv'), index=False)
    print(eff.to_string())

    # Does regime switching alone generate rolling-window "inefficiency"?
    def rejection_rate(y, w=252, step=21):
        zs = [lo_mackinlay_z(y[i:i + w], 2) for i in range(0, len(y) - w, step)]
        zs = np.array(zs)
        return (np.abs(zs) > 1.96).mean() * 100
    rng = np.random.default_rng(7)
    sim2 = [rejection_rate(simulate_hmm(p2, len(x), rng)[0]) for _ in range(10)]
    sim3 = [rejection_rate(simulate_hmm(p3, len(x), rng)[0]) for _ in range(10)]
    art = pd.DataFrame([
        {'series': 'Real S&P 500, 1950-2025', 'pct_windows_rejecting_rw': rejection_rate(x)},
        {'series': 'Simulated HMM-2 (regimes, no within-regime predictability)', 'pct_windows_rejecting_rw': np.mean(sim2)},
        {'series': 'Simulated HMM-3 (regimes, no within-regime predictability)', 'pct_windows_rejecting_rw': np.mean(sim3)},
        {'series': 'Nominal size of the test', 'pct_windows_rejecting_rw': 5.0},
    ]).round(1)
    for era, m in eras.items():
        art.loc[len(art)] = {'series': f'Real S&P 500, {era}', 'pct_windows_rejecting_rw': round(rejection_rate(x[m]), 1)}
    art.to_csv(os.path.join(OUT, 'rolling_vr_rejections.csv'), index=False)
    print('\nRolling 1-year Lo-MacKinlay VR(2) tests, % of windows rejecting at 5%:\n', art.to_string())
    return eff, art


def main(argv=None):
    global OUT, N_SIM, CHUNK, QUICK, B4_FIRST_YEAR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--n-sim', type=int, default=N_SIM, help='simulation replications per DGP')
    ap.add_argument('--chunk', type=int, default=CHUNK, help='series per GPU batch')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 1990+ data, 2 replications, real-time refits for 2023-2025 only')
    a = ap.parse_args(argv)
    if a.device != 'auto':
        G.set_device(a.device)
    N_SIM, CHUNK, QUICK = a.n_sim, a.chunk, a.quick
    r = load_returns()
    if QUICK:
        OUT = os.path.join(ROOT, 'outputs', 'regimes_quick')
        N_SIM = min(N_SIM, 2)
        r = r.loc['1990':]
        B4_FIRST_YEAR = 2023
    os.makedirs(OUT, exist_ok=True)
    print(f'Device: {G.device_summary()} | replications per DGP: {N_SIM} | batch: {CHUNK}')
    print(f'S&P 500 daily log returns: {r.index[0].date()} to {r.index[-1].date()}, n = {len(r)}')
    summ, ktab, true_dur = part_a(r)
    labels, p2, p3, res = part_b(r)
    part_c(r, labels, p2, p3)
    res.update({k: round(v, 1) for k, v in true_dur.items()})
    res['device'] = G.device_summary()
    res['n_sim'] = N_SIM
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print('\nDone.')


if __name__ == '__main__':
    main()
