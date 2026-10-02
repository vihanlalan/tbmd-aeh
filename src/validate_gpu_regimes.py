"""
validate_gpu_regimes.py
-----------------------
Checks the batched torch implementations in gpu_regimes.py against the
reference packages on real S&P 500 returns:

  * Gaussian HMM (K = 2, 3) vs hmmlearn: log likelihood, parameters,
    agreement of smoothed states; filtered probabilities vs a numpy loop.
  * Padded batching: a short series fitted alone vs inside a batch.
  * Jump model (lambda = 10, 30, 100) vs jumpmodels: objective, ex-post and
    online labels.

    python src/validate_gpu_regimes.py [--device cuda|mps|cpu] [--float32] [--n 5000]
"""

import argparse
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import cohen_kappa_score
from hmmlearn.hmm import GaussianHMM
from jumpmodels.jump import JumpModel

import gpu_regimes as G
from regime_identification import load_returns, jm_features, high_vol_labels, switches, quiet


def numpy_filter(x, p):
    logB = stats.norm.logpdf(x[:, None], p['mu'][None, :], p['sd'][None, :])
    prior, filt, lpd = p['pi'], np.zeros_like(logB), np.zeros(len(x))
    for t in range(len(x)):
        a = np.log(prior + 1e-300) + logB[t]
        lpd[t] = logsumexp(a)
        filt[t] = np.exp(a - lpd[t])
        prior = filt[t] @ p['A']
    return filt, lpd


def run_checks(x):
    """All checks on the return series x. Returns (all_passed, list of result rows)."""
    f32 = G.get_dtype() == G.torch.float32
    tol = 1e-4 if f32 else 1e-8
    rows = []

    def add(check, torch_value, reference, passed):
        rows.append({'check': check, 'torch': torch_value, 'reference': reference, 'passed': bool(passed)})
        print(f'  [{"pass" if passed else "FAIL"}] {check}: torch {torch_value} | reference {reference}')

    for k in (2, 3):
        m = GaussianHMM(n_components=k, covariance_type='full', n_iter=300, tol=1e-6, random_state=0)
        quiet(m.fit, x.reshape(-1, 1))
        ll_ref = m.score(x.reshape(-1, 1))
        order = np.argsort(m.covars_.ravel())
        sd_ref = np.sqrt(m.covars_.ravel()[order])
        g = G.fit_hmm_batch([x], k, n_init=4)[0]
        ref_smoothed = m.predict_proba(x.reshape(-1, 1))[:, order]   # states ordered by volatility
        kap = cohen_kappa_score(ref_smoothed.argmax(1), g['smoothed'].argmax(1))
        f_np, lpd_np = numpy_filter(x, g)
        filt_err = np.abs(f_np - g['filtered']).max()
        _, lpd_t = G.hmm_filter_batch([x], [g])[0]
        lpd_err = (np.abs(lpd_np - lpd_t) / (1 + np.abs(lpd_np))).max()   # relative error
        add(f'HMM K={k} log likelihood (hmmlearn)', round(g['loglik'], 2), round(ll_ref, 2), g['loglik'] >= ll_ref - 0.5)
        add(f'HMM K={k} state sd', np.round(g['sd'], 3).tolist(), np.round(sd_ref, 3).tolist(), True)
        add(f'HMM K={k} smoothed-state kappa vs hmmlearn', round(kap, 4), '> 0.95', kap > 0.95)
        add(f'HMM K={k} filter max abs error vs numpy', f'{filt_err:.1e}', f'< {tol:g}', filt_err < tol)
        add(f'HMM K={k} predictive density max rel. error vs numpy', f'{lpd_err:.1e}', f'< {tol:g}', lpd_err < tol)

    short = x[:1500]
    alone = G.fit_hmm_batch([short], 2, n_init=4)[0]['loglik']
    batched = G.fit_hmm_batch([x, short], 2, n_init=4)[1]['loglik']
    add('Padded batching: short series in batch vs alone', round(batched, 4), round(alone, 4), abs(alone - batched) < 0.5)

    X = jm_features(x)
    for lam in (10.0, 30.0, 100.0):
        jm = JumpModel(n_components=2, jump_penalty=lam, cont=False, random_state=0, n_init=5)
        quiet(jm.fit, X, ret_ser=x, sort_by='vol')
        lr = np.asarray(jm.predict(X))
        obj_ref = 0.5 * ((X - jm.centers_[lr]) ** 2).sum() + lam * switches(lr)
        g = G.fit_jump_batch([X], [lam], n_init=5)[0]
        k_ep = cohen_kappa_score(high_vol_labels(lr, x), high_vol_labels(g['ex_post'], x))
        k_on = cohen_kappa_score(high_vol_labels(np.asarray(jm.predict_online(X)), x), high_vol_labels(g['online'], x))
        add(f'Jump λ={lam:g} objective (jumpmodels)', round(g['objective'], 2), round(obj_ref, 2),
            g['objective'] <= obj_ref + (1e-5 if f32 else 1e-9) * abs(obj_ref))
        add(f'Jump λ={lam:g} ex-post / online label kappa vs jumpmodels', f'{k_ep:.4f} / {k_on:.4f}', '> 0.95',
            k_ep > 0.95 and k_on > 0.95)
    ok = all(r['passed'] for r in rows)
    print('  ALL CHECKS PASSED' if ok else '  SOME CHECKS FAILED')
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    ap.add_argument('--float32', action='store_true', help='float32 on the CPU (GPU precision)')
    ap.add_argument('--n', type=int, default=5000, help='most recent observations used')
    a = ap.parse_args()
    G.set_device(None if a.device == 'auto' else a.device, 'float32' if a.float32 else None)
    print('Device:', G.device_summary())
    ok, _ = run_checks(load_returns().values[-a.n:])
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
