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

    python src/validate_gpu_regimes.py [--device mps] [--float32] [--n 5000]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    ap.add_argument('--float32', action='store_true', help='float32 on the CPU (MPS precision)')
    ap.add_argument('--n', type=int, default=5000, help='most recent observations used')
    a = ap.parse_args()
    G.set_device(None if a.device == 'auto' else a.device, 'float32' if a.float32 else None)
    f32 = G.get_dtype() == G.torch.float32
    print('Device:', G.device_summary())
    x = load_returns().values[-a.n:]
    ok = True

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
        f_t, lpd_t = G.hmm_filter_batch([x], [g])[0]
        lpd_err = (np.abs(lpd_np - lpd_t) / (1 + np.abs(lpd_np))).max()   # relative error
        print(f'HMM K={k}: loglik hmmlearn {ll_ref:.2f} torch {g["loglik"]:.2f} | sd {np.round(sd_ref, 3)} vs '
              f'{np.round(g["sd"], 3)} | state kappa {kap:.4f} | max filter error {filt_err:.1e} | '
              f'max predictive log density rel. error {lpd_err:.1e}')
        tol = 1e-4 if f32 else 1e-8
        ok &= g['loglik'] >= ll_ref - 0.5 and kap > 0.95 and filt_err < tol and lpd_err < tol

    short = x[:1500]
    alone = G.fit_hmm_batch([short], 2, n_init=4)[0]['loglik']
    batched = G.fit_hmm_batch([x, short], 2, n_init=4)[1]['loglik']
    print(f'Padded batching: short series alone {alone:.4f}, in batch {batched:.4f}')
    ok &= abs(alone - batched) < 0.5

    X = jm_features(x)
    for lam in (10.0, 30.0, 100.0):
        jm = JumpModel(n_components=2, jump_penalty=lam, cont=False, random_state=0, n_init=5)
        quiet(jm.fit, X, ret_ser=x, sort_by='vol')
        lr = np.asarray(jm.predict(X))
        obj_ref = 0.5 * ((X - jm.centers_[lr]) ** 2).sum() + lam * switches(lr)
        g = G.fit_jump_batch([X], [lam], n_init=5)[0]
        k_ep = cohen_kappa_score(high_vol_labels(lr, x), high_vol_labels(g['ex_post'], x))
        k_on = cohen_kappa_score(high_vol_labels(np.asarray(jm.predict_online(X)), x), high_vol_labels(g['online'], x))
        print(f'Jump λ={lam:g}: objective jumpmodels {obj_ref:.1f} torch {g["objective"]:.1f} | '
              f'kappa ex post {k_ep:.4f}, online {k_on:.4f}')
        ok &= g['objective'] <= obj_ref + (1e-5 if f32 else 1e-9) * abs(obj_ref) and k_ep > 0.95 and k_on > 0.95

    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
