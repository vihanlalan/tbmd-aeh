"""
gpu_regimes.py
--------------
Batched, device-agnostic implementations of the regime models used in
regime_identification.py. Runs on an Apple-silicon GPU (MacBook M1-M4)
through PyTorch's Metal backend (MPS) when one is available and on the CPU
otherwise.

MPS has no float64, so on the GPU everything runs in float32. To keep that
accurate over long series, emissions are rescaled by their per-step maximum
before the log-space scans (the offsets are added back to the likelihood),
and predictive densities are computed per step rather than by differencing
a cumulative log likelihood.

  * Gaussian hidden Markov model (Baum-Welch EM). The forward-backward
    recursion is computed with a parallel prefix scan in the log semiring,
    so a T-step sequential loop becomes O(log T) batched tensor operations.
  * Statistical jump model (Nystrup, Lindstrom & Madsen 2020; Shu, Yu &
    Mulvey 2024). The dynamic program is computed with a min-plus prefix scan.

Many series (simulation replications, random restarts, expanding windows,
penalty values) are fitted together in one batch. Series of unequal length
are right-padded and handled with a validity mask: padded steps emit
nothing and carry the state forward unchanged.
"""

import os
import math
import platform
import numpy as np

# run any operator MPS lacks on the CPU instead of raising
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import torch

_DEVICE = None
_DTYPE = None


def mps_available():
    return getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('mps' if mps_available() else 'cpu')
    return _DEVICE


def get_dtype():
    """float32 on MPS (no float64 support there), float64 on the CPU unless overridden."""
    if _DTYPE is not None:
        return _DTYPE
    return torch.float32 if get_device().type == 'mps' else torch.float64


def set_device(name, dtype=None):
    """name: 'mps', 'cpu', or None for automatic selection.
    dtype: 'float32' / 'float64' to override the default for the device."""
    global _DEVICE, _DTYPE
    if name == 'mps' and not mps_available():
        raise RuntimeError('MPS requested but torch.backends.mps.is_available() is False. '
                           'This needs an Apple-silicon Mac, macOS 12.3+ and an arm64 '
                           '(not Rosetta) Python with PyTorch >= 2.1.')
    _DEVICE = torch.device(name) if name else None
    _DTYPE = {None: None, 'float32': torch.float32, 'float64': torch.float64}[dtype]
    if get_device().type == 'mps' and _DTYPE == torch.float64:
        raise RuntimeError('MPS does not support float64.')
    return get_device()


def device_summary():
    d = get_device()
    prec = str(get_dtype()).replace('torch.', '')
    if d.type == 'mps':
        return f'mps (Apple GPU, {platform.machine()}, {prec})'
    return f'cpu ({prec}; no Apple GPU available)' if not mps_available() else f'cpu ({prec})'


def _tiny():
    return torch.finfo(get_dtype()).tiny


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64), device='cpu').to(get_device(), get_dtype())


def _mask_t(m):
    return torch.as_tensor(np.asarray(m, bool)).to(get_device())


def pad_batch(series):
    """List of 1-D arrays -> (B, T) array right-padded with 0 and a (B, T) validity mask."""
    T = max(len(s) for s in series)
    X = np.zeros((len(series), T))
    M = np.zeros((len(series), T), bool)
    for i, s in enumerate(series):
        X[i, :len(s)] = s
        M[i, :len(s)] = True
    return X, M


# ------------------------------------------------------------------
# Semiring prefix scans
# ------------------------------------------------------------------

def _log_matmul(a, b):
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def _min_matmul(a, b):
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).amin(dim=-2)


def _scan(M, op, reverse=False):
    """
    Inclusive scan over dim 1 of a (B, T, K, K) stack of matrices.
    forward:  P[t] = M[0] op M[1] op ... op M[t]
    reverse:  S[t] = M[t] op M[t+1] op ... op M[T-1]
    Hillis-Steele doubling: ceil(log2 T) steps.
    """
    P = M.flip(1) if reverse else M
    T = P.shape[1]
    d = 1
    while d < T:
        new = op(P[:, d:], P[:, :-d]) if reverse else op(P[:, :-d], P[:, d:])
        P = torch.cat([P[:, :d], new], dim=1)
        d *= 2
    return P.flip(1) if reverse else P


# ------------------------------------------------------------------
# Gaussian HMM
# ------------------------------------------------------------------

def _log_emissions(x, mask, mu, var):
    """Per-step rescaled log emissions (max over states = 0) and the offsets.
    Padded steps have zero emissions and zero offset."""
    logb = -0.5 * (torch.log(2 * math.pi * var[:, None, :])
                   + (x[..., None] - mu[:, None, :]) ** 2 / var[:, None, :])
    c = logb.amax(dim=-1) * mask
    return (logb - c[..., None]) * mask[..., None], c


def _log_identity(K):
    eye = torch.full((K, K), -math.inf, dtype=get_dtype(), device=get_device())
    eye.fill_diagonal_(0.0)
    return eye


def _forward(x, mask, logpi, logA, mu, var):
    """Log forward variables alpha (B, T, K) of the rescaled model, the
    transition stack M, rescaled emissions and the emission offsets c.
    True log likelihood = logsumexp(alpha[:, -1]) + c.sum(1)."""
    logb, c = _log_emissions(x, mask, mu, var)
    K = mu.shape[1]
    M = logA[:, None] + logb[:, 1:, None, :]
    M = torch.where(mask[:, 1:, None, None], M, _log_identity(K))
    alpha0 = logpi + logb[:, 0]
    if x.shape[1] == 1:
        return alpha0[:, None], M, logb, c
    P = _scan(M, _log_matmul)
    alpha_rest = torch.logsumexp(alpha0[:, None, :, None] + P, dim=-2)
    return torch.cat([alpha0[:, None], alpha_rest], dim=1), M, logb, c


def _estep(x, mask, logpi, logA, mu, var):
    alpha, M, _, c = _forward(x, mask, logpi, logA, mu, var)
    B, T, K = alpha.shape
    S = _scan(M, _log_matmul, reverse=True)
    beta = torch.cat([torch.logsumexp(S, dim=-1), torch.zeros(B, 1, K, dtype=alpha.dtype, device=x.device)], dim=1)
    logZs = torch.logsumexp(alpha[:, -1], dim=-1)          # rescaled model
    log_gamma = alpha + beta - logZs[:, None, None]
    log_xi = alpha[:, :-1, :, None] + M + beta[:, 1:, None, :] - logZs[:, None, None, None]
    log_xi = torch.where(mask[:, 1:, None, None], log_xi, torch.tensor(-math.inf, dtype=alpha.dtype, device=x.device))
    # offsets summed in float64 on the CPU so the likelihood keeps full precision
    logZ = logZs.double().cpu() + c.double().cpu().sum(1)
    return log_gamma.exp() * mask[..., None], torch.logsumexp(log_xi, dim=1), logZ


def fit_hmm_batch(series, K, n_init=4, n_iter=300, tol=1e-6, seed=0, var_floor=1e-3):
    """
    Fit a K-state Gaussian HMM to each 1-D series in `series` (list of arrays),
    with n_init random restarts per series, all in one batch.

    Returns a list (one entry per series) of dicts with keys
      pi, A, mu, sd   parameters, states ordered by increasing volatility
      loglik          maximized log likelihood
      smoothed        (T, K) posterior state probabilities
      filtered        (T, K) filtered state probabilities (uses data up to t only)
    """
    X, Mk = pad_batch(series)
    n = len(series)
    X = np.repeat(X, n_init, axis=0)
    Mk = np.repeat(Mk, n_init, axis=0)
    B, T = X.shape
    x, mask = _t(X), _mask_t(Mk)
    dt = get_dtype()
    # in float32 the likelihood is only resolved to about 1e-7 relative
    tol = max(tol, 1e-7) if dt == torch.float32 else tol

    # initial values drawn in float64 on the CPU so every device starts identically
    g = torch.Generator(device='cpu').manual_seed(seed)
    Xc, Mc = torch.as_tensor(X), torch.as_tensor(Mk)
    cnt = Mc.sum(1).double()
    m0 = (Xc * Mc).sum(1) / cnt
    v0 = (((Xc - m0[:, None]) ** 2) * Mc).sum(1) / cnt
    spread = torch.linspace(-1.0, 1.0, K, dtype=torch.float64)
    noise = torch.randn(B, K, generator=g, dtype=torch.float64) * 0.3
    var = _t(v0[:, None] * torch.exp(spread[None, :] + noise))
    mu = _t(m0[:, None] + torch.randn(B, K, generator=g, dtype=torch.float64) * 0.1 * v0[:, None].sqrt())
    A = torch.full((B, K, K), 0.05 / max(K - 1, 1), dtype=torch.float64)
    A[:, range(K), range(K)] = 0.95 if K > 1 else 1.0
    logA = _t(torch.log(A))
    logpi = torch.full((B, K), -math.log(K), dtype=dt, device=get_device())

    prev = torch.full((B,), -math.inf, dtype=torch.float64)
    for _ in range(n_iter):
        gamma, log_xi_sum, logZ = _estep(x, mask, logpi, logA, mu, var)
        Nk = gamma.sum(1).clamp_min(1e-10)
        mu = (gamma * x[..., None]).sum(1) / Nk
        var = (gamma * (x[..., None] - mu[:, None]) ** 2).sum(1) / Nk + var_floor
        logpi = torch.log(gamma[:, 0].clamp_min(_tiny()))
        log_xi_sum = log_xi_sum.clamp_min(-700.0 if dt == torch.float64 else -80.0)
        logA = log_xi_sum - torch.logsumexp(log_xi_sum, dim=-1, keepdim=True)
        if torch.all((logZ - prev).abs() < tol * logZ.abs().clamp_min(1.0)):
            break
        prev = logZ

    gamma, _, logZ = _estep(x, mask, logpi, logA, mu, var)
    alpha = _forward(x, mask, logpi, logA, mu, var)[0]
    filt = torch.softmax(alpha, dim=-1)

    logZ = logZ.view(n, n_init)
    best = logZ.argmax(1)
    out = []
    for i in range(n):
        b = i * n_init + int(best[i])
        L = int(Mk[b].sum())
        order = torch.argsort(var[b])
        oi = order.cpu().numpy()
        out.append(dict(
            pi=logpi[b].exp()[order].cpu().numpy(),
            A=logA[b].exp()[order][:, order].cpu().numpy(),
            mu=mu[b][order].cpu().numpy(),
            sd=var[b][order].sqrt().cpu().numpy(),
            loglik=float(logZ[i, best[i]]),
            smoothed=gamma[b, :L].cpu().numpy()[:, oi],
            filtered=filt[b, :L].cpu().numpy()[:, oi],
        ))
    return out


def hmm_filter_batch(series, params, priors=None):
    """
    Filtered state probabilities and one-step-ahead predictive log densities
    for each series under fixed parameters (one params dict per series).
    priors: optional list of initial state distributions (defaults to params['pi']).
    """
    X, Mk = pad_batch(series)
    x, mask = _t(X), _mask_t(Mk)
    logpi = _t(np.stack([np.log(np.maximum(priors[i] if priors is not None else p['pi'], 1e-300))
                         for i, p in enumerate(params)]).clip(min=-80.0))
    logA = _t(np.log(np.maximum(np.stack([p['A'] for p in params]), 1e-300)).clip(min=-80.0))
    mu = _t(np.stack([p['mu'] for p in params]))
    var = _t(np.stack([p['sd'] ** 2 for p in params]))
    alpha, _, logb, c = _forward(x, mask, logpi, logA, mu, var)
    filt = torch.softmax(alpha, dim=-1)
    # one-step predictive density, per step: p(x_t | x_<t) = sum_k pred_t(k) b_t(k)
    log_filt = torch.log_softmax(alpha, dim=-1)
    log_pred = torch.cat([logpi[:, None], _log_matmul(log_filt[:, :-1, None, :], logA[:, None])[:, :, 0]], dim=1)
    log_pred = log_pred - torch.logsumexp(log_pred, dim=-1, keepdim=True)
    lpd = (torch.logsumexp(log_pred + logb, dim=-1).double().cpu() + c.double().cpu())
    res = []
    for i, s in enumerate(series):
        L = len(s)
        res.append((filt[i, :L].cpu().numpy(), lpd[i, :L].cpu().numpy()))
    return res


# ------------------------------------------------------------------
# Statistical jump model
# ------------------------------------------------------------------

def _jump_values(loss, mask, lam):
    """Min-plus forward values V (B, T, K) for per-step losses (B, T, K).
    Loss is 0.5 * squared Euclidean distance, matching the jumpmodels package."""
    B, T, K = loss.shape
    off = 1.0 - torch.eye(K, dtype=loss.dtype, device=loss.device)
    M = lam[:, None, None, None] * off + loss[:, 1:, None, :]
    ident = torch.full((K, K), math.inf, dtype=loss.dtype, device=loss.device)
    ident.fill_diagonal_(0.0)
    M = torch.where(mask[:, 1:, None, None], M, ident)
    V0 = loss[:, 0]
    if T == 1:
        return V0[:, None]
    P = _scan(M, _min_matmul)
    rest = (V0[:, None, :, None] + P).amin(dim=-2)
    return torch.cat([V0[:, None], rest], dim=1)


def _backtrack(V, lam, lengths):
    """Ex-post optimal state path from forward values (numpy, vectorized over the batch)."""
    V = V.cpu().numpy()
    lam = lam.cpu().numpy()
    B, T, K = V.shape
    s = np.zeros((B, T), int)
    rows = np.arange(B)
    last = lengths - 1
    s[rows, last] = V[rows, last].argmin(1)
    pen = lam[:, None, None] * (1 - np.eye(K))[None]
    for t in range(T - 2, -1, -1):
        active = t < last
        nxt = s[:, t + 1]
        cand = V[:, t] + pen[rows, :, nxt]
        s[active, t] = cand[active].argmin(1)
    return s


def fit_jump_batch(features, lams, K=2, n_init=5, n_iter=30, seed=0, n_fit=None):
    """
    Fit a statistical jump model to each feature matrix in `features`
    (list of (T_i, d) arrays) with jump penalty lams[i].
    n_fit: optional list; if given, only the first n_fit[i] rows are used to
    estimate the centroids, and the online labels cover all rows.

    Returns a list of dicts: centroids, ex_post (labels over the fitted rows),
    online (real-time labels over all rows), objective.
    """
    n = len(features)
    d = features[0].shape[1]
    T = max(len(f) for f in features)
    lengths_all = np.array([len(f) for f in features])
    lengths_fit = np.array(n_fit) if n_fit is not None else lengths_all
    F = np.zeros((n, T, d))
    for i, f in enumerate(features):
        F[i, :len(f)] = f
    rng = np.random.default_rng(seed)

    F_rep = np.repeat(F, n_init, axis=0)
    len_fit = np.repeat(lengths_fit, n_init)
    lam = _t(np.repeat(np.asarray(lams, float), n_init))
    B = len(F_rep)
    Xf = _t(F_rep)
    mask_fit = _mask_t(np.arange(T)[None, :] < len_fit[:, None])

    cent = np.zeros((B, K, d))
    for b in range(B):
        cent[b] = F_rep[b, rng.choice(len_fit[b], K, replace=False)]
    C = _t(cent)
    prev_labels = None
    for _ in range(n_iter):
        loss = 0.5 * ((Xf[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1) * mask_fit[..., None]
        V = _jump_values(loss, mask_fit, lam)
        labels = _backtrack(V, lam, len_fit)
        lab_t = torch.as_tensor(labels).to(get_device())
        onehot = torch.nn.functional.one_hot(lab_t, K).to(get_dtype()) * mask_fit[..., None]
        cnt = onehot.sum(1)
        newC = torch.einsum('btk,btd->bkd', onehot, Xf) / cnt.clamp_min(1)[..., None]
        C = torch.where(cnt[..., None] > 0, newC, C)
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels

    loss = 0.5 * ((Xf[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1) * mask_fit[..., None]
    V = _jump_values(loss, mask_fit, lam)
    labels = _backtrack(V, lam, len_fit)
    sw = np.array([(np.diff(labels[b, :len_fit[b]]) != 0).sum() for b in range(B)])
    obj = V.cpu().numpy()[np.arange(B), len_fit - 1].min(1)
    obj = obj.reshape(n, n_init)
    best = obj.argmin(1)

    out = []
    for i in range(n):
        b = i * n_init + int(best[i])
        out.append(dict(centroids=C[b].cpu().numpy(), ex_post=labels[b, :len_fit[b]],
                        objective=float(obj[i, best[i]]), n_switches=int(sw[b])))

    # online (real-time) labels over all rows with the fitted centroids
    Cbest = _t(np.stack([o['centroids'] for o in out]))
    Xall = _t(F)
    mask_all = _mask_t(np.arange(T)[None, :] < lengths_all[:, None])
    loss = 0.5 * ((Xall[:, :, None, :] - Cbest[:, None, :, :]) ** 2).sum(-1) * mask_all[..., None]
    V = _jump_values(loss, mask_all, _t(np.asarray(lams, float)))
    online = V.argmin(-1).cpu().numpy()
    for i in range(n):
        out[i]['online'] = online[i, :lengths_all[i]]
    return out
