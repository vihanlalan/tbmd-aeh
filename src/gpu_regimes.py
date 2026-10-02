"""
gpu_regimes.py
--------------
Batched, device-agnostic implementations of the regime models used in
regime_identification.py. Runs on an NVIDIA GPU through CUDA when one is
available and on the CPU otherwise.

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

import math
import numpy as np
import torch

_DEVICE = None
DTYPE = torch.float64


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


def set_device(name):
    """'cuda', 'cpu', or None for automatic selection."""
    global _DEVICE
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False. '
                           'Install a CUDA build of PyTorch: https://pytorch.org/get-started/locally/')
    _DEVICE = torch.device(name) if name else None
    return get_device()


def device_summary():
    d = get_device()
    if d.type == 'cuda':
        return f'cuda ({torch.cuda.get_device_name(d)})'
    return 'cpu (no CUDA device available)'


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=DTYPE, device=get_device())


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
    logb = -0.5 * (torch.log(2 * math.pi * var[:, None, :])
                   + (x[..., None] - mu[:, None, :]) ** 2 / var[:, None, :])
    return logb * mask[..., None]


def _log_identity(K):
    eye = torch.full((K, K), -math.inf, dtype=DTYPE, device=get_device())
    eye.fill_diagonal_(0.0)
    return eye


def _forward(x, mask, logpi, logA, mu, var):
    """Log forward variables alpha (B, T, K) and the transition stack M."""
    logb = _log_emissions(x, mask, mu, var)
    K = mu.shape[1]
    M = logA[:, None] + logb[:, 1:, None, :]
    M = torch.where(mask[:, 1:, None, None], M, _log_identity(K))
    alpha0 = logpi + logb[:, 0]
    if x.shape[1] == 1:
        return alpha0[:, None], M
    P = _scan(M, _log_matmul)
    alpha_rest = torch.logsumexp(alpha0[:, None, :, None] + P, dim=-2)
    return torch.cat([alpha0[:, None], alpha_rest], dim=1), M


def _estep(x, mask, logpi, logA, mu, var):
    alpha, M = _forward(x, mask, logpi, logA, mu, var)
    B, T, K = alpha.shape
    S = _scan(M, _log_matmul, reverse=True)
    beta = torch.cat([torch.logsumexp(S, dim=-1), torch.zeros(B, 1, K, dtype=DTYPE, device=x.device)], dim=1)
    logZ = torch.logsumexp(alpha[:, -1], dim=-1)
    log_gamma = alpha + beta - logZ[:, None, None]
    log_xi = alpha[:, :-1, :, None] + M + beta[:, 1:, None, :] - logZ[:, None, None, None]
    log_xi = torch.where(mask[:, 1:, None, None], log_xi, torch.tensor(-math.inf, dtype=DTYPE, device=x.device))
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
    x, mask = _t(X), torch.as_tensor(Mk, device=get_device())

    g = torch.Generator(device='cpu').manual_seed(seed)
    cnt = mask.sum(1).to(DTYPE)
    m0 = (x * mask).sum(1) / cnt
    v0 = (((x - m0[:, None]) ** 2) * mask).sum(1) / cnt
    spread = torch.linspace(-1.0, 1.0, K, dtype=DTYPE)
    noise = torch.randn(B, K, generator=g, dtype=DTYPE) * 0.3
    var = (v0[:, None].cpu() * torch.exp(spread[None, :] + noise)).to(get_device())
    mu = (m0[:, None].cpu() + torch.randn(B, K, generator=g, dtype=DTYPE) * 0.1 * v0[:, None].cpu().sqrt()).to(get_device())
    A = torch.full((B, K, K), 0.05 / max(K - 1, 1), dtype=DTYPE)
    A[:, range(K), range(K)] = 0.95 if K > 1 else 1.0
    logA = torch.log(A).to(get_device())
    logpi = torch.full((B, K), -math.log(K), dtype=DTYPE, device=get_device())

    prev = torch.full((B,), -math.inf, dtype=DTYPE, device=get_device())
    for _ in range(n_iter):
        gamma, log_xi_sum, logZ = _estep(x, mask, logpi, logA, mu, var)
        Nk = gamma.sum(1).clamp_min(1e-10)
        mu = (gamma * x[..., None]).sum(1) / Nk
        var = (gamma * (x[..., None] - mu[:, None]) ** 2).sum(1) / Nk + var_floor
        logpi = torch.log(gamma[:, 0].clamp_min(1e-300))
        log_xi_sum = log_xi_sum.clamp_min(-700.0)
        logA = log_xi_sum - torch.logsumexp(log_xi_sum, dim=-1, keepdim=True)
        if torch.all((logZ - prev).abs() < tol * logZ.abs().clamp_min(1.0)):
            break
        prev = logZ

    gamma, _, logZ = _estep(x, mask, logpi, logA, mu, var)
    alpha, _ = _forward(x, mask, logpi, logA, mu, var)
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
    x, mask = _t(X), torch.as_tensor(Mk, device=get_device())
    logpi = _t(np.stack([np.log(np.maximum(priors[i] if priors is not None else p['pi'], 1e-300))
                         for i, p in enumerate(params)]))
    logA = _t(np.log(np.maximum(np.stack([p['A'] for p in params]), 1e-300)))
    mu = _t(np.stack([p['mu'] for p in params]))
    var = _t(np.stack([p['sd'] ** 2 for p in params]))
    alpha, _ = _forward(x, mask, logpi, logA, mu, var)
    logZt = torch.logsumexp(alpha, dim=-1)
    lpd = torch.diff(logZt, dim=1, prepend=torch.zeros(len(series), 1, dtype=DTYPE, device=x.device))
    filt = torch.softmax(alpha, dim=-1)
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
    off = 1.0 - torch.eye(K, dtype=DTYPE, device=loss.device)
    M = lam[:, None, None, None] * off + loss[:, 1:, None, :]
    ident = torch.full((K, K), math.inf, dtype=DTYPE, device=loss.device)
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
    mask_fit = torch.as_tensor(np.arange(T)[None, :] < len_fit[:, None], device=get_device())

    cent = np.zeros((B, K, d))
    for b in range(B):
        cent[b] = F_rep[b, rng.choice(len_fit[b], K, replace=False)]
    C = _t(cent)
    prev_labels = None
    for _ in range(n_iter):
        loss = 0.5 * ((Xf[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1) * mask_fit[..., None]
        V = _jump_values(loss, mask_fit, lam)
        labels = _backtrack(V, lam, len_fit)
        lab_t = torch.as_tensor(labels, device=get_device())
        onehot = torch.nn.functional.one_hot(lab_t, K).to(DTYPE) * mask_fit[..., None]
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
    mask_all = torch.as_tensor(np.arange(T)[None, :] < lengths_all[:, None], device=get_device())
    loss = 0.5 * ((Xall[:, :, None, :] - Cbest[:, None, :, :]) ** 2).sum(-1) * mask_all[..., None]
    V = _jump_values(loss, mask_all, _t(np.asarray(lams, float)))
    online = V.argmin(-1).cpu().numpy()
    for i in range(n):
        out[i]['online'] = online[i, :lengths_all[i]]
    return out
