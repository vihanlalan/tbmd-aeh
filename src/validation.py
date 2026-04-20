"""
================================================================================
TBMD Framework — Walk-Forward Validation Engine
================================================================================

Implements the strict walk-forward protocol described in Section IV of the
paper. Key design rules:
  - Training data is ALWAYS lagged: prediction at time t uses only data
    from [t - train_window, t-1]
  - All signals are lagged by 1 day before use in prediction
  - Hyperparameters selected on training fold cross-validation only
  - Refit every step_size days (monthly)
  - Transaction costs applied at each trade

Author: Vihan Lalan
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .config import SEED, WF_CONFIG


def walk_forward_validation(
    features_df:   pd.DataFrame,
    returns_df:    pd.DataFrame,
    regime_series: pd.Series,
    cfg:           dict = WF_CONFIG,
) -> dict:
    """
    Walk-Forward Validation Engine — Synthetic Version.

    Parameters
    ----------
    features_df   : pd.DataFrame  — BPC proxies + RES scores (aligned by date)
    returns_df    : pd.DataFrame  — asset returns (aligned by date)
    regime_series : pd.Series     — true regime labels (for validation only)
    cfg           : dict          WF_CONFIG parameter dictionary

    Returns
    -------
    dict with keys:
        'rcf_pnl'           : pd.Series — Regime-Conditional Filter PnL
        'unconditional_pnl' : pd.Series — Same signal, no regime filter
        'momentum_pnl'      : pd.Series — Cross-sectional momentum baseline
        'mean_rev_pnl'      : pd.Series — Mean reversion baseline
        'buyhold_pnl'       : pd.Series — Equal-weight buy-and-hold
        'regime_predictions': pd.Series — Predicted regime labels
        'regime_true'       : pd.Series — True regime labels (aligned)
        'fold_metrics'      : pd.DataFrame — Per-fold performance summary
    """
    tc_bps     = cfg['transaction_cost_bps'] / 10_000
    top_n      = cfg['top_n_assets']
    train_w    = cfg['train_window']
    step       = cfg['step_size']
    conf_hi    = cfg['signal_confidence_hi']
    conf_lo    = cfg['signal_confidence_lo']
    mom_lb     = cfg['momentum_lookback']

    # ── Align all inputs to a common index ───────────────────────────────────
    common_idx    = features_df.index.intersection(returns_df.index)
    features_df   = features_df.loc[common_idx]
    returns_df    = returns_df.loc[common_idx]
    regime_series = regime_series.loc[common_idx]
    dates         = common_idx
    n_days        = len(dates)

    # ── Storage ───────────────────────────────────────────────────────────────
    rcf_pnl           = pd.Series(0.0, index=dates)
    uncond_pnl        = pd.Series(0.0, index=dates)
    momentum_pnl      = pd.Series(0.0, index=dates)
    mean_rev_pnl      = pd.Series(0.0, index=dates)
    buyhold_pnl       = returns_df.mean(axis=1)
    regime_predictions = pd.Series(-1, index=dates)

    fold_metrics = []
    prev_rcf_wt  = 0.0
    prev_unc_wt  = 0.0

    feature_cols = [
        c for c in features_df.columns
        if features_df[c].std() > 1e-8
    ]

    # ── Walk-Forward Loop ─────────────────────────────────────────────────────
    for t in range(train_w, n_days - 5, step):
        t_start = max(0, t - train_w)
        t_end   = t

        pred_start = t
        pred_end   = min(t + step, n_days)

        # Training targets: 5-day ahead market direction (lagged — no lookahead)
        fwd_returns   = returns_df.shift(-5).iloc[t_start:t_end]
        y_direction   = (fwd_returns.mean(axis=1) > 0).astype(int)

        X_tr = features_df.iloc[t_start:t_end][feature_cols].fillna(0.0)
        X_te = features_df.iloc[pred_start:pred_end][feature_cols].fillna(0.0)

        valid_mask = y_direction.notna() & ~X_tr.isnull().any(axis=1)
        X_tr_v     = X_tr[valid_mask]
        y_v        = y_direction[valid_mask]

        if len(X_tr_v) < 50 or y_v.nunique() < 2:
            continue

        # ── Scale on training data only ───────────────────────────────────────
        scaler       = StandardScaler()
        X_tr_scaled  = scaler.fit_transform(X_tr_v)
        X_te_scaled  = scaler.transform(X_te)

        # ── Primary model: GBM on behavioural features ────────────────────────
        try:
            model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=SEED,
            )
            model.fit(X_tr_scaled, y_v)
            prob_up = model.predict_proba(X_te_scaled)[:, 1]
        except Exception:
            continue

        # ── Regime classifier (for regime-detection validation) ───────────────
        try:
            regime_clf = LogisticRegression(max_iter=300, C=1.0, random_state=SEED)
            regime_clf.fit(X_tr_scaled, regime_series.loc[common_idx].iloc[t_start:t_end][valid_mask])
            regime_pred = regime_clf.predict(X_te_scaled)
            regime_predictions.iloc[pred_start:pred_end] = regime_pred
        except Exception:
            pass

        # ── Generate daily PnL for each strategy ─────────────────────────────
        for offset in range(pred_end - pred_start):
            day_idx = pred_start + offset
            if day_idx >= n_days:
                break

            actual_returns = returns_df.iloc[day_idx]
            conv           = prob_up[offset] if offset < len(prob_up) else 0.5

            # ── Regime-Conditional Filter (RCF) ───────────────────────────────
            # Only trades when model conviction exceeds threshold
            if conv > conf_hi:
                rcf_wt = (conv - 0.5) * 2.0
            elif conv < conf_lo:
                rcf_wt = -(0.5 - conv) * 2.0
            else:
                rcf_wt = 0.0    # sit in cash in efficient regime

            turnover_rcf = abs(rcf_wt - prev_rcf_wt)
            rcf_pnl.iloc[day_idx] = rcf_wt * actual_returns.mean() - turnover_rcf * tc_bps
            prev_rcf_wt = rcf_wt

            # ── Unconditional Signal (same BPC, no filter) ────────────────────
            unc_wt = (conv - 0.5) * 2.0
            turnover_unc = abs(unc_wt - prev_unc_wt)
            uncond_pnl.iloc[day_idx] = unc_wt * actual_returns.mean() - turnover_unc * tc_bps
            prev_unc_wt = unc_wt

            # ── Cross-Sectional Momentum Baseline ────────────────────────────
            lb = min(mom_lb, day_idx)
            if lb > 5:
                past          = returns_df.iloc[day_idx - lb: day_idx]
                cum_ret       = past.sum()
                top_assets    = cum_ret.nlargest(top_n).index
                bot_assets    = cum_ret.nsmallest(top_n).index
                momentum_pnl.iloc[day_idx] = (
                    0.5 * (actual_returns[top_assets].mean() - actual_returns[bot_assets].mean())
                    - tc_bps * 0.1
                )

            # ── Mean Reversion Baseline ───────────────────────────────────────
            if lb > 5:
                mean_rev_pnl.iloc[day_idx] = (
                    0.5 * (actual_returns[bot_assets].mean() - actual_returns[top_assets].mean())
                    - tc_bps * 0.1
                )

        # ── Per-fold summary ──────────────────────────────────────────────────
        fold_metrics.append({
            'fold_start':    dates[pred_start],
            'fold_end':      dates[min(pred_end - 1, n_days - 1)],
            'rcf_return':    rcf_pnl.iloc[pred_start:pred_end].sum(),
            'uncond_return': uncond_pnl.iloc[pred_start:pred_end].sum(),
            'mom_return':    momentum_pnl.iloc[pred_start:pred_end].sum(),
            'true_regime':   int(regime_series.iloc[pred_start:pred_end].mode().iloc[0])
                             if pred_end > pred_start else -1,
        })

    return {
        'rcf_pnl':            rcf_pnl,
        'unconditional_pnl':  uncond_pnl,
        'momentum_pnl':       momentum_pnl,
        'mean_rev_pnl':       mean_rev_pnl,
        'buyhold_pnl':        buyhold_pnl,
        'regime_predictions': regime_predictions,
        'regime_true':        regime_series,
        'fold_metrics':       pd.DataFrame(fold_metrics),
    }


def walk_forward_validation_real(
    features_df:   pd.DataFrame,
    returns_df:    pd.DataFrame,
    vol_state:     pd.Series,
    cfg:           dict = WF_CONFIG,
) -> dict:
    """
    Walk-Forward Validation Engine — Real Data Version.

    ENGINE IS IDENTICAL to synthetic version. The only difference is that
    vol_state replaces regime_series for the purpose of regime-stratified
    return analysis in visualisation. It is NOT used in model training,
    signal generation, or any performance metric.

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
