"""
================================================================================
TBMD Framework — Visualization
================================================================================

Generates publication-quality figures for both synthetic and real-data
validation pipelines.

Author: Vihan Lalan
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from .config import (
    SEED, REGIME_PARAMS, OUTPUT_DIR, SYNTHETIC_RESULTS,
    START_DATE, END_DATE,
)
from .statistics import compute_performance_metrics, deflated_sharpe_ratio


def plot_framework_results(
    returns_df:       pd.DataFrame,
    regime_series:    pd.Series,
    regimes:          dict,
    bpc_features:     pd.DataFrame,
    res_scores:       pd.DataFrame,
    wf_results:       dict,
    output_path:      str = OUTPUT_DIR + 'tbmd_validation_figures.png',
) -> None:
    """
    Generate the eight-panel validation figure for the synthetic paper.
    """
    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)

    PALETTE = {
        'RCF Strategy':        '#1f77b4',
        'Unconditional BPC':   '#d62728',
        'Momentum':            '#ff7f0e',
        'Mean Reversion':      '#2ca02c',
        'Buy & Hold':          '#9467bd',
    }
    REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    REGIME_NAMES  = {0: 'Efficient Bull', 1: 'Trending/Momentum', 2: 'Behavioral Crash'}

    # ── Panel 1: Market price path with true regime shading ──────────────────
    ax1 = fig.add_subplot(gs[0, :])
    mkt_ret = returns_df.mean(axis=1)
    cum_mkt = (1 + mkt_ret).cumprod()
    ax1.plot(cum_mkt.index, cum_mkt.values, 'k-', lw=1.2, alpha=0.8, label='Market')

    for rid, col in REGIME_COLORS.items():
        mask = regime_series == rid
        if mask.any():
            rdates = regime_series[mask].index
            for i in range(len(rdates) - 1):
                ax1.axvspan(rdates[i], rdates[i+1], alpha=0.10, color=col)

    legend_patches = [
        Patch(facecolor=REGIME_COLORS[i], alpha=0.5, label=REGIME_NAMES[i])
        for i in REGIME_COLORS
    ]
    ax1.legend(handles=legend_patches + [plt.Line2D([0],[0], color='k', label='Market')],
               loc='upper left', fontsize=8)
    ax1.set_title('Figure 1: Market Price Path with True Regime Labels (Ground Truth)',
                  fontweight='bold', fontsize=11)
    ax1.set_ylabel('Cumulative Return (Normalized)')
    ax1.tick_params(axis='x', rotation=30)

    # ── Panel 2: BPC components ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bpc_cols = ['herding_z', 'return_autocorr', 'sentiment', 'BPC_composite']
    bpc_plot = bpc_features[bpc_cols].rolling(21).mean().dropna()
    for col in bpc_cols[:-1]:
        ax2.plot(bpc_plot.index, bpc_plot[col], alpha=0.4, lw=0.8)
    ax2.plot(bpc_plot.index, bpc_plot['BPC_composite'], 'k-', lw=1.6,
             label='BPC Composite')
    ax2.axhline(0, color='k', lw=0.5, ls='--')
    ax2.set_title('Figure 2: BPC Proxy Components (21-day rolling)', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Standardized Score')
    ax2.legend(['Herding', 'Autocorr', 'Sentiment', 'BPC Composite'], fontsize=7, loc='upper right')
    ax2.tick_params(axis='x', rotation=30)

    # ── Panel 3: RES vs true regime ───────────────────────────────────────────
    ax3  = fig.add_subplot(gs[1, 1])
    ax3b = ax3.twinx()
    eff_plot    = res_scores['efficiency_score'].rolling(21).mean().dropna()
    regime_num  = regime_series.map({0: 1.0, 1: 0.5, 2: 0.0}).reindex(eff_plot.index).ffill()
    ax3.plot(eff_plot.index, eff_plot.values, 'b-', lw=1.2, alpha=0.8, label='RES')
    ax3b.fill_between(regime_num.index, regime_num.values, alpha=0.2, color='orange')
    ax3.set_ylabel('Efficiency Score (0=Inefficient, 1=Efficient)', color='blue')
    ax3b.set_ylabel('True Efficiency (1=Efficient, 0=Crash)', color='orange')
    corr = eff_plot.corr(regime_num)
    ax3.text(0.02, 0.05, f'Corr: {corr:.3f}', transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax3.set_title('Figure 3: RES vs True Regime (Validation of Core Claim)',
                  fontweight='bold', fontsize=10)
    ax3.tick_params(axis='x', rotation=30)

    # ── Panel 4: Hurst exponent ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    hurst_plot = res_scores['hurst'].rolling(21).mean().dropna()
    ax4.plot(hurst_plot.index, hurst_plot.values, 'purple', lw=1.2)
    ax4.axhline(0.5, color='k',   ls='--', lw=1,   label='H=0.5 (Efficient RW)')
    ax4.axhline(0.6, color='red', ls=':',  lw=0.8, label='H=0.6 (Trending)')
    ax4.fill_between(hurst_plot.index, 0.5, hurst_plot.values,
                     where=hurst_plot.values > 0.5, alpha=0.25, color='red', label='Trending')
    ax4.fill_between(hurst_plot.index, hurst_plot.values, 0.5,
                     where=hurst_plot.values < 0.5, alpha=0.25, color='blue', label='Mean-Rev')
    ax4.set_ylabel('Hurst Exponent')
    ax4.set_ylim([0.2, 0.8])
    ax4.set_title('Figure 4: Hurst Exponent (R/S Analysis)\nH > 0.5 = Trending | H < 0.5 = Mean-Reverting',
                  fontweight='bold', fontsize=10)
    ax4.legend(fontsize=7)
    ax4.tick_params(axis='x', rotation=30)

    # ── Panel 5: Walk-forward PnL comparison ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    strategies_pnl = {
        'RCF Strategy':       wf_results['rcf_pnl'],
        'Unconditional BPC':  wf_results['unconditional_pnl'],
        'Momentum':           wf_results['momentum_pnl'],
        'Buy & Hold':         wf_results['buyhold_pnl'],
    }
    pnl_start = wf_results['rcf_pnl'].ne(0).idxmax()
    for sname, pnl in strategies_pnl.items():
        cum = pnl.loc[pnl_start:].cumsum()
        lw  = 2.0 if sname == 'RCF Strategy' else 1.2
        ax5.plot(cum.index, cum.values, lw=lw, color=PALETTE[sname],
                 label=sname, alpha=0.9)
    ax5.axhline(0, color='k', lw=0.5)
    ax5.set_title('Figure 5: Walk-Forward Cumulative PnL\n(Net of 10bps Transaction Costs)',
                  fontweight='bold', fontsize=10)
    ax5.set_ylabel('Cumulative Log Return')
    ax5.legend(fontsize=8)
    ax5.tick_params(axis='x', rotation=30)

    # ── Panel 6: Rolling 6-month Sharpe ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    for sname, pnl in list(strategies_pnl.items())[:3]:
        roll_sr = (pnl.rolling(126).mean() /
                   (pnl.rolling(126).std() + 1e-10) * np.sqrt(252))
        ax6.plot(roll_sr.index, roll_sr.values, lw=1.2, color=PALETTE[sname],
                 label=sname, alpha=0.8)
    ax6.axhline(0,   color='k',     lw=0.8, ls='--')
    ax6.axhline(1.0, color='green', lw=0.8, ls=':',  alpha=0.7, label='SR=1.0')
    ax6.set_title('Figure 6: Rolling 6-Month Sharpe Ratio', fontweight='bold', fontsize=10)
    ax6.set_ylabel('Annualized Sharpe')
    ax6.set_ylim([-3, 4])
    ax6.legend(fontsize=8)
    ax6.tick_params(axis='x', rotation=30)

    # ── Panel 7: Return by regime ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 1])
    regime_perf = {}
    for rid in range(3):
        mask = regime_series == rid
        regime_perf[REGIME_NAMES[rid]] = {
            'RCF':        wf_results['rcf_pnl'][mask].mean() * 252 * 100,
            'Uncond BPC': wf_results['unconditional_pnl'][mask].mean() * 252 * 100,
            'Momentum':   wf_results['momentum_pnl'][mask].mean() * 252 * 100,
        }
    rdf = pd.DataFrame(regime_perf).T
    x   = np.arange(len(rdf))
    w   = 0.25
    cols = ['#1f77b4', '#d62728', '#ff7f0e']
    for i, col_name in enumerate(rdf.columns):
        ax7.bar(x + i*w, rdf[col_name].values, w, label=col_name,
                alpha=0.8, color=cols[i])
    ax7.axhline(0, color='k', lw=0.8)
    ax7.set_xticks(x + w)
    ax7.set_xticklabels(rdf.index, rotation=15, fontsize=8)
    ax7.set_title('Figure 7: Ann. Return (%) by True Market Regime\n(Regime-Stratified Analysis)',
                  fontweight='bold', fontsize=10)
    ax7.set_ylabel('Annualized Return (%)')
    ax7.legend(fontsize=8)

    # ── Panel 8: Performance table ────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')

    all_strats = {
        'RCF Strategy':       wf_results['rcf_pnl'],
        'Unconditional BPC':  wf_results['unconditional_pnl'],
        'Momentum':           wf_results['momentum_pnl'],
        'Mean Reversion':     wf_results['mean_rev_pnl'],
        'Buy & Hold':         wf_results['buyhold_pnl'],
    }
    table_rows = []
    for sname, pnl in all_strats.items():
        m   = compute_performance_metrics(pnl[pnl != 0], sname)
        if not m:
            continue
        sr, dsr = deflated_sharpe_ratio(pnl[pnl != 0], n_trials=12)
        table_rows.append([
            sname,
            f"{m['Ann_Return_%']:.2f}%",
            f"{m['Ann_Vol_%']:.2f}%",
            f"{m['Sharpe']:.3f}",
            f"{dsr:.3f}" if dsr and not np.isnan(dsr) else 'N/A',
            f"{m['Max_Drawdown_%']:.2f}%",
            f"{m['Win_Rate_%']:.1f}%",
            f"{m['P_value']:.4f}",
            str(m['Significant_5pct']),
        ])

    col_labels = ['Strategy', 'Ann Ret', 'Ann Vol', 'Sharpe', 'DSR*',
                  'Max DD', 'Win %', 'P-value', 'Sig@5%']
    tbl = ax8.table(cellText=table_rows, colLabels=col_labels,
                    cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.90])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    ax8.set_title(
        'Table 1: Full Walk-Forward Performance Statistics (Net of 10bps Costs)\n'
        '* DSR = Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) — '
        'primary metric; N_trials=12; > 0.95 = credible signal',
        fontweight='bold', fontsize=9, pad=14,
    )

    plt.suptitle(
        'TBMD Framework: Rigorous Empirical Validation\n'
        'Hamilton (1989) Regime-Switching Simulation | Walk-Forward | '
        'Net of Transaction Costs | BH Multiple-Testing Correction',
        fontsize=13, fontweight='bold', y=0.998,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure saved → {output_path}")


def plot_multiple_testing_analysis(
    n_trials:  int   = 420,
    n_obs:     int   = 750,
    output_path: str = OUTPUT_DIR + 'tbmd_multiple_testing.png',
) -> None:
    """
    Visualize the multiple testing problem.

    Shows why an uncorrected SR=1.2 is not evidence of a real signal
    when 420 parameter combinations are tested.
    """
    np.random.seed(SEED)

    null_sharpes = np.random.normal(0, 1.0 / np.sqrt(n_obs), size=n_trials)
    null_pvals   = 2 * (1 - norm.cdf(np.abs(null_sharpes * np.sqrt(n_obs))))
    sorted_pvals = np.sort(null_pvals)
    bh_thresh    = np.arange(1, n_trials + 1) * 0.05 / n_trials
    bonferroni   = 0.05 / n_trials

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('white')

    # Panel A: Null SR distribution
    ax = axes[0]
    ax.hist(null_sharpes * np.sqrt(252), bins=40, color='steelblue', alpha=0.7, ec='white')
    n_fp = int(np.sum(np.abs(null_sharpes * np.sqrt(252)) > 0.5))
    ax.axvline( 0.5, color='red', ls='--', lw=2, label='SR=0.5 threshold')
    ax.axvline(-0.5, color='red', ls='--', lw=2)
    ax.axvline( 1.2, color='orange', ls='-', lw=2, label='Paper claims SR=1.2')
    ax.set_title(
        f'A: Null SR Distribution — 420 Random Strategies\n'
        f'{n_fp} strategies show SR > 0.5 by pure chance',
        fontsize=10, fontweight='bold',
    )
    ax.set_xlabel('Annualized Sharpe Ratio')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

    # Panel B: BH correction
    ax = axes[1]
    k  = np.arange(1, n_trials + 1)
    ax.plot(k, sorted_pvals, 'b-',  lw=1.5, label='Sorted p-values (420 null strategies)')
    ax.plot(k, bh_thresh,    'r--', lw=2,   label='BH threshold (FDR=5%)')
    ax.axhline(bonferroni, color='orange', ls=':', lw=2,
               label=f'Bonferroni ({bonferroni:.5f})')
    ax.axhline(0.05, color='green', ls='-.', lw=1.5, label='Naive p<0.05')
    naive_rej = int(np.sum(sorted_pvals < 0.05))
    bh_rej    = int(np.sum(sorted_pvals <= bh_thresh))
    ax.set_title(
        f'B: Multiple Testing Correction\n'
        f'Naive: {naive_rej} "significant" | BH-corrected: {bh_rej}',
        fontsize=10, fontweight='bold',
    )
    ax.set_xlabel('Rank of p-value')
    ax.set_ylabel('p-value')
    ax.set_ylim([0, 0.15])
    ax.legend(fontsize=8)
    ax.text(0.35, 0.72,
            f'Naive: {naive_rej} significant\nAfter BH: {bh_rej} significant\n\n'
            f'The paper uses naive p<0.05\nwithout any correction.',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', fc='#fadbd8', alpha=0.9))

    # Panel C: Expected max SR vs n_trials
    ax          = axes[2]
    n_range     = np.arange(10, 1001, 10)
    exp_max_sr  = [norm.ppf(1 - 0.05 / n) / np.sqrt(n_obs) * np.sqrt(252) for n in n_range]
    at_420      = norm.ppf(1 - 0.05 / 420) / np.sqrt(n_obs) * np.sqrt(252)
    ax.plot(n_range, exp_max_sr, 'b-', lw=2, label='Expected max SR by chance')
    ax.axhline(1.2, color='red',    ls='--', lw=2, label='Paper claims SR=1.2')
    ax.axhline(0.8, color='orange', ls=':',  lw=1.5, label='Minimum "interesting" SR')
    ax.axvline(420, color='purple', ls='--', lw=1.5, label='Paper grid size (420)')
    ax.scatter([420], [at_420], color='red', s=100, zorder=5)
    ax.annotate(
        f'Expected max SR\nby chance: {at_420:.2f}\n(> paper\'s claimed 1.2!)',
        xy=(420, at_420), xytext=(520, at_420 + 0.5),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=9, color='red',
    )
    ax.set_title(
        'C: Expected Max SR by Chance vs #Strategies Tried\n'
        'Bailey & Lopez de Prado (2014) Framework',
        fontsize=10, fontweight='bold',
    )
    ax.set_xlabel('Number of strategies / parameters tested')
    ax.set_ylabel('Expected maximum SR (by chance)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim([0, 800])

    plt.suptitle(
        'Why the Original Paper\'s SR=1.2 is Not Evidence of a Real Signal\n'
        f'With 420 tested combinations over {n_obs} obs., expected max SR by chance = {at_420:.2f}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure saved → {output_path}")


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
    output_path:  str = 'tbmd_realdata_figures.png',
) -> None:
    """
    8-Panel validation figure — Real Market Data.

    Layout mirrors the synthetic paper figure for direct visual comparison.
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
    output_path:  str = 'tbmd_realdata_comparison.png',
) -> None:
    """
    Side-by-side comparison of synthetic vs real Sharpe and DSR.
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
    ax2.barh([s for s, _ in directions_match],
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
