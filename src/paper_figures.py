"""
paper_figures.py
----------------
Builds the figures used in main.tex from the regime-study outputs.

    python src/paper_figures.py [results_dir]     # default: outputs/regimes

Writes to figures/:
    F1_regimes_timeline.png      copied from the results (made by regime_identification.py)
    F2_simulation_accuracy.png   redrawn with the legend outside the plot area
    F3_agreement_vs_null.png     copied from the results
    F4_efficiency_by_regime.png  AR(1) by real-time state and the regime-vs-volatility comparison
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'outputs', 'regimes')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

for f in ('F1_regimes_timeline.png', 'F3_agreement_vs_null.png'):
    shutil.copy(os.path.join(RES, f), os.path.join(FIG, f))

# F2: simulation accuracy
acc = pd.read_csv(os.path.join(RES, 'A1_sim_accuracy_summary.csv'))
meth = ['HMM-2 ex post (smoothed)', 'HMM-2 real time (filtered)', 'Jump λ=30 ex post', 'Jump λ=30 real time',
        'Change-point (PELT) ex post']
dg = ['HMM2 (correctly specified)', 'HMM2 with t(4) shocks (misspecified)', 'HMM3 (three regimes)']
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.8 / len(meth)
for j, m in enumerate(meth):
    sub = acc[acc.method == m].set_index('dgp').reindex(dg)
    ax.bar(np.arange(len(dg)) + j * w, sub.balanced_accuracy, w, yerr=sub.ba_sd, label=m, capsize=2)
ax.axhline(0.8, color='grey', ls='--', lw=0.8)
ax.set_xticks(np.arange(len(dg)) + 0.4 - w / 2)
ax.set_xticklabels(dg, fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('balanced accuracy, high-volatility state')
ax.legend(fontsize=8, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F2_simulation_accuracy.png'), dpi=200)
plt.close(fig)

# F4: efficiency by real-time state, and regime label vs volatility
eff = pd.read_csv(os.path.join(RES, 'C1_ar1_by_regime.csv'))
race = pd.read_csv(os.path.join(RES, 'C3_regime_vs_volatility.csv'))
eff = eff[eff['type'] == 'real time'].copy()
eff['period'] = eff['period'].str.replace('all (', '', regex=False).str.replace(')', '', regex=False)
eff.loc[eff.period == '1950-1979', 'period'] = '1970-1979'   # real-time labels start in 1970
order = ['1970-2025', '1970-1979', '1980-1999', '2000-2025']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [1.25, 1]})
for i, lab in enumerate(['HMM-2 real time', 'Jump λ=30 real time']):
    sub = eff[eff['labels'] == lab].set_index('period').reindex(order)
    x = np.arange(len(order)) + (i - 0.5) * 0.36
    for col, tcol, mk, name in [('ar1_low', 't_low', 'o', 'low-volatility state'),
                                ('ar1_high', 't_high', 's', 'high-volatility state')]:
        se = (sub[col] / sub[tcol]).abs()
        a1.errorbar(x + (0.08 if mk == 's' else -0.08), sub[col], yerr=1.96 * se, fmt=mk, ms=5, capsize=2,
                    color=('tab:blue' if mk == 'o' else 'tab:red'), alpha=1.0 if i == 0 else 0.55,
                    label=f'{name} ({lab.split(" real")[0]})')
a1.axhline(0, color='black', lw=0.6)
a1.set_xticks(np.arange(len(order)))
a1.set_xticklabels(order, fontsize=9)
a1.set_ylabel('AR(1) of daily returns (95% CI)')
a1.set_title('Autocorrelation by real-time volatility state', fontsize=10)
a1.legend(fontsize=7, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
sub = race[race.period.str.endswith('2025') & race.period.str.startswith('1970')].set_index('labels')
labs = ['HMM-2 real time', 'Jump λ=30 real time']
cols = [('regime_only_t', 'regime label alone'), ('vol_only_t', 'volatility alone'),
        ('both_regime_t', 'both: regime label'), ('both_vol_t', 'both: volatility')]
w = 0.2
for j, (c, name) in enumerate(cols):
    a2.bar(np.arange(len(labs)) + (j - 1.5) * w, sub.reindex(labs)[c], w, label=name)
for y in (-1.96, 1.96):
    a2.axhline(y, color='grey', ls='--', lw=0.8)
a2.axhline(0, color='black', lw=0.6)
a2.set_xticks(np.arange(len(labs)))
a2.set_xticklabels(labs, fontsize=9)
a2.set_ylabel('White t of interaction with r(t-1)')
a2.set_title('Regime label vs volatility level, 1970-2025', fontsize=10)
a2.legend(fontsize=7, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F4_efficiency_by_regime.png'), dpi=200)
plt.close(fig)
print('figures written to', os.path.normpath(FIG))
