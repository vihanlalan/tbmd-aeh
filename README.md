# TBMD Framework

**Operationalizing the Adaptive Markets Hypothesis: Observable Behavioral Proxies, Real-Time Efficiency Scoring, and Regime-Conditional Alpha Generation**

*Vihan Lalan — March 2026*

---

## Overview

This repository contains the complete empirical framework for the TBMD paper. The framework provides a testable, computable operationalization of Lo's (2004, 2017) **Adaptive Markets Hypothesis (AMH)** by replacing unobservable behavioral equations with five literature-grounded proxies.

### Core Contributions

1. **Behavioral Proxy Composite (BPC)** — Five observable proxies that replace the abstract BAM equation:
   - Herding (Christie & Huang 1995)
   - Loss Aversion (Ang et al. 2006)
   - Momentum Bias (Lo & MacKinlay 1988)
   - Investor Sentiment (Baker & Wurgler 2007)
   - Volatility Regime (Schwert 1989)

2. **Real-Time Efficiency Score (RES)** — A composite of Hurst exponent, Variance Ratio, and Ljung-Box test that quantifies market efficiency on a 0-to-1 scale in real time.

3. **Regime-Conditional Filter (RCF)** — A trading strategy that only deploys signals when the efficiency score indicates an inefficient regime, sitting in cash otherwise.

### Core Claim

> The Regime-Conditional Filter (RCF) produces a higher risk-adjusted return than deploying the same BPC signal unconditionally.

This is validated on both synthetic data (with ground-truth regime labels) and real S&P 100 data (2005–2024).

---

## Repository Structure

```
tbmd-aeh/
├── src/                          # Modular source code
│   ├── config.py                 # All hyperparameters
│   ├── simulation.py             # Hamilton (1989) regime-switching
│   ├── bpc.py                    # Behavioral Proxy Composite
│   ├── res.py                    # Real-Time Efficiency Score
│   ├── validation.py             # Walk-forward engine
│   ├── statistics.py             # DSR, VR test, BH correction
│   ├── data_ingestion.py         # Yahoo Finance, FRED data
│   └── visualization.py          # Publication-quality figures
│
├── scripts/                      # Entry points
│   ├── run_synthetic.py          # Synthetic validation pipeline
│   └── run_realdata.py           # Real S&P 100 validation pipeline
│
├── results/                      # Output figures and tables
│   └── realdata/
│       ├── tbmd_realdata_figures.png
│       ├── tbmd_realdata_comparison.png
│       └── tbmd_realdata_results.csv
│
├── data/                         # Data documentation
│   └── README.md
│
├── docs/                         # Methodology and reproducibility
│   ├── methodology.md
│   └── reproducibility.md
│
├── tbmd_framework.py             # Original monolithic synthetic code
├── tbmd_realdata.py              # Original monolithic real-data code
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/vihanlalan/tbmd-aeh.git
cd tbmd-aeh
pip install -r requirements.txt
```

### Run Synthetic Validation

```bash
python scripts/run_synthetic.py
```

This generates all figures and prints performance statistics for the Hamilton (1989) regime-switching simulation.

### Run Real-Data Validation

```bash
python scripts/run_realdata.py
```

Downloads S&P 100 data from Yahoo Finance (cached after first run), then runs the full walk-forward validation. Runtime: ~15-25 minutes.

---

## Key Results

### Synthetic Validation

| Strategy | Sharpe | DSR | P-value |
|----------|--------|-----|---------|
| RCF Strategy | 0.43 | 0.71 | 0.040 |
| Unconditional BPC | -2.47 | 0.00 | — |
| Buy & Hold | 0.24 | 0.70 | — |

### Real Data Validation (S&P 100, 2005–2024)

| Strategy | Sharpe | DSR | P-value |
|----------|--------|-----|---------|
| RCF Strategy | 0.610 | 0.980 | 0.040 |
| Unconditional BPC | 0.451 | 0.972 | 0.056 |
| Buy & Hold | 0.520 | 0.989 | 0.024 |

**Core claim: RCF Sharpe (0.610) > Unconditional Sharpe (0.451) ✓ PASS**

⚠️ *Survivorship bias warning: Results use current S&P 100 constituents. Actual returns likely 1-3% lower annualized.*

---

## Design Principles

- **Zero lookahead bias**: Every signal is lagged by ≥1 day before use
- **Walk-forward only**: No in-sample performance is reported
- **Multiple-testing corrected**: Benjamini-Hochberg FDR at 5%
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014) is the primary metric
- **All parameters stated explicitly** in `src/config.py`

---

## Key References

- Lo, A. W. (2004). "The Adaptive Markets Hypothesis." *Journal of Portfolio Management*.
- Lo, A. W. (2017). *Adaptive Markets: Financial Evolution at the Speed of Thought*. Princeton University Press.
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94–107.
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 357–384.
- Lo, A. W., & MacKinlay, A. C. (1988). "Stock market prices do not follow random walks." *Review of Financial Studies*, 1(1), 41–66.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
