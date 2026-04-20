# Methodology

## Framework Architecture

The TBMD framework operationalizes the Adaptive Markets Hypothesis (Lo 2004, 2017) through three interconnected components:

### 1. Behavioral Proxy Composite (BPC)

The AMH posits a behavioral market balance equation:

```
MB(b,t) = Σ_i B(i,b,t) · W(i,t)
```

where B(i,b,t) represents individual behavioral biases and W(i,t) their market weights. This is unobservable. The BPC replaces it with five computable proxies:

| Proxy | Formula | Behavioral Construct |
|-------|---------|---------------------|
| Herding (Z) | -σ_cs(r_t) rolling Z-score | Cross-sectional dispersion → conformity |
| Loss Aversion | σ_down / σ_up - 1 | Downside/upside vol asymmetry |
| Autocorrelation | ρ(r_t, r_{t-1}) over 60 days | Momentum bias / information lag |
| Sentiment | UMCSENT Z-score (lagged 1m) | Investor sentiment |
| Vol Ratio | σ_20 / σ_252 | Stress regime detection |

The composite is formed by equal-weight averaging the rolling Z-scores of four core proxies.

### 2. Real-Time Efficiency Score (RES)

Three statistical tests, each measuring a different aspect of market efficiency:

```
Hurst_eff(t)  = 1 - 2|H(t) - 0.5|
VR_eff(t)     = 1 / (1 + |VR(t) - 1|)
LB_eff(t)     = p-value of Ljung-Box test

RES(t) = (1/3) · [Hurst_eff + VR_eff + LB_eff]
```

- **RES ≈ 1.0**: Market near random walk → signals unlikely predictive
- **RES ≈ 0.0**: Behavioral regime → BPC signals potentially informative

### 3. Regime-Conditional Filter (RCF)

The RCF deploys the BPC signal only when model conviction exceeds a threshold:

```
if P(up) > 0.60:  weight = (P(up) - 0.5) × 2     # long
if P(up) < 0.40:  weight = -(0.5 - P(up)) × 2    # short
otherwise:         weight = 0                       # cash
```

## Validation Protocol

### Walk-Forward Design

- **Training window**: 252 days (1 year), rolling
- **Refit frequency**: Every 21 days (monthly)
- **Signal lag**: All features lagged by ≥1 day
- **Transaction costs**: 10 bps one-way applied at every trade
- **Model**: Gradient Boosting Classifier (50 trees, depth 3)
- **Target**: 5-day ahead market direction (binary)

### Statistical Testing

- **Primary metric**: Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
- **Multiple testing**: Benjamini-Hochberg FDR at 5%
- **Significance**: One-sample t-test on daily returns (p < 0.05)

### Dual Validation Approach

1. **Synthetic** (Hamilton 1989 simulation): Provides ground-truth regime labels for validating RES detection accuracy
2. **Real data** (S&P 100, 2005–2024): Validates the core tradeable claim (RCF > Unconditional) without requiring regime labels
