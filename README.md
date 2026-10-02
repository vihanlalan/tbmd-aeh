# TBMD: Testing an Operationalization of the Adaptive Markets Hypothesis

**Can the Adaptive Markets Hypothesis Be Operationalized? A Pre-Specified, Out-of-Sample Test of Behavioral Regime Detection and Regime-Conditional Trading**

*Vihan Lalan*

This repository contains the code behind the paper `Revised_Paper_TBMD_Operationalizing_AMH.docx`. Every table and in-text number in the paper is generated from the analysis outputs by `src/build_paper.py`.

## What is tested

1. **Real-Time Efficiency Score (RES):** rolling Hurst, Lo-MacKinlay variance-ratio and Ljung-Box statistics on the market return.
2. **Behavioral Proxy Composite (BPC):** equal-weighted Z-scores of herding, autocorrelation, loss-aversion, VIX-based sentiment and Amihud illiquidity proxies.
3. **Regime-Conditional Filter (RCF):** a long/short decile strategy that is held only when RES is below a trailing threshold. It is compared with the identical strategy deployed unconditionally.

Design: 12 pre-declared configurations; selection on 2005–2020 only; a single evaluation on a 2021–2025 lockbox; unchanged application to DAX constituents; four independent U.S. stress definitions; transaction-cost sensitivity.

## Key results

| | Result |
|---|---|
| RES AUC vs. U.S. stress (VIX, realized vol, drawdown, NBER) | 0.44–0.58; every 95% CI includes 0.5 |
| RES AUC in Hamilton simulation (20 seeds) | mean 0.545 |
| Selected RCF Sharpe, development 2005–2020 | 0.14 (DSR 0.35) |
| Selected RCF Sharpe, lockbox 2021–2025 | −1.08 (95% CI −1.74 to −0.40) |
| RCF − Unconditional Sharpe (lockbox / full / Germany) | −0.56 / +0.19 / −0.41; none significant |
| Illiquidity proxy AUC vs. stress | 0.82–0.96 |

The regime-conditional filter does not survive out-of-sample testing. See the paper for the full results and limitations; survivorship bias is the main one, because constituents are current members.

## Reproduce

```bash
pip install -r requirements.txt
python src/build_dataset.py        # downloads Yahoo Finance + FRED data to data/cache/
python src/run_paper_analysis.py   # writes outputs/paper/*.csv
python src/build_paper.py          # regenerates the manuscript from those outputs
```

Yahoo Finance data is not redistributed. The paper's results use data downloaded on 26 September 2026, and re-downloads may differ slightly because of revisions to adjusted prices.

## Regime identification (work in progress, GPU)

`src/regime_identification.py` asks whether market regimes can be identified objectively, and how accurately. It uses simulations with known regimes, S&P 500 data from 1950 to 2025, agreement across methods, and comparisons of real-time and ex-post labels. The HMM and jump-model fits are batched in PyTorch (`src/gpu_regimes.py`) and run on an NVIDIA GPU through CUDA when one is available, otherwise on the CPU.

```bash
python src/validate_gpu_regimes.py                 # checks against hmmlearn / jumpmodels
python src/regime_identification.py --quick        # smoke test -> outputs/regimes_quick/
python src/regime_identification.py --device cuda  # full run -> outputs/regimes/
```

No results from the full run are reported yet.

## Files

- `src/build_dataset.py`: data download and cache.
- `src/run_paper_analysis.py`: all analyses reported in the paper.
- `src/build_paper.py`: manuscript generator.
- `src/behavioral_proxies.py`, `src/efficiency_score.py`, `src/backtest_engine.py`: framework components.
- `src/tbmd_framework.py`: Hamilton (1989) regime-switching simulator (the paper uses only its simulator).
- `src/main_analysis.py`, `src/tbmd_realdata.py`, `outputs/*.csv`: earlier exploratory pipeline, not used for the paper's results.

## License

MIT. See [LICENSE](LICENSE).
