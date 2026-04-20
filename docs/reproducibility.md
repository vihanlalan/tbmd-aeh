# Reproducing Results

## Prerequisites

- Python >= 3.9
- Internet connection (for first-run data download)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run Synthetic Validation

```bash
python scripts/run_synthetic.py
```

**Expected output**: Performance statistics printed to stdout, plus two figures saved:
- `tbmd_validation_figures.png` — 8-panel validation chart
- `tbmd_multiple_testing.png` — Multiple testing analysis

**Runtime**: ~2-5 minutes

**Random seed**: Fixed at 42 (`src/config.py`). Results are deterministic.

## Step 3: Run Real-Data Validation

```bash
python scripts/run_realdata.py
```

**Expected output**:
- `tbmd_realdata_figures.png` — 8-panel real-data validation chart
- `tbmd_realdata_comparison.png` — Synthetic vs real comparison
- `tbmd_realdata_results.csv` — Full performance table

**Runtime**: ~15-25 minutes (walk-forward over ~4,800 trading days)

**First run**: Downloads ~9 MB of price data from Yahoo Finance. Data is cached to `price_cache.csv`, `spy_cache.csv`, and `sentiment_cache.csv` for subsequent runs.

## Reproducibility Notes

1. **Random seed**: All stochastic components use `SEED = 42`.
2. **Data caching**: Yahoo Finance data may differ slightly between download dates due to price adjustments. The cached CSVs in the repository correspond to the paper's reported results.
3. **Library versions**: Results were generated with NumPy 1.24, Pandas 2.0, SciPy 1.11, scikit-learn 1.3.
4. **Survivorship bias**: Results on real data use current S&P 100 constituents. For bias-free results, use CRSP/Compustat data via WRDS.

## Using the Original Monolithic Files

The original single-file implementations are preserved in the repository root:

```bash
python tbmd_framework.py    # Synthetic validation (standalone)
python tbmd_realdata.py     # Real-data validation (standalone)
```

These produce identical results to the modular `scripts/` versions.
