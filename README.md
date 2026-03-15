# CSD x MAG Energy Solutions — Data Challenge 2026

## Objective

Select **profitable electricity trading opportunities** (FTRs) for the following month, based on simulation data and historical market data available at decision time.

An opportunity is a triplet `(EID, MONTH, PEAKID)` and is profitable when `|PR| - C > 0`.

## Approach

**Two-stage pipeline** with rank-based selection:

1. **Classifier** (LightGBM) — predicts P(profitable) per opportunity
2. **Regressor** (LightGBM) — predicts expected profit magnitude (log1p scale)
3. **Selection** — filters by classifier confidence, ranks by blended score `alpha * rank(proba) + (1-alpha) * rank(profit)`, selects top-K per month (constrained to [10, 100])

Hyperparameters optimized end-to-end with **Optuna** (200 trials, combined F1 + profit objective).

## Results (2023-H2 test set)

| Metric | Value |
|--------|-------|
| F1-score (avg ON/OFF) | 0.259 |
| Net Profit | 711,987 |
| ROC AUC | 0.874 |

## Usage

### Generate predictions for a target period

```bash
pip install -r requirements.txt
python main.py --start-month 2024-01 --end-month 2024-12
```

This produces `opportunities.csv` with columns: `TARGET_MONTH`, `PEAK_TYPE`, `EID`.

### Evaluate predictions

```bash
python evaluate.py opportunities.csv --start-month 2024-01 --end-month 2024-12
```

## Data Structure

```
data/
├── costs/          # Monthly costs (Parquet)
├── prices/         # Realized hourly prices (Parquet)
├── sim_monthly/    # Monthly simulations (Parquet, by year)
└── sim_daily/      # Daily simulations (Parquet, by year)
```

Data files are not included in this repository (too large). Place parquet files in the above structure.

## Key Design Decisions

- **Anti-leakage**: cutoff at day 7 of month M. PR/PROFIT lags use `shift(2)` (not shift(1)). Only cost C uses shift(1).
- **Walk-forward CV**: expanding window, temporal split. No future data leaks into training.
- **Hybrid EID universe**: union of market-validated EIDs (costs/prices) + strong-simulation EIDs.
- **Imbalance handling**: `scale_pos_weight=16.6` (Optuna-optimized), not SMOTE.
- **Feature engineering**: 52 features from sim_daily (PSD, impacts), sim_monthly (PSM), historical lags, and calendar.

## Files

| File | Description |
|------|-------------|
| `main.py` | Production pipeline (raw data to opportunities.csv) |
| `evaluate.py` | Jury evaluation script (F1 + profit) |
| `requirements.txt` | Python dependencies |
| `notebook/modeling.ipynb` | Full modeling notebook with baselines, Optuna, SHAP |
| `notebook/master_dataset_hybrid_v3.ipynb` | Master dataset construction |
| `notebook/EDA_sim_daily.ipynb` | Exploratory data analysis |
| `FINDINGS.md` | Analytical journal (all findings documented) |

## Team

CSD Data Challenge 2026 Edition — University competition.

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list
