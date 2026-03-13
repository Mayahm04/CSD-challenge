# CLAUDE.md — MAG Energy Solutions Data Challenge

## Project Overview

University data science competition (CSD x MAG Energy Solutions, 2026 Edition). We build an algorithm that selects **profitable electricity trading opportunities** (FTRs — Financial Transmission Rights) for the following month, based on simulation data available at decision time.

**My role**: EDA and feature engineering on the `sim_daily` dataset. Team of 4, each member handles one dataset (costs, prices, sim_monthly, sim_daily). **Deadline: <1 week.**

---

## Problem Definition

An **opportunity** is a triplet `(EID, MONTH, PEAKID)` where:
- `EID`: network element identifier
- `MONTH`: target month (YYYY-MM)
- `PEAKID`: 0 = OFF-Peak, 1 = ON-Peak

An opportunity is **profitable** when: `PROFIT = |PR| - |C| > 0` *(updated formula confirmed by jury)*
- `|PR|` = ABS(SUM(PRICEREALIZED)) for that (EID, MONTH, PEAKID) — absolute value of monthly sum
- `|C|` = ABS(cost) from costs table
- Sign of prices reflects energy flow direction only; **magnitude determines profitability**

**Selection constraint**: between 10 and 100 opportunities per target month (total ON + OFF).

**Profitability rate**: ~14.78% with updated formula (was 1.01% with old PR - C formula). Still imbalanced but learnable.

### EID Universe — Hybrid approach
The universe is NOT limited to the 927 EIDs in costs. It is the **union** of:
- **Pool 1 (market-validated)**: EIDs in `costs` OR `prices` (known to have traded)
- **Pool 2 (strong-sim)**: EIDs with high ACTIVATIONLEVEL AND high |PSM|/|PSD| (p80 thresholds) from simulations, not in Pool 1
- `is_sim_only` flag distinguishes the two pools (model feature)

### Modeling strategy — Two-stage filter
To optimize both F1 (25%) and Net Profit (25%):
1. **Classifier** (e.g. LightGBM, `class_weight='balanced'`) → P(profitable) per opportunity
2. **Regressor** (e.g. LightGBM on `PROFIT`) → expected profit magnitude
3. **Selection**: filter by classifier confidence → rank by predicted PROFIT → select top-K, K ∈ [10, 100]

Master dataset keeps both `TARGET` (binary) and `PROFIT` (continuous) + `PR_signed` (directional).

---

## Anti-Leakage Rule (CRITICAL)

Decision is made on the **7th day of month M (inclusive)** for target month **M+1**.

### Allowed at cutoff (7th of M, 23:59:59):
- Full history of realized prices and costs up to M-1
- Realized prices from M up to the 7th (HE convention: last valid timestamp = M 8th at 00:00:00)
- Cost C for month M
- **All monthly simulations** (historical + forward-looking for M and M+1)
- **Daily simulations up to the 7th day of M only**
- Historical daily simulations for all past months

### FORBIDDEN:
- Realized prices for M+1
- Realized prices for M after the 7th
- Cost C for M+1
- Daily simulations for days after the 7th of M
- Any aggregate computed from forbidden data

**Hour Ending (HE) convention**: timestamp indicates the END of the hour. Last valid entry for day 7 = `YYYY-MM-08 00:00:00`.

### Lag feature leakage trap (CRITICAL)
The **full monthly price for month M** is NOT available at the cutoff (day 7 of M). Only days 1-7 of M are observable. Therefore:
- PR/PROFIT lags must use **shift(2)**, not shift(1) — nearest fully available month is **M-1**
- Cost C for month M IS available at cutoff (per case spec) → `c_lag1 = shift(1)` is safe
- `pr_partial_current` = ABS(SUM(prices for days 1-7 of M)) is a cutoff-safe current signal
- `has_history` flags (binary) must be created **BEFORE** any `fillna(0)` to distinguish new EIDs from zero-profit EIDs

---

## Data Structure

```
data/
├── costs/          # Monthly exposure costs (Parquet)
│   └── Columns: EID, MONTH, PEAKID, C
├── prices/         # Realized hourly prices (Parquet)
│   └── Columns: EID, DATETIME, PEAKID, PRICEREALIZED
├── sim_monthly/    # Monthly forward-looking simulations (Parquet, by year)
│   └── Columns: SCENARIOID, EID, DATETIME, PEAKID, ACTIVATIONLEVEL,
│                 WINDIMPACT, SOLARIMPACT, HYDROIMPACT, NONRENEWBALIMPACT,
│                 EXTERNALIMPACT, LOADIMPACT, TRANSMISSIONOUTAGEIMPACT, PSM
└── sim_daily/      # Daily short-term simulations (Parquet, by year)
    └── Columns: SCENARIOID, EID, DATETIME, PEAKID, ACTIVATIONLEVEL,
                  WINDIMPACT, SOLARIMPACT, HYDROIMPACT, NONRENEWBALIMPACT,
                  EXTERNALIMPACT, LOADIMPACT, TRANSMISSIONOUTAGEIMPACT, PSD
```

### Key data rules:
- **Sparsified data**: only non-zero values stored. Missing = 0.
- `PEAKID`: 0 = OFF, 1 = ON
- `ACTIVATIONLEVEL` and impacts are documented as **percentages**, but extreme outliers exist (HYDRO: [-1M, +111K] while p99 ~17). Bulk of distribution is consistent with percentages; extreme spikes may be real grid events or simulation artifacts — handled via `log_abs_mean` + `abs_max` features.
- **Source-based impacts** (WIND, SOLAR, HYDRO, NONRENEWBAL, EXTERNAL): partial sum → ACTIVATIONLEVEL
- **Explanatory variables** (LOAD, TRANSMISSIONOUTAGE): overlap with source-based, do NOT sum together
- `PSD`: simulated price from daily sims (3 scenarios)
- `PSM`: simulated price from monthly sims (3 scenarios)
- Simulations structured **by year** (4 files: 2020, 2021, 2022, 2023), each ~1.3GB
- **sim_monthly** is NOT available locally (too large to download). Code must handle `SIM_MONTHLY_AVAILABLE = False` gracefully.

---

## My Focus: sim_daily Dataset

### What sim_daily contains:
- Short-term simulations refined with most recent information
- Produced day J-1 for day J
- 3 scenarios (SCENARIOID = 1, 2, 3)
- Hourly granularity
- **At cutoff, only days 1-7 of month M are available**

### EDA objectives:
1. **Structure & dimensions**: row counts, distinct EIDs per year, scenarios confirmation
2. **Temporal coverage**: which days are covered per month, verify HE convention
3. **Variable distributions**: PSD, ACTIVATIONLEVEL, all impact variables — per scenario, per PEAKID
4. **Sparsity analysis**: % zeros for each variable
5. **Inter-scenario concordance**: do scenarios agree or diverge? Divergence = uncertainty signal
6. **Correlation structure**: between impact variables, between PSD and ACTIVATIONLEVEL
7. **Seasonal/monthly patterns**: does the signal change by month or season?
8. **Signal quality**: are daily sims for first 7 days predictive of monthly outcomes? (cross with prices dataset later)

### Feature engineering — finalized decisions:

**PSD features (days 1-7 of decision month M):**
- Magnitude: `psd_abs_sum`, `psd_abs_nonzero_mean`, `psd_abs_nonzero_std`, `psd_abs_max`
- Signed: `psd_signed_mean` (directional analysis)
- Activity: `psd_nonzero_count`
- Scenarios: `psd_abs_s1_mean` vs `psd_abs_s23_mean` (S1 differs from S2/S3, r=0.22 vs r=0.87)
- Uncertainty: `psd_scenario_spread` (inter-scenario std)
- Trend: `psd_abs_early` (days 1-3) vs `psd_abs_late` (days 4-7) — momentum within cutoff window

**Impact features:**
- All 7 impacts: `abs_mean` = AVG(ABS(impact))
- Top 3 enriched (HYDRO, WIND, LOAD — heaviest tails, 69-70% sparse):
  - `log_abs_mean` = AVG(LN(1 + ABS(impact))) — robust to extreme outliers
  - `abs_max` = MAX(ABS(impact)) — captures extreme events
- SOLAR (85% sparse): `abs_mean` only — too sparse to enrich
- ACTIVATIONLEVEL: `mean`, `max`, `nonzero_count` (already enriched)

**Historical lag features (leakage-corrected):**
- `pr_lag1..3` = shift(2,3,4) — nearest full month is M-1, NOT M
- `c_lag1` = shift(1) — cost of M is available at cutoff
- `profit_lag1..2`, `target_lag1` = shift(2,3) — same correction as PR
- Rolling: `pr_rolling3_mean`, `profit_rolling3_mean`, `profitable_count_3m` — window starts at shift(2)
- `pr_partial_current` = partial-month M price (days 1-7), cutoff-safe
- `has_pr_history`, `has_profit_history`, `has_cost_history` — binary flags, created BEFORE fillna

**Calendar:** `month_of_year`, `year`, `season`, `season_encoded`

---

## Technical Stack & Constraints

### Primary tools:
- **DuckDB** — query Parquet files directly without loading into memory (essential for 1.3GB files)
- **Polars** — fast DataFrame operations with lazy evaluation when needed
- **pandas** — only for small extracts and compatibility with plotting
- **matplotlib / seaborn** — visualizations

### Memory management rules:
- NEVER load full yearly parquet files into pandas
- Use DuckDB `read_parquet('path/*.parquet')` with SQL aggregations
- Extract small samples for visualization only after filtering via SQL
- Process year by year when iteration is needed
- **Set DuckDB memory limit explicitly** to avoid RAM exhaustion (see pattern below)
- **Avoid nested subqueries with `USING SAMPLE` on the full view** — sample from a filtered subset or a single year instead
- **Clear notebook outputs** (`Cell > All Outputs > Clear`) before asking Claude Code to read `.ipynb` files — embedded base64 images can crash the extension

### DuckDB patterns:
```python
import duckdb
con = duckdb.connect()

# IMPORTANT: cap DuckDB memory to avoid system-wide RAM pressure
con.execute("SET memory_limit = '2GB'")
con.execute("SET threads = 4")

# Create view over all yearly files
con.execute("""
    CREATE VIEW sim_daily AS
    SELECT * FROM read_parquet('data/sim_daily/*.parquet')
""")

# Query without loading everything
result = con.execute("SELECT ... FROM sim_daily WHERE ...").fetchdf()

# For heavy queries, prefer year-by-year iteration:
for year in [2020, 2021, 2022, 2023]:
    result = con.execute(f"""
        SELECT ... FROM read_parquet('data/sim_daily/sim_daily_{year}.parquet')
        WHERE ...
    """).fetchdf()
```

### Avoiding Claude Code crashes:
- The "Claude Code process exited with code 1" error is typically an extension OOM crash
- **Main trigger**: Claude reading notebooks with large embedded plot outputs (base64 images)
- **Prevention**: clear cell outputs before committing or before asking Claude to analyze the notebook
- **DuckDB memory limit** (`SET memory_limit = '2GB'`) prevents DuckDB from consuming all available RAM during Python execution
- If a query is slow or heavy, break it into per-year passes instead of using the wildcard view

---

## Output Expectations

### Final deliverable (team):
- `main.py` with `--start-month` and `--end-month` arguments
- `requirements.txt`
- `README.md`
- `opportunities.csv` with columns: `TARGET_MONTH`, `PEAK_TYPE` (ON/OFF), `EID`

### My deliverable:
- `notebook/master_dataset_hybrid_v3.ipynb` — master dataset pipeline combining all 4 datasets
- `notebook/EDA_sim_daily.ipynb` — structured EDA of sim_daily
- `data/master_dataset.parquet` — exported master dataset ready for modeling
- Key findings in `FINDINGS.md` for team and jury

---

## Evaluation Grid

- **Axis 1 — F1-score (25%)**: ability to correctly identify profitable opportunities (TP, FP, FN). Aggregated by PEAKID over full eval period.
- **Axis 2 — Net Profit (25%)**: sum of (PR - C) for all selected opportunities. Losses penalize.
- **Axis 3 — Jury (50%)**: methodology (30pts), technical quality (25pts), solution relevance (25pts), presentation (20pts).

**Jury is 50% → code quality, interpretability, and presentation matter as much as model performance.**

---

## Coding Standards

- Python 3.10+
- Clear docstrings and comments (bilingual OK: French for business context, English for code)
- Modular functions, not monolithic scripts
- Type hints where practical
- Reproducible: random seeds, explicit file paths
- Print progress for long operations

---

## Documentation Rule

**Tous les findings, hypotheses et decisions d'analyse doivent etre documentes dans `FINDINGS.md`** au fur et a mesure de l'avancement. Ce fichier sert de journal de bord pour l'equipe et le jury. Chaque section doit etre datee et attribuee (qui a trouve quoi). Mettre a jour ce fichier a chaque nouvelle decouverte significative.

---

## Key Reminders

1. **Anti-leakage is non-negotiable** — always filter by cutoff date before any computation
2. **PR lag shift(2), NOT shift(1)** — full month M price is unavailable at cutoff day 7. Only C uses shift(1).
3. **Sparse data** — missing = 0, handle accordingly in all aggregations
4. **`has_history` flags BEFORE fillna(0)** — distinguish new EIDs from zero-profit established EIDs
5. **10-100 opportunities per month** — enforce this constraint in any selection logic
6. **3 scenarios**: S1 vs S2/S3 grouping (r=0.22 vs r=0.87). Do NOT average blindly.
7. **PEAKID in data (0/1) vs PEAK_TYPE in output (OFF/ON)** — mapping required
8. **Daily sims are complementary** to monthly sims — monthly sims are the core signal for M+1, daily sims provide recent context for first 7 days of M only
9. **Two-model strategy** — classifier for F1 + regressor for net profit. Both targets in master dataset.
10. **Impact extreme values** — use `log_abs_mean` (robust to outliers) + `abs_max` (captures extremes) for HYDRO, WIND, LOAD
