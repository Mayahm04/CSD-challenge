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

An opportunity is **profitable** when: `PR_o - C_o > 0`
- `PR_o` = sum of realized hourly prices for that (EID, MONTH, PEAKID)
- `C_o` = monthly exposure cost assigned by the market

**Selection constraint**: between 10 and 100 opportunities per target month (total ON + OFF).

**Historically, fewer than 5% of opportunities are profitable** → highly imbalanced, selectivity is key.

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
- `ACTIVATIONLEVEL` and impacts are in **percentages**
- **Source-based impacts** (WIND, SOLAR, HYDRO, NONRENEWBAL, EXTERNAL): partial sum → ACTIVATIONLEVEL
- **Explanatory variables** (LOAD, TRANSMISSIONOUTAGE): overlap with source-based, do NOT sum together
- `PSD`: simulated price from daily sims (3 scenarios)
- `PSM`: simulated price from monthly sims (3 scenarios)
- Simulations structured **by year** (4 files: 2020, 2021, 2022, 2023), each ~1.3GB

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

### Feature engineering ideas for modeling phase:
- Aggregate PSD over days 1-7 of month M per (EID, PEAKID, SCENARIOID): mean, sum, std
- Mean ACTIVATIONLEVEL over days 1-7
- Inter-scenario spread: std(PSD) across 3 scenarios for same (EID, hour)
- Dominant impact source per EID
- Trend: compare daily sim values early in month vs later (day 1-3 vs 4-7)

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

### My deliverable (today):
- Jupyter notebook with structured EDA of sim_daily
- Key findings summary for team
- Recommended features to extract from sim_daily for the model
- Code snippets ready to integrate into the team pipeline

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
2. **Sparse data** — missing = 0, handle accordingly in all aggregations
3. **10-100 opportunities per month** — enforce this constraint in any selection logic
4. **3 scenarios are black-box** — free to combine, average, or use individually
5. **PEAKID in data (0/1) vs PEAK_TYPE in output (OFF/ON)** — mapping required
6. **Daily sims are complementary** to monthly sims — monthly sims are the core signal for M+1, daily sims provide recent context for first 7 days of M only
