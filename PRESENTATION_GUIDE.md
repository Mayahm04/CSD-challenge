# Presentation Guide — CSD x MAG Energy Solutions Data Challenge 2026

> **Purpose**: This file is both a presentation schema AND a prompt. Copy-paste it into Claude (web) and ask: *"Build me a PowerPoint presentation following this guide exactly. Use a professional dark blue/teal energy trading theme. Each slide section below = 1 slide unless noted otherwise."*

---

## CONTEXT FOR THE AI BUILDING THE SLIDES

**Who we are**: A team of 4 university students acting as junior quantitative analysts / data scientists for an energy trading firm. The tone should be professional, confident, and grounded — like a consulting deliverable for a trading desk, not an academic paper.

**Jury evaluation grid (100 pts total)**:
- Methodology (30 pts): rigor, anti-leakage, validation strategy, feature engineering rationale
- Technical quality (25 pts): code quality, reproducibility, bug-fixing process, optimization
- Solution relevance (25 pts): F1-score + net profit performance, business applicability
- Presentation (20 pts): clarity, storytelling, visual quality, ability to explain to non-technical stakeholders

**Visual style**: Dark blue / teal / white color scheme. Clean, minimal slides. Large numbers for KPIs. Diagrams over walls of text. Use icons where appropriate (lightning bolt for energy, target for selection, shield for anti-leakage).

---

## SLIDE 1 — Title Slide

**Title**: Algorithmic Selection of Profitable FTR Opportunities
**Subtitle**: CSD x MAG Energy Solutions — Data Challenge 2026
**Bottom**: Team names, University, Date (March 2026)
**Visual**: Subtle background of electricity grid / transmission lines (abstract, dark)

---

## SLIDE 2 — The Business Problem

**Header**: "Which FTRs should we buy next month?"

**Content** (use a clean diagram):
- An **FTR (Financial Transmission Right)** = a bet on congestion between two grid nodes
- An **opportunity** = triplet (EID, Month, Peak Type)
- **Profitable** when: `|Price Realized| - Cost > 0`
- **Constraint**: select between **10 and 100 opportunities per month**
- **Challenge**: only **~5-7% of opportunities are profitable** — needle in a haystack

**Key visual**: Funnel diagram showing ~4,000 monthly candidates narrowed to 80 selected opportunities

**Speaker note**: "Imagine you're a trader on the 7th of the month. You have 7 days of market data, simulation forecasts, and historical patterns. You must commit to your FTR portfolio for next month. Our algorithm automates this decision."

---

## SLIDE 3 — Data Landscape

**Header**: "4 datasets, 616M+ rows, 4 years of history"

**Content** (table or 4-panel layout):

| Dataset | Volume | What it tells us |
|---------|--------|-----------------|
| **sim_daily** | 616M rows, 7,379 EIDs | Short-term congestion forecasts (3 scenarios, hourly, produced D-1 for D) |
| **sim_monthly** | ~500M rows | Monthly forward-looking simulations (broader horizon) |
| **prices** | 453K rows, 3,065 EIDs | Realized hourly congestion prices (ground truth for revenue) |
| **costs** | 9,092 rows, 927 EIDs | Monthly FTR acquisition costs |

**Key insight callout box**: "Only 927 EIDs have costs (tradable universe). Simulations cover 7,000+ — most of the grid is noise."

**Speaker note**: "The data is massively sparse — 99.8% of simulated prices are zero. This isn't missing data, it means no congestion was simulated at that hour. Our feature engineering had to extract signal from this extreme sparsity."

---

## SLIDE 4 — Anti-Leakage Framework (CRITICAL — shows rigor to jury)

**Header**: "Information cutoff: day 7 of month M, predicting month M+1"

**Visual**: Timeline diagram showing:
```
|-------- Month M-1 (fully known) --------|--- Month M (days 1-7 only) ---|--- Month M+1 (TARGET) ---|
         PR fully available                  PR partial (7 days)             FORBIDDEN
         Cost fully available                Cost M available                Cost M+1 FORBIDDEN
         All sims available                  Daily sims days 1-7 only        No data
                                             Monthly sims available
```

**Three rules callout**:
1. **PR/Profit lags = shift(2)**, not shift(1) — full month M price is NOT available at cutoff
2. **Cost lag = shift(1)** — cost of month M IS available (per case specification)
3. **Daily sims**: only days 1-7 of month M (Hour Ending convention: last valid = 8th at 00:00)

**Speaker note**: "This is the single most important methodological choice. Many teams will use shift(1) for price lags and unknowingly leak future information. We caught this during our cross-review process and corrected it. The difference between shift(1) and shift(2) is the difference between a model that works in backtest and one that works in production."

---

## SLIDE 5 — Feature Engineering Philosophy

**Header**: "52 features across 4 signal families"

**Visual**: 4 columns or quadrants:

**1. Simulation Signals (PSD + PSM)** — 25 features
- PSD (daily): magnitude, signed mean, activity count, scenario spread, early/late trend
- PSM (monthly): same structure, stronger predictive power
- Key insight: Scenario 1 differs from Scenarios 2-3 (r=0.22 vs r=0.87) — treated separately

**2. Historical Performance** — 12 features
- Price lags (shift 2,3,4), cost lag (shift 1)
- Rolling averages, profitable count in last 3 months
- Partial current month price (days 1-7)
- Binary flags: `has_history` created BEFORE fillna(0) to distinguish new EIDs from zero-profit EIDs

**3. Impact Variables** — 11 features
- 7 grid factors: Wind, Solar, Hydro, Load, NonRenewable, External, Transmission Outage
- Heavy tails (Hydro: range [-1M, +111K] vs p99=17) handled with `log(1+|x|)` + `max(|x|)`
- Robust to outliers while preserving extreme event signals

**4. Calendar & Structure** — 4 features
- Month, year, season, `is_sim_only` flag

**Speaker note**: "Each feature family captures a different aspect of the trading decision. Simulations tell us what the market expects. History tells us what happened before. Impacts tell us what's driving congestion. And calendar captures seasonality. The key innovation is our treatment of extreme values — rather than clipping outliers, we use a dual encoding (log mean + absolute max) that preserves both the typical signal and the extreme events that drive the biggest profits."

---

## SLIDE 6 — Key EDA Findings (use visuals from notebook if available)

**Header**: "Three critical discoveries from exploratory analysis"

**Finding 1 — Concept Drift** (bar chart):
| Year | Positive Rate |
|------|--------------|
| 2020 | 10.7% |
| 2021 | 9.2% |
| 2022 | 8.0% |
| 2023 | 4.9% |

Caption: "Profitability halved in 3 years — the market is getting more efficient"

**Finding 2 — Profit Distribution** (histogram sketch):
- Median profit: 342, Mean: 2,372, Max: 212,803
- "A few 'jackpot' opportunities dominate total profit — we need a regressor, not just a classifier"

**Finding 3 — Feature #1**: `profitable_count_3m` (r=0.365 with target)
- "Past profitability is the strongest predictor — persistence matters in FTR markets"

**Speaker note**: "These findings directly shaped our modeling strategy. The concept drift means we can't train on 2020 and expect it to generalize to 2024 — we need walk-forward validation. The skewed profit distribution means a binary classifier alone will miss the magnitude signal. And the persistence of profitability tells us that some EIDs are structurally advantaged — the model learns to identify these."

---

## SLIDE 7 — Modeling Architecture

**Header**: "Two-stage pipeline: Filter then Rank"

**Visual**: Flow diagram (this is the centerpiece slide):

```
[All ~4,000 monthly opportunities]
            |
            v
  +-------------------+
  |   LightGBM        |    P(profitable) per opportunity
  |   CLASSIFIER       |    → eval_metric: AUC
  |   (depth=4)        |    → scale_pos_weight: 16.6
  +-------------------+
            |
     Filter: P > 0.19
            |
            v
  +-------------------+
  |   LightGBM        |    Predicted log(1 + profit)
  |   REGRESSOR        |    → target: log1p(PROFIT)
  |   (depth=8)        |    → captures magnitude
  +-------------------+
            |
            v
  +-------------------+
  |   RANK-BASED       |    score = 0.71 * rank(proba)
  |   SELECTION        |         + 0.29 * rank(pred_profit)
  |   Top-K per month  |    K = 80, constrained [10, 100]
  +-------------------+
            |
            v
  [80 selected opportunities per month]
```

**Why two models?** box:
- Classifier optimizes **F1** (25% of grade) — who is profitable?
- Regressor optimizes **Profit** (25% of grade) — how profitable?
- Rank blending combines both signals smoothly

**Speaker note**: "The key insight is that the classifier and regressor serve fundamentally different purposes, which is why Optuna found different architectures for each. The classifier is shallow (depth 4) because the decision boundary is simple — it's mostly about whether an EID has been profitable recently. The regressor is deep (depth 8) because predicting magnitude requires capturing complex non-linear interactions between simulation signals and market conditions."

---

## SLIDE 8 — Validation Strategy

**Header**: "Walk-forward expanding window — no future leakage, ever"

**Visual**: Timeline diagram showing 12 monthly folds:
```
Fold 1:  [2020-01 ————————— 2022-06] → Val: 2022-07
Fold 2:  [2020-01 ————————— 2022-07] → Val: 2022-08
Fold 3:  [2020-01 ————————— 2022-08] → Val: 2022-09
...
Fold 12: [2020-01 ————————— 2023-06] → Val: 2023-07

Final:   [2020-01 ————————— 2023-06] → Test: 2023-07 to 2023-12
```

**Why not k-fold?**: "Time series data with concept drift. Random splits would leak future patterns into training."

**Optuna optimization**: 4 quarterly folds (speed), validated on 12 monthly folds (precision).

**Speaker note**: "Every design choice in our validation mirrors the real trading workflow. On the 7th of each month, you only know the past. Our walk-forward CV replicates this exactly. The expanding window also means the model sees more data over time, which helps it adapt to the evolving market."

---

## SLIDE 9 — The Bugs We Found and Fixed (shows technical maturity)

**Header**: "3 critical bugs caught during development"

**Use a 3-column layout, each with a red "before" and green "after":**

**Bug 1 — Early Stopping Trap**
- Before: Classifier stopped at 1-2 iterations (F1 = 0.03)
- Cause: `binary_logloss` + `scale_pos_weight` = instant logloss increase → early stop
- Fix: `eval_metric = 'AUC'` (ranking quality, not calibration)

**Bug 2 — Unfair Model Comparison**
- Before: LogReg appeared better than LightGBM (F1: 0.21 vs 0.17)
- Cause: LogReg grid-searched K=75, LightGBM hardcoded K=50 (caps recall at 14%)
- Fix: Same grid search for all approaches → LightGBM +55% more profit

**Bug 3 — Focal Loss Default Metric**
- Before: Custom focal loss stopped at 1 iteration
- Cause: LightGBM auto-inserts `binary_logloss` before custom metrics
- Fix: `metric='none'` to suppress default metric

**Speaker note**: "We're showing you our bugs on purpose. In quantitative trading, the most dangerous model is one that looks good due to a methodological error. Our iterative debugging process — documented in our FINDINGS.md journal — demonstrates exactly the kind of rigor required in production trading systems. Each bug was root-caused, fixed, and verified."

---

## SLIDE 10 — Hyperparameter Optimization

**Header**: "Optuna: 200 trials, 20 hyperparameters, 3h12min"

**Key visual**: Optimization history plot (if available from notebook, otherwise describe):
- X-axis: trial number, Y-axis: combined score
- Show convergence from ~0.45 to 0.62

**Best trial**: #183 / 200, combined score = **0.6199**

**Key findings table** (highlight surprising values):
| Parameter | Default | Optuna | Insight |
|-----------|---------|--------|---------|
| scale_pos_weight | 9.5 | **16.6** | Much more aggressive minority upweighting |
| threshold | 0.30 | **0.19** | Cast a wider net, let the ranker sort |
| K | 50 | **80** | Select near the maximum allowed |
| alpha | 0.50 | **0.71** | Probability dominates over profit magnitude |
| clf depth | 6 | **4** | Simpler classifier (avoid overfitting) |
| reg depth | 6 | **8** | Complex regressor (capture non-linearities) |

**Objective**: `0.5 * F1 + 0.5 * normalized_profit` — directly aligned with jury weights (25% each)

**Speaker note**: "The Optuna objective was deliberately designed to mirror the jury's evaluation. We're not optimizing F1 alone or profit alone — we optimize their weighted combination. The most surprising finding is that the optimal strategy is to be aggressive: low threshold, high K, and trust the probability ranking (alpha=0.71). This is consistent with the trading intuition that it's better to cast a wide net with a good ranking than to be overly selective."

---

## SLIDE 11 — Results

**Header**: "Test Performance (2023 H2 — out-of-sample)"

**Large KPI boxes** (make these visually prominent):

| F1-Score (avg ON/OFF) | Net Profit | ROC AUC |
|:---------------------:|:----------:|:-------:|
| **0.259** | **711,987** | **0.874** |

**Comparison table** (shows ML value-add):
| Approach | F1 | Profit | Comment |
|----------|------|--------|---------|
| Random baseline | ~0.05 | Negative | Floor |
| Heuristic (top by history) | ~0.12 | ~200K | No ML needed |
| Logistic Regression | ~0.21 | ~500K | Simple ML |
| **LightGBM Two-Stage** | **0.259** | **712K** | **+42% profit vs LogReg** |

**Speaker note**: "Our model delivers 0.259 F1 on completely unseen data (July-December 2023), with 712K net profit. To put this in context: a random selection strategy would lose money. A simple heuristic based on historical profitability gets you to ~200K. Logistic regression reaches ~500K. Our two-stage LightGBM pipeline adds another 42% on top. The ROC AUC of 0.874 confirms the model has strong discriminative power — it reliably ranks profitable opportunities above unprofitable ones."

---

## SLIDE 12 — Feature Importance & Interpretability

**Header**: "What drives the model's decisions?"

**Visual**: SHAP beeswarm plot (screenshot from notebook cell 26 if available, otherwise describe the layout)

**Top 5 features narrative** (use a numbered list with icons):

1. **`profitable_count_3m`** (Persistence) — EIDs that were profitable recently tend to stay profitable. This is the #1 signal. Captures structural advantages of certain grid elements.

2. **`psm_abs_max`** (Forward simulation peak) — High peak simulated prices from monthly forecasts predict high realized prices. The simulation is informative.

3. **`psd_nonzero_count`** (Daily simulation activity) — More non-zero daily simulated prices = more congestion activity = higher chance of profit.

4. **`pr_lag1`** (Historical revenue, shift=2) — Recent realized price levels are predictive of future levels. Mean-reversion is weak in FTR markets.

5. **`c_lag1`** (Current month cost) — Cost of acquiring the FTR is a strong negative predictor. Expensive FTRs are less likely to be profitable.

**Business narrative box**:
> "The model selects opportunities where: (a) the EID has a track record of profitability, (b) forward simulations show strong congestion signals, and (c) acquisition costs are reasonable relative to expected revenue. It avoids EIDs with no history and weak simulation signals."

**Speaker note**: "This interpretability is critical for a real trading desk. A trader won't trust a black box. Our SHAP analysis shows the model follows economically intuitive logic: buy where congestion is expected, where history supports profitability, and where the price is right. This is essentially what a human trader would do — but at scale across 4,000 opportunities per month."

---

## SLIDE 13 — Production Pipeline

**Header**: "From raw data to trading decisions in one command"

**Visual**: Pipeline diagram:
```
Raw Parquet files (costs, prices, sim_daily, sim_monthly)
    |
    v  [DuckDB — 2GB memory limit, year-by-year processing]
Feature Engineering (52 features, anti-leakage enforced)
    |
    v
Model Training (LightGBM classifier + regressor, walk-forward)
    |
    v
Opportunity Selection (threshold → rank → top-K)
    |
    v
opportunities.csv (TARGET_MONTH, PEAK_TYPE, EID)
```

**Command**:
```bash
python main.py --start-month 2025-01 --end-month 2025-12
```

**Technical highlights**:
- DuckDB for memory-efficient processing of multi-GB parquet files
- Absolute paths, cross-platform compatible
- Handles missing sim_monthly gracefully (falls back to sim_daily)
- Reproducible: fixed random seed (42), deterministic pipeline

**Speaker note**: "The jury asked for a main.py that takes start and end months. Our script handles everything end-to-end: it reads raw parquet files, engineers all 52 features respecting the anti-leakage cutoff, trains both models on all available history, and outputs the selection CSV. It's designed to work on 2025 data without any code changes."

---

## SLIDE 14 — Robustness & Limitations

**Header**: "What we know, and what we'd do with more time"

**Strengths**:
- Anti-leakage rigorously enforced (shift(2), HE convention, day 1-7 filter)
- Walk-forward validation matches real trading workflow
- Two-stage pipeline optimizes both F1 and profit simultaneously
- Documented bug-fixing process (FINDINGS.md as audit trail)
- Production-ready code (main.py works on unseen data)

**Limitations & future work**:
- Concept drift (10.7% → 4.9%): model may need recalibration as market evolves
- sim-only EIDs (Pool 2): 0% positive rate in training — we exclude them but they could become profitable
- No ensemble: single LightGBM model, no stacking or blending
- Threshold sensitivity: optimal threshold (0.19) may shift with market conditions
- Per-PEAKID models could capture ON/OFF asymmetries better

**Speaker note**: "Transparency about limitations is part of professional quantitative analysis. The concept drift is our biggest concern — the market is getting more efficient over time, which compresses profit margins. In a production setting, we'd implement rolling retraining and threshold recalibration on a monthly basis."

---

## SLIDE 15 — Conclusion

**Header**: "Key Takeaways"

**Three large callout points**:

1. **Methodological rigor** — Anti-leakage framework with shift(2), walk-forward CV, and systematic bug-fixing documented in FINDINGS.md (35 findings across 6 rounds)

2. **Business-aligned optimization** — Two-stage pipeline with Optuna (200 trials) directly optimizing the jury's combined F1 + profit metric

3. **Production-ready delivery** — One-command pipeline (main.py) processing raw data to trading decisions, tested on 2024 data, ready for 2025 evaluation

**Final KPI bar** (large, centered):

| F1 = 0.259 | Profit = 712K | ROC AUC = 0.874 |

**Closing line**: "Our algorithm doesn't just predict — it selects opportunities that make money."

---

## SLIDE 16 — Appendix / Backup Slides (if jury asks questions)

### A1. Full Feature List (52 features)
- PSD features (13): psd_nonzero_count, psd_abs_nonzero_mean, psd_abs_nonzero_std, psd_abs_sum, psd_signed_mean, psd_abs_max, psd_abs_s1_mean, psd_abs_s23_mean, psd_scenario_spread, psd_abs_early, psd_abs_late, activation_mean, activation_max
- PSM features (16): psm_nonzero_count, psm_abs_nonzero_mean, psm_abs_nonzero_std, psm_abs_sum, psm_signed_mean, psm_abs_max, psm_activation_mean, psm_activation_max, psm_wind_abs_mean, psm_solar_abs_mean, psm_hydro_abs_mean, psm_nonrenew_abs_mean, psm_external_abs_mean, psm_abs_s1_mean, psm_abs_s23_mean, psm_scenario_spread
- Impact features (11): wind_abs_mean, solar_abs_mean, hydro_abs_mean, nonrenew_abs_mean, external_abs_mean, load_abs_mean, transoutage_abs_mean, hydro_log_abs_mean, hydro_abs_max, wind_log_abs_mean, wind_abs_max, load_log_abs_mean, load_abs_max
- Historical (12): pr_lag1-3, c_lag1, profit_lag1-2, target_lag1, pr_rolling3_mean, profit_rolling3_mean, profitable_count_3m, pr_partial_current, has_pr_history, has_profit_history, has_cost_history
- Calendar (4): month_of_year, year, season, is_sim_only

### A2. Optuna Full Parameter Table
(Copy from FINDINGS.md F33)

### A3. Confusion Matrix on Test Set
(Screenshot from notebook if available)

### A4. Monthly Breakdown (Test Period)
| Month | Selected | TP | FP | Profit |
|-------|----------|----|----|--------|
| 2023-07 | 80 | ... | ... | ... |
| ... | ... | ... | ... | ... |

### A5. Team Contribution
| Member | Dataset | Deliverable |
|--------|---------|-------------|
| Sofiane | sim_daily | EDA, feature engineering, modeling notebook, production pipeline |
| Rachid | sim_monthly + master dataset | PSM features, master dataset construction |
| [Name] | costs | Cost analysis, ... |
| [Name] | prices | Price analysis, ... |

---

## PROMPT TO GENERATE THE POWERPOINT

Copy everything above into Claude (web) and add this instruction:

> "Create a professional PowerPoint presentation following the guide above. Use a dark blue (#1B2A4A) and teal (#2ECCC7) color scheme with white text. Each numbered slide section = 1 slide. Use clean layouts with large KPI numbers, minimal text, and professional diagrams. For the pipeline diagram (Slide 7), use connected boxes with arrows. For the timeline (Slide 8), use a horizontal bar chart. Make the title slide impactful. Total: 15 main slides + appendix. Export-ready quality."

---

## KEY TALKING POINTS FOR Q&A

**If asked "Why not XGBoost / Random Forest / Neural Network?"**
> "LightGBM was chosen for three reasons: (1) native handling of sparse features, which is critical given 99.8% zeros in PSD, (2) built-in categorical handling and histogram-based splitting that scales to our 200K+ training rows, and (3) the sklearn API compatibility needed for our Optuna integration. We considered neural networks but our dataset of 200K rows with 52 tabular features is well within the regime where gradient-boosted trees consistently outperform deep learning."

**If asked "How do you handle the class imbalance?"**
> "We use scale_pos_weight=16.6, which Optuna found to be optimal. This is significantly more aggressive than the naive inverse ratio (~12.2). We deliberately chose this over SMOTE or undersampling because synthetic oversampling can create artifacts in temporal data with concept drift, and undersampling would discard valuable negative examples. The key insight is that imbalance handling interacts with the threshold — our low threshold (0.19) combined with high upweighting is effectively saying 'be sensitive to any hint of profitability, then let the ranker sort'."

**If asked "What if the model fails on 2025 data?"**
> "The concept drift we documented (10.7% to 4.9%) is our primary risk. Our mitigations are: (1) walk-forward CV that explicitly tests on recent data, (2) the production pipeline retrains on all available history including 2024, (3) the selection parameters (threshold, K, alpha) were optimized on recent validation folds. In a production setting, we'd add monthly recalibration of the threshold based on the most recent profitable rate."

**If asked "Why shift(2) and not shift(1)?"**
> "On the 7th of month M, you're making decisions for month M+1. The full price for month M won't be known until the end of M. So the most recent complete month of price data is M-1, which is shift(2) from the target month M+1. Using shift(1) would leak 23 days of future price data. This is the single most common data leakage mistake in FTR modeling."

**If asked "What is alpha=0.71 doing?"**
> "Alpha controls the blend between classifier probability and regressor profit prediction in our ranking formula. Alpha=0.71 means the probability signal gets 71% weight. Intuitively, this makes sense: it's more important to select opportunities that ARE profitable than to chase the biggest profits. A false positive (selecting unprofitable) always costs money, while the difference between a medium and large profit is less consequential for F1."
