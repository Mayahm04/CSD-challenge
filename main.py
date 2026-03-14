"""
main.py — FTR Opportunity Selection Pipeline
CSD x MAG Energy Solutions Data Challenge 2026

Selects profitable electricity trading opportunities (FTRs) for target months.
Two-stage strategy: LightGBM classifier (filter) + regressor (rank) + selection.

Usage:
    python main.py --start-month 2024-01 --end-month 2024-12
    python main.py --start-month 2025-01 --end-month 2025-12

Expects raw data in data/ directory:
    data/costs/costs*.parquet
    data/prices/prices*.parquet
    data/sim_daily/sim_daily_*.parquet
    data/sim_monthly/sim_monthly_*.parquet  (optional)
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("data")

# Features dropped for multicollinearity (r > 0.99)
DROP_FEATURES = ['has_profit_history', 'psm_abs_nonzero_std', 'psd_abs_nonzero_std']

# Column roles
ID_COLS = ['EID', 'MONTH', 'PEAKID', 'DECISION_MONTH']
TARGET_COLS = ['TARGET', 'PROFIT', 'PR', 'PR_signed', 'C']
META_COLS = ['is_sim_only', 'season']

# Universe thresholds (p80 for strong-sim EIDs)
SIM_ACTIVATION_PCT = 80
SIM_PSM_PCT = 80

# Selection defaults (will be optimized if training data available)
DEFAULT_THRESHOLD = 0.15
DEFAULT_K = 75
DEFAULT_ALPHA = 0.5


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def setup_duckdb(data_dir: Path) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB with views over raw data files."""
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 4")

    # Costs
    costs_path = str(data_dir / "costs" / "costs*.parquet")
    con.execute(f"CREATE VIEW costs AS SELECT * FROM read_parquet('{costs_path}')")

    # Prices
    prices_path = str(data_dir / "prices" / "prices*.parquet")
    con.execute(f"""
        CREATE VIEW prices AS
        SELECT *, STRFTIME(DATETIME, '%Y-%m') AS MONTH
        FROM read_parquet('{prices_path}')
    """)

    # Sim daily
    sim_daily_path = str(data_dir / "sim_daily" / "sim_daily_*.parquet")
    con.execute(f"""
        CREATE VIEW sim_daily AS
        SELECT *, YEAR(DATETIME) AS YEAR, MONTH(DATETIME) AS MO, DAY(DATETIME) AS DAY
        FROM read_parquet('{sim_daily_path}')
    """)

    # Sim monthly (optional)
    sim_monthly_dir = data_dir / "sim_monthly"
    sim_monthly_available = sim_monthly_dir.exists() and any(sim_monthly_dir.glob("*.parquet"))
    if sim_monthly_available:
        sim_monthly_path = str(sim_monthly_dir / "sim_monthly_*.parquet")
        con.execute(f"""
            CREATE VIEW sim_monthly AS
            SELECT *, YEAR(DATETIME) AS YEAR, MONTH(DATETIME) AS MO
            FROM read_parquet('{sim_monthly_path}')
        """)

    return con, sim_monthly_available


def detect_years(data_dir: Path) -> list:
    """Detect available years from sim_daily files."""
    sim_dir = data_dir / "sim_daily"
    years = sorted(
        int(f.stem.split("_")[-1])
        for f in sim_dir.glob("sim_daily_*.parquet")
    )
    return years


def build_universe(con, years: list, sim_monthly_available: bool) -> pd.DataFrame:
    """Build hybrid universe: market-validated + strong-sim EIDs."""
    print("Building hybrid universe...")

    # Pool 1: market-validated EIDs
    market_eids = con.execute("""
        SELECT DISTINCT EID FROM costs
        UNION
        SELECT DISTINCT EID FROM prices
    """).fetchdf()
    con.register('market_eids', market_eids)
    print(f"  Pool 1 (market-validated): {len(market_eids):,} EIDs")

    # Compute percentile thresholds from most recent year
    ref_year = max(years)
    if sim_monthly_available:
        thresholds = con.execute(f"""
            SELECT
                APPROX_QUANTILE(ACTIVATIONLEVEL, {SIM_ACTIVATION_PCT / 100.0}) AS act_thresh,
                APPROX_QUANTILE(ABS(PSM), {SIM_PSM_PCT / 100.0}) AS psm_thresh
            FROM read_parquet('data/sim_monthly/sim_monthly_{ref_year}.parquet')
            WHERE ACTIVATIONLEVEL > 0 AND PSM != 0
        """).fetchone()
    else:
        thresholds = con.execute(f"""
            SELECT
                APPROX_QUANTILE(ACTIVATIONLEVEL, {SIM_ACTIVATION_PCT / 100.0}) AS act_thresh,
                APPROX_QUANTILE(ABS(PSD), {SIM_PSM_PCT / 100.0}) AS psm_thresh
            FROM read_parquet('data/sim_daily/sim_daily_{ref_year}.parquet')
            WHERE ACTIVATIONLEVEL > 0 AND PSD != 0
        """).fetchone()

    act_thresh, psm_thresh = thresholds[0], thresholds[1]

    # Pool 2: strong-sim EIDs
    strong_sim_list = []
    for year in years:
        if sim_monthly_available:
            df_y = con.execute(f"""
                SELECT DISTINCT EID
                FROM read_parquet('data/sim_monthly/sim_monthly_{year}.parquet')
                WHERE ACTIVATIONLEVEL >= {act_thresh} AND ABS(PSM) >= {psm_thresh}
                  AND EID NOT IN (SELECT EID FROM market_eids)
            """).fetchdf()
        else:
            df_y = con.execute(f"""
                SELECT DISTINCT EID
                FROM read_parquet('data/sim_daily/sim_daily_{year}.parquet')
                WHERE DAY(DATETIME) BETWEEN 1 AND 7
                  AND ACTIVATIONLEVEL >= {act_thresh} AND ABS(PSD) >= {psm_thresh}
                  AND EID NOT IN (SELECT EID FROM market_eids)
            """).fetchdf()
        strong_sim_list.append(df_y)

    strong_sim_eids = pd.concat(strong_sim_list, ignore_index=True).drop_duplicates()
    con.register('strong_sim_eids', strong_sim_eids)
    print(f"  Pool 2 (strong-sim): {len(strong_sim_eids):,} EIDs")

    # Union
    candidate_eids = con.execute("""
        SELECT EID, 0 AS is_sim_only FROM market_eids
        UNION ALL
        SELECT EID, 1 AS is_sim_only FROM strong_sim_eids
    """).fetchdf()
    con.register('candidate_eids', candidate_eids)

    # Cross with sim month/PEAKID grid
    source = 'sim_monthly' if sim_monthly_available else 'sim_daily'
    universe_parts = []
    for year in years:
        if sim_monthly_available:
            path = f"data/sim_monthly/sim_monthly_{year}.parquet"
        else:
            path = f"data/sim_daily/sim_daily_{year}.parquet"
        df_y = con.execute(f"""
            SELECT DISTINCT s.EID,
                   STRFTIME(s.DATETIME, '%Y-%m') AS MONTH,
                   s.PEAKID, c.is_sim_only
            FROM read_parquet('{path}') s
            INNER JOIN candidate_eids c ON s.EID = c.EID
            WHERE YEAR(s.DATETIME) = {year}
        """).fetchdf()
        universe_parts.append(df_y)

    universe_df = (pd.concat(universe_parts, ignore_index=True)
                   .drop_duplicates(subset=['EID', 'MONTH', 'PEAKID'])
                   .sort_values(['MONTH', 'EID', 'PEAKID'])
                   .reset_index(drop=True))

    print(f"  Universe: {len(universe_df):,} rows, {universe_df['EID'].nunique():,} EIDs")
    return universe_df


def build_targets(con, universe_df: pd.DataFrame) -> pd.DataFrame:
    """Compute PR, C, PROFIT, TARGET and partial PR for all universe rows."""
    print("Computing targets...")
    con.register('universe', universe_df)

    monthly_pr = con.execute("""
        SELECT EID, MONTH, PEAKID,
               ABS(SUM(PRICEREALIZED)) AS PR,
               SUM(PRICEREALIZED) AS PR_signed
        FROM prices GROUP BY EID, MONTH, PEAKID
    """).fetchdf()
    con.register('monthly_pr', monthly_pr)

    partial_pr = con.execute("""
        SELECT EID, MONTH, PEAKID,
               ABS(SUM(PRICEREALIZED)) AS pr_partial_current
        FROM prices
        WHERE DAY(DATETIME) BETWEEN 1 AND 7
        GROUP BY EID, MONTH, PEAKID
    """).fetchdf()
    con.register('partial_pr', partial_pr)

    target_df = con.execute("""
        SELECT u.EID, u.MONTH, u.PEAKID, u.is_sim_only,
               COALESCE(pr.PR, 0.0) AS PR,
               COALESCE(pr.PR_signed, 0.0) AS PR_signed,
               ABS(COALESCE(c.C, 0.0)) AS C,
               COALESCE(pr.PR, 0.0) - ABS(COALESCE(c.C, 0.0)) AS PROFIT,
               CASE WHEN COALESCE(pr.PR, 0.0) - ABS(COALESCE(c.C, 0.0)) > 0
                    THEN 1 ELSE 0 END AS TARGET,
               COALESCE(pp.pr_partial_current, 0.0) AS pr_partial_current
        FROM universe u
        LEFT JOIN monthly_pr pr ON u.EID = pr.EID AND u.MONTH = pr.MONTH AND u.PEAKID = pr.PEAKID
        LEFT JOIN costs c ON u.EID = c.EID AND u.MONTH = c.MONTH AND u.PEAKID = c.PEAKID
        LEFT JOIN partial_pr pp ON u.EID = pp.EID
            AND pp.MONTH = STRFTIME(STRPTIME(u.MONTH, '%Y-%m') - INTERVAL 1 MONTH, '%Y-%m')
            AND u.PEAKID = pp.PEAKID
        ORDER BY u.MONTH, u.EID, u.PEAKID
    """).fetchdf()

    # Add DECISION_MONTH
    target_df['DECISION_MONTH'] = (
        pd.to_datetime(target_df['MONTH'] + '-01') - pd.DateOffset(months=1)
    ).dt.strftime('%Y-%m')

    print(f"  Target table: {len(target_df):,} rows, "
          f"positive rate: {target_df['TARGET'].mean() * 100:.2f}%")
    return target_df


def extract_sim_daily_features(con, years: list) -> pd.DataFrame:
    """Extract PSD + impact features from sim_daily (days 1-7 only)."""
    print("Extracting sim_daily features...")

    SIM_DAILY_QUERY = """
        SELECT EID,
               STRFTIME(DATETIME, '%Y-%m') AS DECISION_MONTH,
               PEAKID,
               SUM(CASE WHEN PSD != 0 THEN 1 ELSE 0 END) AS psd_nonzero_count,
               AVG(CASE WHEN PSD != 0 THEN ABS(PSD) END) AS psd_abs_nonzero_mean,
               STDDEV(CASE WHEN PSD != 0 THEN ABS(PSD) END) AS psd_abs_nonzero_std,
               SUM(ABS(PSD)) AS psd_abs_sum,
               AVG(CASE WHEN PSD != 0 THEN PSD END) AS psd_signed_mean,
               MAX(ABS(PSD)) AS psd_abs_max,
               AVG(ACTIVATIONLEVEL) AS activation_mean,
               MAX(ACTIVATIONLEVEL) AS activation_max,
               SUM(CASE WHEN ACTIVATIONLEVEL > 0 THEN 1 ELSE 0 END) AS activation_nonzero_count,
               AVG(ABS(WINDIMPACT)) AS wind_abs_mean,
               AVG(ABS(SOLARIMPACT)) AS solar_abs_mean,
               AVG(ABS(HYDROIMPACT)) AS hydro_abs_mean,
               AVG(ABS(NONRENEWBALIMPACT)) AS nonrenew_abs_mean,
               AVG(ABS(EXTERNALIMPACT)) AS external_abs_mean,
               AVG(ABS(LOADIMPACT)) AS load_abs_mean,
               AVG(ABS(TRANSMISSIONOUTAGEIMPACT)) AS transoutage_abs_mean,
               AVG(LN(1 + ABS(HYDROIMPACT))) AS hydro_log_abs_mean,
               MAX(ABS(HYDROIMPACT)) AS hydro_abs_max,
               AVG(LN(1 + ABS(WINDIMPACT))) AS wind_log_abs_mean,
               MAX(ABS(WINDIMPACT)) AS wind_abs_max,
               AVG(LN(1 + ABS(LOADIMPACT))) AS load_log_abs_mean,
               MAX(ABS(LOADIMPACT)) AS load_abs_max,
               AVG(CASE WHEN SCENARIOID = 1 THEN ABS(PSD) END) AS psd_abs_s1_mean,
               AVG(CASE WHEN SCENARIOID IN (2,3) THEN ABS(PSD) END) AS psd_abs_s23_mean,
               STDDEV(CASE WHEN PSD != 0 THEN PSD END) AS psd_scenario_spread,
               AVG(CASE WHEN DAY(DATETIME) <= 3 THEN ABS(PSD) END) AS psd_abs_early,
               AVG(CASE WHEN DAY(DATETIME) BETWEEN 4 AND 7 THEN ABS(PSD) END) AS psd_abs_late
        FROM read_parquet('data/sim_daily/sim_daily_{year}.parquet')
        WHERE DAY(DATETIME) BETWEEN 1 AND 7 AND YEAR(DATETIME) = {year}
        GROUP BY EID, DECISION_MONTH, PEAKID
    """

    parts = []
    for year in years:
        query = SIM_DAILY_QUERY.replace('{year}', str(year))
        df_y = con.execute(query).fetchdf()
        parts.append(df_y)
        print(f"  {year}: {len(df_y):,} rows")

    result = pd.concat(parts, ignore_index=True)
    print(f"  Total: {result.shape}")
    return result


def extract_sim_monthly_features(con, years: list) -> pd.DataFrame:
    """Extract PSM features from sim_monthly."""
    print("Extracting sim_monthly features...")

    SIM_MONTHLY_QUERY = """
        SELECT EID,
               STRFTIME(DATETIME, '%Y-%m') AS TARGET_MONTH,
               PEAKID,
               SUM(CASE WHEN PSM != 0 THEN 1 ELSE 0 END) AS psm_nonzero_count,
               AVG(CASE WHEN PSM != 0 THEN ABS(PSM) END) AS psm_abs_nonzero_mean,
               STDDEV(CASE WHEN PSM != 0 THEN ABS(PSM) END) AS psm_abs_nonzero_std,
               SUM(ABS(PSM)) AS psm_abs_sum,
               AVG(CASE WHEN PSM != 0 THEN PSM END) AS psm_signed_mean,
               MAX(ABS(PSM)) AS psm_abs_max,
               AVG(ACTIVATIONLEVEL) AS psm_activation_mean,
               MAX(ACTIVATIONLEVEL) AS psm_activation_max,
               AVG(ABS(WINDIMPACT)) AS psm_wind_abs_mean,
               AVG(ABS(SOLARIMPACT)) AS psm_solar_abs_mean,
               AVG(ABS(HYDROIMPACT)) AS psm_hydro_abs_mean,
               AVG(ABS(NONRENEWBALIMPACT)) AS psm_nonrenew_abs_mean,
               AVG(ABS(EXTERNALIMPACT)) AS psm_external_abs_mean,
               AVG(CASE WHEN SCENARIOID = 1 THEN ABS(PSM) END) AS psm_abs_s1_mean,
               AVG(CASE WHEN SCENARIOID IN (2,3) THEN ABS(PSM) END) AS psm_abs_s23_mean,
               STDDEV(CASE WHEN PSM != 0 THEN PSM END) AS psm_scenario_spread
        FROM read_parquet('data/sim_monthly/sim_monthly_{year}.parquet')
        WHERE YEAR(DATETIME) = {year}
        GROUP BY EID, TARGET_MONTH, PEAKID
    """

    parts = []
    for year in years:
        query = SIM_MONTHLY_QUERY.replace('{year}', str(year))
        df_y = con.execute(query).fetchdf()
        parts.append(df_y)
        print(f"  {year}: {len(df_y):,} rows")

    result = pd.concat(parts, ignore_index=True)
    print(f"  Total: {result.shape}")
    return result


def compute_historical_lags(target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-corrected historical lag features."""
    print("Computing historical lag features...")
    df = target_df.sort_values(['EID', 'PEAKID', 'MONTH']).copy()
    grouped = df.groupby(['EID', 'PEAKID'])

    # has_history flags BEFORE fillna
    df['has_pr_history'] = grouped['PR'].shift(2).notna().astype(int)
    df['has_profit_history'] = grouped['PROFIT'].shift(2).notna().astype(int)
    df['has_cost_history'] = grouped['C'].shift(1).notna().astype(int)

    # Leakage-corrected lags: PR/PROFIT use shift(2), C uses shift(1)
    df['pr_lag1'] = grouped['PR'].shift(2)
    df['pr_lag2'] = grouped['PR'].shift(3)
    df['pr_lag3'] = grouped['PR'].shift(4)
    df['c_lag1'] = grouped['C'].shift(1)
    df['profit_lag1'] = grouped['PROFIT'].shift(2)
    df['profit_lag2'] = grouped['PROFIT'].shift(3)
    df['target_lag1'] = grouped['TARGET'].shift(2)

    # Rolling windows from shift(2)
    df['pr_rolling3_mean'] = grouped['PR'].transform(
        lambda x: x.shift(2).rolling(3, min_periods=1).mean())
    df['profit_rolling3_mean'] = grouped['PROFIT'].transform(
        lambda x: x.shift(2).rolling(3, min_periods=1).mean())
    df['profitable_count_3m'] = grouped['TARGET'].transform(
        lambda x: x.shift(2).rolling(3, min_periods=1).sum())

    print(f"  Lag features computed for {len(df):,} rows")
    return df


def merge_master_dataset(target_with_hist: pd.DataFrame,
                         sim_daily_features: pd.DataFrame,
                         sim_monthly_features: pd.DataFrame | None) -> pd.DataFrame:
    """Merge all features into master dataset."""
    print("Merging master dataset...")
    master = target_with_hist.copy()

    # Join sim_daily on DECISION_MONTH
    master = master.merge(sim_daily_features,
                          on=['EID', 'DECISION_MONTH', 'PEAKID'], how='left')

    # Join sim_monthly on TARGET_MONTH
    if sim_monthly_features is not None:
        master = master.merge(sim_monthly_features,
                              left_on=['EID', 'MONTH', 'PEAKID'],
                              right_on=['EID', 'TARGET_MONTH', 'PEAKID'],
                              how='left')
        master.drop(columns=['TARGET_MONTH'], inplace=True, errors='ignore')

    # Calendar features
    master['month_of_year'] = pd.to_datetime(master['MONTH']).dt.month
    master['year'] = pd.to_datetime(master['MONTH']).dt.year
    master['season'] = master['month_of_year'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    master['season_encoded'] = master['season'].map(
        {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3})

    # Fill NaN (excluding has_history flags)
    has_hist_cols = [c for c in master.columns if c.startswith('has_')]
    sim_cols = [c for c in master.columns
                if c.startswith(('psd_', 'psm_', 'activation', 'wind_', 'solar_',
                                 'hydro_', 'nonrenew_', 'external_', 'load_', 'transoutage_'))]
    master[sim_cols] = master[sim_cols].fillna(0)

    lag_cols = [c for c in master.columns
                if c.startswith(('pr_lag', 'c_lag', 'profit_lag', 'target_lag',
                                 'pr_rolling', 'profit_rolling', 'profitable_count'))
                and c not in has_hist_cols]
    master[lag_cols] = master[lag_cols].fillna(0)

    print(f"  Master dataset: {master.shape}")
    return master


def get_feature_columns(master_df: pd.DataFrame) -> list:
    """Get final feature columns (excluding dropped multicollinear features)."""
    all_features = [c for c in master_df.columns
                    if c not in ID_COLS + TARGET_COLS + META_COLS]
    return [f for f in all_features if f not in DROP_FEATURES]


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def train_models(master_df: pd.DataFrame, feature_cols: list,
                 train_end: str) -> tuple:
    """
    Train classifier + regressor on data before train_end.
    Uses AUC for classifier early stopping (not logloss — see FINDINGS F25).

    Returns: (clf, reg, best_params)
    """
    df_train_all = master_df[(master_df['is_sim_only'] == 0) &
                             (master_df['MONTH'] < train_end)].copy()

    # Internal split for early stopping: last 3 months as internal validation
    train_months = sorted(df_train_all['MONTH'].unique())
    if len(train_months) > 6:
        internal_val_start = train_months[-3]
    else:
        internal_val_start = train_months[-1]

    df_tr = df_train_all[df_train_all['MONTH'] < internal_val_start]
    df_vl = df_train_all[df_train_all['MONTH'] >= internal_val_start]

    scale_pos = (df_tr['TARGET'] == 0).sum() / max((df_tr['TARGET'] == 1).sum(), 1)
    print(f"  Training: {len(df_tr):,} rows, validation: {len(df_vl):,} rows")
    print(f"  scale_pos_weight = {scale_pos:.2f}")

    # Classifier
    clf_params = {
        'objective': 'binary', 'verbosity': -1, 'seed': SEED,
        'n_estimators': 2000, 'learning_rate': 0.05,
        'max_depth': 6, 'num_leaves': 31,
        'min_child_samples': 50, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos,
    }

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(
        df_tr[feature_cols], df_tr['TARGET'],
        eval_set=[(df_vl[feature_cols], df_vl['TARGET'])],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, first_metric_only=True),
                   lgb.log_evaluation(50)]
    )
    clf_iters = clf.best_iteration_
    print(f"  Classifier: best_iteration={clf_iters}")

    # Regressor
    reg_params = {
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'seed': SEED,
        'n_estimators': 2000, 'learning_rate': 0.05,
        'max_depth': 6, 'num_leaves': 31,
        'min_child_samples': 50, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
    }

    y_tr_profit = np.log1p(np.maximum(df_tr['PROFIT'].values, 0))
    y_vl_profit = np.log1p(np.maximum(df_vl['PROFIT'].values, 0))

    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(
        df_tr[feature_cols], y_tr_profit,
        eval_set=[(df_vl[feature_cols], y_vl_profit)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    reg_iters = reg.best_iteration_
    print(f"  Regressor: best_iteration={reg_iters}")

    # Retrain on ALL training data with optimal iterations
    print(f"  Retraining on full training set ({len(df_train_all):,} rows)...")
    final_clf = lgb.LGBMClassifier(**{**clf_params, 'n_estimators': clf_iters})
    final_clf.fit(df_train_all[feature_cols], df_train_all['TARGET'])

    y_all_profit = np.log1p(np.maximum(df_train_all['PROFIT'].values, 0))
    final_reg = lgb.LGBMRegressor(**{**reg_params, 'n_estimators': reg_iters})
    final_reg.fit(df_train_all[feature_cols], y_all_profit)

    return final_clf, final_reg


# ═══════════════════════════════════════════════════════════════
# SELECTION & OUTPUT
# ═══════════════════════════════════════════════════════════════

def select_opportunities(proba, pred_profit, threshold, K, alpha=0.5):
    """Rank-based selection for one month."""
    candidates = np.where(proba >= threshold)[0]
    if len(candidates) < 10:
        candidates = np.argsort(proba)[-10:]

    r_proba = rankdata(proba[candidates])
    r_profit = rankdata(pred_profit[candidates])
    combined = alpha * r_proba + (1 - alpha) * r_profit

    n_select = min(max(K, 10), 100, len(candidates))
    top_idx = candidates[np.argsort(combined)[-n_select:]]
    return top_idx


def generate_opportunities(master_df, clf, reg, feature_cols,
                           target_months, threshold, K, alpha):
    """Generate opportunities.csv for target months."""
    print(f"\nGenerating selections for {len(target_months)} months...")
    output_rows = []

    for month in sorted(target_months):
        df_month = master_df[master_df['MONTH'] == month].copy()
        if len(df_month) == 0:
            print(f"  {month}: no data, skipping")
            continue

        # Exclude sim-only EIDs (0% positive in training)
        df_pred = df_month[df_month['is_sim_only'] == 0].copy()
        if len(df_pred) == 0:
            df_pred = df_month.copy()  # Fallback

        proba = clf.predict_proba(df_pred[feature_cols])[:, 1]
        pred_profit = np.expm1(reg.predict(df_pred[feature_cols]))

        selected_idx = select_opportunities(proba, pred_profit, threshold, K, alpha)
        df_selected = df_pred.iloc[selected_idx]

        for _, row in df_selected.iterrows():
            output_rows.append({
                'TARGET_MONTH': row['MONTH'],
                'PEAK_TYPE': 'ON' if row['PEAKID'] == 1 else 'OFF',
                'EID': row['EID'],
            })

        n_sel = len(selected_idx)
        print(f"  {month}: {n_sel} opportunities selected "
              f"(from {len(df_pred)} candidates)")

    result = pd.DataFrame(output_rows)
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="FTR Opportunity Selection — CSD x MAG Energy Solutions")
    parser.add_argument("--start-month", required=True,
                        help="Start of target period (YYYY-MM)")
    parser.add_argument("--end-month", required=True,
                        help="End of target period (YYYY-MM)")
    parser.add_argument("--output", default="opportunities.csv",
                        help="Output CSV path (default: opportunities.csv)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Probability threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help=f"Max selections per month (default: {DEFAULT_K})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"Proba/profit blend (default: {DEFAULT_ALPHA})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Target months
    target_months = pd.date_range(
        args.start_month + "-01", args.end_month + "-01", freq="MS"
    ).strftime("%Y-%m").tolist()
    print(f"Target period: {args.start_month} to {args.end_month} ({len(target_months)} months)")

    # Detect available data
    years = detect_years(DATA_DIR)
    print(f"Available years: {years}")

    # Setup DuckDB
    print("\nSetting up data views...")
    con, sim_monthly_available = setup_duckdb(DATA_DIR)
    print(f"  sim_monthly: {'available' if sim_monthly_available else 'NOT available'}")

    # Build features
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    universe_df = build_universe(con, years, sim_monthly_available)
    target_df = build_targets(con, universe_df)
    sim_daily_feats = extract_sim_daily_features(con, years)

    sim_monthly_feats = None
    if sim_monthly_available:
        sim_monthly_feats = extract_sim_monthly_features(con, years)

    target_with_hist = compute_historical_lags(target_df)
    master_df = merge_master_dataset(target_with_hist, sim_daily_feats, sim_monthly_feats)
    feature_cols = get_feature_columns(master_df)
    print(f"\n  Final features: {len(feature_cols)}")

    con.close()

    # Determine training period: all months BEFORE the target period
    # (use all available labeled data for maximum training signal)
    train_end = args.start_month
    print(f"\n{'=' * 60}")
    print(f"MODEL TRAINING (data < {train_end})")
    print("=" * 60)

    final_clf, final_reg = train_models(master_df, feature_cols, train_end)

    # Generate selections
    print(f"\n{'=' * 60}")
    print("OPPORTUNITY SELECTION")
    print("=" * 60)

    opportunities = generate_opportunities(
        master_df, final_clf, final_reg, feature_cols,
        target_months, args.threshold, args.K, args.alpha
    )

    # Validate constraints
    sel_per_month = opportunities.groupby('TARGET_MONTH').size()
    print(f"\n  Selections per month: min={sel_per_month.min()}, "
          f"max={sel_per_month.max()}, mean={sel_per_month.mean():.0f}")

    # Save
    opportunities.to_csv(args.output, index=False)
    print(f"\n  Saved to: {args.output}")
    print(f"  Total rows: {len(opportunities):,}")

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  Months: {args.start_month} to {args.end_month}")
    print(f"  Selection params: threshold={args.threshold}, K={args.K}, alpha={args.alpha}")
    print(f"\n  To evaluate: python evaluate.py {args.output} "
          f"--start-month {args.start_month} --end-month {args.end_month}")


if __name__ == "__main__":
    main()
