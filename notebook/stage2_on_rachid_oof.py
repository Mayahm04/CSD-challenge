"""
Stage 2 pipeline on top of Rachid's RF OOF predictions.

What Rachid did:
  - RF grid-search (8 configs), best = config_id 7
    (n_estimators=300, max_depth=None, min_samples_leaf=20, max_features='sqrt')
  - Saved OOF probabilities for all 8 configs

What this script adds:
  - Stage 2 regressor (LightGBM + XGBoost, Optuna-tuned)
  - Score = rf_prob × max(0, pred_profit)
  - Per-month top-K selection (K ∈ [10, 100], Optuna-optimised)
  - 2-fold walk-forward: Fold1 val=2022, Fold2 val=2023
  - Compares 4 strategies: rf_prob only, xgb_ev (Rachid), rf_ev (ours), lgbm_ev (ours)
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import json
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, mean_absolute_error,
                             mean_squared_error, r2_score)
import lightgbm as lgb
from xgboost import XGBRegressor

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/master_dataset.parquet"
OOF_PATH       = "model_artifact/rf_tuning_oof_store.joblib"
RESULTS_PATH   = "model_artifact/rf_tuning_results.parquet"
BEST_CONFIG_ID = 7          # highest AUC from Rachid's grid (AUC=0.8447, profit=11.7M)
STAGE2_RESULTS_PATH = "model_artifact/stage2_results.parquet"
STAGE2_COMPARE_PATH = "model_artifact/stage2_strategy_comparison.csv"
STAGE2_BEST_PATH    = "model_artifact/best_stage2_strategy.json"

# ─── features (mirror Rachid's drop list) ─────────────────────────────────────
DROP_COLS = {"EID", "MONTH", "DECISION_MONTH", "PR", "PR_signed", "C",
             "PROFIT", "TARGET", "season", "season_encoded", "is_sim_only",
             "PEAKID"}  # identifier, not a feature

# ─── walk-forward folds ───────────────────────────────────────────────────────
FOLDS = [
    {"name": "Fold1", "train_end": "2021-12", "val_start": "2022-01", "val_end": "2022-12"},
    {"name": "Fold2", "train_end": "2022-12", "val_start": "2023-01", "val_end": "2023-12"},
]

K_MIN, K_MAX = 10, 100


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_features(df):
    num_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [c for c in num_cols if c not in DROP_COLS]


def per_month_topk_profit(df_val, score_col, k):
    """Select top-k per month, return total realized profit."""
    total = 0.0
    for _, grp in df_val.groupby("MONTH"):
        top = grp.nlargest(k, score_col)
        total += top["PROFIT"].sum()
    return total


def topk_classification_metrics(df_val, score_col, k):
    """Precision, Recall, F1 treating top-k selected per month as predicted positives."""
    y_true_all, y_pred_all = [], []
    for _, grp in df_val.groupby("MONTH"):
        selected = grp.nlargest(k, score_col).index
        pred = np.zeros(len(grp), dtype=int)
        pred[np.isin(grp.index, selected)] = 1
        y_true_all.extend(grp["TARGET"].values.tolist())
        y_pred_all.extend(pred.tolist())
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f1


def optimize_k(df_val, score_col, k_min=K_MIN, k_max=K_MAX):
    """Grid-search K in [k_min, k_max] to maximise total realized profit."""
    best_k, best_profit = k_min, -np.inf
    for k in range(k_min, k_max + 1):
        p = per_month_topk_profit(df_val, score_col, k)
        if p > best_profit:
            best_profit, best_k = p, k
    return best_k, best_profit


def build_strategy_comparison(res_df):
    """Aggregate fold-level results into a strategy comparison table."""
    strategies = ["rf_prob", "lgbm_ev", "xgb_ev", "ens_ev"]
    rows = []
    for s in strategies:
        row = {"strategy": s}
        for fold in res_df.index.tolist():
            row[f"{fold}_k"] = int(res_df.loc[fold, f"{s}_k"])
            row[f"{fold}_profit"] = float(res_df.loc[fold, f"{s}_profit"])
            row[f"{fold}_top10"] = float(res_df.loc[fold, f"{s}_top10"])
        row["total_profit"] = float(sum(row[f"{fold}_profit"] for fold in res_df.index.tolist()))
        row["mean_top10_profit"] = float(np.mean([row[f"{fold}_top10"] for fold in res_df.index.tolist()]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total_profit", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — LightGBM regressor (Optuna-tuned on training positives)
# ══════════════════════════════════════════════════════════════════════════════

def tune_lgbm_regressor(X_tr, y_tr, n_trials=40):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 120),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "verbosity": -1,
            "n_jobs": -1,
        }
        # 3-fold CV on training positives only
        n = len(X_tr)
        fold_size = n // 3
        scores = []
        for i in range(3):
            va_idx = slice(i * fold_size, (i + 1) * fold_size)
            tr_idx = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, n))
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
                  eval_set=[(X_tr.iloc[va_idx], y_tr.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])
            pred = m.predict(X_tr.iloc[va_idx])
            # metric: pearson correlation (proxy for ranking quality)
            corr = np.corrcoef(y_tr.iloc[va_idx], pred)[0, 1]
            scores.append(corr if np.isfinite(corr) else -1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb_regressor(X_tr, y_tr, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 7),
            "learning_rate":   trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": -1,
        }
        n = len(X_tr)
        fold_size = n // 3
        scores = []
        for i in range(3):
            va_idx = slice(i * fold_size, (i + 1) * fold_size)
            tr_idx = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, n))
            m = XGBRegressor(**params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
                  eval_set=[(X_tr.iloc[va_idx], y_tr.iloc[va_idx])],
                  verbose=False)
            pred = m.predict(X_tr.iloc[va_idx])
            corr = np.corrcoef(y_tr.iloc[va_idx], pred)[0, 1]
            scores.append(corr if np.isfinite(corr) else -1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Stage 2 pipeline — continuing from Rachid's RF OOF (config_id=7)")
    print("=" * 70)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    df = pd.read_parquet(DATA_PATH)
    df["MONTH"] = df["MONTH"].astype(str)
    df["TARGET"] = df["TARGET"].astype(int)
    df["PROFIT"] = pd.to_numeric(df["PROFIT"], errors="coerce").fillna(0)
    features = get_features(df)
    print(f"  {len(df)} rows, {len(features)} features")

    # ── load Rachid's OOF ─────────────────────────────────────────────────────
    print("\n[2/5] Loading Rachid's OOF (best config_id=7) …")
    oof_store = joblib.load(OOF_PATH)
    oof = oof_store[BEST_CONFIG_ID][["TARGET", "PROFIT", "MONTH", "rf_prob"]].copy()

    # OOF covers rows from 2020-07+ (first 6 months are always training in Rachid's CV)
    df_filtered = df[df["TARGET"].isin([0, 1])].copy()
    df_oof_base = df_filtered[df_filtered["MONTH"] >= "2020-07"].reset_index(drop=True)
    oof = oof.reset_index(drop=True)
    assert len(df_oof_base) == len(oof), (
        f"Row count mismatch: df_07+={len(df_oof_base)}, oof={len(oof)}"
    )
    assert (df_oof_base["TARGET"].values == oof["TARGET"].values).all(), \
        "TARGET mismatch between df and OOF — wrong alignment"
    df_oof_base["rf_prob"] = oof["rf_prob"].values

    # For stage-2 training we also need the first 6 months (no OOF prob → use 0.0)
    df_early = df_filtered[df_filtered["MONTH"] < "2020-07"].copy()
    df_early["rf_prob"] = 0.0
    df_filtered = pd.concat([df_early, df_oof_base], ignore_index=True)

    print(f"  OOF rf_prob range (2020-07+): [{df_oof_base['rf_prob'].min():.4f}, "
          f"{df_oof_base['rf_prob'].max():.4f}]")

    # ── walk-forward stage 2 ──────────────────────────────────────────────────
    print("\n[3/5] Walk-forward Stage 2 (regressor on positives) …")
    fold_results = []

    for fold in FOLDS:
        fname    = fold["name"]
        tr_mask  = df_filtered["MONTH"] <= fold["train_end"]
        va_mask  = ((df_filtered["MONTH"] >= fold["val_start"]) &
                    (df_filtered["MONTH"] <= fold["val_end"]))

        df_tr = df_filtered[tr_mask].copy()
        df_va = df_filtered[va_mask].copy()

        print(f"\n  ── {fname}: train ≤{fold['train_end']}, "
              f"val {fold['val_start']}–{fold['val_end']} ──")
        print(f"     train {len(df_tr)} rows ({df_tr['TARGET'].sum()} pos), "
              f"val {len(df_va)} rows ({df_va['TARGET'].sum()} pos)")

        # Stage 1 AUC (Rachid's RF, no retraining needed)
        auc1 = roc_auc_score(df_va["TARGET"], df_va["rf_prob"])
        rf_pred_bin = (df_va["rf_prob"] >= 0.5).astype(int)
        f1_s1  = f1_score(df_va["TARGET"], rf_pred_bin, zero_division=0)
        prec_s1 = precision_score(df_va["TARGET"], rf_pred_bin, zero_division=0)
        rec_s1  = recall_score(df_va["TARGET"], rf_pred_bin, zero_division=0)
        print(f"     Stage-1 RF  AUC={auc1:.4f}  F1={f1_s1:.4f}  "
              f"Prec={prec_s1:.4f}  Rec={rec_s1:.4f}")

        # ── train stage-2 regressors on all training rows ─────────────────────
        X_pos  = df_tr[features]
        y_pos  = df_tr["PROFIT"]
        print(f"     Tuning LGBM regressor on {len(df_tr)} rows (all) …", end=" ", flush=True)
        lgbm_params = tune_lgbm_regressor(X_pos, y_pos, n_trials=40)
        lgbm_reg = lgb.LGBMRegressor(**lgbm_params, verbosity=-1, n_jobs=-1)
        lgbm_reg.fit(X_pos, y_pos)
        print("done")

        print(f"     Tuning XGB  regressor on {len(df_tr)} rows (all) …", end=" ", flush=True)
        xgb_params  = tune_xgb_regressor(X_pos, y_pos, n_trials=30)
        xgb_reg = XGBRegressor(**xgb_params, tree_method="hist", verbosity=0, n_jobs=-1)
        xgb_reg.fit(X_pos, y_pos)
        print("done")

        # ── Stage-2 regression metrics (on validation positives) ──────────────
        pos_va = df_va[df_va["TARGET"] == 1]
        X_pos_va = pos_va[features]
        y_pos_va = pos_va["PROFIT"]
        for reg_name, reg_model in [("LGBM", lgbm_reg), ("XGB", xgb_reg)]:
            preds_va = reg_model.predict(X_pos_va)
            mae  = mean_absolute_error(y_pos_va, preds_va)
            rmse = np.sqrt(mean_squared_error(y_pos_va, preds_va))
            r2   = r2_score(y_pos_va, preds_va)
            print(f"     {reg_name} regressor (val positives) — "
                  f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}")

        # ── validation predictions ─────────────────────────────────────────────
        X_va = df_va[features]
        lgbm_profit = lgbm_reg.predict(X_va)
        xgb_profit  = xgb_reg.predict(X_va)

        # Ensemble regressor (average)
        ens_profit = (lgbm_profit + xgb_profit) / 2

        # Scores
        df_va = df_va.copy()
        df_va["lgbm_ev"]  = df_va["rf_prob"] * np.maximum(0, lgbm_profit)
        df_va["xgb_ev"]   = df_va["rf_prob"] * np.maximum(0, xgb_profit)
        df_va["ens_ev"]   = df_va["rf_prob"] * np.maximum(0, ens_profit)

        # ── per-month top-K optimisation ──────────────────────────────────────
        strategies = {
            "rf_prob":  "rf_prob",
            "lgbm_ev":  "lgbm_ev",
            "xgb_ev":   "xgb_ev",
            "ens_ev":   "ens_ev",
        }

        print(f"\n     {'Strategy':<12}  {'Best K':>6}  {'Total Profit':>14}  "
              f"{'Top-10/month profit':>20}  {'Prec@K':>7}  {'Rec@K':>7}  {'F1@K':>7}")
        print("     " + "-" * 80)

        fold_row = {"fold": fname, "stage1_auc": auc1,
                    "stage1_f1": f1_s1, "stage1_prec": prec_s1, "stage1_rec": rec_s1}
        for strat_name, score_col in strategies.items():
            best_k, best_profit = optimize_k(df_va, score_col)
            fixed_profit = per_month_topk_profit(df_va, score_col, 10)
            prec_k, rec_k, f1_k = topk_classification_metrics(df_va, score_col, best_k)
            fold_row[f"{strat_name}_k"]      = best_k
            fold_row[f"{strat_name}_profit"] = best_profit
            fold_row[f"{strat_name}_top10"]  = fixed_profit
            fold_row[f"{strat_name}_prec"]   = prec_k
            fold_row[f"{strat_name}_rec"]    = rec_k
            fold_row[f"{strat_name}_f1"]     = f1_k
            print(f"     {strat_name:<12}  {best_k:>6}  {best_profit:>14,.0f}  "
                  f"{fixed_profit:>20,.0f}  {prec_k:>7.4f}  {rec_k:>7.4f}  {f1_k:>7.4f}")

        fold_results.append(fold_row)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — total realized profit over 2022 + 2023")
    print("=" * 70)
    res_df = pd.DataFrame(fold_results).set_index("fold")

    strategies_order = ["rf_prob", "lgbm_ev", "xgb_ev", "ens_ev"]
    total_profits = {}
    for s in strategies_order:
        total = res_df[f"{s}_profit"].sum()
        total_profits[s] = total

    print(f"\n  {'Strategy':<12}  {'Fold1 profit':>14}  {'Fold2 profit':>14}  {'TOTAL':>14}")
    print("  " + "-" * 60)
    for s in strategies_order:
        p1 = res_df.loc["Fold1", f"{s}_profit"]
        p2 = res_df.loc["Fold2", f"{s}_profit"]
        tot = p1 + p2
        flag = "  ★" if tot == max(total_profits.values()) else ""
        print(f"  {s:<12}  {p1:>14,.0f}  {p2:>14,.0f}  {tot:>14,.0f}{flag}")

    best_strat = max(total_profits, key=total_profits.get)
    print(f"\n  Best strategy: {best_strat}  "
          f"(total profit = {total_profits[best_strat]:,.0f} €)")

    # best K per fold for best strategy
    print(f"\n  Optimal K for '{best_strat}':")
    for f in ["Fold1", "Fold2"]:
        k = res_df.loc[f, f"{best_strat}_k"]
        print(f"    {f}: K = {k}")

    comparison_df = build_strategy_comparison(res_df)
    stage2_only = comparison_df[comparison_df["strategy"].isin(["lgbm_ev", "xgb_ev", "ens_ev"])].copy()
    best_stage2 = stage2_only.iloc[0].to_dict()

    print("\n  Stage-2-only ranking:")
    print(stage2_only[["strategy", "total_profit", "mean_top10_profit"]].to_string(index=False))
    print(f"\n  Best Stage-2 strategy: {best_stage2['strategy']} "
          f"(total profit = {best_stage2['total_profit']:,.0f} €)")

    # ── save ──────────────────────────────────────────────────────────────────
    res_df.to_parquet(STAGE2_RESULTS_PATH)
    comparison_df.to_csv(STAGE2_COMPARE_PATH, index=False)
    with open(STAGE2_BEST_PATH, "w") as f:
        json.dump(best_stage2, f, indent=2)

    print(f"\n  Results saved → {STAGE2_RESULTS_PATH}")
    print(f"  Strategy table saved → {STAGE2_COMPARE_PATH}")
    print(f"  Best stage-2 summary saved → {STAGE2_BEST_PATH}")


if __name__ == "__main__":
    main()
