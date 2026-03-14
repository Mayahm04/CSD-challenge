I executed the beginning of the notebook on collab quickly instead of locally, and here as some results that I am still making me doubt or question how you are training and validating the model (the way you implemented the walk forward cross validation) because the F1 is very low. Here are some of the raw output of the cells : 
-  → Aucune fuite detectee. 60 features valides.
- === Taux de profitabilite par annee (CONCEPT DRIFT) ===
  2020:  28,767 lignes | 3,092 positifs (10.7%)
  2021:  44,246 lignes | 4,062 positifs (9.2%)
  2022:  59,288 lignes | 4,736 positifs (8.0%)
  2023:  71,860 lignes | 3,516 positifs (4.9%)

=== Taux par PEAKID ===
  PEAKID=0 (OFF): 7.53% positif
  PEAKID=1 (ON): 7.56% positif
→ Le taux positif chute de 10.7% (2020) a 4.9% (2023).
  Le modele entraine sur 2020-2022 (~8.5%) sera evalue sur 2023 (~4.9%).
  Walk-forward CV et calibration du seuil sur donnees recentes sont obligatoires.

=== Paires avec |r| > 0.95 (candidates a suppression) ===
  has_pr_history                 <-> has_profit_history              r=1.000
  psm_abs_nonzero_std            <-> psm_scenario_spread             r=0.999
  psd_abs_nonzero_std            <-> psd_scenario_spread             r=0.997
  psm_abs_nonzero_mean           <-> psm_signed_mean                 r=0.968

→ 3 features supprimees (r > 0.99)
→ 57 features finales retenues
  Supprimees: ['has_profit_history', 'psm_abs_nonzero_std', 'psd_abs_nonzero_std']
=== Split temporel ===
  Train: 100,797 lignes | 9,623 positifs (9.5%) | 2020-01 → 2022-06
  Val  :  66,308 lignes | 4,130 positifs (6.2%) | 2022-07 → 2023-06
  Test :  37,056 lignes | 1,653 positifs (4.5%) | 2023-07 → 2023-12

  Walk-forward: 12 folds mensuels, 4 folds trimestriels
  Test: 6 mois (intouchable jusqu'a l'eval finale)
=== Baselines HEURISTIQUES (sans ML, K=50) ===
  Strategie                                 F1       Profit
  -------------------------------------------------------
  A — profitable_count_3m               0.1366      502,267
  B — psm_abs_max                       0.0841      350,098
  C — pr_lag1                           0.1070     -215,630

  → La persistance historique est un signal fort meme sans ML.

======================================================================
  BASELINE — Regression Logistique (t=0.20, K=75)
  F1-score: 0.2103  |  Profit net:      949,827
======================================================================
  Mois        Sel   TP   Pos    Prec  Recall       Profit
  ------------------------------------------------------------
  2022-07      75   38   352   50.7%   10.8%       40,374
  2022-08      75   45   314   60.0%   14.3%      -25,538
  2022-09      75   53   403   70.7%   13.2%      226,485
  2022-10      75   43   394   57.3%   10.9%      139,980
  2022-11      75   48   436   64.0%   11.0%      207,253
  2022-12      75   44   368   58.7%   12.0%      305,417
  2023-01      75   37   322   49.3%   11.5%     -283,614
  2023-02      75   53   334   70.7%   15.9%      119,400
  2023-03      75   53   315   70.7%   16.8%      110,803
  2023-04      75   44   377   58.7%   11.7%      209,533
  2023-05      75   28   245   37.3%   11.4%     -128,549
  2023-06      75   43   270   57.3%   15.9%       28,284

scale_pos_weight = 9.47

--- Approche A : Classificateur seul (walk-forward quarterly) ---
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.248161
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
  Fold 2022-07→2022-10: F1=0.0200, Profit=     5,536
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[3]	valid_0's binary_logloss: 0.258647
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
  Fold 2022-10→2023-01: F1=0.0228, Profit=    26,754
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.220081
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
  Fold 2023-01→2023-04: F1=0.0460, Profit=    26,735
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.203948
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  Fold 2023-04→2023-07: F1=0.0325, Profit=    40,562
  → Mean F1=0.0303, Profit=99,587, Combined=0.1512

--- Approche B : Two-stage (walk-forward quarterly) ---
  Test de alpha = blend entre proba (alpha=1) et profit predit (alpha=0)

Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.248161
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[3]	valid_0's binary_logloss: 0.258647
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.220081
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.203948
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  alpha=0.3: F1=0.0303, Profit=    99,587, Combined=0.1512
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.248161
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[3]	valid_0's binary_logloss: 0.258647
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.220081
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.203948
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  alpha=0.5: F1=0.0303, Profit=    99,587, Combined=0.1512
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.248161
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[3]	valid_0's binary_logloss: 0.258647
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.220081
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[2]	valid_0's binary_logloss: 0.203948
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  alpha=0.7: F1=0.0303, Profit=    99,587, Combined=0.1512

  → Meilleur alpha = 0.3

--- Approche C : Focal Loss + Two-stage (walk-forward quarterly) ---
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_288/982304889.py in <cell line: 0>()
     29 
     30 print("--- Approche C : Focal Loss + Two-stage (walk-forward quarterly) ---")
---> 31 res_C = run_walk_forward_cv(df_model, FEATURE_COLS_FINAL,
     32                             FOCAL_CLF_PARAMS, DEFAULT_REG_PARAMS,
     33                             threshold=0.3, K=50, alpha=0.5,

7 frames
/tmp/ipykernel_288/982304889.py in focal_loss_objective(y_true, y_pred, gamma, alpha_fl)
      6     """Focal loss custom pour LightGBM."""
      7     p = 1.0 / (1.0 + np.exp(-y_pred))
----> 8     grad = -(alpha_fl * y_true * (1 - p)**gamma * (gamma * p * np.log(np.maximum(p, 1e-15)) + p - 1)
      9              + (1 - alpha_fl) * (1 - y_true) * p**gamma * (
     10                  -gamma * (1 - p) * np.log(np.maximum(1 - p, 1e-15)) + p))

TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'

## Resultat après correction des bugs : 
scale_pos_weight = 9.47

--- Approche A : Classificateur seul (walk-forward quarterly) ---
Training until validation scores don't improve for 50 rounds
[50]	valid_0's auc: 0.843049	valid_0's binary_logloss: 0.429365
[100]	valid_0's auc: 0.84489	valid_0's binary_logloss: 0.436189
Early stopping, best iteration is:
[91]	valid_0's auc: 0.845054	valid_0's binary_logloss: 0.437445
Evaluated only: auc
    CLF best_iteration=91, best_score=OrderedDict({'auc': np.float64(0.845054215965936), 'binary_logloss': np.float64(0.4374448058066739)})
    Proba stats: min=0.0620, median=0.2379, max=0.9670, std=0.2174
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
  Fold 2022-07→2022-10: CLF(iters=91, AUC=0.8451, F1_pw=0.3732) → Selection F1=0.1559, Profit=   157,077
Training until validation scores don't improve for 50 rounds
[50]	valid_0's auc: 0.821052	valid_0's binary_logloss: 0.390272
Early stopping, best iteration is:
[46]	valid_0's auc: 0.821489	valid_0's binary_logloss: 0.384002
Evaluated only: auc
    CLF best_iteration=46, best_score=OrderedDict({'auc': np.float64(0.8214885689333001), 'binary_logloss': np.float64(0.38400173754684297)})
    Proba stats: min=0.1037, median=0.2012, max=0.8845, std=0.1913
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
  Fold 2022-10→2023-01: CLF(iters=46, AUC=0.8215, F1_pw=0.3986) → Selection F1=0.1409, Profit=   237,886
Training until validation scores don't improve for 50 rounds
[50]	valid_0's auc: 0.854905	valid_0's binary_logloss: 0.351805
[100]	valid_0's auc: 0.857116	valid_0's binary_logloss: 0.362986
[150]	valid_0's auc: 0.858182	valid_0's binary_logloss: 0.3503
[200]	valid_0's auc: 0.85768	valid_0's binary_logloss: 0.340247
Early stopping, best iteration is:
[169]	valid_0's auc: 0.858467	valid_0's binary_logloss: 0.346364
Evaluated only: auc
    CLF best_iteration=169, best_score=OrderedDict({'auc': np.float64(0.8584670489552029), 'binary_logloss': np.float64(0.3463640689409379)})
    Proba stats: min=0.0431, median=0.1695, max=0.9853, std=0.2111
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
  Fold 2023-01→2023-04: CLF(iters=169, AUC=0.8585, F1_pw=0.3917) → Selection F1=0.2141, Profit=   286,265
Training until validation scores don't improve for 50 rounds
[50]	valid_0's auc: 0.838492	valid_0's binary_logloss: 0.31751
Early stopping, best iteration is:
[33]	valid_0's auc: 0.839971	valid_0's binary_logloss: 0.290994
Evaluated only: auc
    CLF best_iteration=33, best_score=OrderedDict({'auc': np.float64(0.8399705330705588), 'binary_logloss': np.float64(0.2909938252474569)})
    Proba stats: min=0.0902, median=0.1505, max=0.8034, std=0.1625
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  Fold 2023-04→2023-07: CLF(iters=33, AUC=0.8400, F1_pw=0.3812) → Selection F1=0.1631, Profit=   120,889

  Classifier: mean AUC=0.8412, mean iters=85

  → Mean F1=0.1685, Profit=802,117, Combined=0.5843


 --- Approche B : Two-stage (walk-forward quarterly) ---
  Test de alpha = blend entre proba (alpha=1) et profit predit (alpha=0)

Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[91]	valid_0's auc: 0.845054	valid_0's binary_logloss: 0.437445
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[46]	valid_0's auc: 0.821489	valid_0's binary_logloss: 0.384002
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[169]	valid_0's auc: 0.858467	valid_0's binary_logloss: 0.346364
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[33]	valid_0's auc: 0.839971	valid_0's binary_logloss: 0.290994
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  alpha=0.3: F1=0.1747, Profit= 1,359,006, Combined=0.5874
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[91]	valid_0's auc: 0.845054	valid_0's binary_logloss: 0.437445
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[46]	valid_0's auc: 0.821489	valid_0's binary_logloss: 0.384002
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[169]	valid_0's auc: 0.858467	valid_0's binary_logloss: 0.346364
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[33]	valid_0's auc: 0.839971	valid_0's binary_logloss: 0.290994
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  alpha=0.5: F1=0.1754, Profit= 1,473,837, Combined=0.5877
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[91]	valid_0's auc: 0.845054	valid_0's binary_logloss: 0.437445
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[46]	valid_0's auc: 0.821489	valid_0's binary_logloss: 0.384002
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[169]	valid_0's auc: 0.858467	valid_0's binary_logloss: 0.346364
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[33]	valid_0's auc: 0.839971	valid_0's binary_logloss: 0.290994
Evaluated only: auc
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  alpha=0.7: F1=0.1702, Profit= 1,099,883, Combined=0.5851

  → Meilleur alpha = 0.5


--- Approche C : Focal Loss + Two-stage (walk-forward quarterly) ---
Training until validation scores don't improve for 50 rounds
[50]	valid_0's binary_logloss: 31.4456	valid_0's auc: 0.5
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.425723	valid_0's auc: 0.199989
Evaluated only: binary_logloss
    CLF best_iteration=1, best_score=OrderedDict({'binary_logloss': np.float64(0.42572302923172045), 'auc': np.float64(0.19998862034660214)})
    Proba stats: min=0.4881, median=0.5334, max=0.5334, std=0.0054
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[45]	valid_0's rmse: 1.45999
  Fold 2022-07→2022-10: CLF(iters=1, AUC=0.2000, F1_pw=0.1217) → Selection F1=0.0295, Profit=     8,031
Training until validation scores don't improve for 50 rounds
[50]	valid_0's binary_logloss: 31.1007	valid_0's auc: 0.497078
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.371241	valid_0's auc: 0.233105
Evaluated only: binary_logloss
    CLF best_iteration=1, best_score=OrderedDict({'binary_logloss': np.float64(0.37124076210378937), 'auc': np.float64(0.23310488820250666)})
    Proba stats: min=0.4898, median=0.5335, max=0.5335, std=0.0047
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[83]	valid_0's rmse: 1.55868
  Fold 2022-10→2023-01: CLF(iters=1, AUC=0.2331, F1_pw=0.1384) → Selection F1=0.0504, Profit=    25,100
Training until validation scores don't improve for 50 rounds
[50]	valid_0's binary_logloss: 31.9613	valid_0's auc: 0.487735
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.324586	valid_0's auc: 0.181301
Evaluated only: binary_logloss
    CLF best_iteration=1, best_score=OrderedDict({'binary_logloss': np.float64(0.3245858856434311), 'auc': np.float64(0.1813006336076529)})
    Proba stats: min=0.4891, median=0.5335, max=0.5335, std=0.0042
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[54]	valid_0's rmse: 1.22852
  Fold 2023-01→2023-04: CLF(iters=1, AUC=0.1813, F1_pw=0.1060) → Selection F1=0.0321, Profit=   -11,115
Training until validation scores don't improve for 50 rounds
[50]	valid_0's binary_logloss: 32.2664	valid_0's auc: 0.492946
Early stopping, best iteration is:
[1]	valid_0's binary_logloss: 0.286203	valid_0's auc: 0.207369
Evaluated only: binary_logloss
    CLF best_iteration=1, best_score=OrderedDict({'binary_logloss': np.float64(0.2862030241947553), 'auc': np.float64(0.2073685902420086)})
    Proba stats: min=0.4895, median=0.5336, max=0.5336, std=0.0042
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[60]	valid_0's rmse: 1.19364
  Fold 2023-04→2023-07: CLF(iters=1, AUC=0.2074, F1_pw=0.0936) → Selection F1=0.0345, Profit=    -5,366

  Classifier: mean AUC=0.2054, mean iters=1

  → Mean F1=0.0367, Profit=16,651, Combined=0.0789 

  ======================================================================
                      COMPARAISON DES APPROCHES                       
======================================================================

  Approche                              F1 moy       Profit    Score
  ---------------------------------------------------------------
  Random                                0.0160       22,048      —
  Heuristique (persist.)                0.1366      502,267      —
  Heuristique (PSM)                     0.0841      350,098      —
  LogReg                                0.2103      949,827      —
  ---------------------------------------------------------------
  A — Clf seul (alpha=1.0)              0.1685      802,117   0.5843
  B — Two-stage (alpha=0.5)             0.1754    1,473,837   0.5877
  C — Focal Loss + Two-stage            0.0367       16,651   0.0789
======================================================================

→ Meilleure approche : B (score combine = 0.5877)