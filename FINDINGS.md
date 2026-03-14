# FINDINGS.md — Journal de bord analytique

## Round 1 : EDA sim_daily (2020 uniquement) — 2026-03-12

### F1. Structure des donnees
- **123.5M lignes** dans sim_daily_2020.parquet, **5,386 EIDs**, 3 scenarios, 2 PEAKIDs
- 15,522 lignes debordent en janvier 2021 : c'est le **HE overflow** du 31 dec 2020 (timestamp 2021-01-01 00:00:00)
- Meme pattern observe pour 2023 → 2024 (22,125 lignes en 2024)
- **Croissance annuelle des EIDs** : 5386 (2020) → 6026 (2021) → 6545 (2022) → 7379 (2023)

### F2. Convention HE confirmee
- Le jour 7 a bien un timestamp le 8 a 00:00:00 (derniere heure du jour)
- ON-Peak (PEAKID=1) : heures 07:00 a 22:00 (16 heures/jour)
- OFF-Peak (PEAKID=0) : heures 00:00-06:00 + 23:00-00:00 (8 heures/jour)
- Ratio ON/OFF ~2:1 en nombre de lignes : coherent

### F3. PSD est 99.79% zeros (CRITIQUE)
- Sur 123.5M lignes stockees, seulement **258,970 ont un PSD non-nul** (0.21%)
- Rappel : les donnees sont sparsifiees (lignes absentes = 0), donc les 123.5M lignes ont deja au moins une variable non-nulle
- PSD=0 signifie probablement "pas de congestion simulee sur cet element a cette heure"
- **Implication** : les features PSD doivent etre calculees sur le sous-ensemble non-nul, ou bien le signal sera noye

### F4. Scenario 1 != Scenarios 2-3 (CRITIQUE)
- S2-S3 correles a **r=0.87** (quasi-redondants)
- S1 faiblement correle avec S2 (r=0.22) et S3 (r=0.26)
- **Hypothese** : S1 represente un regime de marche fondamentalement different (base case vs stress ?)
- **Implication** : ne PAS faire la moyenne aveugle des 3 scenarios. Traiter S1 separement.

### F5. Distributions a queues lourdes
- HYDROIMPACT : range [-1,022,560 ; +111,560] — valeurs extremes
- WINDIMPACT : range [-204,386 ; +173,143]
- PSD moyen legerement negatif (-0.14) : coherent avec <5% d'opportunites profitables
- Medianes toutes a 0 (sparsity)

### F6. Sparsity par variable (parmi les lignes stockees)
| Variable | % zeros |
|----------|---------|
| PSD | 99.79% |
| SOLARIMPACT | 85.10% |
| HYDROIMPACT | 70.43% |
| TRANSMISSIONOUTAGEIMPACT | 70.29% |
| EXTERNALIMPACT | 70.23% |
| NONRENEWBALIMPACT | 69.80% |
| WINDIMPACT | 69.50% |
| LOADIMPACT | 69.09% |
| ACTIVATIONLEVEL | 67.93% |

### F7. 212 EIDs instables (2020 seulement)
- 5,174 EIDs stables sur toute l'annee, 212 apparaissent dans certains mois uniquement
- Tous les 212 sont presents en 2020 mais pas dans le debordement 2021 (disparus avant decembre)

---

## Round 2 : Decouverte multi-dataset — 2026-03-12

### F8. Dimensions croisees (CRITIQUE)
| Dataset | Lignes | EIDs distincts | Periode |
|---------|--------|---------------|---------|
| sim_daily (4 ans) | ~616M | 5,386 → 7,379 | 2020-01 a 2023-12 |
| prices | 453,167 | 3,065 | 2020-01 a 2023-12 |
| costs | 9,092 | **927** | 2020-01 a 2023-12 |

- **Costs ne contient que 927 EIDs** — c'est l'univers des opportunites reellement tradables
- sim_daily a 5000-7000 EIDs : la grande majorite n'ont jamais de cout assigne
- prices a 3,065 EIDs : plus que costs mais moins que sim_daily
- **L'intersection costs ∩ sim_daily definit les opportunites evaluables**

### F9. Volume prices tres reduit
- 453k lignes pour 3,065 EIDs sur 4 ans = ~150 lignes/EID en moyenne
- Donnees aussi sparsifiees : seules les heures avec prix realise non-nul sont stockees
- Beaucoup d'EIDs dans sim_daily n'ont jamais de prix realise

---

## Round 3 : Corrections critiques et decisions de design — 2026-03-13

*Revue croisee avec Rachid (branche Rachid2). 4 bugs corriges, 5 decisions de design.*

### F10. Bug 1 — Univers EID trop restrictif (Rachid)
- L'ancien notebook limitait l'univers aux 927 EIDs de `costs`
- **Fix** : univers hybride = union(EIDs market-validated, EIDs strong-sim)
- Pool 1 (market) : EIDs presents dans `costs` OU `prices`
- Pool 2 (sim-only) : EIDs avec ACTIVATIONLEVEL et |PSM|/|PSD| au-dessus du p80, pas dans Pool 1
- Flag `is_sim_only` pour distinguer les deux pools (feature du modele)
- **H4 partiellement resolue** : l'univers hybride inclut desormais les EIDs a fort signal de simulation

### F11. Bug 2 — Fuite de donnees sur les lags PR/PROFIT (Rachid, CRITIQUE)
- `pr_lag1 = shift(1)` donnait le prix **complet du mois M** pour predire M+1
- Or au cutoff (jour 7 de M), le prix complet de M n'est PAS encore disponible
- **Fix** : PR/PROFIT lags decales a shift(2) → le mois complet le plus recent est M-1
- `c_lag1` reste a shift(1) car le cout de M est disponible au cutoff (spec du cas)
- Ajout de `pr_partial_current` = prix jours 1-7 de M (signal cutoff-safe)

### F12. Bug 3 — Absence de prix partiel du mois courant (Rachid)
- Aucune feature ne capturait le signal de prix disponible au moment de la decision
- **Fix** : `pr_partial_current` = ABS(SUM(PRICEREALIZED)) pour les jours 1-7 du mois M
- Joint au target via DECISION_MONTH = TARGET_MONTH - 1

### F13. Bug 4 — Conflation NaN/zero dans fillna (Rachid)
- `fillna(0)` applique a toutes les colonnes rendait indistinguables :
  - Un EID nouveau (pas d'historique → NaN)
  - Un EID etabli avec zero profit (valeur reelle = 0)
- **Fix** : flags `has_pr_history`, `has_profit_history`, `has_cost_history` crees AVANT fillna
- Ces flags sont exclus du fillna pour preserver leur semantique

### F14. Decision : formule de profit mise a jour par le jury
- `PROFIT = |PR| - |C|` (valeurs absolues)
- Ancien taux de profitabilite : 1.01% → Nouveau : **~14.78%**
- Le signe des prix reflete la direction du flux d'energie, pas la profitabilite
- **|PSD|** devient le signal predictif cle (forte magnitude → |PR| eleve → probable profit)

### F15. Decision : deux modeles (classification + regression)
- Le jury evalue sur F1-score (25%) ET profit net (25%)
- Un classificateur seul perd l'information de magnitude
- **Strategie** : classificateur (filtre par confiance) + regresseur (rang par profit predit)
- Selection : top-K opportunites avec P(profitable) > seuil ET profit predit maximal, K ∈ [10, 100]

### F16. Decision : features d'impact — log_abs_mean + abs_max
- Les variables d'impact ont des queues extremement lourdes (HYDRO: [-1M, +111K]) alors que p99 ~17
- La documentation dit "pourcentages" mais les extremes sont ~60,000x le p99
- `AVG(ABS(...))` domine par les outliers → signal trompe
- **Fix** : pour HYDRO, WIND, LOAD (les 3 impacts les moins sparse apres ACTIVATION) :
  - `log_abs_mean` = AVG(LN(1 + ABS(x))) — moyenne robuste aux outliers
  - `abs_max` = MAX(ABS(x)) — capture les evenements extremes
- SOLAR (85% zeros) : trop sparse, garde seulement `abs_mean`
- Ces deux features sont complementaires : log ignore les extremes, abs_max les capture

### F17. Decision : trend PSD early/late (jours 1-3 vs 4-7)
- `psd_abs_early` vs `psd_abs_late` = momentum dans la fenetre de 7 jours du mois de decision
- Signal de tendance : la congestion augmente-t-elle ou diminue-t-elle dans la semaine ?
- Conserve pour PSD uniquement — pas d'extension aux impacts (14 colonnes supplementaires pour gain marginal, PSD capture deja le trend agrege puisqu'il derive des impacts)

### F18. Decision : scenarios S1 vs S23
- Structure de correlation (F4) : S1 different (r=0.22), S2-S3 quasi-redondants (r=0.87)
- Groupage S1 vs S23 justifie et suffisant — separer S2/S3 ajouterait une feature redondante

---

## Round 4 : Statistiques du master dataset — 2026-03-14

*Inspection du master dataset genere par Rachid. 6 findings critiques pour la modelisation.*

### F19. Dimensions du master dataset
- **211,510 lignes**, **3,173 EIDs**, 48 mois (2020-01 a 2023-12), 2 PEAKIDs, **71 colonnes** (55 features exploitables)
- **0 NaN** dans tout le dataset (fillna applique correctement)
- **PSM features presentes** : Rachid avait les donnees sim_monthly sur sa machine, les 16 colonnes PSM sont remplies
- Lignes par mois : de 2,282 (2020) a 6,344 (2023) — croissance liee a l'augmentation des EIDs

### F20. Taux de profitabilite reel (CORRECTION)
- **7.28% global** (15,406 / 211,510), **7.55% dans le pool market-validated** (15,406 / 204,161)
- L'ancien taux de 14.78% etait base sur l'univers costs-only (927 EIDs). L'univers hybride ajoute des EIDs avec PR=0 → dilution du taux
- `scale_pos_weight ≈ 12.2` pour LightGBM

### F21. sim-only = 0% positif (CRITIQUE)
- Les **7,349 lignes** `is_sim_only=1` n'ont **AUCUN** TARGET=1
- Normal : pas de PR ni C pour ces EIDs → PROFIT=0 par definition
- **Implication** : exclure du training set. Predire avec precaution (aucune donnee de validation)

### F22. Concept drift temporel (CRITIQUE)
| Annee | Lignes | Positifs | Taux |
|-------|--------|----------|------|
| 2020 | 28,767 | 3,092 | **10.7%** |
| 2021 | 44,246 | 4,062 | **9.2%** |
| 2022 | 59,288 | 4,736 | **8.0%** |
| 2023 | 71,860 | 3,516 | **4.9%** |

- La profitabilite **diminue chaque annee** : cause possible = marche plus efficient, plus de competition
- **Implication** : le modele entraine sur 2020-2022 (~9%) sera evalue sur 2023 (~5%). Walk-forward CV obligatoire. Le seuil de decision doit etre calibre sur les donnees recentes, pas sur la moyenne historique.

### F23. Top features predictives (correlation Pearson avec TARGET)
| Feature | r |
|---------|---|
| `profitable_count_3m` | **+0.365** |
| `psm_abs_max` | +0.189 |
| `psm_abs_nonzero_mean` | +0.180 |
| `psd_nonzero_count` | +0.180 |
| `psm_abs_sum` | +0.135 |
| `psd_abs_max` | +0.129 |
| `pr_rolling3_mean` | +0.122 |
| `psd_signed_mean` | -0.121 |
| `psd_scenario_spread` | +0.108 |
| `pr_lag1` | +0.107 |

- `profitable_count_3m` domine : fort signal de persistence (un EID profitable le reste souvent)
- Les features PSM (~0.18-0.19) sont plus predictives que les features PSD (~0.13-0.18)
- Les features d'impact brutes (hydro_abs_mean, load_abs_mean) ont une correlation tres faible (<0.02)
- Les features engineerees (log, max, scenario spread) importent plus que les moyennes brutes

### F24. Distribution PROFIT (TARGET=1 uniquement)
- min=0, **median=342**, mean=2,372, **max=212,803**, std=7,442
- Distribution **tres asymetrique a droite** : quelques opportunites extremement profitables dominent
- **Implication** : utiliser `log1p(PROFIT)` comme cible du regresseur pour compresser les extremes
- Le modele doit identifier ces rares "jackpots" pour maximiser le profit net

---

## Round 5 : Modelisation — Choix techniques et corrections — 2026-03-14

*Pipeline de modelisation LightGBM two-stage. 3 bugs critiques identifies et corriges.*

### F25. Bug critique : early stopping a 1-2 iterations (Sofiane)
- **Symptome** : le classificateur LightGBM s'arretait apres 1-2 iterations (decision stump), produisant F1=0.03
- **Cause** : `eval_metric='binary_logloss'` + `scale_pos_weight=9.47` → la logloss penalise les predictions confiantes pour la classe minoritaire upweightee. Apres 1 iteration, logloss augmente monotoniquement → early stopping declenche immediatement
- **Fix** : `eval_metric='auc'` mesure la qualite du ranking, insensible au biais de calibration. Resultat : 33-169 iterations, AUC 0.82-0.86
- **Lecon** : avec class imbalance + scale_pos_weight, JAMAIS utiliser logloss pour l'early stopping

### F26. Bug : focal loss metric par defaut (Sofiane)
- **Symptome** : focal loss stoppait aussi a 1 iteration, AUC=0.20
- **Cause** : `LGBMClassifier` ajoute automatiquement `binary_logloss` comme metric par defaut, AVANT la metric custom. `first_metric_only=True` selectionnait logloss au lieu de notre AUC custom
- **Fix** : `metric='none'` dans les params pour supprimer la metric par defaut

### F27. Bug : comparaison biaisee des approches (Sofiane)
- **Symptome** : LightGBM (F1=0.17) semblait inferieur a LogReg (F1=0.21)
- **Cause** : LogReg optimisait threshold/K via grid search (trouvant t=0.20, K=75) tandis que LightGBM utilisait des valeurs fixes (t=0.30, K=50). Avec ~350 positifs/mois, K=75 → recall max 21%, K=50 → recall max 14%
- **Fix** : grid search sur threshold/K/alpha pour toutes les approches, en reutilisant les predictions pre-calculees (pas de re-entrainement)
- **Resultat apres correction** : LightGBM genere deja +55% de profit vs LogReg (1.47M vs 949K) grace au regressor qui identifie les opportunites a haute valeur

### F28. Choix de l'eval metric : AUC vs logloss
- AUC mesure si le modele **classe correctement** les positifs au-dessus des negatifs
- Logloss mesure la **calibration** des probabilites (proche de la vraie distribution)
- Pour un probleme de selection/ranking, AUC est le bon choix
- La calibration des probas n'est pas critique car on utilise un ranking relatif (alpha * rank(proba) + (1-alpha) * rank(profit))

### F29. Architecture de la pipeline de selection
- **Selection par mois** : pour chaque mois cible, on selectionne independamment
- **Score = alpha * rank(proba) + (1-alpha) * rank(pred_profit)** : rank-based blending, smooth, evite les effets de seuil
- **Contrainte [10, 100]** : enforced per month (ON + OFF combines)
- **F1 de selection** (et non F1 du classificateur) : `f1_score(TARGET, selected)` sur TOUTES les opportunites du mois. C'est le vrai metric du jury

### F30. Evaluation du jury — analyse de evaluate.py
- **F1** : calcule par PEAKID (ON et OFF separement), puis moyenne des deux
- **Profit** : `|PR| - COST` pour chaque opportunite selectionnee, somme totale
- **Ground truth** : outer join entre prices et costs — les EIDs sans prix ont PR=0 (pas profitable)
- **Contrainte** : max 100 selections par mois (enforce dans evaluate.py)
- **Format CSV** : colonnes `TARGET_MONTH`, `PEAK_TYPE` (ON/OFF), `EID`
- **IMPORTANT** : le F1 est calcule sur l'union profitable ∪ selected (outer join), pas seulement sur les selected. Cela signifie que les FN comptent (opportunites profitables non selectionnees)
- **evaluate_pipeline() mis a jour** pour correspondre exactement a cette methode (F1 par PEAKID puis moyenne)

### F31. Pipeline de production — main.py
- Script autonome : `python main.py --start-month 2024-01 --end-month 2024-12`
- Reconstruit le master dataset a partir des donnees brutes (DuckDB, 2GB memory limit)
- Entraine les modeles sur toutes les donnees anterieures au start-month
- Selectionne les opportunites avec le pipeline two-stage
- Output : `opportunities.csv` au format jury (TARGET_MONTH, PEAK_TYPE, EID)
- Compatible avec les donnees 2024 et futures (2025)
- Le jury fournira les donnees 2025 (sim_daily, sim_monthly, costs) ; le script les traitera automatiquement

---

## Hypotheses a verifier

- [ ] H1 : PSD=0 signifie "pas de congestion simulee" (pas un artefact)
- [ ] H2 : S1 est un scenario "base case" et S2-S3 sont des "stress scenarios" (ou inverse)
- [ ] H3 : Les 927 EIDs de costs sont un sous-ensemble stable d'annee en annee
- [~] H4 : Les EIDs avec PSD non-nul recoupent majoritairement les 927 EIDs de costs → partiellement resolue par l'univers hybride (F10)
- [ ] H5 : La dispersion inter-scenario sur PSD!=0 est predictive de la profitabilite

---

## Questions en suspens pour MAG Energy Solutions

1. **Pourquoi seulement 927 EIDs dans costs vs 7000+ dans sim_daily ?** Les sims couvrent-elles tout le reseau alors que seuls certains elements sont tradables ?
2. **Les 3 scenarios sont-ils ordonnes ?** (ex: S1=base, S2=haut, S3=bas) ou purement aleatoires ?
3. **PSD=0 est-il semantique ?** (pas de congestion) ou juste un artefact de la sparsification ?
4. **Les EIDs de costs sont-ils fixes ou changent-ils chaque annee ?**
5. **Y a-t-il un lien entre ACTIVATIONLEVEL et la probabilite d'un PSD non-nul ?**
