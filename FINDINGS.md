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

## Hypotheses a verifier

- [ ] H1 : PSD=0 signifie "pas de congestion simulee" (pas un artefact)
- [ ] H2 : S1 est un scenario "base case" et S2-S3 sont des "stress scenarios" (ou inverse)
- [ ] H3 : Les 927 EIDs de costs sont un sous-ensemble stable d'annee en annee
- [ ] H4 : Les EIDs avec PSD non-nul recoupent majoritairement les 927 EIDs de costs
- [ ] H5 : La dispersion inter-scenario sur PSD!=0 est predictive de la profitabilite

---

## Questions en suspens pour MAG Energy Solutions

1. **Pourquoi seulement 927 EIDs dans costs vs 7000+ dans sim_daily ?** Les sims couvrent-elles tout le reseau alors que seuls certains elements sont tradables ?
2. **Les 3 scenarios sont-ils ordonnes ?** (ex: S1=base, S2=haut, S3=bas) ou purement aleatoires ?
3. **PSD=0 est-il semantique ?** (pas de congestion) ou juste un artefact de la sparsification ?
4. **Les EIDs de costs sont-ils fixes ou changent-ils chaque annee ?**
5. **Y a-t-il un lien entre ACTIVATIONLEVEL et la probabilite d'un PSD non-nul ?**
