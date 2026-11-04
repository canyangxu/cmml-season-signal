# Results

Fold-level AUROC from `validation/` (protocol in `METHODS.md`, section 4); per-fold values in
`results/cv_folds.json`.

## Cross-validation

| model | selection (25 folds) | confirmation (25 folds) |
|---|---:|---:|
| RF on `base` | 0.98699 | 0.98685 |
| CNN, pretrained | 0.98191 | 0.98291 |
| CNN from scratch (ablation, not used) | 0.97969 | 0.97827 |
| LR on `allview` | 0.98365 | 0.98476 |
| HGB on `base` | 0.98347 | 0.98346 |
| CatBoost on `allview` | 0.98620 | 0.98538 |
| HGB_cat on `allview` | 0.98395 | 0.98495 |
| E-style ensemble with LR on `base`, no detail-view members | 0.98739 | 0.98754 |
| **final ensemble** | **0.98847** | **0.98861** |

The final ensemble is above the simpler ensemble on 20 of the 25 confirmation folds. Pretraining
adds about 0.002 to the CNN (0.98191 vs 0.97969).
The member CV runs use 1 seed for RF and CatBoost and 3 for the CNN; the final fit uses 3, 3 and 5.

## Alternatives considered

These were tried on the same selection folds and not adopted. The final recipe scored 0.98847 in
those runs. The exploration code is in the git history (commit `bee1888`).

| alternative | selection AUROC |
|---|---:|
| RF with default settings (800 trees, leaf 3) / ExtraTrees | 0.98659 / 0.98551 |
| CatBoost on `base` / XGBoost / LightGBM / equal-weight XGB+LGBM+CatBoost | 0.98439 / 0.98117 / 0.98456 / 0.98490 |
| TabPFN v2 / EBM / spline GAM | 0.98495 / 0.98542 / 0.98668 (GAM public 0.95809) |
| three ECG-style CNN subnetworks fused / CNN with flat summary input | 0.98083 / 0.98235 |
| nested stacking (fixed42 folds only) / greedy weight search (confirmation) | 0.98575 vs 0.98805 / 0.98843 vs 0.98861 |
| label smoothing (LightGBM, 0.1) / label noise on the members | 0.97952 / no gain (best +0.00006) |
| account-creation frontier flags in LR and HGB_cat | 0.98863 (+0.0002, too small to submit) |

The public half has about 71 buyers, and one buyer moving from bottom to top changes AUROC by about
0.014. The public score is therefore much noisier than the 50-fold CV.

## Leaderboard

| file | what | public AUROC |
|---|---|---:|
| `season_signal_v2_retrained.csv` | final: this repository run end to end (seeded) | **0.97303** |
| `season_signal_v2.csv` | same recipe, first run (CNN head initialisation unseeded) | 0.97283 |
