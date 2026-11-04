# Experiments

Everything explored while building the final model, kept as evidence. None of these scripts is
needed to reproduce the submission (`src/` is self-contained). Numbers are fold-level AUROC on the
25 selection folds unless noted; the full log, including failures, is `results/SUMMARY.md`, and
every fold score is in `results/fold_metrics/`.

| script | what it does | outcome |
|---|---|---|
| `cv.py` | per-scheme CV of the CNN (pretrained and from scratch), LR, RF, smooth RF, HGB and declared rank ensembles | source of the final RF, HGB and CNN members; `ens_final` 0.98739 (confirmation 0.98754) |
| `iterate.py` | fast harness for tabular members and feature views (RF / ExtraTrees grids, HGB, CatBoost, LR, XGBoost, LightGBM, EBM, TabPFN, spline GAM) | chose smooth RF, CatBoost depth 3, HGB with categories, LR on the detail view |
| `extra_features.py` | views not used by the final model: latest-season categories, history-panel score, account-creation frontier, 2014 donations, location counts, rolling windows, no-address view | none adopted (frontier view: +0.0001 only, see v3) |
| `cnn_variants.py` | CNN with the 44-column flat summary in the dense branch | 0.98235 vs 0.98191; no ensemble gain |
| `ecg_fusion.py` | three ECG-style subnetworks (dilated / GeM / multi-kernel), fused | 0.98083, below the single CNN |
| `panel_cv.py` | pooled-season training (history rows + 2014 rows) | no gain |
| `stack.py` | nested, buyer-group-aware logistic stacking | 0.98575 vs equal ranks 0.98805 (fixed42): overfits |
| `hillclimb.py` | greedy ensemble weight search on the selection folds | chose 3 RF : 1 CatBoost : 1 HGB_cat; confirmation 0.98843 vs final 0.98861: not adopted |
| `churn_expert.py` | renewal specialist trained on past renewal decisions only | not adopted |
| `noise.py`, `noise_swap.py` | label noise (output flipping / smearing) on every member, scored and early-stopped on clean labels | no variant improves the ensemble |
| `final_v1.py` | frozen stage for `ens_final` | `season_signal_v1.csv`, not submitted |
| `final_v2_original.py` | original freeze / inference script of the submitted file (commit f383c4a) | `season_signal_v2.csv`, public 0.97283; `src/` reproduces it exactly |
| `final_v3_smooth.py` | frozen stage for v3 and the spline-GAM candidates | `smooth_gam_v1.csv` public 0.95809 (rejected) |
| `summary.py` | prints the per-scheme result tables | - |

The `final_v*` scripts verify file hashes recorded at their freeze time, so after the move into this
folder they are historical records rather than runnable entry points. The other scripts run from the
repository root, for example `python -I experiments/iterate.py --variant cat_d3slow:allview`.
