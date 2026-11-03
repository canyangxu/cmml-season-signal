# Results (fold-level AUROC, the competition metric)

Selection folds = fixed42, fixed2026, rep101-103 (25 folds). Confirmation folds = rep201-205
(25 fresh folds, run once for the chosen recipe).

| model | selection 25 folds | confirmation 25 folds |
|---|---:|---:|
| ens_final = (3 x rf_t + cnn_pre + lr + hgb) / 6, equal-rank | 0.98739 | **0.98754** |
| rf_t (1500 trees, leaf 8, 4 features/split) | 0.98699 | 0.98685 |
| rf (800 trees, leaf 3, sqrt) | 0.98659 | 0.98642 |
| hgb | 0.98347 | 0.98346 |
| cnn_pre (history-pretrained, 3 seeds) | 0.98191 | 0.98291 |
| lr | 0.98099 | 0.98089 |
| cnn_scratch (ablation) | 0.97969 | 0.97827 |

Previous project's best on the two fixed fold sets: 0.98848 / 0.98630; ens_final there:
0.98835 / 0.98619; 0.5 ens_final + 0.5 previous best: 0.98916 / 0.98712.

Tried on the selection folds and rejected (all below rf_base 0.98659 unless noted):
rich signal features 0.98559; venue distances 0.98590; ticket details 0.98528; latest-season
categories with CatBoost 0.98337; CatBoost base 0.98439; ExtraTrees 0.98551; pooled-season
training RF 0.98658 / 0.98478 (fixed folds); history-transfer score 0.98623; monotone HGB 0.98508.

Later attempts (selection folds):
- cnn_flat (CNN with the 44-column flat summary in the dense branch, 5 seeds): 0.98235 vs cnn_pre 0.98191;
  ensemble with it 0.98743 vs 0.98739: no gain.
- nested group-aware stacking (logistic meta-model on member ranks x buyer group), fixed42:
  0.98575 vs equal-rank 0.98805: overfits, rejected.

Final files (outputs/final, frozen before inference; raw-table feature parity check passed):
- submissions/season_signal_v1.csv = ens_final, sha256 6146980986…; test rank corr with the previous
  best submission 0.799 (development OOF 0.808 / 0.821).
- submissions/season_signal_hybrid_v1.csv = 0.5 rank(ens_final) + 0.5 rank(previous best
  half_snapshot_best_v1), sha256 ee829ca5e0…; fixed-fold CV 0.98916 / 0.98712.
