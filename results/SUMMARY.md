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

## Round 3: detail-view boosting members (fold-level AUROC)

| ensemble | selection 25 folds | confirmation 25 folds |
|---|---:|---:|
| ens_final (3 RF_t + CNN + LR + HGB) | 0.98739 | 0.98754 |
| + CatBoost depth 4 on allview (91 cols, native categories) | 0.98794 | 0.98798 |
| + HGB with native categories on allview | 0.98818 | 0.98836 |
| **final2** = (3 x E + CatBoost d3 + HGB_cat) / 5, E with LR on allview | **0.98847** | **0.98861** (20/25 folds above ens_final) |

Members (selection / confirmation): CatBoost d4 0.98561 / 0.98547; CatBoost d3 (2000 it, lr 0.03)
0.98620 / 0.98538; CatBoost d4 3-seed bag 0.98653 / -; HGB_cat 0.98395 / 0.98495; LR allview (C=0.03)
0.98365 / 0.98476. Rejected: RF on allview 0.98635 (no ensemble gain); nested stacking.

On the previous project's fixed folds the new ensemble alone beats its best (0.98917 vs 0.98848;
0.98700 vs 0.98630).

Final file: submissions/season_signal_v2.csv (outputs/final2, frozen before inference; raw-table parity
check passed), sha256 20a26de718…; test rank correlation with the previous best submission 0.884.

## Kaggle public result

season_signal_v2.csv: public **0.97283** (previous best submission four_family_v1 0.97279; 13 submissions
in total now range 0.960-0.9728). The confirmed CV gain (+0.0007 to +0.0011) shows up on the public half
only as +0.00004, consistent with the earlier finding that the public half is dominated by a few buyers
that no model trained on these data ranks well.
