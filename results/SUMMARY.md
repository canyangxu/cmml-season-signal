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

## Round 4: after the v2 submission (selection folds; v2 = 0.98847)

| attempt | alone | in the ensemble |
|---|---:|---:|
| **v3** = v2 with LR and HGB_cat on allview_new (+ account-creation frontier flags) | LR 0.98562, HGB_cat 0.98415 | **0.98863**; confirmation **0.98872** vs v2 0.98861 (20/25 fresh folds above v2) |
| ECG three-network fusion (Net1 dilated / Net3 GeM / Net8 multi-kernel, 3 seeds each) | 0.98083 (single CNN 0.98191) | not adopted |
| TabPFN v2 (public weights, 4 estimators) | 0.98495 | not adopted |
| EBM (interpret, 10 interactions) | 0.98542 | not adopted |
| XGBoost / LightGBM / CatBoost d4 on base | 0.98117 / 0.98456 / 0.98472 | equal-weight trio 0.98490 |
| + rolling mean/std/EWM/slope features (base_roll) | XGB 0.98041, LGBM 0.98343 | not adopted |
| LightGBM, stronger regularization (7 leaves, lambda 10, 40 per leaf) | 0.98473 | not adopted |
| label smoothing, eps 0.1 (LGBM / XGB) / eps 0.02 (LGBM) | 0.97952 / 0.97667 / 0.98326 | not adopted |
| HGB_cat, stronger regularization | 0.98423 | 0.98831 |
| account counts (allview_cnt), 2014 donations (allview_don) | LR 0.98568 / 0.98560 | no gain over allview_new |

v3 is consistently but only marginally above v2 (+0.0001 to +0.00016); on a public half with
~70 buyers that is far below the noise (fold sd ~0.005), so it was not submitted.

## Round 5: label noise on the v3 members (`src/noise.py`, `src/noise_swap.py`)

Rule (user, 2026-11-03): noise only on training targets; evaluation and early stopping on clean
labels. The outer fold is always scored against true labels; boosting iterations and CNN epochs
are chosen on inner splits of the outer-training rows that keep their true labels.
flip r = class-preserving output flipping (P(1->0) = r, P(0->1) = r n1/n0); smear s = Gaussian
output smearing with sd s * sd(y) (Breiman 2000). Noise is redrawn per 10-tree group (RF), per
bag member (boosting) or per mini-batch (CNN). Each row: member alone / v3 with that member swapped in
(v3 = 0.98863; 25 selection folds).

| member | clean control | flip 0.05 | flip 0.1 | flip 0.2 | smear 0.5 | smear 1.0 | smear 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF_t (1500 trees) | 0.98702 / 0.98860 | 0.98670 / 0.98860 | 0.98624 / 0.98851 | 0.98641 / 0.98869 | 0.98687 / 0.98867 | 0.98601 / 0.98851 | 0.98240 / 0.98775 |
| CNN (3 seeds; clean inner early stopping) | 0.98217 / 0.98865 | - | 0.98227 / 0.98866 | - | - | - | - |
| HGB_cat (5-bag, clean early stopping) | 0.98513 / 0.98836 | - | 0.98189 / 0.98797 | - | 0.97986 / 0.98783 | - | - |
| CatBoost d3 (5-bag, clean early stopping) | 0.98647 / 0.98850 | - | 0.98476 / 0.98839 | - | 0.98367 (fixed folds only, vs 0.98617) | - | - |

No variant moves v3 by more than +0.00006 (fold-level sd ~0.005); noise is rejected.

Why it cannot help here:
- RF averages leaf frequencies. Class-preserving flipping and zero-mean smearing change the target
  only by a linear map in expectation, E[y'|x] = a P(y=1|x) + b, which leaves every ranking
  unchanged; what remains is extra variance, so AUROC falls as the noise grows (as observed).
- For log-loss learners (boosting, CNN) noise in expectation is label smoothing: targets bounded away
  from 0 squeeze the logit differences among low-probability accounts, which is where the hard
  history-less buyers sit (most of the remaining AUROC loss). Smoothing LGBM by 0.1 cost 0.005.
- Clean-label early stopping itself helps single boosting members (HGB_cat 0.98415 -> 0.98513 with a
  median of 91 instead of 300 iterations; CatBoost 0.98620 -> 0.98647), but the stopped members are
  smoother and closer to RF, so the ensemble loses diversity (0.98836 / 0.98850 < 0.98863).

## Round 6: smooth additive models (public-board hypothesis)

All 14 earlier submissions (old and new repo) were tree-heavy and land at public 0.970-0.9728 whatever
their CV; a classmate's public repo reports a C = 0.03 logistic regression (CV 0.9786) at public
0.97370, above that classmate's own tree ensemble (CV 0.9780, public 0.97198). Hypothesis: smooth
additive models transfer to the test accounts better than tree interactions.

| model (allview_new) | selection 25 folds | confirmation 25 folds |
|---|---:|---:|
| LR C = 0.1 / 0.03 / 0.01 / 0.003 | 0.98430 / 0.98562 / **0.98606** / 0.98430 | - |
| spline GAM (cubic, 5 uniform knots per numeric feature), C = 0.01 / 0.003 / 0.001 / 0.0003 | 0.98457 / **0.98668** / 0.98652 / 0.98414 | 0.98679 (C = 0.003) |
| GAM + LR(0.01) + CNN (all smooth) | 0.98702 | - |
| 0.25 GAM + 0.75 v3 | 0.98875 (17/25 folds above v3) | - |

GAM vs v3 rank correlation: 0.898 on development OOF, 0.896 on test (v2 file); the test-side
agreement matches development, so inference is consistent. Frozen in outputs/final3 (v2 members
reused after artifact sha256 verification; raw-table parity checks for all four files and for v2).
Files: smooth_gam_v1.csv (613f803e…), smooth_trio_v1.csv (9866fffb…), season_signal_v3.csv
(cf5de04f…), v3_gam25_v1.csv (77982ce5…).

**Public result:** smooth_gam_v1.csv scored **0.95809** on the public board (v2 0.97283). The smooth-model
hypothesis is rejected: the GAM keeps its CV (0.98668 / 0.98679) but loses 0.029 on the public half,
the largest gap of all 15 submissions. With ~71 public buyers one buyer moved to the bottom costs
~0.014, so a few extrapolated accounts (linear spline tails, one-hot categories) are enough.

## Final selection (2026-11-04)

Grading: 70 points if mean(public, private) > 0.97. Final file: **season_signal_v2.csv** (public
0.97283, the highest of all 15 submissions; best confirmed CV among submitted files, 0.98861).
Full marks need private > 0.96717. With fold-level sd ~0.004: P(full) = 0.93 if the private half is
as hard as the public half (mean 0.973), 0.997 at 0.978, ~1.0 at the CV level. smooth_gam_v1 must not be
selected (it would need private > 0.98191). Research stopped here.

## Repository clean-up and reproduction check (2026-11-04)

- `src/` now holds only the final pipeline (isolate, pretrain, train, predict); all exploratory scripts
  moved to `experiments/` (index in `experiments/README.md`), fold scores to `results/fold_metrics/`.
- Re-running isolation and pretraining reproduces `data/isolated` and the pretrained trunk byte for byte.
- `src/predict.py` on the original v2 model files reproduces the submitted `season_signal_v2.csv` byte
  for byte (sha256 20a26de7...).
- A full retrain reproduces every tree / linear member exactly; the CNN head initialisation was unseeded
  in the original run (PyTorch's default generator is seeded per process). It is seeded now: two retrains
  give byte-identical submissions (sha256 1a14bcec...), rank correlation 0.99996 with the submitted file.
- Recorded for completeness: greedy weight search chose 3 RF : 1 CatBoost : 1 HGB_cat (selection 0.98853,
  confirmation 0.98843 vs 0.98861 for the declared weights; not adopted).

**Final:** the retrained file `season_signal_v2_retrained.csv` (this repository's `src/` run end to end,
sha256 1a14bcec...) scored **0.97303** on the public board, the best of all 15 submissions, and is the
final submission (Kaggle auto-selects the best public score when none is ticked).
