# Classical Music Meets Classical ML: season-signal ensemble

Kaggle class competition (AIPI 520, Fall 2026): predict which orchestra patrons buy a 2014-15
subscription. Metric: AUROC.

**Final submission:** `season_signal_v2.csv`, public leaderboard **0.97283** (our best submission).
Cross-validated AUROC **0.98847** on 25 selection folds and **0.98861** on 25 fresh confirmation folds.

## 1. Approach in one paragraph

Each patron's history is treated like a multi-lead ECG. There are 21 seasons (1993-94 to 2013-14),
each with 17 channels: subscription presence, seats, tier, full package, records, price, multiple
subscriptions, six venue series, and ticket presence, seats, records and data availability. A 1D-CNN
is pretrained on the whole history (predict next-season subscription from earlier seasons) and then
fine-tuned on the 2014-15 labels. Tree and linear models use flat summaries of the same signal plus
account features: donations, relationship, address presence and account age from the identifier
order. The final prediction is a fixed rank average of six model families.

## 2. Data handling: test set sealed

- `src/isolate.py` is the only development-stage code that opens the raw files. It reads `test.csv`
  for the ID column only, removes those accounts, and writes `data/isolated/`:
  - 6,941 labelled development accounts (332 buyers, 4.8%);
  - 9,917 unlabelled "source" accounts that are in neither `train.csv` nor `test.csv`;
  - the public catalogues (zip codes, concerts).
- `src/data.py` reads only `data/isolated/` and verifies the file hashes in its manifest.
- Every learned transform (scaling, category levels, early stopping) is fitted on training rows only.
  The CNN stops early on an inner split of the training rows.
- Test features are computed only in the final inference step, after the model is frozen. Nothing
  from the test accounts is used to choose features, members or weights.

## 3. Features

| view | columns | content |
|---|---:|---|
| season signal | 17 x 21 | per-season channels above, log-scaled counts (CNN input) |
| static | 15 | donations (2013, lifetime, count, mean, first-gift age / year), relationship flags, billing / shipping presence, account-age rank |
| `base` | 44 | flat signal summary (seasons, last 3, 2013 / 2012, recency, streak, 2013 and max values per channel, 2013 venue) + static |
| `allview` | 91 | `base` + per-season presence 2009-2013, year-on-year changes, tenure, gaps, longest run, venue switching, distance to venues, ticket details, and as categories: latest-season package / section / price / venue, identifier batch prefix, ZIP3, state |

Account age is the identifier's rank among the development and source accounts. This list is fixed
and label-free, and the test accounts are never part of it.

## 4. Model

```
E     = (3 x RF[base] + CNN + LR[allview] + HGB[base]) / 6
final = (3 x E + CatBoost[allview] + HGB_cat[allview]) / 5        (ranks within the scored set)
```

| member | settings | CV selection / confirmation |
|---|---|---:|
| RF (smooth random forest), 3 seeds | 1500 trees, min leaf 8, 4 features per split | 0.98699 / 0.98685 |
| CNN, 5 fine-tuned copies | 3 parallel Conv1d branches (kernels 2/3/5), max + mean + last-season pooling, static branch; history-pretrained trunk; buyer oversampling 1:3, Min-Max scaling, exponential LR decay, inner early stopping | 0.98191 / 0.98291 |
| LR on `allview` | L2, C = 0.03; median imputation + indicators; one-hot categories | 0.98365 / 0.98476 |
| HGB on `base` | lr 0.05, 300 iterations, 15 leaves | 0.98347 / 0.98346 |
| CatBoost on `allview`, 3 seeds | depth 3, 2000 iterations, lr 0.03, native categories | 0.98620 / 0.98538 |
| HGB_cat on `allview` | as HGB, native categories (levels seen fewer than 10 times pooled) | 0.98395 / 0.98495 |
| **final ensemble** | weights declared before scoring, never fitted | **0.98847 / 0.98861** |

The member CV runs use 1 seed for RF and CatBoost and 3 for the CNN; the final fit uses the seed
counts in the table.

## 5. Validation protocol

- The metric is fold-level AUROC, the competition metric: the mean over folds of each fold's AUROC.
- **Selection folds (25):** the earlier project's two fixed 5-fold assignments (seeds 42 and 2026)
  plus repeated stratified 5-fold with seeds 101-103. Every model, feature and ensemble choice was
  made here.
- **Confirmation folds (25):** seeds 201-205, run once for the chosen recipe only.
- The final ensemble beats the simpler `RF + CNN + LR + HGB` ensemble on 20 of the 25 confirmation
  folds.

## 6. Results and what did not help

| | public LB | CV (selection) |
|---|---:|---:|
| **season_signal_v2 (final)** | **0.97283** | 0.98847 |
| earlier-project best (four-family blend) | 0.97279 | - |
| spline GAM (smooth additive model) | 0.95809 | 0.98668 |

The public half has about 71 buyers. One buyer moving from the bottom to the top changes the AUROC
by about 0.014, so the public score is much noisier than the 50-fold CV.

Tried and rejected, all on the same folds (details and numbers in `results/SUMMARY.md`):
- ECG-style three-network CNN fusion;
- a CNN with the flat summary in its dense branch;
- pooled-season training;
- TabPFN v2, EBM, XGBoost and LightGBM, alone and as an equal-weight trio, including rolling-window features;
- nested stacking and greedy weight search (both overfit);
- a renewal specialist;
- label smoothing and label noise (output flipping and smearing with clean-label early stopping);
- stronger regularization;
- creation-frontier, donation-2014 and count views.

## 7. Reproduce

Put the competition files in `data/raw/for_students/`. Then, with Python 3.12 and
`pip install -r requirements.txt`, run these steps on CPU (about 20 minutes in total):

```bash
python -I src/isolate.py     # raw -> data/isolated (test IDs used only to exclude them)
python -I src/pretrain.py    # history-panel pretraining of the CNN trunk (79,771 rows)
python -I src/train.py       # fit on all development accounts, freeze to outputs/model
python -I src/predict.py     # verify the freeze, rebuild features from raw tables, score test
```

`predict.py` refuses to run if any source file or model file differs from its freeze record. It also
checks that features rebuilt from the raw tables reproduce the frozen development scores.

Verification (2026-11-04):
- Re-running `isolate.py` and `pretrain.py` reproduces `data/isolated` and the pretrained trunk byte
  for byte.
- `predict.py`, run on the original v2 model files (`experiments/repackage_v2_freeze.py`), reproduces
  the submitted `season_signal_v2.csv` byte for byte (sha256 `20a26de7...`).
- A full retrain matches every tree and linear member of the original exactly. The CNN head
  initialisation was unseeded in the original run and is seeded now. Two retrains therefore give
  identical files, with rank correlation 0.99996 to the submitted file.

## 8. Repository layout

```
src/            final pipeline only
  isolate.py      step 1: raw files -> data/isolated (sealed test)
  data.py         loader with manifest hash checks
  signal.py       season signal, static features, flat summary
  features.py     base and allview views
  model.py        1D-CNN and its training loop
  pretrain.py     step 2: CNN trunk pretraining on the history panel
  members.py      the six model families
  ensemble.py     views, fit and scoring of the final ensemble
  train.py        step 3: fit + freeze
  predict.py      step 4: verified inference -> submission
experiments/    every explored idea, kept as evidence (index: experiments/README.md)
results/        SUMMARY.md (full experiment log) and fold_metrics/ (every fold score)
```

`data/`, `outputs/` and `submissions/` are not committed: the competition data may not be
redistributed.
