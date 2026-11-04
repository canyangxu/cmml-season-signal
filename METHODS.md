# Method

Kaggle competition *Classical Music Meets Classical ML Fall 2026*: predict which patrons buy a
2014-15 subscription. The metric is AUROC.

The final Kaggle file is `season_signal_v2_retrained.csv`, produced by running `src/` end to end.
It scored **0.97303** on the public leaderboard, our best of 15 submissions. The same recipe was first
submitted as `season_signal_v2.csv` (public 0.97283); that run left the CNN head initialisation
unseeded, which is fixed here.

## 1. Data and cleaning

**Files used:**
- `train.csv` (account ID, label) and `test.csv` (account ID).
- `account.csv`: addresses and donation history.
- `subscriptions.csv` and `tickets_all.csv`: purchase history by season.
- `zipcodes.csv`: latitude, longitude and state per ZIP code.

The concert files are not used.

**Reading (`src/isolate.py`, `src/data.py`):**
- Every column is read as text.
- Only empty cells count as missing. Strings such as "NA" stay literal values.
- Files are decoded as UTF-8, falling back to cp1252, because `account.csv` has cp1252 city names.
- Integrity checks: all 19,833 account IDs are 15-character and unique. Every train and test ID exists
  in `account.csv`. Train and test do not overlap.

**Test isolation (`src/isolate.py`):** this script is the only development code that opens the raw
folder. It reads `test.csv` for its ID column only, removes those 2,975 accounts from every table, and
writes `data/isolated/`:

| pool | accounts | role |
|---|---:|---|
| development | 6,941 (332 buyers, 4.8%) | training and cross-validation |
| source | 9,917 | accounts in neither `train.csv` nor `test.csv`, no labels; used only for CNN pretraining and as the reference list for account age |

`src/data.py` reads only `data/isolated/` and checks every file against the SHA-256 hashes in the
manifest. Test features are built only in `src/predict.py`, after the model is frozen. Nothing from
the test accounts is used to choose features, models, weights or the stopping point.

**Cleaning and encoding rules (`src/signal.py`, `src/features.py`):**
- **Seasons:** "2013-2014" becomes 2013. Only seasons 1993-2013 are kept (21 steps).
- **Seats, tier and price level:** parsed as numbers; unparseable values become 0. Seat and record counts are
  `log1p`-transformed.
- **Ticket data availability:** ticket records exist only from 2010-11. A `tk_available` channel marks those
  seasons, so "no ticket data yet" is not read as "bought no tickets".
- **Donations:** amounts and counts are numeric (missing becomes 0) and `log1p`-transformed. The first-gift date is
  parsed, and an invalid date counts as unknown. Derived values: gift age in years to the end of 2014,
  "first gift in 2014" and "first-gift date known" flags, and the mean gift.
- **Relationship:** the free-text field becomes three flags (education, vendor, other).
- **Address:** billing and shipping ZIP presence, and whether they are equal. ZIP5 is the billing ZIP,
  else the shipping ZIP. It is joined to `zipcodes.csv` for distances to five venues and an
  in-California flag; distances are missing when the ZIP is unknown.
- **Account age:** the ID is decoded as a base-62 number and ranked among the development and source
  accounts. This is a label-free creation-order proxy, and test accounts never enter the reference list.
- **Categories:** latest- and previous-season package, section, price level and venue (missing becomes `none`),
  the ID batch prefix (first 11 characters), ZIP3 and state.

**Model-specific preprocessing (always fitted on training rows only):**
- CNN: Min-Max scaling.
- Logistic regression: median imputation plus missing-value indicators, then standardisation, and
  one-hot categories with at least 20 occurrences.
- HGB with categories: levels seen fewer than 10 times are pooled into `rare`.
- CatBoost: native categories.
- Tree models take missing values natively.

## 2. Features

Each account's history is treated like a multi-lead ECG signal: 21 seasons x 17 channels.

- Subscription channels: presence, seats, tier, full package, record count, price level, multiple
  subscriptions.
- Six venue series.
- Ticket channels: presence, seats, records, data availability.

| view | columns | content |
|---|---:|---|
| season signal | 17 x 21 | the channels above (CNN input) |
| static | 15 | donations, relationship flags, address presence, account-age rank |
| `base` | 44 | flat signal summary plus static. Summary: seasons, last 3 seasons, 2013 and 2012 presence, recency, current streak, 2013 and maximum value per channel, 2013 venue |
| `allview` | 91 | `base` plus per-season presence 2009-2013, year-on-year changes, tenure, gaps, longest run, venue switching, venue distances and ticket details; as categories: latest-season details, ID prefix, ZIP3, state |

## 3. Model

```
E     = (3 x RF[base] + CNN + LR[allview] + HGB[base]) / 6
final = (3 x E + CatBoost[allview] + HGB_cat[allview]) / 5
```

Each member's scores are converted to ranks within the scored set before averaging, because AUROC
depends only on the order. The weights were declared before scoring and never fitted.

| member | settings (`src/members.py`, `src/model.py`) |
|---|---|
| RF, 3 seeds | 1500 trees, minimum leaf 8, 4 candidate features per split |
| CNN, 5 seeds | 3 parallel Conv1d branches (kernels 2/3/5), each with max, mean and last-season pooling, plus a dense static branch. Trunk pretrained on the history panel, then fine-tuned: buyers oversampled 1:3, Adam, exponential learning-rate decay, early stopping on an inner 20% split of the training rows |
| LR | L2 logistic regression, C = 0.03 |
| HGB | learning rate 0.05, 300 iterations, 15 leaves, L2 1.0 |
| CatBoost, 3 seeds | depth 3, 2000 iterations, learning rate 0.03, L2 5 |
| HGB_cat | as HGB, with native categorical splits |

**CNN pretraining (`src/pretrain.py`):** for every development and source account and every season t
with earlier activity, the input is the signal before t and the target is "subscribed in t". This
gives 79,771 rows. It uses no 2014-15 label and no donation snapshot, and validation AUROC is 0.9706
on 10% held-out accounts.

## 4. Validation

- **Metric:** fold-level AUROC, the mean over folds of each fold's AUROC.
- **Selection folds (25):** two fixed 5-fold assignments (seeds 42 and 2026) plus repeated stratified 5-fold with
  seeds 101-103. All design choices were made on these folds.
- **Confirmation folds (25):** seeds 201-205, used once for the chosen recipe.

Results (member and ensemble AUROC, leaderboard, alternatives considered) are in
`results/SUMMARY.md`; per-fold values are in `results/cv_folds.json`.

## 5. How to run

Put the competition files in `data/raw/for_students/`, then run `pip install -r requirements.txt`
under Python 3.12 (CPU is enough).

```bash
python -I src/isolate.py       # 1. raw -> data/isolated (test IDs used only to exclude them)
python -I src/pretrain.py      # 2. CNN trunk pretraining
python -I src/train.py         # 3. fit on all development accounts, freeze to outputs/model
python -I src/predict.py       # 4. verify freeze, rebuild features from raw tables, score test

python -I validation/cv.py --scheme fixed42           # CNN (pretrained / scratch), HGB, LR on base; run for all 10 schemes
python -I validation/members_cv.py --member rf        # RF, catboost, hgb_cat, logistic; each with and without --confirm
python -I validation/ensemble_cv.py                   # the table in results/SUMMARY.md -> results/cv_folds.json
```

`predict.py` stops if any source or model file differs from its freeze record. It also checks that
features rebuilt from the raw tables reproduce the frozen development scores. Every random step is
seeded: two retrains produce byte-identical submissions (SHA-256 `1a14bcec...`).

## 6. Files

| path | role |
|---|---|
| `src/isolate.py` | step 1: raw files to `data/isolated`, test accounts removed |
| `src/data.py` | reader for `data/isolated` with hash checks |
| `src/signal.py` | season signal, static features, flat summary |
| `src/features.py` | `base` and `allview` views |
| `src/model.py` | 1D-CNN and its training loop |
| `src/pretrain.py` | step 2: CNN trunk pretraining |
| `src/members.py` | model definitions |
| `src/ensemble.py` | inputs, fitting and scoring of the final ensemble |
| `src/train.py` | step 3: fit and freeze |
| `src/predict.py` | step 4: verified inference, writes the submission |
| `validation/` | cross-validation of every member and of the ensemble |
| `results/SUMMARY.md` | cross-validation results, leaderboard, alternatives considered |
| `results/cv_folds.json` | per-fold AUROC behind the results table |

`data/`, `outputs/` and `submissions/` are not committed, because the competition data may not be
redistributed.
