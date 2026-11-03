# cmml-season-signal

A from-scratch rebuild for the Kaggle competition *Classical Music Meets Classical ML Fall 2026*
(Jon Reifschneider, AIPI 520): predict which patrons buy a 2014-15 subscription; metric AUROC.

Each account's history is treated like a multi-lead ECG: 21 seasons (1993-2013) x 17 channels
(subscription presence, seats, tier, full package, records, price, multiple subscriptions,
6 venue series, ticket presence / seats / records, ticket-data availability).

## Data isolation

- `src/isolate.py` is the only development-stage code that opens `data/raw/`. It reads
  `test.csv` for the identity column only, excludes those accounts, and writes
  `data/isolated/` (6941 labelled development accounts, 9917 unlabelled source accounts,
  static tables). Every other module reads through `src/data.py`, which refuses any
  path outside `data/isolated/` and verifies the isolation manifest hashes.
- Test inputs are opened only by the final inference stage, after the recipe is frozen.
- `data/`, `outputs/` and `submissions/` are not committed.

## Evaluation protocol

- Metric: fold-level AUROC (the competition metric), never a substitute.
- Selection folds: the previous project's two fixed 5-fold assignments (seed 42 / 2026)
  plus repeated stratified 5-fold seeds 101-103 (25 folds). All model and ensemble
  choices are made here.
- Confirmation folds: seeds 201-205 (25 more folds), used once for the chosen recipe only.
- Early stopping and scaling are fitted inside training folds only.

## Components

| file | role |
|---|---|
| `src/signal.py` | season signal, static features, flat summary |
| `src/model.py` | 3-branch Conv1d network (kernels 2/3/5; max, mean, last-season pooling) + static branch; oversampling, Min-Max, LR decay, inner early stopping |
| `src/pretrain.py` | pretraining on 79,771 (account, season) rows: next-season subscription from prior seasons, development + source accounts, no 2014-15 labels |
| `src/cv.py` | fine-tuned CNN (3 seeds), scratch CNN, LR, RF, HGB and declared ensembles |
| `src/iterate.py`, `src/features.py` | fast tabular iteration harness and feature views |
| `src/panel_cv.py` | pooled-season training (history rows + 2014 rows) |
