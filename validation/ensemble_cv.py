"""Fold-level AUROC of every member and of the final ensemble on the 25 selection folds and the
25 confirmation folds, from the out-of-fold predictions of validation/cv.py (all 10 schemes) and
validation/members_cv.py (each member, with and without --confirm).  Members are ranked within each validation fold, then averaged with the
final model's fixed weights.  Writes results/cv_folds.json.

    python -I validation/ensemble_cv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SELECT = ('fixed42', 'fixed2026', 'rep101', 'rep102', 'rep103')
CONFIRM = ('rep201', 'rep202', 'rep203', 'rep204', 'rep205')
ID = 'account.id'


def scheme_scores(scheme):
    a = pd.read_csv(ROOT/f'outputs/cv/oof_{scheme}.csv', dtype={ID: str})  # rows in development-account order
    tag = '_confirm' if scheme in CONFIRM else ''
    b = pd.DataFrame({k: pd.read_csv(ROOT/f'outputs/members_cv/{k}{tag}_oof.csv')[scheme] for k in ('rf', 'catboost', 'hgb_cat', 'logistic')})
    if len(a) != len(b):
        sys.exit('OOF files have different lengths')
    y, f = a.label.to_numpy(), a.fold.to_numpy()
    rk = lambda v: pd.Series(np.asarray(v, float)).groupby(f).rank(pct=True).to_numpy()
    m = {'RF (3x weight in E)': rk(b.rf), 'CNN pretrained': rk(a.cnn_pre), 'CNN from scratch (ablation)': rk(a.cnn_scratch),
         'LR allview': rk(b.logistic), 'HGB base': rk(a.hgb), 'CatBoost allview': rk(b.catboost), 'HGB_cat allview': rk(b.hgb_cat)}
    m['base-view ensemble (RF, CNN, LR base, HGB)'] = (3*m['RF (3x weight in E)'] + m['CNN pretrained'] + rk(a.lr) + m['HGB base'])/6
    E = (3*m['RF (3x weight in E)'] + m['CNN pretrained'] + m['LR allview'] + m['HGB base'])/6
    m['FINAL = (3E + CatBoost + HGB_cat)/5'] = (3*E + m['CatBoost allview'] + m['HGB_cat allview'])/5
    return {k: [roc_auc_score(y[f == q], v[f == q]) for q in range(1, 6)] for k, v in m.items()}


def main():
    out = {}
    for name, schemes in (('selection', SELECT), ('confirmation', CONFIRM)):
        per = [scheme_scores(s) for s in schemes]
        out[name] = {k: sum((p[k] for p in per), []) for k in per[0]}
    final = 'FINAL = (3E + CatBoost + HGB_cat)/5'
    print(f"{'model':44s} {'selection':>10s} {'confirmation':>13s}")
    for k in out['selection']:
        print(f"{k:44s} {np.mean(out['selection'][k]):10.5f} {np.mean(out['confirmation'][k]):13.5f}")
    base = 'base-view ensemble (RF, CNN, LR base, HGB)'
    wins = sum(a > b for a, b in zip(out['confirmation'][final], out['confirmation'][base]))
    print(f'final above the base-view ensemble on {wins}/25 confirmation folds')
    (ROOT/'results/cv_folds.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
