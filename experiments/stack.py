"""Nested stacking: a logistic meta-model learns, inside each outer training fold, how to
weight member ranks by buyer group (subscription history / ticket-only / none).

Inner 5-fold OOF of rf_t, lr, hgb on the outer-training rows feeds the meta-model;
members refitted on all outer-training rows score the outer-validation rows.
Compared against the equal-rank blend of the same members on the same folds.

    python -I experiments/stack.py --scheme fixed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cv import folds  # noqa: E402
from experiments.iterate import prepare  # noqa: E402
from src.features import base  # noqa: E402
from src.signal import CH  # noqa: E402

OUT = ROOT/'outputs/stack'


def members(seed):
    return {'rf_t': RandomForestClassifier(n_estimators=1500, min_samples_leaf=8, max_features=4, n_jobs=8, random_state=seed),
            'lr': make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
            'hgb': HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)}


def ranks(models, X):
    return np.column_stack([pd.Series(m.predict_proba(X)[:, 1]).rank(pct=True).to_numpy() for m in models.values()])


def meta_x(R, g):
    onehot = np.column_stack([(g == k).astype(float) for k in ('sub', 'tk', 'none')])
    return np.column_stack([R] + [R*onehot[:, [j]] for j in range(3)] + [onehot])


def run(scheme):
    OUT.mkdir(parents=True, exist_ok=True)
    d = prepare(); y = d['y']; tab = base(d); s = d['sig']
    g = np.where(s[:, CH['sub_present']].max(1) > 0, 'sub', np.where(s[:, CH['tk_present']].max(1) > 0, 'tk', 'none'))
    fold = folds(scheme, d['ids'], y)
    stacked, equal = np.zeros(len(y)), np.zeros(len(y))
    for f in range(1, 6):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = np.zeros((len(tr), 3))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=f).split(tr, y[tr]):
            ms = {k: m.fit(tab.iloc[tr[a]], y[tr[a]]) for k, m in members(f).items()}
            inner[b] = ranks(ms, tab.iloc[tr[b]])
        meta = LogisticRegression(C=1.0, max_iter=5000).fit(meta_x(inner, g[tr]), y[tr])
        ms = {k: m.fit(tab.iloc[tr], y[tr]) for k, m in members(f).items()}
        R = ranks(ms, tab.iloc[va])
        stacked[va] = meta.predict_proba(meta_x(R, g[va]))[:, 1]
        equal[va] = (3*R[:, 0] + R[:, 1] + R[:, 2])/5
        print(scheme, f, 'stacked', round(roc_auc_score(y[va], stacked[va]), 4), 'equal', round(roc_auc_score(y[va], equal[va]), 4), flush=True)
    res = {k: [roc_auc_score(y[fold == q], v[fold == q]) for q in range(1, 6)] for k, v in (('stacked', stacked), ('equal', equal))}
    print(scheme, 'MEAN', {k: round(float(np.mean(v)), 5) for k, v in res.items()}, flush=True)
    (OUT/f'metrics_{scheme}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', required=True)
    run(p.parse_args().scheme)
