"""Renewal specialist: learn churn from every past renewal decision (history only).

Rows: (account, season t) with a subscription in t-1, for t = 1996..2013, development + source
accounts; features as of t (records before t, donation amounts unknown); target = renewed in t.
Scored on the 2014 development rows of 2013-14 subscribers: AUC renew vs churn, compared with the
2014-trained models on the same accounts.  No 2014-15 label is used to fit the specialist.

    python -I src/churn_expert.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load  # noqa: E402
from src.panel_cv import build  # noqa: E402


def main():
    c = load()
    history, present, dev = build(c)
    y = c.labels.reindex(dev).to_numpy()
    ren = history[history.sub_2013 == 1]  # 'sub_2013' = subscribed in the season before the target
    X, Y = ren.drop(columns='y'), ren.y.to_numpy()
    print('renewal decisions', len(Y), 'renewal rate', round(Y.mean(), 3), flush=True)
    by_year = ren.groupby('target_year').y.agg(['size', 'mean'])
    print('renewal rate by target year (last 8):', {int(k): round(v, 2) for k, v in by_year['mean'].tail(8).items()})
    mask = present.sub_2013.to_numpy() == 1
    P = present[mask].copy()
    snap = ['don_2013', 'donated_2013', 'don_lifetime', 'don_count', 'don_mean']
    for name, m in (('hgb', HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)),
                    ('rf', RandomForestClassifier(n_estimators=1000, min_samples_leaf=8, max_features=4, n_jobs=8, random_state=0))):
        Xf = X.fillna(-1) if name == 'rf' else X
        m.fit(Xf, Y)
        Pf = P.copy(); Pf[snap] = np.nan
        p = m.predict_proba(Pf.fillna(-1) if name == 'rf' else Pf)[:, 1]
        print(f'specialist {name}: AUC among 2014 2013-subscribers = {roc_auc_score(y[mask], p):.3f} (n={mask.sum()}, churn={int((y[mask]==0).sum())})', flush=True)


if __name__ == '__main__':
    main()


def specialist_scores(c):
    """Fitted once on history renewal decisions; returns (dev ids, mask of 2013 subscribers, score)."""
    history, present, dev = build(c)
    ren = history[history.sub_2013 == 1]
    m = RandomForestClassifier(n_estimators=1000, min_samples_leaf=8, max_features=4, n_jobs=8, random_state=0)
    m.fit(ren.drop(columns='y').fillna(-1), ren.y.to_numpy())
    P = present.copy(); P[['don_2013', 'donated_2013', 'don_lifetime', 'don_count', 'don_mean']] = np.nan
    return dev, present.sub_2013.to_numpy() == 1, m.predict_proba(P.fillna(-1))[:, 1]


def rerank_block(score, block, expert, w=0.5):
    """Keep the block's score values; reorder them inside the block by a blend of ranks."""
    out = score.copy()
    idx = np.where(block)[0]
    blend = (1 - w)*pd.Series(score[idx]).rank(pct=True).to_numpy() + w*pd.Series(expert[idx]).rank(pct=True).to_numpy()
    out[idx[np.argsort(blend)]] = np.sort(score[idx])
    return out
