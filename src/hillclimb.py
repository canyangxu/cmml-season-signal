"""Caruana-style greedy ensemble selection (with replacement) on per-fold rank OOF.

Weights are searched on the 25 selection folds only (mean fold AUROC), then the frozen weights
are scored once on the 25 confirmation folds against the v2 recipe.

    python -I src/hillclimb.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SELECT = ('fixed42', 'fixed2026', 'rep101', 'rep102', 'rep103')
CONFIRM = ('rep201', 'rep202', 'rep203', 'rep204', 'rep205')
IT = ROOT/'outputs/iterate'


def members(schemes, confirm):
    pre = 'confirm_' if confirm else ''
    tab = {'cat_d3': f'{pre}cat_d3slow_allview', 'cat_d4': f'{pre}cat_d4_allview', 'hgbcat': f'{pre}hgbcat_allview',
           'lrall': f'{pre}lrall_c003_allview'}
    files = {k: pd.read_csv(IT/f'{v}_oof.csv') for k, v in tab.items()}
    rft = None if confirm else pd.read_csv(IT/'rfg_4_8_1500_base_oof.csv')
    out = []
    for s in schemes:
        o = pd.read_csv(ROOT/f'outputs/cv/oof_{s}.csv'); y, f = o.label.to_numpy(), o.fold.to_numpy()
        rk = lambda v: pd.Series(np.asarray(v, float)).groupby(f).rank(pct=True).to_numpy()
        M = {'rf_t': rk(o.rf_t if confirm else rft[s]), 'cnn': rk(o.cnn_pre), 'lr': rk(o.lr), 'hgb': rk(o.hgb)}
        M.update({k: rk(v[s]) for k, v in files.items()})
        out.append((y, f, M))
    return out


def score(data, w):
    vals = []
    for y, f, M in data:
        v = sum(w[k]*M[k] for k in w if w[k]) / sum(w.values())
        vals += [roc_auc_score(y[f == q], v[f == q]) for q in range(1, 6)]
    return np.array(vals)


def v2(M):
    E = (3*M['rf_t'] + M['cnn'] + M['lrall'] + M['hgb'])/6
    return (3*E + M['cat_d3'] + M['hgbcat'])/5


def main():
    sel, con = members(SELECT, False), members(CONFIRM, True)
    names = list(sel[0][2])
    w = {k: 0 for k in names}; w['rf_t'] = 1
    best = score(sel, w).mean()
    for step in range(30):
        cand = []
        for k in names:
            t = dict(w); t[k] += 1
            cand.append((score(sel, t).mean(), k))
        s, k = max(cand)
        if s <= best + 1e-6:
            break
        w[k] += 1; best = s
    print('selected weights', {k: v for k, v in w.items() if v}, 'selection ALL', round(best, 5))
    hc = score(con, w)
    base = np.array([roc_auc_score(y[f == q], v2(M)[f == q]) for y, f, M in con for q in range(1, 6)])
    print(f'CONFIRM hill-climb {hc.mean():.5f}  v2 recipe {base.mean():.5f}  folds hill>v2 {(hc > base).sum()}/25')


if __name__ == '__main__':
    main()
