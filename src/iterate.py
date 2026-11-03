"""Fast iteration harness for tabular models on the season-signal features.

Selection schemes (fixed42, fixed2026, rep101-103 = 25 folds) are used to choose;
confirmation schemes (rep201-205) are reserved for one final check of the chosen
recipe only, so that selection over many variants cannot inflate the reported score.

    python -I src/iterate.py --variant rf_base [--confirm]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cv import folds  # noqa: E402
from src.data import load, ID  # noqa: E402
from src.features import FEATURE_SETS  # noqa: E402
from src.signal import signals, static  # noqa: E402

SELECT = ('fixed42', 'fixed2026', 'rep101', 'rep102', 'rep103')
CONFIRM = ('rep201', 'rep202', 'rep203', 'rep204', 'rep205')
OUT = ROOT/'outputs/iterate'
INCREASING = ('sub_seasons', 'sub_last3', 'sub_2013', 'sub_2012', 'sub_streak', 'tk_2013', 'tk_last3', 'don_2013', 'donated_2013',
              'don_lifetime', 'don_count', 'first_donated_2014', 'sub_seats_2013', 'sub_full_2013')
DECREASING = ('sub_recency', 'tk_recency')
MONO = None  # set per feature table in evaluate()


def model(kind, seed):
    if kind == 'rf_base':
        return RandomForestClassifier(n_estimators=800, min_samples_leaf=3, max_features='sqrt', n_jobs=8, random_state=seed)
    if kind == 'et_base':
        return ExtraTreesClassifier(n_estimators=1500, min_samples_leaf=2, max_features='sqrt', n_jobs=8, random_state=seed)
    if kind == 'rf_wide':
        return RandomForestClassifier(n_estimators=1500, min_samples_leaf=2, max_features=0.3, n_jobs=8, random_state=seed)
    if kind == 'rf_balanced':
        return RandomForestClassifier(n_estimators=1500, min_samples_leaf=3, max_features='sqrt', class_weight='balanced_subsample',
                                      n_jobs=8, random_state=seed)
    if kind == 'et_wide':
        return ExtraTreesClassifier(n_estimators=2000, min_samples_leaf=1, max_features=0.3, n_jobs=8, random_state=seed)
    if kind.startswith('rfg_'):  # rfg_<max_features>_<min_samples_leaf>_<trees>
        _, mf, leaf, trees = kind.split('_')
        mf = mf if mf in ('sqrt', 'log2') else (float(mf) if '.' in mf else int(mf))
        return RandomForestClassifier(n_estimators=int(trees), min_samples_leaf=int(leaf), max_features=mf, n_jobs=8, random_state=seed)
    if kind.startswith('hgbmono'):
        from sklearn.ensemble import HistGradientBoostingClassifier
        lr, leaves, it = {'hgbmono': (0.05, 15, 300), 'hgbmono_slow': (0.02, 15, 900), 'hgbmono_small': (0.03, 7, 700)}[kind]
        return HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, l2_regularization=1.0,
                                              min_samples_leaf=20, random_state=seed, monotonic_cst=MONO)
    if kind == 'cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.04, l2_leaf_reg=5.0, random_seed=seed,
                                  thread_count=8, verbose=False, allow_writing_files=False)
    raise ValueError(kind)


def evaluate(kind, feature_set, schemes, data):
    global MONO
    ids, y, tab = data['ids'], data['y'], FEATURE_SETS[feature_set](data)
    MONO = [1 if c in INCREASING else -1 if c in DECREASING else 0 for c in tab.columns]
    res, oofs = {}, {}
    for scheme in schemes:
        fold = folds(scheme, ids, y)
        oof = np.zeros(len(y))
        for f in range(1, 6):
            tr, va = fold != f, fold == f
            m = model(kind, f)
            if kind == 'cat':
                cats = [c for c in tab.columns if tab[c].dtype == object]
                m.fit(tab[tr], y[tr], cat_features=cats)
            else:
                m.fit(tab[tr], y[tr])
            oof[va] = m.predict_proba(tab[va])[:, 1]
        res[scheme] = [roc_auc_score(y[fold == f], oof[fold == f]) for f in range(1, 6)]
        oofs[scheme] = (fold, oof)
    return tab.shape[1], res, oofs


def prepare():
    c = load()
    ids = c.dev_accounts[ID].tolist()
    permitted = ids + c.source_accounts[ID].tolist()
    return {'corpus': c, 'ids': ids, 'y': c.labels.reindex(ids).to_numpy(), 'permitted': permitted,
            'sig': signals(ids, c.dev_subs, c.dev_tickets), 'stat': static(ids, c.dev_accounts, permitted, full=True)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, help='model kind, optionally kind:feature_set')
    p.add_argument('--confirm', action='store_true')
    a = p.parse_args()
    kind, _, fs = a.variant.partition(':')
    fs = fs or 'base'
    n, res, oofs = evaluate(kind, fs, CONFIRM if a.confirm else SELECT, prepare())
    allf = np.concatenate(list(res.values()))
    tag = ('CONFIRM ' if a.confirm else '') + f'{kind}:{fs}'
    print(f'{tag:34s} features {n:3d}  ' + ' '.join(f'{s}={np.mean(v):.5f}' for s, v in res.items())
          + f'  ALL={allf.mean():.5f} sd={allf.std():.4f}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{'confirm_' if a.confirm else ''}{kind}_{fs}"
    (OUT/f'{name}.json').write_text(json.dumps({'variant': a.variant, 'features': n, 'folds': res}, indent=1))
    pd.DataFrame({s: o for s, (_, o) in oofs.items()}).to_csv(OUT/f'{name}_oof.csv', index=False)
