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
import os
DEVICE = os.environ.get('TABPFN_DEVICE', 'cpu')
SMOOTH = float(os.environ.get('LABEL_SMOOTH', '0.1'))


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
    if kind == 'hgbcat_reg':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, max_leaf_nodes=7, l2_regularization=5.0,
                                              min_samples_leaf=40, categorical_features='from_dtype', random_state=seed)
    if kind in ('hgbcat', 'hgbcat_slow'):
        from sklearn.ensemble import HistGradientBoostingClassifier
        lr, it = (0.05, 300) if kind == 'hgbcat' else (0.02, 900)
        return HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                              min_samples_leaf=20, categorical_features='from_dtype', random_state=seed)
    if kind.startswith('lrall'):
        from sklearn.compose import make_column_transformer, make_column_selector
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        C = {'lrall': 0.1, 'lrall_c03': 0.3, 'lrall_c003': 0.03}[kind]
        pre = make_column_transformer(
            (make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), make_column_selector(dtype_include='number')),
            (OneHotEncoder(handle_unknown='ignore', min_frequency=20), make_column_selector(dtype_include=object)))
        return make_pipeline(pre, LogisticRegression(C=C, max_iter=10000))
    if kind == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.6,
                             min_child_weight=3, reg_lambda=2.0, eval_metric='auc', n_jobs=8, random_state=seed)
    if kind == 'xgb_slow':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=1200, learning_rate=0.015, max_depth=3, subsample=0.8, colsample_bytree=0.5,
                             min_child_weight=5, reg_lambda=5.0, eval_metric='auc', n_jobs=6, random_state=seed)
    if kind in ('lgbm_reg', 'lgbm_smooth'):
        from lightgbm import LGBMClassifier, LGBMRegressor
        params = dict(n_estimators=1200, learning_rate=0.02, num_leaves=7, min_child_samples=40, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.5, reg_lambda=10.0, n_jobs=8, random_state=seed, verbose=-1)
        return LGBMClassifier(**params) if kind == 'lgbm_reg' else LGBMRegressor(objective='cross_entropy', **params)
    if kind == 'xgb_smooth':
        from xgboost import XGBRegressor
        return XGBRegressor(objective='binary:logistic', n_estimators=1200, learning_rate=0.015, max_depth=3, subsample=0.8,
                            colsample_bytree=0.5, min_child_weight=5, reg_lambda=5.0, n_jobs=6, random_state=seed)
    if kind == 'lgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, n_jobs=8, random_state=seed, verbose=-1)
    if kind.startswith('ebm'):
        from interpret.glassbox import ExplainableBoostingClassifier
        inter = {'ebm': 10, 'ebm0': 0}[kind]
        return ExplainableBoostingClassifier(interactions=inter, outer_bags=8, n_jobs=4, random_state=seed)
    if kind.startswith('tabpfn'):
        from tabpfn import TabPFNClassifier
        n = {'tabpfn': 4, 'tabpfn8': 8}[kind]
        return TabPFNClassifier(device=DEVICE, n_estimators=n, random_state=seed, ignore_pretraining_limits=True)
    if kind == 'cat_d3slow':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=2000, depth=3, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=seed,
                                  thread_count=5, verbose=False, allow_writing_files=False)
    if kind == 'cat_d4bag':  # three seeds averaged inside one estimator call (see evaluate)
        from catboost import CatBoostClassifier
        return [CatBoostClassifier(iterations=1000, depth=4, learning_rate=0.04, l2_leaf_reg=5.0, random_seed=100*seed + k,
                                   thread_count=5, verbose=False, allow_writing_files=False) for k in range(3)]
    if kind in ('cat', 'cat_d4', 'cat_d6'):
        from catboost import CatBoostClassifier
        depth = {'cat': 5, 'cat_d4': 4, 'cat_d6': 6}[kind]
        return CatBoostClassifier(iterations=1000, depth=depth, learning_rate=0.04, l2_leaf_reg=5.0, random_seed=seed,
                                  thread_count=8, verbose=False, allow_writing_files=False)
    raise ValueError(kind)


def evaluate(kind, feature_set, schemes, data):
    global MONO
    ids, y, tab = data['ids'], data['y'], FEATURE_SETS[feature_set](data)
    MONO = [1 if c in INCREASING else -1 if c in DECREASING else 0 for c in tab.columns]
    if kind.startswith('rfg_') and any(tab[c].dtype == object for c in tab.columns):
        tab = tab.copy()
        for c in [c for c in tab.columns if tab[c].dtype == object]:
            tab[c] = tab[c].astype('category').cat.codes  # ordinal codes for the forest
    if kind.startswith('hgbcat'):
        tab = tab.copy()
        for c in [c for c in tab.columns if tab[c].dtype == object]:
            counts = tab[c].value_counts()  # label-free: pool rare values so every column has < 255 levels
            tab[c] = tab[c].where(tab[c].map(counts) >= 10, 'rare').astype('category')
    res, oofs = {}, {}
    for scheme in schemes:
        fold = folds(scheme, ids, y)
        oof = np.zeros(len(y))
        for f in range(1, 6):
            tr, va = fold != f, fold == f
            m = model(kind, f)
            if isinstance(m, list):
                cats = [c for c in tab.columns if tab[c].dtype == object]
                oof[va] = np.mean([pd.Series(e.fit(tab[tr], y[tr], cat_features=cats).predict_proba(tab[va])[:, 1]).rank(pct=True)
                                   for e in m], axis=0)
                continue
            if kind.endswith('_smooth'):  # label smoothing: y' = y(1-eps) + eps/2, fitted with a logistic loss
                m.fit(tab[tr], y[tr]*(1 - SMOOTH) + SMOOTH/2)
                oof[va] = m.predict(tab[va])
                continue
            if kind.startswith('cat'):
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
