"""Out-of-fold predictions of the tree and linear members of the final model, built with the
constructors in src/members.py, on the same folds as validation/cv.py: RF on `base`; CatBoost, HGB
with native categories and logistic regression on `allview`.  As in the recorded runs: CatBoost uses 5 threads; for HGB_cat, category
levels seen fewer than 10 times among all development accounts are pooled into 'rare' (label-free).

Writes outputs/members_cv/<member>[_confirm]_oof.csv with one column per fold scheme.

    python -I validation/members_cv.py --member rf [--confirm]      (members: rf, catboost, hgb_cat, logistic)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import ID, load  # noqa: E402
from src.features import allview, base  # noqa: E402
from src.members import catboost, hist_gb_cat, logistic, random_forest  # noqa: E402
from src.signal import signals, static  # noqa: E402
from validation.cv import folds  # noqa: E402

OUT = ROOT/'outputs/members_cv'
SELECT = ('fixed42', 'fixed2026', 'rep101', 'rep102', 'rep103')
CONFIRM = ('rep201', 'rep202', 'rep203', 'rep204', 'rep205')


def table(member, c, ids, permitted):
    d = {'corpus': c, 'ids': ids, 'sig': signals(ids, c.dev_subs, c.dev_tickets), 'stat': static(ids, c.dev_accounts, permitted, full=True)}
    if member == 'rf':
        return base(d)
    av = allview(d)
    if member == 'hgb_cat':
        av = av.copy()
        for col in [col for col in av.columns if av[col].dtype == object]:
            counts = av[col].value_counts()
            av[col] = av[col].where(av[col].map(counts) >= 10, 'rare').astype('category')
    return av


def fit_predict(member, f, a, ya, b):
    if member == 'rf':
        return random_forest(f).fit(a, ya).predict_proba(b)[:, 1]
    if member == 'catboost':
        cats = [col for col in a.columns if a[col].dtype == object]
        return catboost(f).set_params(thread_count=5).fit(a, ya, cat_features=cats).predict_proba(b)[:, 1]
    if member == 'hgb_cat':
        return hist_gb_cat().set_params(random_state=f).fit(a, ya).predict_proba(b)[:, 1]
    return logistic().fit(a, ya).predict_proba(b)[:, 1]


def run(member, confirm):
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    permitted = ids + c.source_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    av = table(member, c, ids, permitted)
    oofs, res = {}, {}
    for scheme in (CONFIRM if confirm else SELECT):
        fold = folds(scheme, ids, y)
        oof = np.zeros(len(y))
        for f in range(1, 6):
            tr, va = fold != f, fold == f
            oof[va] = fit_predict(member, f, av[tr], y[tr], av[va])
        res[scheme] = [roc_auc_score(y[fold == q], oof[fold == q]) for q in range(1, 6)]
        oofs[scheme] = oof
        print(member, scheme, round(float(np.mean(res[scheme])), 5), flush=True)
    name = f"{member}{'_confirm' if confirm else ''}"
    pd.DataFrame(oofs).to_csv(OUT/f'{name}_oof.csv', index=False)
    (OUT/f'{name}.json').write_text(json.dumps({'member': member, 'folds': res}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--member', required=True, choices=['rf', 'catboost', 'hgb_cat', 'logistic'])
    p.add_argument('--confirm', action='store_true')
    a = p.parse_args()
    run(a.member, a.confirm)
