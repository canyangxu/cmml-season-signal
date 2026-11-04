"""Pooled-season training: every past season becomes training data for the 2014 model.

Rows (account, target season t) for t = 1996..2013 cover every permitted account
(development + independent source, never test) that already has a record before t;
features are computed from its records before t only, the target is whether it
subscribed in t.  Donation snapshot amounts are unknown as of t and set missing;
first-donation timing is recomputed relative to t (the analogue of "first donated in
2014").  The 2014 rows are the development accounts with their real labels.  Each
outer fold trains on all history rows + the 2014 training-fold rows and is scored
on the 2014 validation-fold rows only.  Variants are declared before scoring.

    python -I experiments/panel_cv.py --schemes fixed42 fixed2026
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cv import folds  # noqa: E402
from src.data import load, ID  # noqa: E402
from src.signal import CH, YEARS, flat, signals, static  # noqa: E402

OUT = ROOT/'outputs/panel_cv'
SNAPSHOT = ['don_2013', 'donated_2013', 'don_lifetime', 'don_count', 'don_mean']
VARIANTS = {'rf_2014only': ('rf', 0.0), 'rf_panel': ('rf', 0.5), 'hgb_2014only': ('hgb', 0.0), 'hgb_panel': ('hgb', 0.5)}


def as_of(stat, first, year):
    """Static features as known at target season `year` (amounts missing before 2014)."""
    s = stat.copy()
    end = pd.Timestamp(f'{year}-12-31')
    known = first.notna() & (first <= end)
    s['first_donated_known'] = known.astype(np.float32).to_numpy()
    s['first_donated_age'] = np.where(known, (end - first).dt.days/365.25, 0).astype(np.float32)
    s['first_donated_2014'] = (first.dt.year == year).astype(np.float32).to_numpy()  # "first donated in the target year"
    if year < 2014:
        s[SNAPSHOT] = np.nan
    s['target_year'] = np.float32(year)
    return s


def build(c):
    accounts = pd.concat([c.dev_accounts, c.source_accounts])
    ids = accounts[ID].tolist()
    dev = c.dev_accounts[ID].tolist()
    sig = signals(ids, pd.concat([c.dev_subs, c.source_subs]), pd.concat([c.dev_tickets, c.source_tickets]))
    stat = static(ids, accounts, ids, full=True)
    first = pd.to_datetime(accounts.set_index(ID).reindex(ids)['first.donated'], errors='coerce').reset_index(drop=True)
    rows = []
    for t in range(3, len(YEARS)):
        seen = (sig[:, CH['sub_present'], :t].max(1) > 0) | (sig[:, CH['tk_present'], :t].max(1) > 0)
        i = np.where(seen)[0]
        x = np.zeros((len(i), sig.shape[1], len(YEARS)), dtype=np.float32)
        x[:, :, len(YEARS) - t:] = sig[i, :, :t]
        x[:, CH['tk_available']] = 0
        x[:, CH['tk_available'], len(YEARS) - t:] = sig[0, CH['tk_available'], :t]
        st = as_of(stat.iloc[i].reset_index(drop=True), first.iloc[i].reset_index(drop=True), YEARS[t]).set_axis(stat.index[i])
        f = flat(x, st)
        f['y'] = sig[i, CH['sub_present'], t]
        rows.append(f)
    history = pd.concat(rows)
    d = [ids.index(a) for a in dev]
    st = as_of(stat.iloc[d].reset_index(drop=True), first.iloc[d].reset_index(drop=True), 2014).set_axis(stat.index[d])
    present = flat(sig[d], st)
    return history, present, dev


def model(kind, seed):
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=800, min_samples_leaf=3, max_features='sqrt', n_jobs=8, random_state=seed)
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)


def run(schemes):
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    history, present, dev = build(c)
    y = c.labels.reindex(dev).to_numpy()
    hx, hy = history.drop(columns='y'), history.y.to_numpy()
    print('history rows', len(hx), 'positive rate', round(hy.mean(), 3), '| 2014 rows', len(present), flush=True)
    report = {}
    for scheme in schemes:
        fold = folds(scheme, dev, y)
        oof = {k: np.zeros(len(y)) for k in VARIANTS}
        for f in range(1, 6):
            tr, va = fold != f, fold == f
            for name, (kind, w) in VARIANTS.items():
                if w:
                    X = pd.concat([hx, present[tr]]); Y = np.concatenate([hy, y[tr]])
                    W = np.concatenate([np.full(len(hx), w), np.ones(tr.sum())])
                else:
                    X, Y, W = present[tr], y[tr], np.ones(tr.sum())
                oof[name][va] = model(kind, f).fit(X, Y, sample_weight=W).predict_proba(present[va])[:, 1]
            print(scheme, f, {k: round(roc_auc_score(y[va], v[va]), 4) for k, v in oof.items()}, flush=True)
        report[scheme] = {k: [roc_auc_score(y[fold == q], v[fold == q]) for q in range(1, 6)] for k, v in oof.items()}
        print(scheme, 'MEAN', {k: round(float(np.mean(v)), 5) for k, v in report[scheme].items()}, flush=True)
        pd.DataFrame({ID: dev, 'label': y, 'fold': fold, **oof}).to_csv(OUT/f'oof_{scheme}.csv', index=False)
    (OUT/f'metrics_{"_".join(schemes)}.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--schemes', nargs='+', default=['fixed42', 'fixed2026'])
    run(p.parse_args().schemes)
