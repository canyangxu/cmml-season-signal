"""Swap label-noise members into v3 and compare on the same folds (clean labels throughout).

v3:  E  = (3 x RF_t[base] + CNN_pre + LR[allview_new] + HGB[base]) / 6
     v3 = (3 x E + CatBoost_d3[allview] + HGB_cat[allview_new]) / 5         (per-fold equal-rank)

    python -I src/noise_swap.py [--confirm]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.iterate import CONFIRM, SELECT  # noqa: E402

IT, NO = ROOT/'outputs/iterate', ROOT/'outputs/noise'
SLOT = {'rf': 'rft', 'cnn': 'cnn', 'hgbcat': 'hcn', 'cat': 'c3', 'hgb': 'hgb'}


def noise_oof(member, noise, schemes, tag):
    whole = NO/f'{tag}{member}_{noise}_oof.csv'
    if whole.exists():
        return pd.read_csv(whole)
    cols = {}
    for p in NO.glob(f'{tag}{member}_{noise}_*_oof.csv'):  # runs over scheme subsets
        o = pd.read_csv(p)
        cols.update({s: o[s] for s in o.columns if s in schemes})
    return pd.DataFrame(cols) if all(s in cols for s in schemes) else None


def base_members(scheme, tag):
    o = pd.read_csv(ROOT/f'outputs/cv/oof_{scheme}.csv')
    rd = lambda n: pd.read_csv(IT/f'{tag}{n}_oof.csv')[scheme]
    rft = rd('rfg_4_8_1500_base') if (IT/f'{tag}rfg_4_8_1500_base_oof.csv').exists() else o.rf_t  # same model, same seed
    return o.label.to_numpy(), o.fold.to_numpy(), {
        'rft': rft, 'cnn': o.cnn_pre, 'lan': rd('lrall_c003_allview_new'), 'hgb': o.hgb,
        'c3': rd('cat_d3slow_allview'), 'hcn': rd('hgbcat_allview_new')}


def v3(m, f):
    rk = lambda v: pd.Series(np.asarray(v, float)).groupby(f).rank(pct=True).to_numpy()
    E = (3*rk(m['rft']) + rk(m['cnn']) + rk(m['lan']) + rk(m['hgb']))/6
    return (3*E + rk(m['c3']) + rk(m['hcn']))/5


def main(confirm):
    schemes, tag = (CONFIRM, 'confirm_') if confirm else (SELECT, '')
    variants = sorted({(p.name[len(tag):].split('_')[0], p.name[len(tag):].split('_')[1])
                       for p in NO.glob(f'{tag}*_oof.csv') if (not confirm) or p.name.startswith('confirm_')}
                      - ({('confirm', x) for x in ('rf', 'cnn', 'hgb', 'hgbcat', 'cat')} if not confirm else set()))
    rows = {'v3': []}
    alone = {}
    for s in schemes:
        y, f, m = base_members(s, tag)
        rows['v3'] += [roc_auc_score(y[f == q], v3(m, f)[f == q]) for q in range(1, 6)]
        for member, noise in variants:
            o = noise_oof(member, noise, schemes, tag)
            if o is None:
                continue
            mm = dict(m); mm[SLOT[member]] = o[s]
            rows.setdefault(f'{member}:{noise}', []).extend(roc_auc_score(y[f == q], v3(mm, f)[f == q]) for q in range(1, 6))
            alone.setdefault(f'{member}:{noise}', []).extend(roc_auc_score(y[f == q], o[s].to_numpy()[f == q]) for q in range(1, 6))
    b = np.array(rows['v3'])
    print(f"{'v3':22s} ALL={b.mean():.5f}")
    for k, v in rows.items():
        if k != 'v3' and len(v) == len(b):
            v = np.array(v)
            print(f'{k:22s} ALL={v.mean():.5f}  diff={v.mean() - b.mean():+.5f}  folds>v3 {(v > b).sum()}/{len(b)}'
                  f'  member alone {np.mean(alone[k]):.5f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--confirm', action='store_true')
    main(p.parse_args().confirm)
