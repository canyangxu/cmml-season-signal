"""Final stage v3: smooth-model candidates and v3, from one freeze.

Hypothesis tested on the public board: smooth additive models transfer to the test accounts better
than the tree-heavy ensembles (all 14 earlier submissions were tree-heavy; a classmate's public
repo reports LR above its own tree ensemble on the public board at equal CV).

Reused from the v2 freeze (outputs/final2, artifact sha256 verified): 3 x RF_t[base], HGB[base],
3 x CatBoost d3[allview], 5 x CNN.  Fitted here on all development accounts:
  gam      cubic-spline additive logistic model on allview_new (C = 0.003)   CV 0.98668
  lr001    logistic regression on allview_new (C = 0.01)                      CV 0.98606
  lr003n   logistic regression on allview_new (C = 0.03)    v3 member         CV 0.98562
  hgbcatn  HGB, native categories, allview_new              v3 member         CV 0.98415
Files (rank-based; weights declared before inference; 25 selection folds):
  smooth_gam_v1.csv    = gam                                      0.98668
  smooth_trio_v1.csv   = (gam + lr001 + cnn) / 3                  0.98702
  season_signal_v3.csv = v3                                       0.98863 (confirmation 0.98872)
  v3_gam25_v1.csv      = 0.25 gam + 0.75 v3                       0.98875

    python -I src/final3.py train
    python -I src/final3.py infer
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load, ID  # noqa: E402
from src.features import allview_new, base, frontier  # noqa: E402
from src.final2 import encode, read_raw  # noqa: E402
from src.iterate import model as make  # noqa: E402
from src.model import Net, Trunk, predict  # noqa: E402
from src.signal import CHANNELS, signals, static  # noqa: E402

OUT = ROOT/'outputs/final3'
V2 = ROOT/'outputs/final2'
SUB = ROOT/'submissions'
SOURCES = ('final3.py', 'final2.py', 'features.py', 'signal.py', 'model.py', 'data.py', 'iterate.py')
FILES = ('smooth_gam_v1.csv', 'smooth_trio_v1.csv', 'season_signal_v3.csv', 'v3_gam25_v1.csv')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def views(ids, accounts, subs, tickets, permitted):
    corpus = SimpleNamespace(dev_accounts=accounts[accounts[ID].isin(ids)], dev_subs=subs[subs[ID].isin(ids)],
                             dev_tickets=tickets[tickets[ID].isin(ids)])
    d = {'corpus': corpus, 'ids': ids, 'sig': signals(ids, subs, tickets), 'stat': static(ids, accounts, permitted, full=True)}
    pre = np.load(ROOT/'outputs/pretrain/scaler.npz')
    seq = ((d['sig'] - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)
    avn = allview_new(d)
    return seq, d['stat'].to_numpy(), base(d).reset_index(drop=True), avn.drop(columns=['created_after_frontier', 'near_frontier']), avn


def load_v2():
    fz = json.loads((V2/'freeze.json').read_text())
    if {p.name: sha(p) for p in sorted(V2.glob('*')) if p.suffix in ('.pkl', '.pt')} != fz['artifacts']:
        sys.exit('v2 artifact verification failed')
    with open(V2/'models.pkl', 'rb') as f:
        saved = pickle.load(f)
    nets = []
    for i in range(5):
        n = Net(Trunk(len(CHANNELS)), len(saved['scaler'].st_lo[0])); n.load_state_dict(torch.load(V2/f'cnn_{i}.pt')); nets.append(n)
    return fz, saved, nets


def score(v2, nets, new, seq, st, tab, av, avn):
    m, levels, scaler = v2['m'], v2['levels'], v2['scaler']
    r = lambda v: rankdata(v)/len(v)
    rf = np.mean([r(m[f'rf_{s}'].predict_proba(tab)[:, 1]) for s in range(3)], axis=0)
    _, sts = scaler(seq, st)
    cnn = r(np.mean([predict(n, seq, sts) for n in nets], axis=0))
    cat = r(np.mean([r(c.predict_proba(av)[:, 1]) for c in m['cat']], axis=0))
    hgb = r(m['hgb'].predict_proba(tab)[:, 1])
    gam, lr001 = r(new['gam'].predict_proba(avn)[:, 1]), r(new['lr001'].predict_proba(avn)[:, 1])
    lr003n = r(new['lr003n'].predict_proba(avn)[:, 1])
    hcn = r(new['hgbcatn'].predict_proba(encode(avn, new['levels']))[:, 1])
    E = (3*rf + cnn + lr003n + hgb)/6
    v3 = (3*E + cat + hcn)/5
    return {'smooth_gam_v1.csv': gam, 'smooth_trio_v1.csv': (gam + lr001 + cnn)/3,
            'season_signal_v3.csv': v3, 'v3_gam25_v1.csv': .25*gam + .75*v3}


def train():
    if OUT.exists():
        sys.exit('outputs/final3 exists: never overwritten')
    OUT.mkdir(parents=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    permitted = ids + c.source_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    fz2, v2, nets = load_v2()
    if fz2['dev_ids'] != ids or fz2['permitted'] != permitted:
        sys.exit('account lists differ from the v2 freeze')
    seq, st, tab, av, avn = views(ids, c.dev_accounts, c.dev_subs, c.dev_tickets, permitted)
    cats = [col for col in avn.columns if avn[col].dtype == object]
    levels = {col: sorted(v for v, n in avn[col].value_counts().items() if n >= 10) for col in cats}  # label-free pooling
    new = {'gam': make('gam_c0003', 0).fit(avn, y), 'lr001': make('lrall_c001', 0).fit(avn, y),
           'lr003n': make('lrall_c003', 0).fit(avn, y), 'hgbcatn': make('hgbcat', 0).fit(encode(avn, levels), y), 'levels': levels}
    with open(OUT/'models.pkl', 'wb') as f:
        pickle.dump(new, f)
    check = score(v2, nets, new, seq, st, tab, av, avn)
    fmax, q99 = frontier()
    (OUT/'freeze.json').write_text(json.dumps({'status': 'frozen_before_inference', 'dev_ids': ids, 'permitted': permitted,
        'frontier': [fmax, q99], 'base_cols': list(tab.columns), 'allview_new_cols': list(avn.columns),
        'check': {k: v[:300].tolist() for k, v in check.items()}, 'v2_check': fz2['check'],
        'sources': {n: sha(ROOT/'src'/n) for n in SOURCES}, 'artifacts': {'models.pkl': sha(OUT/'models.pkl')},
        'v2_artifacts': fz2['artifacts']}))
    print('frozen', len(ids), {k: round(float(pd.Series(v).corr(pd.Series(check['season_signal_v3.csv']), method='spearman')), 3)
                               for k, v in check.items()}, flush=True)


def infer():
    fz = json.loads((OUT/'freeze.json').read_text())
    if {n: sha(ROOT/'src'/n) for n in SOURCES} != fz['sources'] or sha(OUT/'models.pkl') != fz['artifacts']['models.pkl']:
        sys.exit('freeze verification failed')
    fz2, v2, nets = load_v2()
    if fz2['artifacts'] != fz['v2_artifacts'] or list(frontier()) != fz['frontier']:
        sys.exit('v2 artifacts or frontier changed')
    with open(OUT/'models.pkl', 'rb') as f:
        new = pickle.load(f)
    accounts, subs, tickets = read_raw('account.csv'), read_raw('subscriptions.csv'), read_raw('tickets_all.csv')
    dev, permitted = fz['dev_ids'], fz['permitted']
    again = score(v2, nets, new, *views(dev, accounts, subs, tickets, permitted))
    for k in FILES:  # raw-table development features reproduce the frozen scores
        np.testing.assert_allclose(again[k][:300], fz['check'][k], atol=1e-9)
    from src.final2 import score as score_v2  # the v2 ensemble itself must also reproduce its own freeze
    s2, st2, tab2, av2, _ = views(dev, accounts, subs, tickets, permitted)
    np.testing.assert_allclose(score_v2(v2['m'], v2['levels'], v2['scaler'], nets, s2, st2, tab2, av2)[:300], fz['v2_check'], atol=1e-9)
    test_ids = read_raw('test.csv')['ID'].tolist()  # official test identities and inputs, used only here
    sample = read_raw('sample_submission.csv')
    if set(test_ids) & set(permitted) or sample.ID.tolist() != test_ids:
        sys.exit('test identity check failed')
    seq, st, tab, av, avn = views(test_ids, accounts, subs, tickets, permitted)
    if list(tab.columns) != fz['base_cols'] or list(avn.columns) != fz['allview_new_cols']:
        sys.exit('feature schema differs')
    pred = score(v2, nets, new, seq, st, tab, av, avn)
    SUB.mkdir(exist_ok=True)
    rep = {}
    for k in FILES:
        out = SUB/k
        pd.DataFrame({'ID': test_ids, 'Predicted': pred[k]}).to_csv(out, index=False, float_format='%.17g', mode='x')
        rep[k] = {'sha256': sha(out), 'rows': len(test_ids)}
    (OUT/'inference.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train', 'infer'])
    a = p.parse_args()
    train() if a.stage == 'train' else infer()
