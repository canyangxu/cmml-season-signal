"""Final stage v2: the confirmed three-view ensemble.

  E     = (3 x RF_t[base] + CNN_pre + LR[allview] + HGB[base]) / 6        (equal-rank)
  final = (3 x E + CatBoost_d3[allview] + HGB_cat[allview]) / 5

Selection folds 0.98847; members confirmed on fresh folds.  `train` reads only
data/isolated; category pooling and encodings are fitted on development rows and
applied unchanged to test rows.  `infer` verifies the freeze, re-derives development
features from the raw files (must reproduce the frozen check), then scores test.

    python -I experiments/final_v2_original.py train
    python -I experiments/final_v2_original.py infer
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
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load, ID  # noqa: E402
from src.features import allview, base  # noqa: E402
from experiments.iterate import model as make  # noqa: E402
from src.model import Net, Scaler, Trunk, fit, predict  # noqa: E402
from src.signal import CHANNELS, signals, static  # noqa: E402

OUT = ROOT/'outputs/final2'
RAW = ROOT/'data/raw/for_students'
SUB = ROOT/'submissions'
SOURCES = ('final2.py', 'features.py', 'signal.py', 'model.py', 'data.py', 'iterate.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def views(ids, accounts, subs, tickets, permitted):
    corpus = SimpleNamespace(dev_accounts=accounts[accounts[ID].isin(ids)], dev_subs=subs[subs[ID].isin(ids)],
                             dev_tickets=tickets[tickets[ID].isin(ids)])
    d = {'corpus': corpus, 'ids': ids, 'sig': signals(ids, subs, tickets), 'stat': static(ids, accounts, permitted, full=True)}
    pre = np.load(ROOT/'outputs/pretrain/scaler.npz')
    seq = ((d['sig'] - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)
    return seq, d['stat'].to_numpy(), base(d).reset_index(drop=True), allview(d)


def encode(av, levels):
    out = av.copy()
    for c, keep in levels.items():
        out[c] = pd.Categorical(out[c].where(out[c].isin(keep), 'rare'), categories=keep + ['rare'])
    return out


def fit_all(seq, st, tab, av, y):
    cats = [c for c in av.columns if av[c].dtype == object]
    levels = {c: sorted(v for v, n in av[c].value_counts().items() if n >= 10) for c in cats}
    m = {f'rf_{s}': RandomForestClassifier(n_estimators=1500, min_samples_leaf=8, max_features=4, n_jobs=8, random_state=s).fit(tab, y)
         for s in range(3)}
    m['lrall'] = make('lrall_c003', 0).fit(av, y)
    m['hgb'] = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                              random_state=0).fit(tab, y)
    m['cat'] = [CatBoostClassifier(iterations=2000, depth=3, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=s, thread_count=8,
                                   verbose=False, allow_writing_files=False).fit(av, y, cat_features=cats) for s in range(3)]
    m['hgbcat'] = make('hgbcat', 0).fit(encode(av, levels), y)
    scaler = Scaler().fit(seq, st)
    _, sts = scaler(seq, st)
    nets = []
    for s in range(5):
        t = Trunk(len(CHANNELS)); t.load_state_dict(torch.load(ROOT/'outputs/pretrain/trunk.pt'))
        n = Net(t, st.shape[1]); fit(n, seq, sts, y, seed=100 + s, trunk_lr=3e-4); nets.append(n)
    return m, levels, scaler, nets


def score(m, levels, scaler, nets, seq, st, tab, av):
    r = lambda v: rankdata(v)/len(v)
    rf = np.mean([r(m[f'rf_{s}'].predict_proba(tab)[:, 1]) for s in range(3)], axis=0)
    _, sts = scaler(seq, st)
    cnn = r(np.mean([predict(n, seq, sts) for n in nets], axis=0))
    E = (3*rf + cnn + r(m['lrall'].predict_proba(av)[:, 1]) + r(m['hgb'].predict_proba(tab)[:, 1]))/6
    cat = r(np.mean([r(c.predict_proba(av)[:, 1]) for c in m['cat']], axis=0))
    hc = r(m['hgbcat'].predict_proba(encode(av, levels))[:, 1])
    return (3*E + cat + hc)/5


def train():
    if OUT.exists():
        sys.exit('outputs/final2 exists: never overwritten')
    OUT.mkdir(parents=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    permitted = ids + c.source_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    seq, st, tab, av = views(ids, c.dev_accounts, c.dev_subs, c.dev_tickets, permitted)
    m, levels, scaler, nets = fit_all(seq, st, tab, av, y)
    with open(OUT/'models.pkl', 'wb') as f:
        pickle.dump({'m': m, 'levels': levels, 'scaler': scaler}, f)
    for i, n in enumerate(nets):
        torch.save(n.state_dict(), OUT/f'cnn_{i}.pt')
    check = score(m, levels, scaler, nets, seq, st, tab, av)
    (OUT/'freeze.json').write_text(json.dumps({'status': 'frozen_before_inference', 'dev_ids': ids, 'permitted': permitted,
        'base_cols': list(tab.columns), 'allview_cols': list(av.columns), 'check': check[:300].tolist(),
        'sources': {n: sha(ROOT/'src'/n) for n in SOURCES},
        'artifacts': {p.name: sha(p) for p in sorted(OUT.glob('*')) if p.suffix in ('.pkl', '.pt')}}))
    print('frozen', len(ids), flush=True)


def read_raw(name):
    for enc in ('utf-8-sig', 'cp1252'):
        try:
            return pd.read_csv(RAW/name, dtype=str, keep_default_na=False, na_values=[''], encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(name)


def infer():
    fz = json.loads((OUT/'freeze.json').read_text())
    if {n: sha(ROOT/'src'/n) for n in SOURCES} != fz['sources'] or \
            {p.name: sha(p) for p in sorted(OUT.glob('*')) if p.suffix in ('.pkl', '.pt')} != fz['artifacts']:
        sys.exit('freeze verification failed')
    with open(OUT/'models.pkl', 'rb') as f:
        saved = pickle.load(f)
    nets = []
    for i in range(5):
        n = Net(Trunk(len(CHANNELS)), len(saved['scaler'].st_lo[0])); n.load_state_dict(torch.load(OUT/f'cnn_{i}.pt')); nets.append(n)
    accounts, subs, tickets = read_raw('account.csv'), read_raw('subscriptions.csv'), read_raw('tickets_all.csv')
    dev, permitted = fz['dev_ids'], fz['permitted']
    again = score(saved['m'], saved['levels'], saved['scaler'], nets, *views(dev, accounts, subs, tickets, permitted))
    np.testing.assert_allclose(again[:300], fz['check'], atol=1e-9)
    test_ids = read_raw('test.csv')['ID'].tolist()  # official test identities and inputs, used only here
    sample = read_raw('sample_submission.csv')
    if set(test_ids) & set(permitted) or sample.ID.tolist() != test_ids:
        sys.exit('test identity check failed')
    seq, st, tab, av = views(test_ids, accounts, subs, tickets, permitted)
    if list(tab.columns) != fz['base_cols'] or list(av.columns) != fz['allview_cols']:
        sys.exit('feature schema differs')
    pred = score(saved['m'], saved['levels'], saved['scaler'], nets, seq, st, tab, av)
    SUB.mkdir(exist_ok=True)
    out = SUB/'season_signal_v2.csv'
    pd.DataFrame({'ID': test_ids, 'Predicted': pred}).to_csv(out, index=False, float_format='%.17g', mode='x')
    rep = {'file': out.name, 'sha256': sha(out), 'rows': len(test_ids)}
    (OUT/'inference.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train', 'infer'])
    a = p.parse_args()
    train() if a.stage == 'train' else infer()
