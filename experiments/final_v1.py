"""Final stage: fit the chosen ensemble on all development accounts, freeze, then infer.

Recipe (chosen on the selection folds, confirmed on rep201-205): equal-rank mean of
3 x RF(1500 trees, leaf 8, 4 features/split) + pretrained season CNN + LR + HGB.
`train` reads only data/isolated.  `infer` verifies the freeze, then opens the raw
official files and computes test features with the same functions; the account-age
reference list stays development + source accounts (test accounts never enter it).

    python -I experiments/final_v1.py train
    python -I experiments/final_v1.py infer [--hybrid PATH_TO_PRIOR_SUBMISSION]
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load, ID  # noqa: E402
from src.model import Net, Scaler, Trunk, fit, predict  # noqa: E402
from src.signal import CHANNELS, flat, signals, static  # noqa: E402

OUT = ROOT/'outputs/final'
RAW = ROOT/'data/raw/for_students'
SUB = ROOT/'submissions'
CNN_SEEDS = (0, 1, 2, 3, 4)
SOURCES = ('final.py', 'signal.py', 'model.py', 'data.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(ids, accounts, subs, tickets, permitted):
    sig = signals(ids, subs, tickets)
    stat = static(ids, accounts, permitted, full=True)
    pre = np.load(ROOT/'outputs/pretrain/scaler.npz')
    seq = ((sig - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)
    return seq, stat.to_numpy(), flat(sig, stat)


def members(seq, st, tab, y):
    models = {f'rf_{s}': RandomForestClassifier(n_estimators=1500, min_samples_leaf=8, max_features=4, n_jobs=8, random_state=s).fit(tab, y)
              for s in range(3)}
    models['lr'] = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(tab, y)
    models['hgb'] = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                                   random_state=0).fit(tab, y)
    scaler = Scaler().fit(seq, st)
    _, sts = scaler(seq, st)
    nets = []
    for s in CNN_SEEDS:
        trunk = Trunk(len(CHANNELS)); trunk.load_state_dict(torch.load(ROOT/'outputs/pretrain/trunk.pt'))
        net = Net(trunk, st.shape[1])
        fit(net, seq, sts, y, seed=100 + s, trunk_lr=3e-4)
        nets.append(net)
    return models, scaler, nets


def score(models, scaler, nets, seq, st, tab):
    r = lambda v: rankdata(v)/len(v)
    rf = np.mean([r(models[f'rf_{s}'].predict_proba(tab)[:, 1]) for s in range(3)], axis=0)
    _, sts = scaler(seq, st)
    cnn = r(np.mean([predict(n, seq, sts) for n in nets], axis=0))
    parts = {'rf': rf, 'cnn': cnn, 'lr': r(models['lr'].predict_proba(tab)[:, 1]), 'hgb': r(models['hgb'].predict_proba(tab)[:, 1])}
    return (3*parts['rf'] + parts['cnn'] + parts['lr'] + parts['hgb'])/6, parts


def train():
    if OUT.exists():
        sys.exit('outputs/final exists: the freeze is never overwritten')
    OUT.mkdir(parents=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    permitted = ids + c.source_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    seq, st, tab = features(ids, c.dev_accounts, c.dev_subs, c.dev_tickets, permitted)
    models, scaler, nets = members(seq, st, tab, y)
    with open(OUT/'models.pkl', 'wb') as f:
        pickle.dump({'models': models, 'scaler': scaler}, f)
    for i, n in enumerate(nets):
        torch.save(n.state_dict(), OUT/f'cnn_{i}.pt')
    check, _ = score(models, scaler, nets, seq, st, tab)
    (OUT/'freeze.json').write_text(json.dumps({'status': 'frozen_before_inference', 'dev_ids': ids, 'permitted': permitted,
        'features': list(tab.columns), 'check': check[:200].tolist(), 'sources': {n: sha(ROOT/'src'/n) for n in SOURCES},
        'artifacts': {p.name: sha(p) for p in sorted(OUT.glob('*')) if p.suffix in ('.pkl', '.pt')}}))
    print('frozen', len(ids), 'accounts', flush=True)


def read_raw(name):
    for enc in ('utf-8-sig', 'cp1252'):
        try:
            return pd.read_csv(RAW/name, dtype=str, keep_default_na=False, na_values=[''], encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(name)


def infer(hybrid):
    fz = json.loads((OUT/'freeze.json').read_text())
    if fz['status'] != 'frozen_before_inference' or {n: sha(ROOT/'src'/n) for n in SOURCES} != fz['sources'] or \
            {p.name: sha(p) for p in sorted(OUT.glob('*')) if p.suffix in ('.pkl', '.pt')} != fz['artifacts']:
        sys.exit('freeze verification failed')
    with open(OUT/'models.pkl', 'rb') as f:
        saved = pickle.load(f)
    nets = []
    for i in range(len(CNN_SEEDS)):
        n = Net(Trunk(len(CHANNELS)), len(saved['scaler'].st_lo[0])); n.load_state_dict(torch.load(OUT/f'cnn_{i}.pt')); nets.append(n)
    accounts, subs, tickets = read_raw('account.csv'), read_raw('subscriptions.csv'), read_raw('tickets_all.csv')
    dev, permitted = fz['dev_ids'], fz['permitted']
    seq, st, tab = features(dev, accounts, subs, tickets, permitted)
    again, _ = score(saved['models'], saved['scaler'], nets, seq, st, tab)
    np.testing.assert_allclose(again[:200], fz['check'], atol=1e-9)  # raw-table features equal isolated ones
    # Official test identities and inputs, used only here.
    test_ids = read_raw('test.csv')['ID'].tolist()
    sample = read_raw('sample_submission.csv')
    if set(test_ids) & set(permitted) or sample.ID.tolist() != test_ids:
        sys.exit('test identity check failed')
    seq, st, tab = features(test_ids, accounts, subs, tickets, permitted)
    if list(tab.columns) != fz['features']:
        sys.exit('feature schema differs')
    pred, parts = score(saved['models'], saved['scaler'], nets, seq, st, tab)
    SUB.mkdir(exist_ok=True)
    out = SUB/'season_signal_v1.csv'
    pd.DataFrame({'ID': test_ids, 'Predicted': pred}).to_csv(out, index=False, float_format='%.17g', mode='x')
    report = {'file': out.name, 'sha256': sha(out), 'rows': len(test_ids)}
    if hybrid:
        prior = pd.read_csv(hybrid, dtype={'ID': str}).set_index('ID').Predicted.reindex(test_ids).to_numpy()
        hyb = .5*pred + .5*rankdata(prior)/len(prior)
        hout = SUB/'season_signal_hybrid_v1.csv'
        pd.DataFrame({'ID': test_ids, 'Predicted': hyb}).to_csv(hout, index=False, float_format='%.17g', mode='x')
        report.update(hybrid_file=hout.name, hybrid_sha256=sha(hout), prior=str(hybrid), prior_sha256=sha(hybrid))
    (OUT/'inference.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train', 'infer'])
    p.add_argument('--hybrid', default='')
    a = p.parse_args()
    train() if a.stage == 'train' else infer(a.hybrid)
