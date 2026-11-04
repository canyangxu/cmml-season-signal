"""Cross-validated evaluation on the 2014-15 development labels.

Schemes: the two fixed fold assignments of the previous project (seed 42 / 2026, for
comparability) and fresh repeated stratified 5-fold splits.  Per outer fold:
  cnn_pre     - pretrained trunk fine-tuned, 3 seeds (inner early stopping only)
  cnn_scratch - same network from random initialisation, 3 seeds (ablation)
  lr, rf, hgb - reference models on a flat summary of the same signal + static
Equal rank-mean ensembles are declared here, before any score is seen.

    python -I experiments/cv.py --scheme fixed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import ISO, load, ID  # noqa: E402
from src.model import Net, Scaler, Trunk, fit, predict  # noqa: E402
from src.signal import CHANNELS, flat, signals, static  # noqa: E402

OUT = ROOT/'outputs/cv'
SEEDS = (0, 1, 2)
ENSEMBLES = {'ens_cnn_lr_rf_hgb': ('cnn_pre', 'lr', 'rf', 'hgb'), 'ens_cnn_lr_rf': ('cnn_pre', 'lr', 'rf'),
             'ens_lr_rf_hgb': ('lr', 'rf', 'hgb'),
             # chosen on the 25 selection folds (fixed42, fixed2026, rep101-103); confirmed once on rep201-205
             'ens_final': ('rf_t', 'rf_t', 'rf_t', 'cnn_pre', 'lr', 'hgb')}


def folds(scheme, ids, y):
    if scheme.startswith('fixed'):
        ref = pd.read_csv(ISO/f'reference_folds_seed{scheme[5:]}.csv', dtype={ID: str}).set_index(ID).fold
        return ref.reindex(ids).to_numpy()
    fold = np.zeros(len(y), int)
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=int(scheme[3:])).split(ids, y), 1):
        fold[va] = f
    return fold


def cnn(seq, st, y, tr, va, pretrained, seed):
    st_scaler = Scaler().fit(seq[tr], st[tr])
    _, st_tr = st_scaler(seq[tr], st[tr]); _, st_va = st_scaler(seq[va], st[va])
    trunk = Trunk(len(CHANNELS))
    if pretrained:
        trunk.load_state_dict(torch.load(ROOT/'outputs/pretrain/trunk.pt'))
    net = Net(trunk, st.shape[1])
    inner, epochs = fit(net, seq[tr], st_tr, y[tr], seed=seed, trunk_lr=3e-4 if pretrained else None)
    return predict(net, seq[va], st_va), inner, epochs


def run(scheme):
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    permitted = ids + c.source_accounts[ID].tolist()
    sig = signals(ids, c.dev_subs, c.dev_tickets)
    stat = static(ids, c.dev_accounts, permitted, full=True)
    pre = np.load(ROOT/'outputs/pretrain/scaler.npz')
    seq = ((sig - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)  # pretraining scale
    st = stat.to_numpy()
    tab = flat(sig, stat)
    fold = folds(scheme, ids, y)
    oof = {k: np.zeros(len(y)) for k in ('cnn_pre', 'cnn_scratch', 'lr', 'rf', 'rf_t', 'hgb')}
    audit = []
    for f in range(1, 6):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        for name, pretrained in (('cnn_pre', True), ('cnn_scratch', False)):
            runs = []
            for s in SEEDS:
                p, inner, epochs = cnn(seq, st, y, tr, va, pretrained, 1000*f + s)
                runs.append(pd.Series(p).rank(pct=True).to_numpy())
                audit.append(dict(fold=f, model=name, seed=s, inner_auc=inner, epochs=epochs))
            oof[name][va] = np.mean(runs, axis=0)
        models = {'lr': make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
                  'rf': RandomForestClassifier(n_estimators=800, min_samples_leaf=3, max_features='sqrt', n_jobs=2, random_state=f),
                  'rf_t': RandomForestClassifier(n_estimators=1500, min_samples_leaf=8, max_features=4, n_jobs=2, random_state=f),
                  'hgb': HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                        l2_regularization=1.0, random_state=f)}
        for name, m in models.items():
            oof[name][va] = m.fit(tab.iloc[tr], y[tr]).predict_proba(tab.iloc[va])[:, 1]
        print(scheme, f, {k: round(roc_auc_score(y[va], v[va]), 4) for k, v in oof.items()}, flush=True)
    rank = lambda v: pd.Series(v).groupby(fold).rank(pct=True).to_numpy()
    for name, members in ENSEMBLES.items():
        oof[name] = np.mean([rank(oof[m]) for m in members], axis=0)
    res = {k: [roc_auc_score(y[fold == f], v[fold == f]) for f in range(1, 6)] for k, v in oof.items()}
    summary = {k: float(np.mean(v)) for k, v in res.items()}
    print(scheme, 'MEAN', {k: round(v, 5) for k, v in summary.items()}, flush=True)
    pd.DataFrame({ID: ids, 'label': y, 'fold': fold, **oof}).to_csv(OUT/f'oof_{scheme}.csv', index=False)
    (OUT/f'metrics_{scheme}.json').write_text(json.dumps({'scheme': scheme, 'mean': summary, 'folds': res, 'audit': audit}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', required=True)
    run(p.parse_args().scheme)
