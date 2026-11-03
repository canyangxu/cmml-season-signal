"""CNN variants on the same folds as src/cv.py, to lift the weakest ensemble member.

cnn_flat: pretrained trunk + the 44-column flat summary (incl. static) in the dense
branch instead of the 18 static columns; 5 seeds.  Everything else as in cv.cnn.

    python -I src/cnn_variants.py --scheme fixed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cv import folds  # noqa: E402
from src.data import load, ID  # noqa: E402
from src.model import Net, Scaler, Trunk, fit, predict  # noqa: E402
from src.signal import CHANNELS, flat, signals, static  # noqa: E402

OUT = ROOT/'outputs/cnn_variants'
SEEDS = (0, 1, 2, 3, 4)


def run(scheme):
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    permitted = ids + c.source_accounts[ID].tolist()
    sig = signals(ids, c.dev_subs, c.dev_tickets)
    stat = static(ids, c.dev_accounts, permitted, full=True)
    pre = np.load(ROOT/'outputs/pretrain/scaler.npz')
    seq = ((sig - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)
    dense = flat(sig, stat).to_numpy()
    fold = folds(scheme, ids, y)
    oof = np.zeros(len(y))
    for f in range(1, 6):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        sc = Scaler().fit(seq[tr], dense[tr])
        _, d_tr = sc(seq[tr], dense[tr]); _, d_va = sc(seq[va], dense[va])
        runs = []
        for s in SEEDS:
            trunk = Trunk(len(CHANNELS)); trunk.load_state_dict(torch.load(ROOT/'outputs/pretrain/trunk.pt'))
            net = Net(trunk, dense.shape[1])
            fit(net, seq[tr], d_tr, y[tr], seed=1000*f + s, trunk_lr=3e-4)
            runs.append(pd.Series(predict(net, seq[va], d_va)).rank(pct=True).to_numpy())
        oof[va] = np.mean(runs, axis=0)
        print(scheme, f, round(roc_auc_score(y[va], oof[va]), 4), flush=True)
    res = [roc_auc_score(y[fold == q], oof[fold == q]) for q in range(1, 6)]
    print(scheme, 'MEAN cnn_flat', round(float(np.mean(res)), 5), flush=True)
    pd.DataFrame({ID: ids, 'label': y, 'fold': fold, 'cnn_flat': oof}).to_csv(OUT/f'oof_{scheme}.csv', index=False)
    (OUT/f'metrics_{scheme}.json').write_text(json.dumps({'scheme': scheme, 'folds': res}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', required=True)
    run(p.parse_args().scheme)
