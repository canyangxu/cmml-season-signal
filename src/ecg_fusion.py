"""Three heterogeneous 1D-CNN subnetworks, as in the user's ECG heartbeat project, adapted to the
21-season x 17-channel patron signal.

  Net1: shallow small kernels + max pooling, then a dilated convolution (wider receptive field)
  Net3: deeper stack with batch norm and learnable GeM pooling
  Net8: parallel kernels 3/5/7, each with parallel max + average pooling
Each subnetwork has its own dropout (simpler net, less dropout), is pretrained on the history
panel (next-season subscription, development + source accounts, no 2014-15 labels), then
fine-tuned per outer fold with Min-Max scaling, buyer oversampling, exponential learning-rate
decay (0.9 / epoch) and early stopping on an inner validation split.  3 seeds per subnetwork;
the three subnetworks are fused with equal weights (declared before scoring).

    python -I src/ecg_fusion.py pretrain
    python -I src/ecg_fusion.py cv --scheme fixed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cv import folds  # noqa: E402
from src.data import load, ID  # noqa: E402
from src.model import Net, Scaler, fit, predict  # noqa: E402
from src.pretrain import panel  # noqa: E402
from src.signal import CHANNELS, signals, static  # noqa: E402

OUT = ROOT/'outputs/ecg_fusion'
C = len(CHANNELS)
DROPOUT = {'net1': 0.2, 'net3': 0.3, 'net8': 0.4}


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x):
        return x.clamp(min=1e-6).pow(self.p).mean(-1).pow(1/self.p)


class Net1(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(C, w, 3, padding=1), nn.ReLU(), nn.Conv1d(w, w, 3, padding=1), nn.ReLU(),
                                  nn.MaxPool1d(2), nn.Conv1d(w, w, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(w, w, 3, padding=2, dilation=2), nn.ReLU())
        self.out_dim = 2*w

    def forward(self, x):
        h = self.body(x)
        return torch.cat([h.amax(-1), h[..., -1]], 1)


class Net3(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        layers, cin = [], C
        for k in (3, 5, 3, 3):
            layers += [nn.Conv1d(cin, w, k, padding=k//2), nn.BatchNorm1d(w), nn.ReLU()]
            cin = w
        self.body = nn.Sequential(*layers)
        self.pool = GeM()
        self.out_dim = 2*w

    def forward(self, x):
        h = self.body(x)
        return torch.cat([self.pool(h), h[..., -1]], 1)


class Net8(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        self.paths = nn.ModuleList([nn.Sequential(nn.Conv1d(C, w, k, padding=k//2), nn.ReLU(),
                                                  nn.Conv1d(w, w, k, padding=k//2), nn.ReLU()) for k in (3, 5, 7)])
        self.out_dim = 3*3*w

    def forward(self, x):
        parts = []
        for p in self.paths:
            h = p(x)
            parts += [h.amax(-1), h.mean(-1), h[..., -1]]
        return torch.cat(parts, 1)


TRUNKS = {'net1': Net1, 'net3': Net3, 'net8': Net8}


def fit_decay(net, *args, **kw):
    return fit(net, *args, decay=0.9, **kw)


def pretrain():
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    accounts = pd.concat([c.dev_accounts, c.source_accounts]); ids = accounts[ID].tolist()
    sig = signals(ids, pd.concat([c.dev_subs, c.source_subs]), pd.concat([c.dev_tickets, c.source_tickets]))
    x, s, y, owner = panel(sig, static(ids, accounts, ids, full=False).to_numpy())
    held = np.random.default_rng(0).random(len(ids)) < 0.1
    tr, va = ~held[owner], held[owner]
    sc = Scaler().fit(x[tr], s[tr]); xt, st_ = sc(x[tr], s[tr]); xv, sv = sc(x[va], s[va])
    np.savez(OUT/'scaler.npz', seq_lo=sc.seq_lo, seq_hi=sc.seq_hi)
    report = {}
    for name, T in TRUNKS.items():
        torch.manual_seed(0)
        net = Net(T(), s.shape[1], dropout=DROPOUT[name])
        best, ep = fit_decay(net, xt, st_, y[tr], seed=0, epochs=40, patience=4, oversample=0, batch=512, val=(xv, sv, y[va]))
        torch.save(net.trunk.state_dict(), OUT/f'{name}_trunk.pt')
        report[name] = {'validation_auroc': best, 'epochs': ep}
        print(name, 'pretrain AUROC', round(best, 4), 'epochs', ep, flush=True)
    (OUT/'pretrain.json').write_text(json.dumps(report, indent=1))


def cv(scheme):
    c = load()
    ids = c.dev_accounts[ID].tolist(); y = c.labels.reindex(ids).to_numpy()
    permitted = ids + c.source_accounts[ID].tolist()
    sig = signals(ids, c.dev_subs, c.dev_tickets)
    st = static(ids, c.dev_accounts, permitted, full=True).to_numpy()
    pre = np.load(OUT/'scaler.npz')
    seq = ((sig - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)
    fold = folds(scheme, ids, y)
    oof = {k: np.zeros(len(y)) for k in TRUNKS}
    for f in range(1, 6):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        sc = Scaler().fit(seq[tr], st[tr]); _, s_tr = sc(seq[tr], st[tr]); _, s_va = sc(seq[va], st[va])
        for name, T in TRUNKS.items():
            runs = []
            for s in range(3):
                trunk = T(); trunk.load_state_dict(torch.load(OUT/f'{name}_trunk.pt'))
                net = Net(trunk, st.shape[1], dropout=DROPOUT[name])
                fit_decay(net, seq[tr], s_tr, y[tr], seed=1000*f + s, trunk_lr=3e-4, patience=6)
                runs.append(pd.Series(predict(net, seq[va], s_va)).rank(pct=True).to_numpy())
            oof[name][va] = np.mean(runs, axis=0)
        print(scheme, f, {k: round(roc_auc_score(y[va], v[va]), 4) for k, v in oof.items()}, flush=True)
    rank = lambda v: pd.Series(v).groupby(fold).rank(pct=True).to_numpy()
    oof['ecg_fusion'] = np.mean([rank(oof[k]) for k in TRUNKS], axis=0)
    res = {k: [roc_auc_score(y[fold == q], v[fold == q]) for q in range(1, 6)] for k, v in oof.items()}
    print(scheme, 'MEAN', {k: round(float(np.mean(v)), 5) for k, v in res.items()}, flush=True)
    pd.DataFrame({ID: ids, 'label': y, 'fold': fold, **oof}).to_csv(OUT/f'oof_{scheme}.csv', index=False)
    (OUT/f'metrics_{scheme}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['pretrain', 'cv'])
    p.add_argument('--scheme', default='fixed42')
    a = p.parse_args()
    pretrain() if a.stage == 'pretrain' else cv(a.scheme)
