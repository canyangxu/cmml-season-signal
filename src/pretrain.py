"""Self-supervised-style pretraining on the multi-season history panel.

For every permitted account (development + independent source, never test) and every
target season t in 1995..2013 where the account already has a record before t, the
input is its season signal for seasons before t (right-aligned) and the target is
whether it subscribed in season t.  Only history records are used: no 2014-15 label,
no donation snapshot.  The fitted trunk initialises the 2014 fine-tuning model.

    python -I src/pretrain.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load, ID  # noqa: E402
from src.model import Net, Scaler, Trunk, fit, predict  # noqa: E402
from src.signal import CH, CHANNELS, YEARS, signals, static  # noqa: E402

OUT = Path(__file__).resolve().parents[1]/'outputs/pretrain'


def panel(sig, stat):
    xs, ss, ys, accounts = [], [], [], []
    for t in range(2, len(YEARS)):
        seen = (sig[:, CH['sub_present'], :t].max(1) > 0) | (sig[:, CH['tk_present'], :t].max(1) > 0)
        i = np.where(seen)[0]
        x = np.zeros((len(i), sig.shape[1], len(YEARS)), dtype=np.float32)
        x[:, :, len(YEARS) - t:] = sig[i, :, :t]
        x[:, CH['tk_available']] = 0
        x[:, CH['tk_available'], len(YEARS) - t:] = sig[0, CH['tk_available'], :t]  # absolute-year data mask
        xs.append(x)
        ss.append(np.column_stack([stat[i], np.full(len(i), (YEARS[t] - YEARS[0])/20, np.float32)]))
        ys.append(sig[i, CH['sub_present'], t].astype(int))
        accounts.append(i)
    return np.concatenate(xs), np.concatenate(ss).astype(np.float32), np.concatenate(ys), np.concatenate(accounts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    accounts = pd.concat([c.dev_accounts, c.source_accounts])
    ids = accounts[ID].tolist()
    sig = signals(ids, pd.concat([c.dev_subs, c.source_subs]), pd.concat([c.dev_tickets, c.source_tickets]))
    stat = static(ids, accounts, ids, full=False).to_numpy()
    x, s, y, owner = panel(sig, stat)
    print('panel', x.shape, 'positive rate', round(y.mean(), 3), flush=True)
    rng = np.random.default_rng(0)
    held = rng.random(len(ids)) < 0.1  # account-grouped validation split
    tr, va = ~held[owner], held[owner]
    scaler = Scaler().fit(x[tr], s[tr])
    xt, st_ = scaler(x[tr], s[tr]); xv, sv = scaler(x[va], s[va])
    torch.manual_seed(0)
    net = Net(Trunk(len(CHANNELS)), s.shape[1])
    best, epochs = fit(net, xt, st_, y[tr], seed=0, epochs=40, patience=4, lr=1e-3, oversample=0, batch=512,
                       val=(xv, sv, y[va]))
    print('pretrain validation AUROC', round(best, 4), 'epochs', epochs, flush=True)
    torch.save(net.trunk.state_dict(), OUT/'trunk.pt')
    np.savez(OUT/'scaler.npz', seq_lo=scaler.seq_lo, seq_hi=scaler.seq_hi)
    (OUT/'pretrain.json').write_text(json.dumps({'examples': int(len(y)), 'positive_rate': float(y.mean()),
        'validation_accounts': int(held.sum()), 'validation_auroc': best, 'epochs': epochs, 'channels': CHANNELS}, indent=1))


if __name__ == '__main__':
    main()
