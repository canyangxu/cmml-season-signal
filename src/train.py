"""Step 3: fit the final ensemble on all 6,941 labelled development accounts and freeze it.

Reads only data/isolated (via src.data) and the pretrained trunk.  Writes to --out:
models.pkl, cnn_0..4.pt, seq_scale.npz and freeze.json (account lists, feature schema, the first
300 development scores, sha256 of the source files and of every artifact).  An existing freeze is
never overwritten.

    python -I src/train.py [--pretrain outputs/pretrain] [--out outputs/model]
"""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import ID, load  # noqa: E402
from src.ensemble import fit_all, score, views  # noqa: E402

SOURCES = ('ensemble.py', 'members.py', 'features.py', 'signal.py', 'model.py', 'data.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(pretrain, out):
    if out.exists():
        sys.exit(f'{out} exists: a freeze is never overwritten')
    out.mkdir(parents=True)
    c = load()
    ids = c.dev_accounts[ID].tolist()
    permitted = ids + c.source_accounts[ID].tolist()
    y = c.labels.reindex(ids).to_numpy()
    shutil.copy(pretrain/'scaler.npz', out/'seq_scale.npz')
    scale = np.load(out/'seq_scale.npz')
    seq, st, tab, av = views(ids, c.dev_accounts, c.dev_subs, c.dev_tickets, permitted, (scale['seq_lo'], scale['seq_hi']))
    m, levels, scaler, nets = fit_all(seq, st, tab, av, y, pretrain/'trunk.pt')
    with open(out/'models.pkl', 'wb') as f:
        pickle.dump({'m': m, 'levels': levels, 'scaler': scaler}, f)
    for i, n in enumerate(nets):
        torch.save(n.state_dict(), out/f'cnn_{i}.pt')
    check = score(m, levels, scaler, nets, seq, st, tab, av)
    (out/'freeze.json').write_text(json.dumps({
        'status': 'frozen_before_inference', 'dev_ids': ids, 'permitted': permitted,
        'base_cols': list(tab.columns), 'allview_cols': list(av.columns), 'check': check[:300].tolist(),
        'pretrain': {'trunk.pt': sha(pretrain/'trunk.pt'), 'scaler.npz': sha(pretrain/'scaler.npz')},
        'sources': {n: sha(ROOT/'src'/n) for n in SOURCES},
        'artifacts': {p.name: sha(p) for p in sorted(out.glob('*')) if p.suffix in ('.pkl', '.pt', '.npz')}}))
    print('frozen', len(ids), 'development accounts ->', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain', default=str(ROOT/'outputs/pretrain'))
    p.add_argument('--out', default=str(ROOT/'outputs/model'))
    a = p.parse_args()
    main(Path(a.pretrain), Path(a.out))
