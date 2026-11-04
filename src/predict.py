"""Step 4: score the official test accounts with the frozen ensemble.

1. Verify the freeze: source files and every artifact must match their recorded sha256.
2. Rebuild the development features from the raw official tables; the frozen model must reproduce
   its recorded development scores (proves the raw-table path equals the isolated one).
3. Only then read test.csv, check the identities (no overlap with any training or reference account,
   same order as sample_submission.csv), compute test features with the same functions and write
   the submission.  Nothing computed on the test accounts flows back into any earlier step.

    python -I src/predict.py [--model outputs/model] [--out submissions/season_signal_v2.csv]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ensemble import CNN_SEEDS, score, views  # noqa: E402
from src.model import Net, Trunk  # noqa: E402
from src.signal import CHANNELS  # noqa: E402
from src.train import SOURCES, sha  # noqa: E402

RAW = ROOT/'data/raw/for_students'


def read_raw(name):
    for enc in ('utf-8-sig', 'cp1252'):
        try:
            return pd.read_csv(RAW/name, dtype=str, keep_default_na=False, na_values=[''], encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(name)


def main(model, out):
    fz = json.loads((model/'freeze.json').read_text())
    if fz['status'] != 'frozen_before_inference' or {n: sha(ROOT/'src'/n) for n in SOURCES} != fz['sources'] or \
            {p.name: sha(p) for p in sorted(model.glob('*')) if p.suffix in ('.pkl', '.pt', '.npz')} != fz['artifacts']:
        sys.exit('freeze verification failed')
    with open(model/'models.pkl', 'rb') as f:
        saved = pickle.load(f)
    nets = []
    for i in range(len(CNN_SEEDS)):
        n = Net(Trunk(len(CHANNELS)), len(saved['scaler'].st_lo[0])); n.load_state_dict(torch.load(model/f'cnn_{i}.pt')); nets.append(n)
    scale = np.load(model/'seq_scale.npz'); scale = (scale['seq_lo'], scale['seq_hi'])
    frozen = (saved['m'], saved['levels'], saved['scaler'], nets)
    accounts, subs, tickets = read_raw('account.csv'), read_raw('subscriptions.csv'), read_raw('tickets_all.csv')
    dev, permitted = fz['dev_ids'], fz['permitted']
    again = score(*frozen, *views(dev, accounts, subs, tickets, permitted, scale))
    np.testing.assert_allclose(again[:300], fz['check'], atol=1e-9)
    test_ids = read_raw('test.csv')['ID'].tolist()  # official test identities and inputs: used only from here on
    sample = read_raw('sample_submission.csv')
    if set(test_ids) & set(permitted) or sample.ID.tolist() != test_ids or set(test_ids) - set(accounts['account.id']):
        sys.exit('test identity check failed')
    seq, st, tab, av = views(test_ids, accounts, subs, tickets, permitted, scale)
    if list(tab.columns) != fz['base_cols'] or list(av.columns) != fz['allview_cols']:
        sys.exit('feature schema differs')
    pred = score(*frozen, seq, st, tab, av)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ID': test_ids, 'Predicted': pred}).to_csv(out, index=False, float_format='%.17g', mode='x')
    rep = {'file': out.name, 'sha256': sha(out), 'rows': len(test_ids), 'model': str(model)}
    (model/'inference.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(ROOT/'outputs/model'))
    p.add_argument('--out', default=str(ROOT/'submissions/season_signal_v2.csv'))
    a = p.parse_args()
    main(Path(a.model), Path(a.out))
