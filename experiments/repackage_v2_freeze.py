"""Repackage the original v2 freeze (outputs/final2, commit f383c4a) for the cleaned src/predict.py.

The model files are copied byte for byte; only the freeze record is rewritten in the new format,
keeping the original development check scores.  src/predict.py then verifies the copies, reproduces
those check scores from the raw tables with the cleaned code, and writes the test predictions, which
must equal the submitted season_signal_v2.csv byte for byte.

    python -I experiments/repackage_v2_freeze.py
    python -I src/predict.py --model outputs/model_v2 --out submissions/season_signal_v2_repro.csv
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import SOURCES, sha  # noqa: E402

V2, OUT, PRE = ROOT/'outputs/final2', ROOT/'outputs/model_v2', ROOT/'outputs/pretrain'


def main():
    if OUT.exists():
        sys.exit(f'{OUT} exists')
    old = json.loads((V2/'freeze.json').read_text())
    if {p.name: sha(p) for p in sorted(V2.glob('*')) if p.suffix in ('.pkl', '.pt')} != old['artifacts']:
        sys.exit('original v2 artifacts do not match their freeze record')
    OUT.mkdir(parents=True)
    for name in old['artifacts']:
        shutil.copy(V2/name, OUT/name)
    shutil.copy(PRE/'scaler.npz', OUT/'seq_scale.npz')
    (OUT/'freeze.json').write_text(json.dumps({
        'status': 'frozen_before_inference', 'origin': 'outputs/final2 (commit f383c4a), model files copied unchanged',
        'dev_ids': old['dev_ids'], 'permitted': old['permitted'], 'base_cols': old['base_cols'],
        'allview_cols': old['allview_cols'], 'check': old['check'],
        'pretrain': {'trunk.pt': sha(PRE/'trunk.pt'), 'scaler.npz': sha(PRE/'scaler.npz')},
        'sources': {n: sha(ROOT/'src'/n) for n in SOURCES},
        'artifacts': {p.name: sha(p) for p in sorted(OUT.glob('*')) if p.suffix in ('.pkl', '.pt', '.npz')}}))
    print('repackaged', len(old['artifacts']), 'model files ->', OUT)


if __name__ == '__main__':
    main()
