"""Aggregate cross-validated results over all schemes (fold-level mean, spread, wins).

    python -I src/summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = {'fixed42': 0.988483091060, 'fixed2026': 0.986300382169}  # previous project's best, same folds


def main():
    runs = {p.stem.removeprefix('metrics_'): json.loads(p.read_text()) for p in sorted((ROOT/'outputs/cv').glob('metrics_*.json'))}
    models = list(next(iter(runs.values()))['folds'])
    print(f"{'model':20s} " + ' '.join(f'{s:>10s}' for s in runs) + f" {'all-folds':>10s} {'sd':>7s}")
    for m in models:
        folds = np.concatenate([r['folds'][m] for r in runs.values()])
        print(f'{m:20s} ' + ' '.join(f"{r['mean'][m]:10.5f}" for r in runs.values()) + f' {folds.mean():10.5f} {folds.std():7.4f}')
    print('prior best on fixed folds:', PRIOR)


if __name__ == '__main__':
    sys.exit(main())
