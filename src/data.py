"""Read-only access to the isolated corpus.  Modelling code imports only this.

`load()` verifies the isolation manifest hashes and returns development and source
tables.  Any attempt to read from the raw folder is refused: the raw folder (and so
the sealed test accounts) is used only by src/isolate.py and the final-inference
stage src/final.py.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ISO = ROOT/'data/isolated'
ID = 'account.id'


@dataclass
class Corpus:
    dev_accounts: pd.DataFrame
    dev_subs: pd.DataFrame
    dev_tickets: pd.DataFrame
    labels: pd.Series
    source_accounts: pd.DataFrame
    source_subs: pd.DataFrame
    source_tickets: pd.DataFrame


def _read(rel):
    path = (ISO/rel).resolve()
    if ISO.resolve() not in path.parents:
        raise PermissionError('Only data/isolated may be read during development')
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])


def load():
    manifest = json.loads((ISO/'manifest.json').read_text())
    for rel, digest in manifest['files'].items():
        if hashlib.sha256((ISO/rel).read_bytes()).hexdigest() != digest:
            raise ValueError('Isolated corpus changed: ' + rel)
    labels = _read('dev/labels.csv').set_index(ID).label.astype(int)
    return Corpus(_read('dev/accounts.csv'), _read('dev/subscriptions.csv'), _read('dev/tickets.csv'), labels,
                  _read('source/accounts.csv'), _read('source/subscriptions.csv'), _read('source/tickets.csv'))
