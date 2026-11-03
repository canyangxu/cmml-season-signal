"""Isolation step: the only development-stage code that opens the raw folder.

It reads test.csv for its identity column alone, to exclude those accounts, and
writes development accounts (with labels) and independent source accounts (neither
train nor test, no labels) to data/isolated/.  Nothing about the test accounts is
written, counted beyond their number, or passed on.  Modelling code reads only
data/isolated/ (see src/data.py).

    python -I src/isolate.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/for_students'
OUT = ROOT/'data/isolated'
ID = 'account.id'


def read(name):
    for encoding in ('utf-8-sig', 'cp1252'):  # account.csv carries cp1252 city names
        try:
            return pd.read_csv(RAW/name, dtype=str, keep_default_na=False, na_values=[''], encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError('Undecodable ' + name)


def main():
    if OUT.exists():
        sys.exit('data/isolated already exists; delete it deliberately to rebuild')
    train = read('train.csv')
    sealed = set(read('test.csv')['ID'])  # identity column only, for exclusion
    accounts, subs, tickets = read('account.csv'), read('subscriptions.csv'), read('tickets_all.csv')
    dev = set(train[ID])
    if dev & sealed:
        sys.exit('train and test overlap')
    source = set(accounts[ID]) - dev - sealed
    for pool, ids in (('dev', dev), ('source', source)):
        d = OUT/pool
        d.mkdir(parents=True)
        for name, table in (('accounts', accounts), ('subscriptions', subs), ('tickets', tickets)):
            part = table[table[ID].isin(ids)]
            if set(part[ID]) & sealed:
                sys.exit('sealed account leaked into ' + pool)
            part.to_csv(d/f'{name}.csv', index=False)
    train.to_csv(OUT/'dev/labels.csv', index=False)
    (OUT/'static').mkdir()
    for name in ('zipcodes.csv', 'concerts.csv', 'concerts_2014-15.csv'):
        read(name).to_csv(OUT/'static'/name, index=False)
    manifest = {'dev_accounts': len(dev), 'source_accounts': len(source), 'sealed_accounts_excluded': len(sealed),
                'dev_positives': int(train.label.astype(int).sum()),
                'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*.csv'))}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=1))
    print({k: v for k, v in manifest.items() if k != 'files'})


if __name__ == '__main__':
    main()
