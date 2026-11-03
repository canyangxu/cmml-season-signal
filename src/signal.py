"""Season signals and static features, computed per account from its own records.

Signal: 17 channels over seasons 1993-2013 (21 steps), like a multi-lead ECG.
Static: donation behaviour, relationship, address presence, account age.  Account
age is the identifier's rank among the permitted (development + source) accounts:
no labels, and the sealed test accounts are never part of the reference list.
"""
from bisect import bisect_left

import numpy as np
import pandas as pd

ID = 'account.id'
YEARS = np.arange(1993, 2014)
LOCATIONS = ('San Francisco', 'Berkeley Sunday', 'Berkeley Saturday', 'Peninsula', 'Contra Costa', 'Santa Rosa')
CHANNELS = (['sub_present', 'sub_seats', 'sub_tier', 'sub_full', 'sub_records', 'sub_price', 'sub_multiple']
            + [f'loc_{l}' for l in LOCATIONS] + ['tk_present', 'tk_seats', 'tk_records', 'tk_available'])
CH = {c: i for i, c in enumerate(CHANNELS)}
LOOKUP = {c: i for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')}


def _year(t):
    return t.season.str[:4].astype(int)


def signals(ids, subs, tickets):
    """Array (n_accounts, 17, 21) of raw per-season values."""
    pos = {a: i for i, a in enumerate(ids)}
    x = np.zeros((len(ids), len(CHANNELS), len(YEARS)), dtype=np.float32)

    def put(op, ch, i, t, v):
        op.at(x, (i, np.full(len(i), CH[ch]), t), np.asarray(v, dtype=np.float32))

    s = subs[subs[ID].isin(pos)].assign(y=_year)
    s = s[s.y.between(YEARS[0], YEARS[-1])]
    i, t = s[ID].map(pos).to_numpy(), (s.y - YEARS[0]).to_numpy()
    num = lambda c: pd.to_numeric(s[c], errors='coerce').fillna(0).to_numpy()
    put(np.maximum, 'sub_present', i, t, np.ones(len(i)))
    put(np.add, 'sub_seats', i, t, num('no.seats'))
    put(np.maximum, 'sub_tier', i, t, num('subscription_tier'))
    put(np.maximum, 'sub_full', i, t, s.package.eq('Full').to_numpy())
    put(np.add, 'sub_records', i, t, np.ones(len(i)))
    put(np.maximum, 'sub_price', i, t, num('price.level'))
    put(np.maximum, 'sub_multiple', i, t, s['multiple.subs'].eq('yes').to_numpy())
    for loc in LOCATIONS:
        put(np.maximum, f'loc_{loc}', i, t, s.location.eq(loc).to_numpy())
    k = tickets[tickets[ID].isin(pos)].assign(y=_year)
    k = k[k.y.between(YEARS[0], YEARS[-1])]
    i, t = k[ID].map(pos).to_numpy(), (k.y - YEARS[0]).to_numpy()
    put(np.maximum, 'tk_present', i, t, np.ones(len(i)))
    put(np.add, 'tk_seats', i, t, pd.to_numeric(k['no.seats'], errors='coerce').fillna(0).to_numpy())
    put(np.add, 'tk_records', i, t, np.ones(len(i)))
    x[:, CH['tk_available'], YEARS >= 2010] = 1  # ticket records exist only from 2010-11
    for ch in ('sub_seats', 'sub_records', 'tk_seats', 'tk_records'):
        x[:, CH[ch]] = np.log1p(x[:, CH[ch]])
    return x


def id_rank(ids, permitted):
    """Share of permitted accounts with a smaller identifier (creation-order proxy)."""
    nums = sorted(sum(LOOKUP[c]*62**k for k, c in enumerate(reversed(p))) for p in permitted)
    return np.array([bisect_left(nums, sum(LOOKUP[c]*62**k for k, c in enumerate(reversed(q))))/len(nums) for q in ids])


def static(ids, accounts, permitted, full=True):
    """Static features.  full=False drops the 2014-snapshot donation fields (pretraining)."""
    a = accounts.set_index(ID).reindex(ids)
    f = pd.DataFrame(index=pd.Index(ids, name=ID))
    rel = a.relationship.fillna('')
    f['rel_education'] = rel.str.contains('Education').astype(float)
    f['rel_vendor'] = rel.str.contains('Vendor').astype(float)
    f['rel_other'] = ((rel != '') & (f.rel_education == 0) & (f.rel_vendor == 0)).astype(float)
    f['billing_present'] = a['billing.zip.code'].notna().astype(float)
    f['shipping_present'] = a['shipping.zip.code'].notna().astype(float)
    f['billing_eq_shipping'] = (a['billing.zip.code'] == a['shipping.zip.code']).astype(float)
    f['account_age_rank'] = id_rank(ids, permitted)
    if full:
        num = lambda c: pd.to_numeric(a[c], errors='coerce').fillna(0)
        f['don_2013'] = np.log1p(num('amount.donated.2013'))
        f['donated_2013'] = (num('amount.donated.2013') > 0).astype(float)
        f['don_lifetime'] = np.log1p(num('amount.donated.lifetime'))
        f['don_count'] = np.log1p(num('no.donations.lifetime'))
        f['don_mean'] = np.log1p(num('amount.donated.lifetime')/np.maximum(num('no.donations.lifetime'), 1))
        first = pd.to_datetime(a['first.donated'], errors='coerce')
        f['first_donated_known'] = first.notna().astype(float)
        f['first_donated_age'] = ((pd.Timestamp('2014-12-31') - first).dt.days/365.25).fillna(0)
        f['first_donated_2014'] = first.dt.year.eq(2014).astype(float)
    return f.astype(np.float32)


def flat(sig, stat):
    """Flat summary of the signal for the linear / forest reference models."""
    s = sig
    out = pd.DataFrame(index=stat.index)
    for name, ch in (('sub', 'sub_present'), ('tk', 'tk_present')):
        p = s[:, CH[ch]]
        out[f'{name}_seasons'] = p.sum(1)
        out[f'{name}_last3'] = p[:, -3:].sum(1)
        out[f'{name}_2013'] = p[:, -1]
        out[f'{name}_2012'] = p[:, -2]
        last = np.where(p.any(1), len(YEARS) - 1 - np.argmax(p[:, ::-1], axis=1), -1)
        out[f'{name}_recency'] = np.where(last >= 0, len(YEARS) - 1 - last, 30)
    p = s[:, CH['sub_present']]
    streak = np.zeros(len(s))
    still = np.ones(len(s), bool)
    for t in range(len(YEARS) - 1, -1, -1):
        still &= p[:, t] > 0
        streak += still
    out['sub_streak'] = streak
    for ch in ('sub_seats', 'sub_tier', 'sub_full', 'sub_price', 'sub_multiple', 'tk_seats'):
        out[f'{ch}_2013'] = s[:, CH[ch], -1]
        out[f'{ch}_max'] = s[:, CH[ch]].max(1)
    for loc in LOCATIONS:
        out[f'loc_{loc}_2013'] = s[:, CH[f'loc_{loc}'], -1]
    return pd.concat([out, stat], axis=1).astype(np.float32)
