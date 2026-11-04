"""Tabular views of the season signal + static features for the tree models.

base: the flat summary used in src/cv.py.
rich: base + per-season presence for 2009-2013, year-on-year changes, tenure, gaps,
      longest run, location diversity / switching, full-package share, ticket trend.
"""
import numpy as np
import pandas as pd

from src.signal import CH, LOCATIONS, YEARS, flat


def base(d):
    return flat(d['sig'], d['stat'])


def rich(d):
    s, out = d['sig'], base(d).copy()
    p = s[:, CH['sub_present']]
    for k in range(1, 6):
        out[f'sub_{YEARS[-k]}_present'] = p[:, -k]
    out['tk_2012_present'] = s[:, CH['tk_present'], -2]
    out['tk_2011_present'] = s[:, CH['tk_present'], -3]
    for ch in ('sub_seats', 'sub_tier', 'sub_price', 'sub_full', 'tk_seats', 'tk_records'):
        out[f'{ch}_delta'] = s[:, CH[ch], -1] - s[:, CH[ch], -2]
    first = np.where(p.any(1), np.argmax(p > 0, axis=1), len(YEARS))
    out['sub_tenure'] = np.where(p.any(1), len(YEARS) - first, 0)
    seasons = p.sum(1)
    out['sub_gaps'] = np.where(p.any(1), out['sub_tenure'] - seasons, 0)
    run = best = np.zeros(len(s))
    for t in range(len(YEARS)):
        run = np.where(p[:, t] > 0, run + 1, 0)
        best = np.maximum(best, run)
    out['sub_longest_run'] = best
    locs = np.stack([s[:, CH[f'loc_{l}']] for l in LOCATIONS], 1)
    out['sub_locations_ever'] = (locs.max(2) > 0).sum(1)
    out['sub_location_switch'] = ((locs[:, :, -1] != locs[:, :, -2]).any(1) & (p[:, -1] > 0) & (p[:, -2] > 0)).astype(float)
    out['sub_full_share'] = np.where(seasons > 0, s[:, CH['sub_full']].sum(1)/np.maximum(seasons, 1), 0)
    out['sub_seats_mean3'] = s[:, CH['sub_seats'], -3:].mean(1)
    out['tk_seasons'] = s[:, CH['tk_present']].sum(1)
    out['returning_lapsed'] = ((p[:, -1] == 0) & (p.any(1)) & (s[:, CH['tk_present'], -1] > 0)).astype(float)
    return out.astype(np.float32)


FEATURE_SETS = {'base': base, 'rich': rich}


def latest_categories(d):
    """Latest- and previous-season subscription details as categories (section, package, price, location)."""
    c, ids = d['corpus'], d['ids']
    s = c.dev_subs.assign(y=lambda t: t.season.str[:4].astype(int))
    out = pd.DataFrame(index=pd.Index(ids, name='account.id'))
    for tag, year in (('2013', 2013), ('2012', 2012)):
        g = s[s.y == year].sort_values(['account.id', 'package', 'section']).groupby('account.id')
        for col in ('package', 'section', 'price.level', 'location'):
            out[f'{col}_{tag}'] = g[col].first()
    last = s[s.y == s.groupby('account.id').y.transform('max')].sort_values(['account.id', 'package', 'section']).groupby('account.id')
    for col in ('package', 'section', 'price.level', 'location'):
        out[f'{col}_latest'] = last[col].first()
    return out.fillna('none').astype(str)


def catview(d):
    return pd.concat([base(d).reset_index(drop=True), latest_categories(d).reset_index(drop=True)], axis=1)


FEATURE_SETS['catview'] = catview


VENUES = {'sf': (37.7793, -122.4193), 'berkeley': (37.8696, -122.2620), 'palo_alto': (37.4419, -122.1430),
          'walnut_creek': (37.9001, -122.0601), 'santa_rosa': (38.4405, -122.7144)}


def _zip_coords(d):
    from src.data import ISO
    z = pd.read_csv(ISO/'static/zipcodes.csv', dtype=str)
    z['Zipcode'] = z.Zipcode.str.zfill(5)
    z = z.drop_duplicates('Zipcode').set_index('Zipcode')
    a = d['corpus'].dev_accounts.set_index('account.id').reindex(d['ids'])
    zip5 = a['billing.zip.code'].fillna(a['shipping.zip.code']).str[:5]
    lat = pd.to_numeric(z.Lat.reindex(zip5), errors='coerce').to_numpy()
    lon = pd.to_numeric(z.Long.reindex(zip5), errors='coerce').to_numpy()
    state = z.State.reindex(zip5).to_numpy()
    return lat, lon, state


def geo(d):
    out = base(d).copy()
    lat, lon, state = _zip_coords(d)
    dist = {}
    for name, (vlat, vlon) in VENUES.items():
        dlat, dlon = np.radians(lat - vlat), np.radians(lon - vlon)
        h = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(vlat))*np.sin(dlon/2)**2
        dist[name] = 6371*2*np.arcsin(np.sqrt(h))
    out['venue_km_min'] = np.log1p(np.nanmin(np.column_stack(list(dist.values())), axis=1)) if np.isfinite(lat).any() else np.nan
    out['venue_km_sf'] = np.log1p(dist['sf'])
    out['venue_km_berkeley'] = np.log1p(dist['berkeley'])
    out['in_california'] = (state == 'CA').astype(np.float32)
    return out.astype(np.float32)


def tix(d):
    out = base(d).copy()
    t = d['corpus'].dev_tickets.assign(y=lambda f: f.season.str[:4].astype(int),
                                        price=lambda f: pd.to_numeric(f['price.level'], errors='coerce'))
    g = t.groupby('account.id')
    latest = t[t.y == g.y.transform('max')]
    ids = pd.Index(d['ids'])
    out['tk_latest_price'] = latest.groupby('account.id').price.min().reindex(ids).to_numpy()
    for src_name in ('Philharmonia website', 'Other', 'Box office', 'Telemarketing'):
        out[f'tk_src_{src_name.split()[0].lower()}'] = t.assign(m=t['marketing.source'].eq(src_name)).groupby('account.id').m.any().reindex(ids).fillna(False).astype(float).to_numpy()
    out['tk_multiple_any'] = t.assign(m=t['multiple.tickets'].eq('yes')).groupby('account.id').m.any().reindex(ids).fillna(False).astype(float).to_numpy()
    out['tk_sets'] = t.assign(k=t.season + '|' + t.set.astype(str)).groupby('account.id').k.nunique().reindex(ids).fillna(0).to_numpy()
    return out.astype(np.float32)


FEATURE_SETS['geo'] = geo
FEATURE_SETS['tix'] = tix


_PANEL_CACHE = {}


def panel_score(d):
    """History-transfer score: a model fitted only on past-season rows (targets <= 2013-14,
    development + source accounts, never 2014-15 labels) scores each 2014 row."""
    if 'score' not in _PANEL_CACHE:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from src.panel_cv import build
        history, present, dev = build(d['corpus'])
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
        m.fit(history.drop(columns='y'), history.y)
        X = present.copy()
        X[['don_2013', 'donated_2013', 'don_lifetime', 'don_count', 'don_mean']] = np.nan  # as in training rows
        _PANEL_CACHE['score'] = pd.Series(m.predict_proba(X)[:, 1], index=pd.Index(dev)).reindex(d['ids']).to_numpy()
    return _PANEL_CACHE['score']


def base_panel(d):
    out = base(d).copy()
    out['history_transfer_score'] = panel_score(d)
    return out.astype(np.float32)


FEATURE_SETS['base_panel'] = base_panel


def allview(d):
    """Everything: rich signal summary + venue distances + ticket details (numeric) + categorical
    latest-season details, identifier batch prefix, ZIP3 and state (for boosting with native categories)."""
    num = rich(d).reset_index(drop=True)
    g, t = geo(d).reset_index(drop=True), tix(d).reset_index(drop=True)
    num = pd.concat([num, g[[c for c in g if c not in num]], t[[c for c in t if c not in num]]], axis=1)
    cats = latest_categories(d).reset_index(drop=True)
    a = d['corpus'].dev_accounts.set_index('account.id').reindex(d['ids'])
    zip5 = a['billing.zip.code'].fillna(a['shipping.zip.code']).fillna('none').str[:5]
    _, _, state = _zip_coords(d)
    cats['crm_prefix'] = [i[:11] for i in d['ids']]
    cats['zip3'] = zip5.str[:3].to_numpy()
    cats['state'] = pd.Series(state).fillna('none').astype(str).to_numpy()
    return pd.concat([num, cats], axis=1)


FEATURE_SETS['allview'] = allview


_FRONTIER = {}


def frontier():
    """Label-free creation frontier: the identifier rank of accounts with any activity before 2014
    (subscription or ticket record, or a first donation dated before 2014) among the permitted
    development + source accounts.  Returns (max rank, 99th percentile)."""
    if not _FRONTIER:
        from src.data import load, ID
        from src.signal import id_rank
        c = load()
        perm = c.dev_accounts[ID].tolist() + c.source_accounts[ID].tolist()
        acc = pd.concat([c.dev_accounts, c.source_accounts]).set_index(ID).reindex(perm)
        first = pd.to_datetime(acc['first.donated'], errors='coerce')
        active = (set(pd.concat([c.dev_subs, c.source_subs])[ID]) | set(pd.concat([c.dev_tickets, c.source_tickets])[ID])
                  | set(first[first < '2014-01-01'].index))
        ranks = pd.Series(id_rank(perm, perm), index=perm)
        r = ranks[ranks.index.isin(active)]
        _FRONTIER.update(max=float(r.max()), q99=float(r.quantile(0.99)))
    return _FRONTIER['max'], _FRONTIER['q99']


def _newness(age_rank):
    fmax, q99 = frontier()
    return {'created_after_frontier': (age_rank > fmax).astype(np.float32),
            'near_frontier': (age_rank > q99).astype(np.float32)}


def base_new(d):
    out = base(d).copy()
    for k, v in _newness(out['account_age_rank'].to_numpy()).items():
        out[k] = v
    return out.astype(np.float32)


def allview_new(d):
    out = allview(d)
    for k, v in _newness(out['account_age_rank'].to_numpy()).items():
        out[k] = v
    return out


FEATURE_SETS['base_new'] = base_new
FEATURE_SETS['allview_new'] = allview_new


def _don14(d):
    """Infer 2014 giving from the snapshot fields: first gift dated 2014, or first gift in 2013 with a
    lifetime total above the 2013 amount (the excess can only fall after 2013)."""
    a = d['corpus'].dev_accounts.set_index('account.id').reindex(d['ids'])
    num = lambda c: pd.to_numeric(a[c], errors='coerce').fillna(0).to_numpy()
    life, y13 = num('amount.donated.lifetime'), num('amount.donated.2013')
    first = pd.to_datetime(a['first.donated'], errors='coerce').dt.year.to_numpy()
    certain = (first == 2014) | ((first == 2013) & (life > y13 + 1e-9))
    return {'gave_2014_certain': certain.astype(np.float32),
            'gave_only_2013': ((life > 0) & (np.abs(life - y13) < 1e-9)).astype(np.float32),
            'gift_outside_2013_log': np.log1p(np.maximum(life - y13, 0)).astype(np.float32)}


def base_don(d):
    out = base_new(d).copy()
    for k, v in _don14(d).items():
        out[k] = v
    return out.astype(np.float32)


def allview_don(d):
    out = allview_new(d)
    for k, v in _don14(d).items():
        out[k] = v
    return out


FEATURE_SETS['base_don'] = base_don
FEATURE_SETS['allview_don'] = allview_don


_COUNTS = {}


def _count_tables():
    """Label-free counts over the permitted development + source accounts."""
    if not _COUNTS:
        from src.data import load
        c = load()
        acc = pd.concat([c.dev_accounts, c.source_accounts])
        z5 = acc['billing.zip.code'].fillna(acc['shipping.zip.code']).str[:5]
        city = acc['billing.city'].fillna(acc['shipping.city']).str.strip().str.lower()
        _COUNTS.update(zip5=z5.value_counts(), zip3=z5.str[:3].value_counts(), city=city.value_counts())
    return _COUNTS


def count_features(d):
    t = _count_tables()
    a = d['corpus'].dev_accounts.set_index('account.id').reindex(d['ids'])
    z5 = a['billing.zip.code'].fillna(a['shipping.zip.code']).str[:5]
    city = a['billing.city'].fillna(a['shipping.city']).str.strip().str.lower()
    return {'zip5_accounts_log': np.log1p(z5.map(t['zip5']).fillna(0).to_numpy()).astype(np.float32),
            'zip3_accounts_log': np.log1p(z5.str[:3].map(t['zip3']).fillna(0).to_numpy()).astype(np.float32),
            'city_accounts_log': np.log1p(city.map(t['city']).fillna(0).to_numpy()).astype(np.float32)}


def allview_cnt(d):
    out = allview_new(d)
    for k, v in count_features(d).items():
        out[k] = v
    return out


FEATURE_SETS['allview_cnt'] = allview_cnt


ROLL_CHANNELS = ('sub_present', 'sub_seats', 'sub_tier', 'sub_price', 'sub_full', 'tk_present', 'tk_seats', 'tk_records')


def rolling(d):
    """Rolling statistics of each season signal channel, anchored at the latest season (2013-14):
    means over 3/5/10 seasons, std over 5/10, exponentially weighted mean (alpha 0.5), linear
    trend over the last 5 seasons and the 5-season maximum."""
    s, out = d['sig'], {}
    t5 = np.arange(5) - 2.0
    w = 0.5 ** np.arange(len(YEARS))[::-1]  # latest season weight 1, halving backwards
    for ch in ROLL_CHANNELS:
        x = s[:, CH[ch]]
        for k in (3, 5, 10):
            out[f'{ch}_mean{k}'] = x[:, -k:].mean(1)
        for k in (5, 10):
            out[f'{ch}_std{k}'] = x[:, -k:].std(1)
        out[f'{ch}_ewm'] = (x*w).sum(1)/w.sum()
        out[f'{ch}_slope5'] = (x[:, -5:]*t5).sum(1)/(t5**2).sum()
        out[f'{ch}_max5'] = x[:, -5:].max(1)
    return pd.DataFrame(out).astype(np.float32)


def base_roll(d):
    return pd.concat([base(d).reset_index(drop=True), rolling(d)], axis=1)


FEATURE_SETS['base_roll'] = base_roll


ADDRESS = ('venue_km_min', 'venue_km_sf', 'venue_km_berkeley', 'in_california', 'zip3', 'state')


def allview_noaddr(d):
    """allview without geography: no venue distances, California flag, ZIP3 or state."""
    av = allview(d)
    return av.drop(columns=[c for c in ADDRESS if c in av.columns])


FEATURE_SETS['allview_noaddr'] = allview_noaddr
