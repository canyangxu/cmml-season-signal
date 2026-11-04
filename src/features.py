"""Tabular views used by the final model.

base     the flat summary of the season signal plus static account features (44 columns)
allview  rich signal summary + venue distances + ticket details (numeric) + categorical
         latest-season details, identifier batch prefix, ZIP3 and state (91 columns)

`d` is the dict built by the pipeline: corpus (accounts / subscriptions / tickets of the scored
accounts), ids, sig (season signal), stat (static features).  Every feature is computed from an
account's own records or from fixed public tables; nothing depends on which other accounts are
being scored.
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
