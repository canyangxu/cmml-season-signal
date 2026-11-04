"""The final model (submitted as season_signal_v2.csv).

    E     = (3 x RF[base] + CNN + LR[allview] + HGB[base]) / 6
    final = (3 x E + CatBoost[allview] + HGB_cat[allview]) / 5

RF = mean of 3 seeds; CNN = mean of 5 fine-tuned copies of the pretrained trunk; CatBoost = mean of 3
seeds.  Members are combined as ranks within the scored set: AUROC depends only on the order.
Weights were declared before scoring and never fitted.  Every random step is seeded, so a retrain
in a fresh process reproduces the same model.
"""
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import rankdata

from src.data import ID
from src.features import allview, base
from src.members import catboost, category_levels, encode, hist_gb, hist_gb_cat, logistic, random_forest
from src.model import Net, Scaler, Trunk, fit, predict
from src.signal import CHANNELS, signals, static

RF_SEEDS, CAT_SEEDS, CNN_SEEDS = (0, 1, 2), (0, 1, 2), (100, 101, 102, 103, 104)


def views(ids, accounts, subs, tickets, permitted, seq_scale):
    """Inputs of every member for the accounts `ids`, computed from their own records.
    `permitted` (development + source accounts) is the fixed reference list for account age."""
    corpus = SimpleNamespace(dev_accounts=accounts[accounts[ID].isin(ids)], dev_subs=subs[subs[ID].isin(ids)],
                             dev_tickets=tickets[tickets[ID].isin(ids)])
    d = {'corpus': corpus, 'ids': ids, 'sig': signals(ids, subs, tickets), 'stat': static(ids, accounts, permitted, full=True)}
    lo, hi = seq_scale
    seq = ((d['sig'] - lo)/np.maximum(hi - lo, 1e-6)).astype(np.float32)
    return seq, d['stat'].to_numpy(), base(d).reset_index(drop=True), allview(d)


def fit_all(seq, st, tab, av, y, trunk_path):
    """Fit every member on the given accounts (the order is fixed so the CNN seeds reproduce)."""
    cats = [c for c in av.columns if av[c].dtype == object]
    levels = category_levels(av)
    m = {f'rf_{s}': random_forest(s).fit(tab, y) for s in RF_SEEDS}
    m['lrall'] = logistic().fit(av, y)
    m['hgb'] = hist_gb().fit(tab, y)
    m['cat'] = [catboost(s).fit(av, y, cat_features=cats) for s in CAT_SEEDS]
    m['hgbcat'] = hist_gb_cat().fit(encode(av, levels), y)
    scaler = Scaler().fit(seq, st)
    _, sts = scaler(seq, st)
    nets = []
    for s in CNN_SEEDS:
        torch.manual_seed(s)  # seeds the new head's initialisation too (the original v2 run left it unseeded)
        trunk = Trunk(len(CHANNELS)); trunk.load_state_dict(torch.load(trunk_path))
        net = Net(trunk, st.shape[1]); fit(net, seq, sts, y, seed=s, trunk_lr=3e-4); nets.append(net)
    return m, levels, scaler, nets


def score(m, levels, scaler, nets, seq, st, tab, av):
    r = lambda v: rankdata(v)/len(v)
    rf = np.mean([r(m[f'rf_{s}'].predict_proba(tab)[:, 1]) for s in RF_SEEDS], axis=0)
    _, sts = scaler(seq, st)
    cnn = r(np.mean([predict(n, seq, sts) for n in nets], axis=0))
    E = (3*rf + cnn + r(m['lrall'].predict_proba(av)[:, 1]) + r(m['hgb'].predict_proba(tab)[:, 1]))/6
    cat = r(np.mean([r(c.predict_proba(av)[:, 1]) for c in m['cat']], axis=0))
    hc = r(m['hgbcat'].predict_proba(encode(av, levels))[:, 1])
    return (3*E + cat + hc)/5
