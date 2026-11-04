"""Label-noise training ("add noise to y") on the members of the best ensemble (v3).

User guidance (2026-11-03): noise touches the training targets only; evaluation metrics and
early stopping use the noise-free labels.  Here the outer fold is always scored against the
true labels, and every early-stopping decision (boosting iteration, network epoch) is taken on
an inner split carved from the outer-training rows whose labels stay clean.

Noise (drawn independently for every tree group / bag member / mini-batch):
  flip<r>   output flipping (Breiman 2000), class proportions kept: P(1->0)=r, P(0->1)=r*n1/n0
  smear<s>  output smearing (Breiman 2000): y + N(0, (s*sd(y))^2); clipped to [0, 1] for
            cross-entropy members (soft targets)
  none      clean labels: the control, with identical bagging / early stopping

Members (as in v3): rf = RF_t on base; hgbcat = HGB with native categories on allview_new;
hgb = HGB on base; cat = CatBoost depth 3 on allview; cnn = pretrained season CNN with the
same three seeds per fold as src/cv.py (noise=none reproduces cnn_pre).

    python -I experiments/noise.py --member rf --noise flip0.1 [--schemes fixed42,rep101] [--confirm]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cv import folds  # noqa: E402
from experiments.extra_features import FEATURE_SETS  # noqa: E402
from experiments.iterate import CONFIRM, SELECT, prepare  # noqa: E402

OUT = ROOT/'outputs/noise'
NJ = 5
rank = lambda v: pd.Series(np.asarray(v, float)).rank(pct=True).to_numpy()


def parse(noise):
    for k in ('flip', 'smear', 'none'):
        if noise.startswith(k):
            return k, float(noise[len(k):] or 0)
    raise ValueError(noise)


def sampler(y_train, noise, clip=False):
    """Noise drawer with rates fixed from the (clean) training labels."""
    kind, v = parse(noise)
    n1 = float(y_train.sum()); n0 = len(y_train) - n1; sd = float(y_train.std())

    def draw(yb, rng):
        yb = np.asarray(yb, float)
        if kind == 'none':
            return yb
        if kind == 'flip':
            u = rng.random(len(yb))
            return np.where(np.where(yb == 1, u < v, u < v*n1/n0), 1 - yb, yb)
        t = yb + rng.normal(0, v*sd, len(yb))
        return np.clip(t, 0, 1) if clip else t
    return draw


def rf_member(X, y, Xv, noise, seed, groups=150, per=10):
    """1500 trees in groups of 10; each group sees its own noisy copy of the targets."""
    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    kind, _ = parse(noise)
    draw = sampler(y, noise)

    def one(g):
        t = draw(y, np.random.default_rng([seed, g]))
        kw = dict(n_estimators=per, min_samples_leaf=8, max_features=4, random_state=1000*seed + g, n_jobs=1)
        if kind == 'smear':
            return RandomForestRegressor(**kw).fit(X, t).predict(Xv)
        return RandomForestClassifier(**kw).fit(X, t.astype(int)).predict_proba(Xv)[:, 1]
    return np.mean(Parallel(n_jobs=NJ)(delayed(one)(g) for g in range(groups)), axis=0)


def _soft_rows(X, t):
    """Soft targets for a log-loss classifier: each row twice, as label 1 (weight t) and 0 (weight 1-t)."""
    t = np.clip(t, 0, 1)
    return pd.concat([X, X]), np.r_[np.ones(len(t)), np.zeros(len(t))].astype(int), np.r_[t, 1 - t]


def hgb_member(X, y, Xv, noise, seed, cat, K=5, max_iter=900):
    """Inner K-fold bag: noisy targets on K-1 parts; one common stopping iteration chosen on the
    clean held-out parts (mean inner AUROC curve); every bag member predicts at that iteration."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    kind, _ = parse(noise)
    draw = sampler(y, noise)
    curves, models = [], []
    for k, (a, b) in enumerate(StratifiedKFold(K, shuffle=True, random_state=seed).split(X, y)):
        t = draw(y[a], np.random.default_rng([seed, k]))
        Xa, ya, w = _soft_rows(X.iloc[a], t) if kind == 'smear' else (X.iloc[a], t.astype(int), None)
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=15, l2_regularization=1.0,
                                           min_samples_leaf=20, categorical_features='from_dtype' if cat else None,
                                           early_stopping=False, random_state=10*seed + k).fit(Xa, ya, sample_weight=w)
        curves.append([roc_auc_score(y[b], p[:, 1]) for p in m.staged_predict_proba(X.iloc[b])])
        models.append(m)
    best = int(np.argmax(np.mean(curves, axis=0)))
    preds = []
    for m in models:
        for i, p in enumerate(m.staged_predict_proba(Xv)):
            if i == best:
                preds.append(rank(p[:, 1]))
                break
    return np.mean(preds, axis=0), best + 1


def cat_member(X, y, Xv, noise, seed, K=5, max_iter=2000):
    """CatBoost depth 3 (v3 settings) in the same inner-bag / clean early-stopping scheme."""
    from catboost import CatBoostClassifier
    kind, _ = parse(noise)
    draw = sampler(y, noise, clip=True)
    cats = [c for c in X.columns if X[c].dtype == object]
    curves, models = [], []
    for k, (a, b) in enumerate(StratifiedKFold(K, shuffle=True, random_state=seed).split(X, y)):
        t = draw(y[a], np.random.default_rng([seed, k]))
        m = CatBoostClassifier(iterations=max_iter, depth=3, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=10*seed + k,
                               loss_function='CrossEntropy' if kind == 'smear' else 'Logloss', eval_metric='AUC',
                               use_best_model=False, thread_count=NJ, verbose=False, allow_writing_files=False)
        m.fit(X.iloc[a], t if kind == 'smear' else t.astype(int), cat_features=cats, eval_set=(X.iloc[b], y[b]))
        curves.append(m.get_evals_result()['validation']['AUC'])
        models.append(m)
    best = int(np.argmax(np.mean(curves, axis=0)))
    return np.mean([rank(m.predict_proba(Xv, ntree_end=best + 1)[:, 1]) for m in models], axis=0), best + 1


def fit_noisy(net, seq, st, y, seed, noise, epochs=80, patience=10, lr=1e-3, trunk_lr=None, oversample=3, batch=256, decay=0.95):
    """src.model.fit with noisy mini-batch targets; the inner validation split keeps clean labels
    and drives early stopping.  Noise uses its own generator, so noise=none equals model.fit."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.model import T, predict
    torch.manual_seed(seed); rng = np.random.default_rng(seed); nrng = np.random.default_rng([seed, 7])
    tr, va = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(st, y))
    val = (seq[va], st[va], y[va]); seq, st, y = seq[tr], st[tr], y[tr]
    draw = sampler(y, noise, clip=True)
    groups = [{'params': [p for n, p in net.named_parameters() if not n.startswith('trunk.')], 'lr': lr},
              {'params': net.trunk.parameters(), 'lr': trunk_lr or lr}]
    opt = torch.optim.Adam(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, decay)
    loss_fn = nn.BCEWithLogitsLoss()
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    best, state, wait, epoch = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        net.train()
        idx = np.concatenate([neg, rng.choice(pos, len(neg)//oversample, replace=True)]) if oversample else np.arange(len(y))
        rng.shuffle(idx)
        for b0 in range(0, len(idx), batch):
            j = idx[b0:b0+batch]
            opt.zero_grad()
            loss_fn(net(T(seq[j]), T(st[j])), T(draw(y[j], nrng).astype(np.float32))).backward()
            opt.step()
        sched.step()
        auc = roc_auc_score(val[2], predict(net, val[0], val[1]))  # clean labels
        if auc > best + 1e-5:
            best, state, wait = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state)
    return best, epoch


def cnn_inputs(d):
    from src.signal import CHANNELS  # noqa: F401
    pre = np.load(ROOT/'outputs/pretrain/scaler.npz')
    seq = ((d['sig'] - pre['seq_lo'])/np.maximum(pre['seq_hi'] - pre['seq_lo'], 1e-6)).astype(np.float32)
    return seq, d['stat'].to_numpy()


def cnn_member(seq, st, y, tr, va, noise, f):
    import torch
    from src.model import Net, Scaler, Trunk, predict
    from src.signal import CHANNELS
    sc = Scaler().fit(seq[tr], st[tr]); _, s_tr = sc(seq[tr], st[tr]); _, s_va = sc(seq[va], st[va])
    runs, eps = [], []
    for s in (0, 1, 2):
        trunk = Trunk(len(CHANNELS)); trunk.load_state_dict(torch.load(ROOT/'outputs/pretrain/trunk.pt'))
        net = Net(trunk, st.shape[1])
        _, ep = fit_noisy(net, seq[tr], s_tr, y[tr], seed=1000*f + s, noise=noise, trunk_lr=3e-4)
        runs.append(rank(predict(net, seq[va], s_va))); eps.append(ep)
    return np.mean(runs, axis=0), eps


def table(member, d):
    if member == 'rf':
        return FEATURE_SETS['base'](d).reset_index(drop=True)
    if member == 'hgb':
        return FEATURE_SETS['base'](d).reset_index(drop=True)
    if member == 'cat':
        return FEATURE_SETS['allview'](d).reset_index(drop=True)
    tab = FEATURE_SETS['allview_new'](d).reset_index(drop=True).copy()  # hgbcat, as in iterate.evaluate
    for c in [c for c in tab.columns if tab[c].dtype == object]:
        counts = tab[c].value_counts()  # label-free pooling of rare values
        tab[c] = tab[c].where(tab[c].map(counts) >= 10, 'rare').astype('category')
    return tab


def run(member, noise, schemes, tag):
    d = prepare(); ids, y = d['ids'], d['y']
    tab = None if member == 'cnn' else table(member, d)
    if member == 'cnn':
        seq, st = cnn_inputs(d)
    res, oofs, info = {}, {}, {}
    for scheme in schemes:
        fold = folds(scheme, ids, y)
        oof = np.zeros(len(y)); t0 = time.time()
        for f in range(1, 6):
            tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
            if member == 'rf':
                oof[va] = rf_member(tab.iloc[tr], y[tr], tab.iloc[va], noise, f)
            elif member in ('hgb', 'hgbcat'):
                oof[va], info[f'{scheme}_{f}'] = hgb_member(tab.iloc[tr], y[tr], tab.iloc[va], noise, f, cat=member == 'hgbcat')
            elif member == 'cat':
                oof[va], info[f'{scheme}_{f}'] = cat_member(tab.iloc[tr], y[tr], tab.iloc[va], noise, f)
            else:
                oof[va], info[f'{scheme}_{f}'] = cnn_member(seq, st, y, tr, va, noise, f)
        res[scheme] = [roc_auc_score(y[fold == q], oof[fold == q]) for q in range(1, 6)]
        oofs[scheme] = oof
        print(f'{member}:{noise} {scheme} {np.mean(res[scheme]):.5f} ({time.time() - t0:.0f}s)', flush=True)
    allf = np.concatenate(list(res.values()))
    print(f'{tag}{member}:{noise} ALL={allf.mean():.5f} sd={allf.std():.4f}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{tag}{member}_{noise}" + ('' if tuple(schemes) in (SELECT, CONFIRM) else '_' + '-'.join(schemes))
    (OUT/f'{name}.json').write_text(json.dumps({'member': member, 'noise': noise, 'folds': res, 'stops': info}, indent=1))
    pd.DataFrame(oofs).to_csv(OUT/f'{name}_oof.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--member', required=True, choices=['rf', 'hgb', 'hgbcat', 'cat', 'cnn'])
    p.add_argument('--noise', required=True)
    p.add_argument('--schemes', default='')
    p.add_argument('--confirm', action='store_true')
    p.add_argument('--jobs', type=int, default=5)
    a = p.parse_args()
    NJ = a.jobs
    schemes = tuple(a.schemes.split(',')) if a.schemes else (CONFIRM if a.confirm else SELECT)
    run(a.member, a.noise, schemes, 'confirm_' if a.confirm else '')
