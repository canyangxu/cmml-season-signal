"""Season-signal network: three parallel Conv1d branches (kernels 2, 3, 5) with max,
mean and last-season pooling, fused with a dense branch for static features.

Training recipe (from the user's ECG 1D-CNN project): Min-Max scaling fitted on
training rows, random oversampling of buyers, exponential learning-rate decay and
early stopping on an inner validation split carved from the training rows only.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

torch.set_num_threads(1)
try:
    torch.backends.nnpack.set_flags(False)  # tiny 1D convs: direct kernels beat FFT tiles
except Exception:
    pass


class Trunk(nn.Module):
    def __init__(self, channels, width=32, kernels=(2, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(channels, width, k, padding='same'), nn.ReLU(),
                                                     nn.Conv1d(width, width, k, padding='same'), nn.ReLU())
                                       for k in kernels])
        self.out_dim = 3*width*len(kernels)

    def forward(self, x):
        parts = []
        for branch in self.branches:
            h = branch(x)
            parts += [h.amax(-1), h.mean(-1), h[..., -1]]  # local peaks, global trend, latest season
        return torch.cat(parts, 1)


class Net(nn.Module):
    def __init__(self, trunk, static_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.trunk = trunk
        self.static = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(trunk.out_dim + 32, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, seq, st):
        return self.head(torch.cat([self.trunk(seq), self.static(st)], 1)).squeeze(-1)


class Scaler:
    """Min-Max scaling fitted on training rows (per signal channel, per static column)."""
    def fit(self, seq, st):
        self.seq_lo, self.seq_hi = seq.min(axis=(0, 2), keepdims=True), seq.max(axis=(0, 2), keepdims=True)
        self.st_lo, self.st_hi = st.min(0, keepdims=True), st.max(0, keepdims=True)
        return self

    def __call__(self, seq, st):
        f = lambda a, lo, hi: ((a - lo)/np.maximum(hi - lo, 1e-6)).astype(np.float32)
        return f(seq, self.seq_lo, self.seq_hi), f(st, self.st_lo, self.st_hi)


T = lambda a: torch.tensor(a, dtype=torch.float32)


def predict(net, seq, st, batch=2048):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(T(seq[i:i+batch]), T(st[i:i+batch])).numpy() for i in range(0, len(seq), batch)])


def fit(net, seq, st, y, seed, epochs=80, patience=10, lr=1e-3, trunk_lr=None, oversample=3, batch=256,
        decay=0.95, val=None):
    """Train with inner early stopping.  `val` = (seq, st, y) overrides the inner split."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if val is None:
        tr, va = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(st, y))
        val = (seq[va], st[va], y[va]); seq, st, y = seq[tr], st[tr], y[tr]
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
        for b in range(0, len(idx), batch):
            j = idx[b:b+batch]
            opt.zero_grad()
            loss_fn(net(T(seq[j]), T(st[j])), T(y[j].astype(np.float32))).backward()
            opt.step()
        sched.step()
        auc = roc_auc_score(val[2], predict(net, val[0], val[1]))
        if auc > best + 1e-5:
            best, state, wait = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state)
    return best, epoch
