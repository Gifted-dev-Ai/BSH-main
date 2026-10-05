"""Deep sequence models on the **raw** Plasma Nozzle-Head signal stream.

The AutoML / RUL sections collapse each observation window into summary statistics
(min/max/mean/std/skew/kurt/range). Here we instead feed the model the *raw time-
ordered sequence* of the 18 nozzle signals — one hourly-resampled reading per step —
and let recurrent / convolutional networks learn the temporal pattern directly.

Pipeline
--------
1. **Resample** the cached hourly aggregates to a regular 1 h grid; each step is the
   hourly *mean* of every signal (a genuine reading, not a window statistic). Robust
   per-channel z-scoring (median / IQR).
2. **Windows** of ``SEQ_LEN_H`` hours (default 3 d) ending at the cutoff
   ``onset − lead``. Positives = the 15 defect onsets (augmented by jittering the
   window end); negatives = normal grid points away from any defect.
3. **Models** (torch, CPU): LSTM, GRU and a 1-D CNN. Two heads:
   * **classification** — pre-defect vs normal at several lead times (vs the AutoML
     ROC-AUC envelope);
   * **RUL regression** — hours-to-next-onset (vs the tabular RUL baseline).
   Scored by grouped / stratified-grouped cross-validation (whole onsets held out).
4. **Explainability** — channel permutation importance, input-gradient temporal
   saliency (which hours × channels drive the alarm) and **motif discovery** (the
   recurring sub-sequence shape that best marks the lead-up to a defect).

Everything is cached to ``data/interim/seq_models_plasma.pkl``; the report just
renders. Exploratory: only 15 positive events, so treat the numbers as indicative.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import early_factors as ef

INTERIM = ef.INTERIM
CACHE = INTERIM / "seq_models_plasma.pkl"

# --- configuration ---------------------------------------------------------- #
SEQ_LEN_H = 72                 # input window length (hourly steps) = 3 days
SEQ_LEADS_H = (6, 24, 72)      # classification lead times (hours before onset)
SEQ_GRAN_H = 6                 # spacing of normal / RUL sample points
RUL_HORIZON_H = 336            # RUL ceiling = 14 days
POS_JITTER_H = (0, -4, -8, 4, 8)   # augment each onset into several pre-defect windows
N_NORMAL_CAP = 500             # max normal windows
N_SPLITS = 4                   # CV folds
MOTIF_LEN = 12                 # motif sub-sequence length (hours)
SEED = 7

MODELS = ("LSTM", "GRU", "1D-CNN")
_KIND = {"LSTM": "lstm", "GRU": "gru", "1D-CNN": "cnn"}

# training hyper-parameters (kept small — tiny positive class, CPU, full-batch)
_EPOCHS = 80
_PATIENCE = 10
_HIDDEN = 40
_LR = 1e-3
_WD = 1e-4


# --------------------------------------------------------------------------- #
# Raw sequence tensor
# --------------------------------------------------------------------------- #
def _standardized_series():
    """(arr[T, C] z-scored, hourly DatetimeIndex, channel names). Cached per call."""
    noz = ef.hourly_nozzle()
    if getattr(noz.index, "tz", None) is not None:
        noz = noz.copy()
        noz.index = noz.index.tz_localize(None)
    signals = sorted({c.rsplit("__", 1)[0] for c in noz.columns})
    idx = pd.date_range(noz.index.min(), noz.index.max(), freq="h")
    means = pd.DataFrame(index=idx)
    for s in signals:
        cnt = noz[f"{s}__count"].reindex(idx)
        means[s] = (noz[f"{s}__sum"].reindex(idx) / cnt.replace(0, np.nan))
    # impute gaps (downtime hours) then robust z-score per channel
    means = means.ffill().bfill()
    means = means.fillna(means.median(numeric_only=True))
    med = means.median()
    iqr = (means.quantile(0.75) - means.quantile(0.25)).replace(0, np.nan)
    scale = (iqr / 1.349).fillna(means.std().replace(0, 1.0)).replace(0, 1.0)
    arr = ((means - med) / scale).to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, idx, signals


def _windows(end_times, arr, time_index, L):
    """Stack the L-hour window ending at each end time; drop ends without full history."""
    pos = time_index.get_indexer(pd.DatetimeIndex(end_times).floor("h"))
    out, keep = [], []
    for i, p in enumerate(pos):
        if p >= L - 1:
            out.append(arr[p - L + 1: p + 1])
            keep.append(i)
    X = (np.asarray(out, dtype=np.float32) if out
         else np.empty((0, L, arr.shape[1]), np.float32))
    return X, np.asarray(keep, dtype=int)


def _cov_bounds(idx):
    return idx.min(), idx.max()


def build_classification(lead_h: int, L: int = SEQ_LEN_H):
    """X[n, L, C], y, groups for pre-defect (1) vs normal (0) at one lead time."""
    arr, idx, signals = _standardized_series()
    events = ef.defect_events("Plasma")
    onsets = ef.defect_onsets(events).reset_index(drop=True)
    lead = pd.Timedelta(hours=lead_h)

    pos_ends, pos_grp = [], []
    for gi, on in onsets.items():
        for j in POS_JITTER_H:
            pos_ends.append(on - lead - pd.Timedelta(hours=j))
            pos_grp.append(gi)
    cov_start, cov_end = _cov_bounds(idx)
    npts = ef.normal_points(events, SEQ_GRAN_H, cov_start, cov_end)
    neg_ends = pd.DatetimeIndex(npts) - lead

    Xp, kp = _windows(pos_ends, arr, idx, L)
    Xn, kn = _windows(neg_ends, arr, idx, L)
    pos_grp = np.asarray(pos_grp)[kp]
    # cap & group negatives by ISO week so adjacent overlapping windows stay together
    neg_ends_kept = neg_ends[kn]
    if len(Xn) > N_NORMAL_CAP:
        sel = np.sort(np.random.default_rng(SEED).choice(len(Xn), N_NORMAL_CAP, False))
        Xn, neg_ends_kept = Xn[sel], neg_ends_kept[sel]
    neg_grp = 1000 + (neg_ends_kept.isocalendar().week.to_numpy()
                      + 53 * (neg_ends_kept.year.to_numpy() - 2025))

    X = np.concatenate([Xp, Xn], axis=0)
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]).astype(np.float32)
    groups = np.concatenate([pos_grp, neg_grp]).astype(int)
    return X, y, groups, signals


def build_rul(horizon_h: int = RUL_HORIZON_H, L: int = SEQ_LEN_H):
    """X[n, L, C], y (hours-to-onset, clamped), groups (run-to-failure segment)."""
    arr, idx, signals = _standardized_series()
    events = ef.defect_events("Plasma")
    onsets = np.sort(ef.defect_onsets(events).values.astype("datetime64[ns]"))
    cov_start, cov_end = _cov_bounds(idx)
    pts = pd.date_range(cov_start, cov_end, freq=f"{SEQ_GRAN_H}h")
    X, keep = _windows(pts, arr, idx, L)
    pv = pts[keep].values.astype("datetime64[ns]")
    nxt = np.searchsorted(onsets, pv, side="left")
    has_next = nxt < len(onsets)
    X, pv, nxt = X[has_next], pv[has_next], nxt[has_next]
    tt = (onsets[nxt] - pv) / np.timedelta64(1, "h")
    ramp = tt < horizon_h                       # only the degradation ramp
    X, y, groups = X[ramp], np.minimum(tt[ramp], horizon_h).astype(np.float32), nxt[ramp]
    return X, y, groups, signals


# --------------------------------------------------------------------------- #
# Torch models
# --------------------------------------------------------------------------- #
# torch 2.3 (built against NumPy 1.x) cannot bridge NumPy 2.x arrays via
# ``from_numpy``/``.numpy()``; route conversions through a raw buffer instead.
def _to_t(a, requires_grad: bool = False):
    import torch
    a = np.ascontiguousarray(a, dtype=np.float32)
    t = torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).view(*a.shape)
    return t.requires_grad_(True) if requires_grad else t


def _to_np(t):
    return np.array(t.detach().contiguous().flatten().tolist(),
                    dtype=np.float32).reshape(tuple(t.shape))


def _make_model(kind: str, C: int):
    import torch.nn as nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind in ("lstm", "gru"):
                rnn = nn.LSTM if kind == "lstm" else nn.GRU
                self.rnn = rnn(C, _HIDDEN, batch_first=True)
            else:
                self.conv = nn.Sequential(
                    nn.Conv1d(C, _HIDDEN, 5, padding=2), nn.ReLU(),
                    nn.Conv1d(_HIDDEN, _HIDDEN, 5, padding=2), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(
                nn.Linear(_HIDDEN, _HIDDEN), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(_HIDDEN, 1))

        def forward(self, x):                    # x: [B, L, C]
            if self.kind in ("lstm", "gru"):
                out, _ = self.rnn(x)
                h = out[:, -1, :]
            else:
                h = self.conv(x.transpose(1, 2)).squeeze(-1)
            return self.head(h).squeeze(-1)

    return SeqNet()


def _fit(kind, Xtr, ytr, C, task, pos_weight=None, seed=SEED):
    """Train one model with a small internal validation split for early stopping."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    perm = rng.permutation(n)
    nval = max(1, int(0.15 * n))
    vi, ti = perm[:nval], perm[nval:]
    Xt = _to_t(Xtr[ti]); yt = _to_t(ytr[ti])
    Xv = _to_t(Xtr[vi]); yv = _to_t(ytr[vi])

    model = _make_model(kind, C)
    opt = torch.optim.Adam(model.parameters(), lr=_LR, weight_decay=_WD)
    if task == "clf":
        pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.MSELoss()

    best, best_state, bad = float("inf"), None, 0
    for _ in range(_EPOCHS):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xv), yv))
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= _PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(model, X):
    import torch
    with torch.no_grad():
        return _to_np(model(_to_t(X)))


# --------------------------------------------------------------------------- #
# Cross-validated evaluation
# --------------------------------------------------------------------------- #
def _cv_classification(kind, X, y, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    n_pos_grp = len(np.unique(groups[y == 1]))
    n_splits = max(2, min(N_SPLITS, n_pos_grp))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
            continue
        pw = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
        m = _fit(kind, X[tr], y[tr], X.shape[2], "clf", pos_weight=pw)
        oof[te] = 1 / (1 + np.exp(-_predict(m, X[te])))
    ok = np.isfinite(oof)
    if ok.sum() < 5 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], oof[ok]))


def _cv_rul(kind, X, y, groups, horizon):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error
    n_splits = max(2, min(N_SPLITS, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = _fit(kind, X[tr], (y[tr] / horizon), X.shape[2], "reg")
        oof[te] = np.clip(_predict(m, X[te]) * horizon, 0, horizon)
    ok = np.isfinite(oof)
    mae = float(mean_absolute_error(y[ok], oof[ok]))
    base = float(mean_absolute_error(y[ok], np.full(ok.sum(), y[ok].mean())))
    return mae, base


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def _perm_importance(model, X, y, signals, task, horizon=None, n_repeat=4):
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    rng = np.random.default_rng(SEED)
    if task == "clf":
        base = roc_auc_score(y, 1 / (1 + np.exp(-_predict(model, X))))
    else:
        base = mean_absolute_error(y, np.clip(_predict(model, X) * horizon, 0, horizon))
    imp = []
    for c in range(X.shape[2]):
        drops = []
        for _ in range(n_repeat):
            Xp = X.copy()
            Xp[:, :, c] = X[rng.permutation(len(X)), :, c]
            if task == "clf":
                s = roc_auc_score(y, 1 / (1 + np.exp(-_predict(model, Xp))))
                drops.append(base - s)               # AUC lost
            else:
                s = mean_absolute_error(y, np.clip(_predict(model, Xp) * horizon, 0, horizon))
                drops.append(s - base)               # MAE gained
        imp.append(float(np.mean(drops)))
    order = np.argsort(imp)[::-1]
    return [(signals[i], imp[i]) for i in order]


def _saliency(model, Xpos):
    """Mean |d sigmoid(logit) / d input| over positive windows -> [L, C]."""
    import torch
    x = _to_t(Xpos, requires_grad=True)
    torch.sigmoid(model(x)).sum().backward()
    g = np.abs(_to_np(x.grad))
    return g.mean(0), g                              # [L, C], [n, L, C]


def _motif(Xpos, Xneg, ch_idx, sal_per_sample, m=MOTIF_LEN):
    """Dominant pre-defect sub-sequence on the top channel + its defect/normal lift."""
    from sklearn.cluster import KMeans
    L = Xpos.shape[1]
    sal_c = sal_per_sample[:, :, ch_idx]             # [n, L]

    def _znorm(v):
        sd = v.std()
        return (v - v.mean()) / (sd + 1e-8)

    # for each positive window, the m-length slice with the most saliency mass
    subs = []
    win = np.ones(m)
    for i in range(len(Xpos)):
        mass = np.convolve(sal_c[i], win, mode="valid")
        st = int(np.argmax(mass))
        subs.append(_znorm(Xpos[i, st:st + m, ch_idx]))
    subs = np.asarray(subs)
    if len(subs) < 3:
        return None
    k = min(3, len(subs))
    km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(subs)
    big = np.argmax(np.bincount(km.labels_))
    centroid = km.cluster_centers_[big]

    def _min_dist(X):
        out = []
        for i in range(len(X)):
            series = X[i, :, ch_idx]
            best = np.inf
            for st in range(0, L - m + 1):
                d = np.linalg.norm(_znorm(series[st:st + m]) - centroid)
                if d < best:
                    best = d
            out.append(best)
        return np.asarray(out)

    dpos, dneg = _min_dist(Xpos), _min_dist(Xneg)
    thr = float(np.median(dpos))                     # "contains the motif" threshold
    p_pos = float((dpos <= thr).mean())
    p_neg = float((dneg <= thr).mean()) or 1e-3
    return {"centroid": centroid.tolist(), "thr": thr, "m": m,
            "p_pos": p_pos, "p_neg": p_neg, "lift": p_pos / p_neg}


# --------------------------------------------------------------------------- #
# Orchestration (cached)
# --------------------------------------------------------------------------- #
def _fingerprint(signals):
    return "|".join([
        f"L{SEQ_LEN_H}", f"leads{SEQ_LEADS_H}", f"gran{SEQ_GRAN_H}",
        f"hz{RUL_HORIZON_H}", f"jit{len(POS_JITTER_H)}", f"cap{N_NORMAL_CAP}",
        f"models{MODELS}", f"nsig{len(signals)}"])


def compute(refresh: bool = False) -> dict:
    """Run (or load) the full sequence-model study; returns a render-ready dict."""
    arr, idx, signals = _standardized_series()
    fp = _fingerprint(signals)
    if CACHE.exists() and not refresh:
        try:
            cached = pickle.loads(CACHE.read_bytes())
            if cached.get("fingerprint") == fp:
                return cached
        except Exception:
            pass

    import warnings
    import torch
    warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
    torch.manual_seed(SEED)

    # ---- classification: ROC-AUC per (model, lead) -------------------------
    clf_rows, clf_cache = [], {}
    for lead in SEQ_LEADS_H:
        X, y, groups, _ = build_classification(lead)
        clf_cache[lead] = (X, y, groups)
        for mname in MODELS:
            auc = _cv_classification(_KIND[mname], X, y, groups)
            clf_rows.append({"lead_h": lead, "model": mname, "auc": auc,
                             "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())})

    # ---- RUL: MAE per model ------------------------------------------------
    Xr, yr, gr, _ = build_rul()
    rul_rows = []
    for mname in MODELS:
        mae, base = _cv_rul(_KIND[mname], Xr, yr, gr, RUL_HORIZON_H)
        rul_rows.append({"model": mname, "mae": mae, "baseline_mae": base,
                         "n": int(len(yr))})

    # ---- explainability on the best classifier (best lead × model) ---------
    clf_df = pd.DataFrame(clf_rows)
    valid = clf_df.dropna(subset=["auc"])
    explain = {}
    if not valid.empty:
        best = valid.loc[valid["auc"].idxmax()]
        bl, bm = int(best.lead_h), best.model
        X, y, groups = clf_cache[bl]
        pw = float((y == 0).sum() / max(1, (y == 1).sum()))
        full = _fit(_KIND[bm], X, y, X.shape[2], "clf", pos_weight=pw)
        importance = _perm_importance(full, X, y, signals, "clf")
        sal_map, sal_samples = _saliency(full, X[y == 1])
        top_ch = importance[0][0]
        ch_idx = signals.index(top_ch)
        motif = _motif(X[y == 1], X[y == 0], ch_idx, sal_samples[:, :, :])
        explain = {"lead_h": bl, "model": bm, "auc": float(best.auc),
                   "importance": importance, "saliency": sal_map.tolist(),
                   "top_channel": top_ch, "motif": motif,
                   "L": SEQ_LEN_H, "signals": signals}

    result = {
        "fingerprint": fp, "signals": signals, "seq_len_h": SEQ_LEN_H,
        "gran_h": SEQ_GRAN_H, "rul_horizon_h": RUL_HORIZON_H,
        "n_channels": len(signals), "leads": list(SEQ_LEADS_H),
        "classification": clf_rows, "rul": rul_rows, "explain": explain,
    }
    INTERIM.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(result))
    return result
