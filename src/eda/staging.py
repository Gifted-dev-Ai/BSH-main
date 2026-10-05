"""Degradation-**stage** classification for Plasma defects.

Predicting the exact remaining-useful-life in hours is unreliable here (only 15
events; the honest grouped-CV RUL barely beats predict-the-mean). A coarser, more
robust target is the **health stage**: bucket the time-to-next-onset into ordered
maintenance categories and classify *those* instead.

================  ==============================  ==================================
Stage             RUL (hours to next onset)       Meaning / action
================  ==============================  ==================================
Normal            > 168 h  (> 7 d)                healthy operation
Degrading         72 – 168 h  (3 – 7 d)           early drift; schedule a look
Critical          24 – 72 h  (1 – 3 d)            plan intervention
Failure Imminent  0 – 24 h  (< 1 d)               act now
Failed            inside the defect/down window   machine is down
================  ==============================  ==================================

Features are the same windowed Nozzle/MES summary statistics as the RUL section.
Validation is **GroupKFold** over whole run-to-failure segments (a segment and its
trailing failed window share a group), so no autocorrelated window leaks between
train and test. Because the stages are *ordinal* we also report the within-±1-stage
accuracy — the metric that matters when "Critical vs Failure-Imminent" is still an
actionable call. Results cached to ``data/interim/staging_plasma.pkl``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import early_factors as ef

INTERIM = ef.INTERIM
CACHE = INTERIM / "staging_plasma.pkl"

STAGES = ["Normal", "Degrading", "Critical", "Failure Imminent", "Failed"]
STAGE_EDGES_H = (168, 72, 24)        # Normal | Degrading | Critical | Imminent edges
STAGE_RUL_LABEL = ["> 168 h", "72–168 h", "24–72 h", "0–24 h", "in down-window"]
# per-model summary metrics stored in the cached result (the report reads every one of
# these). Part of the cache fingerprint so a cache written before a metric was added
# self-invalidates instead of silently KeyError-ing the whole section.
MODEL_METRIC_KEYS = ("acc", "within1", "bal_acc", "macro_f1", "qwk", "mcc", "stage_mae")
DEFAULT_OBS = 5
GRAN_H = 4
N_SPLITS = 5
SEED = 7

# Focused short-lead alarm: the one configuration that generalises out-of-sample.
# A short observation window of MES sensor/total error counts, collapsed to the
# actionable binary "act now" (≤ 24 h to onset, or already in the down-window).
ALARM_OBS_D = 0.25                       # 6 h window
ALARM_THR_STAGE = 3                      # Failure Imminent or Failed
ALARM_FEATURE_KEYS = ("MES_sensor", "MES_total")


# --------------------------------------------------------------------------- #
# Stage schemes — the report runs both the fine 5-stage split and a coarser 3-stage
# one. A scheme is just the ordered stage names + the RUL edges that bucket the
# hours-to-next-onset; the last stage always absorbs the in-down-window points
# ("Failed" when it is its own class, else the top "Critical" bucket).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageScheme:
    key: str                 # cache key / id suffix
    cache_name: str          # interim cache filename
    stages: tuple            # ordered stage names (index 0 = healthiest)
    edges_h: tuple           # descending RUL edges partitioning the non-event points
    rul_labels: tuple        # per-stage RUL label (for the report table)
    meanings: tuple          # per-stage action text (for the report table)
    colors: tuple            # per-stage colour (for the report charts)
    alarms: tuple            # ((label, thr_stage), …) coarse "approaching failure" AUCs
    alarm_thr_stage: int     # focused-alarm "act now" collapse (stage ≥ this)
    focused: bool            # whether to compute the MES-only focused alarm

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def as_dict(self) -> dict:
        return {"key": self.key, "stages": list(self.stages),
                "edges_h": list(self.edges_h), "rul_labels": list(self.rul_labels),
                "meanings": list(self.meanings), "colors": list(self.colors),
                "alarms": [list(a) for a in self.alarms]}


# Fine 5-stage split (the original): Normal | Degrading | Critical | Imminent | Failed.
SCHEME_5 = StageScheme(
    key="5stage", cache_name="staging_plasma.pkl",
    stages=tuple(STAGES), edges_h=STAGE_EDGES_H, rul_labels=tuple(STAGE_RUL_LABEL),
    meanings=("healthy operation", "early drift — schedule a look",
              "plan intervention", "act now", "machine is down"),
    colors=("#00b3a4", "#ffd166", "#ff9f1c", "#e2001a", "#5a1a1a"),
    alarms=(("le72h", 2), ("le24h", 3)), alarm_thr_stage=3, focused=True)

# Coarse 3-stage split: Normal (>7 d) | Degrading (1–7 d) | Critical (≤24 h or down).
# The in-down-window points fold into Critical (no separate "Failed" class).
SCHEME_3 = StageScheme(
    key="3stage", cache_name="staging_plasma_3stage.pkl",
    stages=("Normal", "Degrading", "Critical"), edges_h=(168, 24),
    rul_labels=("> 168 h", "24–168 h", "≤ 24 h or in down-window"),
    meanings=("healthy operation", "early drift — watch / plan",
              "act now — imminent or down"),
    colors=("#00b3a4", "#ff9f1c", "#e2001a"),
    alarms=(("le168h", 1), ("le24h", 2)), alarm_thr_stage=2, focused=False)

SCHEMES = {SCHEME_5.key: SCHEME_5, SCHEME_3.key: SCHEME_3}


def candidate_classifiers() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier,
                                   HistGradientBoostingClassifier)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "Logistic (L2)": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)),
        "Random forest": RandomForestClassifier(
            n_estimators=350, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=SEED, n_jobs=1),
        "Hist GBM": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            class_weight="balanced", random_state=SEED),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, class_weight="balanced", random_state=SEED,
            n_jobs=1, verbose=-1)
    except Exception:
        pass
    return models


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _stage_labels(tt: np.ndarray, in_event: np.ndarray,
                  scheme: StageScheme = SCHEME_5) -> np.ndarray:
    """Map hours-to-next-onset (+ in-event flag) to an ordinal stage 0..n_stages-1.

    The descending ``edges_h`` bucket the non-event points into the healthier stages;
    the in-down-window points always take the last stage ("Failed" when the scheme has
    a dedicated one, else the top "Critical" bucket).
    """
    y = np.zeros(len(tt), dtype=int)                 # stage 0 = healthiest (Normal)
    for i, edge in enumerate(scheme.edges_h):
        y = np.where(tt <= edge, i + 1, y)
    y[in_event] = scheme.n_stages - 1
    return y


def build_stage_xy(obs_days: int = DEFAULT_OBS, gran_h: int = GRAN_H,
                   group: str = "Plasma", scheme: StageScheme = SCHEME_5):
    """Feature frame + ordinal stage label + run-to-failure group id."""
    events = ef.defect_events(group)
    onset = ef.defect_onsets(events).reset_index(drop=True)
    end = events["end"].dt.tz_convert("UTC").dt.tz_localize(None).reset_index(drop=True)
    ev = pd.DataFrame({"onset": onset, "end": end}).sort_values("onset").reset_index(drop=True)
    onset_arr = ev["onset"].values.astype("datetime64[ns]")
    end_arr = np.maximum(ev["end"].values.astype("datetime64[ns]"), onset_arr)

    nozzle, mes = ef.hourly_nozzle(), ef.hourly_mes()
    signals = sorted({c.rsplit("__", 1)[0] for c in nozzle.columns})
    cmin, cmax = nozzle.index.min(), nozzle.index.max()
    cov_start = cmin.tz_localize(None) if getattr(cmin, "tz", None) else cmin
    cov_end = cmax.tz_localize(None) if getattr(cmax, "tz", None) else cmax
    pts = pd.date_range(cov_start, cov_end, freq=f"{gran_h}h")
    pv = pts.values.astype("datetime64[ns]")

    # "Failed" = inside a defect/down window [onset, end]; tag membership + event id
    in_event = np.zeros(len(pv), dtype=bool)
    ev_idx = np.full(len(pv), -1, dtype=int)
    for i in range(len(onset_arr)):
        m = (pv >= onset_arr[i]) & (pv <= end_arr[i])
        in_event |= m
        ev_idx[m] = i
    # time-to-next-onset and the segment leading to it
    nxt = np.searchsorted(onset_arr, pv, side="left")
    has_next = nxt < len(onset_arr)
    tt = np.full(len(pv), np.inf)
    tt[has_next] = (onset_arr[nxt[has_next]] - pv[has_next]) / np.timedelta64(1, "h")

    y = _stage_labels(tt, in_event, scheme)
    groups = np.where(has_next, nxt, len(onset_arr))     # sentinel for far-normal tail
    groups[in_event] = ev_idx[in_event]

    roll = ef._rolling_features(nozzle, signals, obs_days)
    end_h = pts.floor("h")
    cols = {}
    for stat in ef.SIGNAL_STATS:
        for s in signals:
            cols[f"{s}·{stat}"] = roll[stat][s].reindex(end_h).to_numpy()
    for s in ef.MES_SIGNALS:
        if f"{s}__count" in mes.columns:
            cols[f"{s}·count"] = ef._rolling_count(
                mes, f"{s}__count", obs_days).reindex(end_h).to_numpy()
    feat_cols = list(cols)
    cols["__y"], cols["__grp"] = y, groups
    X = pd.DataFrame(cols)
    X[feat_cols] = X[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.dropna(subset=feat_cols).reset_index(drop=True)
    # winsorize to [0.1, 99.9] pct: caps the numerically-unstable skew/kurt tails
    # (near-constant windows) that overflow float32 in tree models, leaving 99.8%
    # of values untouched.
    q = X[feat_cols].quantile([0.001, 0.999])
    X[feat_cols] = X[feat_cols].clip(lower=q.loc[0.001], upper=q.loc[0.999], axis=1)
    return X, feat_cols


# --------------------------------------------------------------------------- #
# Evaluation (grouped CV)
# --------------------------------------------------------------------------- #
def _cv_predict(model, X, y, groups, method="predict"):
    from sklearn.model_selection import GroupKFold, cross_val_predict
    import warnings
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(model, X, y, groups=groups, method=method,
                                 cv=GroupKFold(n_splits=n_splits), n_jobs=1)


def _alarm_auc(model, X, y, groups, thr_stage: int) -> float:
    """Grouped-CV ROC-AUC for the binary collapse 'stage >= thr_stage'."""
    from sklearn.metrics import roc_auc_score
    b = (y >= thr_stage).astype(int)
    if b.sum() < 5 or (1 - b).sum() < 5:
        return float("nan")
    try:
        proba = _cv_predict(model, X, b, groups, method="predict_proba")[:, 1]
        return float(roc_auc_score(b, proba))
    except Exception:
        return float("nan")


def _metrics(y, pred, present, scheme: StageScheme = SCHEME_5):
    from sklearn.metrics import (balanced_accuracy_score, f1_score, cohen_kappa_score,
                                 matthews_corrcoef, precision_recall_fscore_support,
                                 confusion_matrix)
    labels = list(range(scheme.n_stages))
    acc = float((pred == y).mean())
    within1 = float((np.abs(pred - y) <= 1).mean())          # ordinal tolerance
    bal = float(balanced_accuracy_score(y, pred))
    macro_f1 = float(f1_score(y, pred, labels=present, average="macro"))
    # ordinal- / imbalance-aware summaries
    qwk = float(cohen_kappa_score(y, pred, weights="quadratic"))   # ordinal agreement
    mcc = float(matthews_corrcoef(y, pred))                        # imbalance-robust
    stage_mae = float(np.abs(pred - y).mean())                     # mean ordinal distance
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return {"acc": acc, "within1": within1, "bal_acc": bal, "macro_f1": macro_f1,
            "qwk": qwk, "mcc": mcc, "stage_mae": stage_mae,
            "precision": pr.tolist(), "recall": rc.tolist(), "f1": f1.tolist(),
            "support": sup.tolist(), "confusion": cm.tolist()}


def _binary_grouped_auc(Xf, b, groups, curve: bool = False):
    """Best-of-panel grouped-CV ROC-AUC for a binary target (+ optional ROC/op-point)."""
    from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                                 precision_score, recall_score, average_precision_score)
    best = None
    for name, model in candidate_classifiers().items():
        try:
            proba = _cv_predict(model, Xf, b, groups, method="predict_proba")[:, 1]
            auc = roc_auc_score(b, proba)
        except Exception:
            continue
        if best is None or auc > best[0]:
            best = (auc, name, proba)
    if best is None:
        return None
    auc, name, proba = best
    # average precision is the imbalance-honest summary (baseline = prevalence)
    res = {"auc": float(auc), "model": name,
           "ap": float(average_precision_score(b, proba)),
           "prevalence": float(b.mean())}
    if curve:
        fpr, tpr, thr = roc_curve(b, proba)
        res["fpr"], res["tpr"] = fpr.tolist(), tpr.tolist()
        k = int(np.argmax(tpr - fpr))                    # Youden operating point
        pred = (proba >= thr[k]).astype(int)
        res["op"] = {"tpr": float(tpr[k]), "fpr": float(fpr[k]),
                     "precision": float(precision_score(b, pred, zero_division=0)),
                     "recall": float(recall_score(b, pred, zero_division=0)),
                     "confusion": confusion_matrix(b, pred).tolist()}
    return res


def _focused_alarm(Xf_all, y_all, groups_all, scheme: StageScheme = SCHEME_5) -> dict:
    """Short-window MES-only alarm vs the kitchen-sink baseline, same binary target.

    The "act now" collapse is ``stage >= scheme.alarm_thr_stage`` — i.e. ≤24 h to onset
    or already in the down-window — which is the *same* binary target in every scheme.
    """
    thr = scheme.alarm_thr_stage
    Xs, fcs = build_stage_xy(obs_days=ALARM_OBS_D, scheme=scheme)
    acols = [c for c in fcs if any(k in c for k in ALARM_FEATURE_KEYS)]
    ys, gs = Xs["__y"].to_numpy(), Xs["__grp"].to_numpy()
    bs = (ys >= thr).astype(int)
    focused = _binary_grouped_auc(Xs[acols].to_numpy(), bs, gs, curve=True)
    # baseline: all features over the long (5 d) window, identical target
    baseline = _binary_grouped_auc(Xf_all, (y_all >= thr).astype(int),
                                   groups_all, curve=True)
    return {"obs_d": ALARM_OBS_D, "feature_keys": list(ALARM_FEATURE_KEYS),
            "thr_stage": thr, "n_features": len(acols),
            "npos": int(bs.sum()), "nneg": int((bs == 0).sum()),
            "focused": focused, "baseline": baseline}


def compute(obs_days: int = DEFAULT_OBS, gran_h: int = GRAN_H,
            refresh: bool = False, scheme: StageScheme = SCHEME_5) -> dict:
    """Grouped-CV stage classification across the model panel, for one scheme (cached)."""
    X, feat_cols = build_stage_xy(obs_days, gran_h, scheme=scheme)
    cache = INTERIM / scheme.cache_name
    feat_sig = (f"scheme{scheme.key}|nstg{scheme.n_stages}|obs{obs_days}|gran{gran_h}"
                f"|edges{scheme.edges_h}|nfeat{len(feat_cols)}"
                f"|alarm{ALARM_OBS_D},{scheme.alarm_thr_stage},{ALARM_FEATURE_KEYS}"
                f"|alarms{scheme.alarms}|foc{scheme.focused}"
                f"|metrics{','.join(MODEL_METRIC_KEYS)}")
    if cache.exists() and not refresh:
        try:
            cached = pickle.loads(cache.read_bytes())
            if cached.get("fingerprint") == feat_sig:
                return cached
        except Exception:
            pass

    y = X["__y"].to_numpy()
    groups = X["__grp"].to_numpy()
    Xf = X[feat_cols].to_numpy()
    present = sorted(np.unique(y).tolist())

    # majority-class baseline (predict the most common stage everywhere)
    maj = int(np.bincount(y).argmax())
    base = _metrics(y, np.full_like(y, maj), present, scheme)

    models, results = candidate_classifiers(), []
    best_name, best_score, best_cm, best_per, best_model = None, -1.0, None, None, None
    for name, model in models.items():
        try:
            pred = _cv_predict(model, Xf, y, groups)
        except Exception:
            continue
        m = _metrics(y, pred, present, scheme)
        results.append({"model": name, **{k: m[k] for k in MODEL_METRIC_KEYS}})
        if m["qwk"] > best_score:          # QWK: ordinal- and imbalance-aware selector
            best_name, best_score = name, m["qwk"]
            best_cm, best_per, best_model = m["confusion"], m, model

    # coarse "approaching failure" collapses (stage >= thr), with the best model
    alarm = {}
    if best_model is not None:
        alarm = {label: _alarm_auc(best_model, Xf, y, groups, thr)
                 for label, thr in scheme.alarms}

    # the MES-only focused alarm is scheme-independent (same binary act-now target), so
    # only the default scheme computes/renders it; coarser schemes reuse that story.
    focused_alarm = _focused_alarm(Xf, y, groups, scheme) if scheme.focused else {}

    stage_counts = np.bincount(y, minlength=scheme.n_stages).tolist()
    result = {
        "fingerprint": feat_sig, "scheme": scheme.as_dict(),
        "obs_days": obs_days, "gran_h": gran_h,
        "n_features": len(feat_cols), "n_samples": int(len(y)),
        "stage_counts": stage_counts, "models": results, "alarm": alarm,
        "focused_alarm": focused_alarm,
        "baseline": {k: base[k] for k in MODEL_METRIC_KEYS},
        "best": {"model": best_name, "confusion": best_cm,
                 "precision": best_per["precision"], "recall": best_per["recall"],
                 "f1": best_per["f1"], "support": best_per["support"],
                 "acc": best_per["acc"], "within1": best_per["within1"],
                 "bal_acc": best_per["bal_acc"], "macro_f1": best_per["macro_f1"],
                 "qwk": best_per["qwk"], "mcc": best_per["mcc"],
                 "stage_mae": best_per["stage_mae"]},
    }
    INTERIM.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(result))
    return result
