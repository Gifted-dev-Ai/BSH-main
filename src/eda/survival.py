"""Survival-analysis reformulation of the Plasma run-to-failure segments.

The RUL section frames prognostics as **regression** — predict hours-to-onset, clamp
at a horizon. That discards two things survival analysis keeps: the healthy stretch at
the end of the record is **right-censored** (we know the machine survived ≥ T, not when
it will fail), and we often want a full **survival curve** rather than a single number.
Following the survival reformulation of run-to-failure data in *arXiv 2504.12156*, this
section reframes the same data as time-to-event with censoring and fits a
**Random Survival Forest** — a non-parametric model that captures complex survival
curves without the Cox proportional-hazards assumption.

Reformulation
-------------
Each grid point (every ``SURV_GRAN_H`` hours, the **same** windowed Nozzle/MES summary
features as the RUL section) becomes one right-censored survival observation:

* **time**  ``T``  = hours from the point to the **next defect onset**, OR — for the
  healthy stretch after the last onset — hours to the **end of signal coverage**.
* **event** ``δ``  = 1 if a defect onset is actually reached (uncensored), 0 if the
  record ends first (right-censored).
* **covariates** = the windowed statistics (min/max/mean/std/skew/kurt/range, plus the
  MES error counts), winsorised for numerical stability.

A *run-to-failure segment* (the operation leading to one onset) is the CV group, so the
trailing autocorrelated windows of one failure never leak between train and test.

Models (scikit-survival)
------------------------
* **Random Survival Forest** — non-parametric ensemble of log-rank survival trees; the
  primary model. Each leaf is a Nelson-Aalen cumulative hazard; the ensemble averages
  them. Captures non-linear, non-proportional hazards.
* **Cox proportional hazards** (ridge-penalised) — the classical semi-parametric
  baseline RSF is contrasted against; *assumes* proportional hazards.
* **Kaplan-Meier** — the population survival curve = the "predict-the-mean" baseline
  (concordance 0.5 by construction).

Evaluation is **GroupKFold over whole segments**, scored by Harrell's concordance
index, plus a naive-vs-grouped leakage contrast that mirrors the RUL demo. The
out-of-fold survival functions are reused for the risk-stratified KM curves and the
per-segment failure-probability countdowns (no extra leave-one-segment-out fits).
Results are cached to ``data/interim/survival_plasma.pkl``.

Exploratory: only 15 defect events and sparse, short-horizon censoring, so treat the
numbers as indicative — IPCW calibration metrics (integrated Brier / time-dependent
AUC) are *not* reported because the censoring distribution has too little support here.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import early_factors as ef
from . import rul

INTERIM = ef.INTERIM
CACHE = INTERIM / "survival_plasma.pkl"

# --- configuration ---------------------------------------------------------- #
SURV_OBS_DAYS = 5                 # observation-window length (matches RUL default)
SURV_GRAN_H = 4                   # spacing of survival sample points
PFAIL_HORIZON_H = 168             # "fail within next 7 d" probability for the countdown
N_SPLITS = 5                      # grouped-CV folds
SEED = 7
RSF_TREES = 150
RSF_MIN_LEAF = 15                 # large leaves: tiny event count, avoid overfit splits
IMP_TREES = 100                   # smaller forest for permutation importance
IMP_SUBSAMPLE = 1000              # rows scored when permuting (runtime cap)
N_DEFECT_CASES = 5
N_TERCILES = 3


def have_sksurv() -> bool:
    """True if scikit-survival is importable (the section degrades gracefully if not)."""
    try:
        import sksurv  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def candidate_models() -> dict:
    """Survival estimators. RSF is used bare (trees need no scaling and expose
    ``predict_survival_function``); Cox is standardised + ridge-penalised because the
    142 windowed features are collinear."""
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Random survival forest": RandomSurvivalForest(
            n_estimators=RSF_TREES, min_samples_leaf=RSF_MIN_LEAF,
            max_features="sqrt", random_state=SEED, n_jobs=-1),
        "Cox PH (ridge)": make_pipeline(
            StandardScaler(),
            CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow")),
    }


def _new_rsf(n_estimators: int = RSF_TREES):
    from sksurv.ensemble import RandomSurvivalForest
    return RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=RSF_MIN_LEAF,
                                max_features="sqrt", random_state=SEED, n_jobs=-1)


# --------------------------------------------------------------------------- #
# Dataset — reformulate the run-to-failure grid as right-censored survival data
# --------------------------------------------------------------------------- #
def build_surv(obs_days: int = SURV_OBS_DAYS, gran_h: int = SURV_GRAN_H,
               group: str = "Plasma") -> dict:
    """Right-censored survival dataset from the RUL run-to-failure grid.

    Reuses :func:`rul.build_xy` (same features, same segment groups). The trailing
    points with no future onset are right-censored at the end of signal coverage.
    """
    X, feat_cols = rul.build_xy(obs_days, gran_h, group)
    tt = X["__tt"].to_numpy()                       # hours to next onset (inf if none)
    grp = X["__grp"].to_numpy().astype(int)
    tstamp = pd.DatetimeIndex(X["__t"])

    # censoring time for the healthy tail = hours from the point to the end of coverage
    noz = ef.hourly_nozzle()
    cov_end = noz.index.max()
    cov_end = cov_end.tz_localize(None) if getattr(cov_end, "tz", None) else cov_end
    cens_h = (cov_end - tstamp).to_numpy() / np.timedelta64(1, "h")

    event = np.isfinite(tt)                          # δ = 1 → onset reached
    time = np.where(event, tt, cens_h).astype(float)
    keep = time > 0                                  # drop degenerate zero-duration rows

    F = X.loc[keep, feat_cols].to_numpy()
    # winsorise to [0.1, 99.9] pct — caps the skew/kurt tails of near-constant windows
    # (same guard as the staging section), leaving 99.8 % of values untouched.
    q = np.nanquantile(F, [0.001, 0.999], axis=0)
    F = np.clip(F, q[0], q[1])

    return {
        "F": F, "feat_cols": feat_cols,
        "time": time[keep], "event": event[keep],
        "groups": grp[keep], "tt": tt[keep],
        "t": tstamp[keep], "obs_days": obs_days, "gran_h": gran_h,
        "n_segments": int(np.unique(grp[keep][event[keep]]).size),
    }


def _structured_y(event: np.ndarray, time: np.ndarray):
    from sksurv.util import Surv
    return Surv.from_arrays(event.astype(bool), time.astype(float))


# --------------------------------------------------------------------------- #
# Cross-validated concordance (leakage-free) + out-of-fold survival functions
# --------------------------------------------------------------------------- #
def _grouped_oof(model, F, y, event, time, groups, want_surv=False, horizon=None):
    """Out-of-fold risk (higher = earlier failure); optionally OOF P(fail ≤ horizon)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.base import clone
    import warnings
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    risk = np.full(len(y), np.nan)
    pfail = np.full(len(y), np.nan) if want_surv else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in GroupKFold(n_splits=n_splits).split(F, event, groups):
            m = clone(model)
            m.fit(F[tr], y[tr])
            risk[te] = m.predict(F[te])
            if want_surv:
                for j, fn in zip(te, m.predict_survival_function(F[te])):
                    # clamp into the step-function domain: sksurv raises if the query
                    # exceeds a fold's max training time (the horizon is in-domain here,
                    # but a shorter-tail fold or larger horizon would otherwise crash)
                    h = min(horizon, float(fn.x[-1]))
                    pfail[j] = float(1.0 - fn(h))
    return risk, pfail


def _concordance(event, time, risk) -> float:
    from sksurv.metrics import concordance_index_censored
    ok = np.isfinite(risk)
    if ok.sum() < 5 or event[ok].sum() < 2:
        return float("nan")
    return float(concordance_index_censored(event[ok].astype(bool), time[ok],
                                            risk[ok])[0])


def _naive_concordance(model, F, y, event, time) -> float:
    """Naive shuffled-KFold concordance — the temporal-leakage inflated number."""
    from sklearn.model_selection import KFold
    from sklearn.base import clone
    import warnings
    risk = np.full(len(y), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(F):
            m = clone(model)
            m.fit(F[tr], y[tr])
            risk[te] = m.predict(F[te])
    return _concordance(event, time, risk)


# --------------------------------------------------------------------------- #
# Kaplan-Meier helpers
# --------------------------------------------------------------------------- #
def _km(event, time):
    from sksurv.nonparametric import kaplan_meier_estimator
    t, s = kaplan_meier_estimator(event.astype(bool), time)
    return np.asarray(t, float), np.asarray(s, float)


def _median_survival(t, s) -> float:
    """First time the KM survival drops to ≤ 0.5 (NaN if it never does)."""
    below = np.where(s <= 0.5)[0]
    return float(t[below[0]]) if len(below) else float("nan")


def _risk_terciles(event, time, risk, k: int = N_TERCILES):
    """KM survival curve per out-of-fold-risk tercile (does RSF risk separate?)."""
    ok = np.isfinite(risk)
    if ok.sum() < 3 * k:
        return []
    ev, tm, rk = event[ok], time[ok], risk[ok]
    labels = ["low risk", "medium risk", "high risk"][:k]
    # bin on dense ranks (not raw values) so tied / degenerate risk scores still split
    # into k equal groups — a non-discriminating forest then yields k *overlapping*
    # curves rather than an empty chart.
    ranks = pd.Series(rk).rank(method="first").to_numpy()
    try:
        binned = pd.qcut(ranks, k, labels=labels)
    except ValueError:
        return []
    out = []
    for lab in labels:
        m = (np.asarray(binned) == lab)
        if m.sum() < 5:
            continue
        t, s = _km(ev[m], tm[m])
        out.append({"label": lab, "n": int(m.sum()), "events": int(ev[m].sum()),
                    "time": t.tolist(), "surv": s.tolist(),
                    "median_surv": _median_survival(t, s)})
    return out


# --------------------------------------------------------------------------- #
# Permutation importance (in-sample) — which signals the RSF leans on
# --------------------------------------------------------------------------- #
def _permutation_importance(F, y, event, time, feat_cols, topn: int = 14):
    forest = _new_rsf(IMP_TREES).fit(F, y)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(F), min(IMP_SUBSAMPLE, len(F)), replace=False)
    Fi, ev_i, tm_i = F[idx], event[idx], time[idx]
    base = _concordance(ev_i, tm_i, forest.predict(Fi))
    drops = np.zeros(F.shape[1])
    for c in range(F.shape[1]):
        Fp = Fi.copy()
        Fp[:, c] = Fi[rng.permutation(len(idx)), c]
        drops[c] = base - _concordance(ev_i, tm_i, forest.predict(Fp))
    order = np.argsort(drops)[::-1][:topn]
    return {"base_c": float(base),
            "features": [feat_cols[i] for i in order],
            "drops": [float(drops[i]) for i in order]}, forest


# --------------------------------------------------------------------------- #
# Per-segment failure-probability countdown (from OOF survival functions)
# --------------------------------------------------------------------------- #
def _case_countdowns(data, oof_pfail, horizon, n_defect=N_DEFECT_CASES):
    """For each segment, OOF P(fail ≤ horizon) vs true hours-to-onset (right→left).

    A good prognostic rises toward 1 as the onset approaches. Segments are ranked by
    the Spearman correlation between predicted P(fail) and *closeness* to the onset.
    The censored healthy tail is returned as the 'normal' reference (should stay low).
    """
    from scipy.stats import spearmanr
    tt, grp, event = data["tt"], data["groups"], data["event"]
    onsets = np.sort(ef.defect_onsets(ef.defect_events("Plasma"))
                     .values.astype("datetime64[ns]"))

    defects = []
    for k in np.unique(grp[event]):
        sel = (grp == k) & event & np.isfinite(oof_pfail) & (tt <= 2 * horizon)
        if sel.sum() < 5:
            continue
        ttk, pf = tt[sel], oof_pfail[sel]
        order = np.argsort(ttk)
        rho = spearmanr(pf, -ttk).correlation
        near = pf[ttk <= horizon]
        defects.append({
            "seg": int(k),
            "onset": pd.Timestamp(onsets[k]) if k < len(onsets) else None,
            "tt": ttk[order].tolist(), "pfail": pf[order].tolist(),
            "n": int(sel.sum()), "rho": float(rho) if rho == rho else 0.0,
            "near_pfail": float(np.nanmean(near)) if near.size else float("nan")})
    defects.sort(key=lambda c: -c["rho"])
    defects = defects[:n_defect]

    # healthy reference: the right-censored tail (no failure ahead) — pfail should be low
    cens = (~event) & np.isfinite(oof_pfail)
    normals = []
    if cens.sum() >= 5:
        hb = (data["t"].max() - data["t"][cens]).to_numpy() / np.timedelta64(1, "h")
        pf = oof_pfail[cens]
        order = np.argsort(hb)
        normals.append({"ref": pd.Timestamp(data["t"].max()),
                        "hours_before": hb[order].tolist(),
                        "pfail": pf[order].tolist(),
                        "mean_pfail": float(np.nanmean(pf)), "n": int(cens.sum())})
    return {"defects": defects, "normals": normals}


# --------------------------------------------------------------------------- #
# Orchestration (cached)
# --------------------------------------------------------------------------- #
def _fingerprint(n_feat: int) -> str:
    return (f"obs{SURV_OBS_DAYS}|gran{SURV_GRAN_H}|H{PFAIL_HORIZON_H}|trees{RSF_TREES}"
            f"|leaf{RSF_MIN_LEAF}|nfeat{n_feat}|splits{N_SPLITS}|seed{SEED}"
            f"|terc{N_TERCILES}|defc{N_DEFECT_CASES}|imp{IMP_TREES}-{IMP_SUBSAMPLE}")


def compute(obs_days: int = SURV_OBS_DAYS, gran_h: int = SURV_GRAN_H,
            refresh: bool = False) -> dict:
    """Run (or load) the full survival study; returns a render-ready dict."""
    if not have_sksurv():
        return {"ok": False, "reason": "scikit-survival not installed "
                "(pip install scikit-survival)"}

    data = build_surv(obs_days, gran_h)
    fp = _fingerprint(len(data["feat_cols"]))
    if CACHE.exists() and not refresh:
        try:
            cached = pickle.loads(CACHE.read_bytes())
            if cached.get("fingerprint") == fp:
                return cached
        except Exception:
            pass

    F, event, time, groups = data["F"], data["event"], data["time"], data["groups"]
    y = _structured_y(event, time)
    models = candidate_models()

    # grouped-CV concordance per model; keep RSF out-of-fold risk + survival functions
    model_rows, oof_risk_rsf, oof_pfail = [], None, None
    for name, model in models.items():
        is_rsf = name.startswith("Random survival")
        risk, pfail = _grouped_oof(model, F, y, event, time, groups,
                                   want_surv=is_rsf, horizon=PFAIL_HORIZON_H)
        c = _concordance(event, time, risk)
        model_rows.append({"model": name, "c_index": c})
        if is_rsf:
            oof_risk_rsf, oof_pfail = risk, pfail

    valid = [r for r in model_rows if r["c_index"] == r["c_index"]]
    best = max(valid, key=lambda r: r["c_index"]) if valid else {"model": None,
                                                                  "c_index": float("nan")}

    # temporal-leakage contrast on the RSF (naive shuffled vs grouped)
    naive_c = _naive_concordance(models["Random survival forest"], F, y, event, time)
    grouped_rsf_c = next((r["c_index"] for r in model_rows
                          if r["model"].startswith("Random survival")), float("nan"))

    # population KM, risk-stratified KM, importance, per-segment countdowns
    km_t, km_s = _km(event, time)
    terciles = _risk_terciles(event, time, oof_risk_rsf)
    importance, _ = _permutation_importance(F, y, event, time, data["feat_cols"])
    cases = _case_countdowns(data, oof_pfail, PFAIL_HORIZON_H)

    result = {
        "ok": True, "fingerprint": fp,
        "obs_days": obs_days, "gran_h": gran_h, "pfail_horizon_h": PFAIL_HORIZON_H,
        "n_samples": int(len(time)), "n_events": int(event.sum()),
        "n_censored": int((~event).sum()), "n_segments": data["n_segments"],
        "n_features": len(data["feat_cols"]),
        "censoring_frac": float((~event).mean()),
        "median_event_h": float(np.median(time[event])),
        "models": model_rows, "best": best, "km_baseline_c": 0.5,
        "leakage": {"model": "Random survival forest",
                    "naive_c": naive_c, "grouped_c": grouped_rsf_c},
        "km": {"time": km_t.tolist(), "surv": km_s.tolist(),
               "median_surv": _median_survival(km_t, km_s)},
        "terciles": terciles, "importance": importance, "cases": cases,
    }
    INTERIM.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(result))
    return result
