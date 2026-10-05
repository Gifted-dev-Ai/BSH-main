"""Remaining-Useful-Life (RUL) regression for Plasma defects.

The early-prediction work asked *"will a defect happen within X h?"* (classification).
RUL generalises it to *"how many hours until the next defect?"* (regression) — and
because **every** time point carries a label (time-to-next-onset), there is far more
training data than the 15 binary events.

Setup
-----
* **Sample points** every ``RUL_GRAN_H`` hours across the signal coverage.
* **Label** = hours to the next defect *onset* (downtime-adjusted), clamped to a
  ``horizon`` ceiling (piecewise-linear RUL, as is standard in prognostics — far from
  any failure the exact RUL is unknowable, so it is censored at the horizon).
* **Features** = the same windowed Nozzle/MES summary statistics used for the early
  factors, over an observation window of ``obs_days`` ending at the sample point.
* **Validation** = GroupKFold where a group is the run-to-failure segment leading to
  one defect, so whole segments are held out (no temporal leakage).

We sweep observation length × horizon × model family, and estimate the **useful
prediction horizon**: how far before a failure the model still beats a predict-the-mean
baseline. Results are cached to ``data/interim``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import early_factors as ef

INTERIM = ef.INTERIM

RUL_OBS_DAYS = (3, 5, 10)            # observation-window lengths to compare
RUL_HORIZONS_H = (72, 168, 336)      # RUL ceilings: 3 d, 7 d, 14 d
RUL_GRAN_H = 4                       # spacing of sample points
DEFAULT_OBS = 5
DEFAULT_HORIZON = 168
HORIZON_TOL_H = 24                   # accuracy band for the prognostic-horizon metric


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_xy(obs_days: int, gran_h: int = RUL_GRAN_H, group: str = "Plasma"):
    """Feature frame + time-to-next-onset (hours) + run-to-failure group id."""
    events = ef.defect_events(group)
    onsets = np.sort(ef.defect_onsets(events).values.astype("datetime64[ns]"))
    nozzle, mes = ef.hourly_nozzle(), ef.hourly_mes()
    signals = sorted({c.rsplit("__", 1)[0] for c in nozzle.columns})

    cmin, cmax = nozzle.index.min(), nozzle.index.max()
    cov_start = cmin.tz_localize(None) if getattr(cmin, "tz", None) else cmin
    cov_end = cmax.tz_localize(None) if getattr(cmax, "tz", None) else cmax
    pts = pd.date_range(cov_start, cov_end, freq=f"{gran_h}h")

    # time-to-next-onset (hours) and the segment (next-defect index) for grouping
    pv = pts.values.astype("datetime64[ns]")
    idx = np.searchsorted(onsets, pv, side="left")
    tt = np.full(len(pts), np.inf)
    has_next = idx < len(onsets)
    tt[has_next] = ((onsets[idx[has_next]] - pv[has_next])
                    / np.timedelta64(1, "h")).astype(float)
    groups = idx  # points before the same onset share a group (sentinel = len(onsets))

    roll = ef._rolling_features(nozzle, signals, obs_days)
    end_h = pts.floor("h")
    cols = {}
    for stat in ef.SIGNAL_STATS:
        for s in signals:
            cols[f"{s}·{stat}"] = roll[stat][s].reindex(end_h).to_numpy()
    # MES error counts: total + per fault category
    for s in ef.MES_SIGNALS:
        if f"{s}__count" in mes.columns:
            cols[f"{s}·count"] = ef._rolling_count(
                mes, f"{s}__count", obs_days).reindex(end_h).to_numpy()

    feat_cols = list(cols)
    cols["__tt"] = tt       # time-to-next-onset (inf where no future defect)
    cols["__grp"] = groups
    cols["__t"] = pts       # sample timestamp (for per-case trajectory reconstruction)
    X = pd.DataFrame(cols)  # build every column at once (avoids frame fragmentation)
    X[feat_cols] = X[feat_cols].replace([np.inf, -np.inf], np.nan)
    # keep only rows with complete features (inf RUL labels are kept and clamped later)
    X = X.dropna(subset=feat_cols).reset_index(drop=True)
    return X, feat_cols


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def candidate_regressors() -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import (RandomForestRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "Random forest": RandomForestRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=5,
            random_state=7, n_jobs=1),
        "Hist GBM": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=7),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "ANN (2 layers)": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=1500, random_state=7)),
    }
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, random_state=7, n_jobs=1, verbose=-1)
    except Exception:
        pass
    return models


def _cv_predict(model, X, y, groups):
    from sklearn.model_selection import GroupKFold, cross_val_predict
    import warnings
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=1)


# --------------------------------------------------------------------------- #
# Sweep + predictions
# --------------------------------------------------------------------------- #
def _ramp(obs_days, horizon_h, gran_h):
    """Degradation-ramp subset: points within ``horizon`` of the next defect."""
    X, feat_cols = build_xy(obs_days, gran_h)
    tt = X["__tt"].to_numpy()
    m = tt < horizon_h
    return (X.loc[m, feat_cols].to_numpy(), np.minimum(tt[m], horizon_h),
            X.loc[m, "__grp"].to_numpy(), tt[m])


def predictions(obs_days: int, horizon_h: int, model_name: str, gran_h: int = RUL_GRAN_H
                ) -> pd.DataFrame:
    """Leakage-free (GroupKFold) RUL predictions within the degradation ramp."""
    Xf, y, groups, tt = _ramp(obs_days, horizon_h, gran_h)
    model = candidate_regressors()[model_name]
    pred = np.clip(_cv_predict(model, Xf, y, groups), 0, horizon_h)
    return pd.DataFrame({"true": y, "pred": pred, "tt": tt})


def leakage_demo(obs_days: int, horizon_h: int, model_name: str,
                 gran_h: int = RUL_GRAN_H) -> dict:
    """Naive (shuffled KFold) vs honest (GroupKFold) R² — the temporal-leakage gap."""
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score
    import warnings
    Xf, y, groups, _ = _ramp(obs_days, horizon_h, gran_h)
    model = candidate_regressors()[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pn = np.clip(cross_val_predict(model, Xf, y,
                     cv=KFold(5, shuffle=True, random_state=7), n_jobs=1), 0, horizon_h)
        pg = np.clip(_cv_predict(model, Xf, y, groups), 0, horizon_h)
    return {"naive_r2": float(r2_score(y, pn)), "grouped_r2": float(r2_score(y, pg)),
            "model": model_name}


def _safe_cv_metrics(model, Xf, y, groups, horizon) -> tuple:
    """Top-level (picklable) wrapper: grouped-CV regression metrics for one model."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try:
        pred = np.clip(_cv_predict(model, Xf, y, groups), 0, horizon)
        return (mean_absolute_error(y, pred),
                float(np.sqrt(mean_squared_error(y, pred))), r2_score(y, pred))
    except Exception:
        return (np.nan, np.nan, np.nan)


def sweep(refresh: bool = False) -> pd.DataFrame:
    """Leakage-free CV MAE / RMSE / R² for every (obs, horizon, model) — cached.

    Evaluated within the degradation ramp (RUL < horizon); the mean-RUL baseline MAE
    is recorded so model skill can be judged against it. The model panel for each
    (obs, horizon) cell is evaluated in parallel.
    """
    out = INTERIM / "rul_plasma_sweep.parquet"
    model_names = set(candidate_regressors().keys())
    feat_sig = ",".join(sorted(ef.MES_SIGNALS) + sorted(ef.SIGNAL_STATS))  # feature-schema fingerprint
    if out.exists() and not refresh:
        cached = pd.read_parquet(out)
        # invalidate if the obs/horizon grid, regressor panel, or features changed
        if (ef._cache_covers(cached, obs_days=RUL_OBS_DAYS, horizon_h=RUL_HORIZONS_H)
                and model_names.issubset(set(cached["model"].unique()))
                and str(cached.get("feat_sig", pd.Series(["?"])).iloc[0]) == feat_sig):
            return cached
    from joblib import Parallel, delayed
    rows = []
    for obs in RUL_OBS_DAYS:
        X, feat_cols = build_xy(obs)          # expensive feature build: once per obs
        tt = X["__tt"].to_numpy()
        groups_all = X["__grp"].to_numpy()
        Xf_all = X[feat_cols].to_numpy()
        for horizon in RUL_HORIZONS_H:
            m = tt < horizon
            Xf, y, gr = Xf_all[m], np.minimum(tt[m], horizon), groups_all[m]
            base_mae = float(np.abs(y - y.mean()).mean())
            models = candidate_regressors()
            res = Parallel(n_jobs=-1)(
                delayed(_safe_cv_metrics)(model, Xf, y, gr, horizon)
                for model in models.values())
            for (mname, _), (mae, rmse, r2) in zip(models.items(), res):
                rows.append((obs, horizon, mname, mae, rmse, r2, base_mae, len(y)))
    df = pd.DataFrame(rows, columns=["obs_days", "horizon_h", "model", "mae",
                                     "rmse", "r2", "baseline_mae", "n"])
    df["feat_sig"] = feat_sig
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def prognostic_horizon(pred_df: pd.DataFrame, horizon_h: int,
                       tol_h: float = HORIZON_TOL_H) -> dict:
    """Estimate how far ahead RUL is still useful.

    Within the run-to-failure ramp (true RUL < horizon) bin by true RUL and compare
    the model's MAE to a predict-the-mean baseline. The useful horizon is the largest
    lead time where the model still beats the baseline and stays within ``tol_h``.
    """
    ramp = pred_df[pred_df["tt"] < horizon_h].copy()
    if len(ramp) < 20:
        return {"horizon": np.nan, "bins": pd.DataFrame()}
    edges = [e for e in (0, 6, 12, 24, 48, 72, 96, 168, 336) if e <= horizon_h]
    if edges[-1] < horizon_h:
        edges.append(horizon_h)
    ramp["bin"] = pd.cut(ramp["true"], bins=edges)
    baseline = ramp["true"].mean()
    recs = []
    for b, g in ramp.groupby("bin", observed=True):
        if len(g) < 5:
            continue
        mae = float(np.abs(g["pred"] - g["true"]).mean())
        base_mae = float(np.abs(baseline - g["true"]).mean())
        recs.append({"bin_right": b.right, "center": (b.left + b.right) / 2,
                     "mae": mae, "baseline_mae": base_mae, "n": len(g)})
    bins = pd.DataFrame(recs)
    useful = bins[(bins["mae"] < bins["baseline_mae"]) & (bins["mae"] <= tol_h)]
    horizon = float(useful["bin_right"].max()) if not useful.empty else 0.0
    return {"horizon": horizon, "bins": bins, "baseline": baseline}


# --------------------------------------------------------------------------- #
# Per-case predicted-RUL trajectories (leakage-free)
# --------------------------------------------------------------------------- #
def case_trajectories(obs_days: int, horizon_h: int, model_name: str,
                      gran_h: int = RUL_GRAN_H, n_defect: int = 5, n_normal: int = 2,
                      group: str = "Plasma") -> dict:
    """Predicted-RUL trajectory for selected defect/normal cases (leakage-free).

    * **Defect cases** — for each onset, the model is trained on the *other* segments
      (leave-one-defect-out) and predicts that onset's run-up; the trajectory is the
      predicted RUL vs the true hours-to-onset. The ``n_defect`` best-tracked onsets
      (lowest ramp MAE) are returned.
    * **Normal cases** — the model (trained on all ramps) predicts the longest healthy
      stretches; a well-behaved model keeps the predicted RUL near the horizon (no
      false countdown). The ``n_normal`` highest/steadiest stretches are returned.
    """
    from sklearn.base import clone

    X, feat_cols = build_xy(obs_days, gran_h, group)
    t = pd.DatetimeIndex(X["__t"])
    tt = X["__tt"].to_numpy()
    grp = X["__grp"].to_numpy().astype(int)
    Xf = X[feat_cols].to_numpy()
    base = candidate_regressors()[model_name]

    onsets = np.sort(ef.defect_onsets(ef.defect_events(group)).values.astype("datetime64[ns]"))
    ramp = tt < horizon_h
    y_clamped = np.minimum(tt, horizon_h)
    pred = np.full(len(tt), np.nan)

    # leave-one-defect-out predictions on each onset's ramp
    for k in range(len(onsets)):
        test = ramp & (grp == k)
        train = ramp & (grp != k)
        if test.sum() < 1 or train.sum() < 20:
            continue
        m = clone(base)
        m.fit(Xf[train], y_clamped[train])
        pred[test] = np.clip(m.predict(Xf[test]), 0, horizon_h)

    # global model on all ramps: predicts the healthy points and ranks the features
    # whose values to surface on hover (model importances / |coef|)
    healthy = ~ramp
    m_all = clone(base).fit(Xf[ramp], y_clamped[ramp]) if ramp.sum() >= 20 else None
    if m_all is not None and healthy.any():
        pred[healthy] = np.clip(m_all.predict(Xf[healthy]), 0, horizon_h)
    imp = getattr(m_all, "feature_importances_", None) if m_all is not None else None
    if imp is None and m_all is not None:
        coef = getattr(m_all, "coef_", None)
        imp = np.abs(np.ravel(coef)) if coef is not None else None
    if imp is None:
        imp = np.ones(Xf.shape[1])
    top_idx = np.argsort(imp)[::-1][:8]
    feat_names = [feat_cols[j] for j in top_idx]

    defect_cases = []
    for k in range(len(onsets)):
        sel = ramp & (grp == k) & np.isfinite(pred)
        if sel.sum() < 3:
            continue
        ttk, prk = tt[sel], pred[sel]
        order = np.argsort(ttk)
        mae = float(np.abs(prk - np.minimum(ttk, horizon_h)).mean())
        defect_cases.append({"k": int(k), "onset": pd.Timestamp(onsets[k]),
                             "tt": ttk[order].tolist(), "pred": prk[order].tolist(),
                             "feat_vals": Xf[sel][:, top_idx][order].tolist(),
                             "n": int(sel.sum()), "mae": mae})
    defect_cases.sort(key=lambda c: c["mae"])
    defect_cases = defect_cases[:n_defect]

    # contiguous healthy stretches (in time order); score by steadiness near horizon
    order_t = np.argsort(t.values)
    th = t.values[order_t]
    hh = healthy[order_t]
    prh = pred[order_t]
    step = np.timedelta64(gran_h * 3, "h")          # allow small gaps
    stretches, run = [], []
    for i in range(len(th)):
        if not hh[i]:
            if run:
                stretches.append(run); run = []
            continue
        if run and (th[i] - th[run[-1]]) > step:
            stretches.append(run); run = []
        run.append(i)
    if run:
        stretches.append(run)

    span = np.timedelta64(horizon_h, "h")
    normal_cases = []
    for run in stretches:
        ref = th[run[-1]]
        keep = [i for i in run if (ref - th[i]) <= span and np.isfinite(prh[i])]
        if len(keep) < 5:
            continue
        hours_before = [(ref - th[i]) / np.timedelta64(1, "h") for i in keep]
        prs = [float(prh[i]) for i in keep]
        ob = np.argsort(hours_before)
        fvals = Xf[order_t[np.array(keep)]][:, top_idx]   # original feature rows
        normal_cases.append({
            "ref": pd.Timestamp(ref),
            "hours_before": list(np.array(hours_before)[ob]),
            "pred": list(np.array(prs)[ob]),
            "feat_vals": fvals[ob].tolist(),
            "mean_pred": float(np.mean(prs)), "min_pred": float(np.min(prs)),
            "n": len(keep)})
    # "best" normal = steadiest & highest (closest to the no-failure horizon)
    normal_cases.sort(key=lambda c: -(c["mean_pred"] - (c["mean_pred"] - c["min_pred"])))
    normal_cases = normal_cases[:n_normal]

    return {"model": model_name, "horizon": horizon_h, "obs": obs_days,
            "feat_names": feat_names,
            "defects": defect_cases, "normals": normal_cases}
