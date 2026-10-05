"""Early predictive-factor exploration for Plasma defects.

Goal: find which upstream state variables separate the lead-up to a *defect* from
*normal* operation, and how early that signal is detectable.

Pipeline
--------
1. **Defect events** — cluster the incident log (consecutive rows < 1 day apart =
   one event); the first incident's time is the defect start. Done per group.
2. **Signal sources** (for Plasma): the Plasma-Nozzle-Head state variables and the
   MES error stream (``event_type == 1`` counts). Each is aggregated to **hourly**
   buckets (count/sum/sumsq/min/max) once and cached in ``data/interim``.
3. **Observation windows** — for a lead time *advance* (1..96 h) the window ends at
   ``defect_start − advance`` and spans *observation duration* (1..10 days). Summary
   statistics (min/max/mean/std, error count) over that window are the features.
4. **Normal samples** — time points on a grid (granularity 2..24 h) that do not sit
   in any defect's run-up (≥ max(24 h, downtime) before a start) or recovery
   (≥ 24 h after an end).
5. **Comparison** — for every (advance, observation, granularity, statistic) the
   defect vs. normal feature distributions are compared with the AUC of a
   Mann-Whitney test (0.5 = no separation, →1 higher before defects, →0 lower).

The full AUC grid is cached to ``data/interim`` so the report build is cheap after
the first run. Charts/section live in :func:`build_sections`.

Caveats (surfaced in the report): incident timestamps are treated as UTC to align
with the signal streams; samples are small (15 Plasma events), so this is
exploratory. The **Robot** monitoring file covers only ~3 days (early Mar 2025) and
does not overlap any Robot defect, so robot early-factors cannot be computed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .config import PROJECT_ROOT, RAW_DIR
from .incident_analysis import load_enriched_incidents
from . import mes_text as _mes

INTERIM = PROJECT_ROOT / "data" / "interim"

# MES error signals fed to the models: the total event_type==1 count plus one count
# per extracted fault category (see mes_text). All use the "count" statistic.
MES_SIGNALS = ["MES_total"] + [f"MES_{c}" for c in _mes.MES_FEATURE_CATEGORIES]


def _cache_covers(df: pd.DataFrame, **expected) -> bool:
    """True if a cached frame still matches the current parameter grid.

    Guards against stale ``data/interim`` caches: if e.g. ``ADVANCE_HOURS`` gained a
    value (72 h) after the cache was written, the cached grid no longer covers it and
    must be recomputed. Each kwarg maps a column name to the set of values it must
    contain (``{column: iterable_of_required_values}``).
    """
    for col, required in expected.items():
        if col not in df.columns or not set(required).issubset(set(df[col].unique())):
            return False
    return True


# --- parameter grids --------------------------------------------------------
ADVANCE_HOURS = (1, 6, 12, 24, 48, 72, 96)  # lead time before defect start
OBS_DAYS = (0.5, 1, 2, 3, 5, 10)           # observation-window length (0.5 d = 12 h)
GRAN_HOURS = (2, 6, 12, 24)                # normal-sample grid granularity
SIGNAL_STATS = ("min", "max", "mean", "std", "skew", "kurt", "range")
DEFAULT_OBS_DAYS = 3                        # used for the main heatmap
DEFAULT_GRAN_H = 12
NORMAL_CAP = 1200                          # max normal samples per granularity
CLUSTER_GAP_H = 24                         # incidents < this apart = one event

# Hardware error sentinels: the CheckTheChecker distance channels emit huge negative
# codes (~-1e10 .. -8e10) when a reading fails; legitimate process values are all far
# below this, so anything beyond the cap is masked to NaN before aggregation.
SENTINEL_CAP = 1e6

NOZZLE_FILE = "Plasma_Nozzle_Head_Data.csv"
MES_FILE = "Plasma_Station__MES_error_event.csv"
ROBOT_FILE = "Stäubli_Robot_monitoring_Data.csv"


# --------------------------------------------------------------------------- #
# Defect events
# --------------------------------------------------------------------------- #
def defect_events(group: str) -> pd.DataFrame:
    """Cluster a group's incidents into defect events.

    Columns: ``start`` (first incident = discovery time), ``end`` (last incident),
    ``downtime_min`` (total cluster downtime, for the normal-state exclusion buffer)
    and ``first_downtime_min`` (downtime of the first incident — how long the machine
    had already been down at discovery; used to back-date the true fault onset).
    """
    df = load_enriched_incidents().dropna(subset=["_ts"])
    g = df[df["Groupdescr"] == group].sort_values("_ts")
    # incident times are naive; treat as UTC to align with the signal streams
    ts = g["_ts"].dt.tz_localize("UTC")
    events = []
    start = end = None
    downtime = first_dt = 0.0
    prev = None
    for t, dt in zip(ts, g["Calcdowntime"]):
        if start is None:
            start = end = t
            downtime = first_dt = float(dt)
        elif (t - prev).total_seconds() < CLUSTER_GAP_H * 3600:
            end = t
            downtime += float(dt)
        else:
            events.append((start, end, downtime, first_dt))
            start = end = t
            downtime = first_dt = float(dt)
        prev = t
    if start is not None:
        events.append((start, end, downtime, first_dt))
    return pd.DataFrame(events, columns=["start", "end", "downtime_min",
                                         "first_downtime_min"])


def defect_onsets(events: pd.DataFrame) -> pd.Series:
    """True fault-onset reference time = discovery − downtime-at-discovery.

    The recorded defect time is when the defect was *discovered*; by then the machine
    has already been down for the first incident's downtime, so the lead-time windows
    must be measured from this earlier onset, not from the raw defect time.
    """
    starts = events["start"].dt.tz_convert("UTC").dt.tz_localize(None)
    return starts - pd.to_timedelta(events["first_downtime_min"], unit="m")


# --------------------------------------------------------------------------- #
# Hourly aggregation of the signal sources (cached)
# --------------------------------------------------------------------------- #
def _nozzle_signals(columns) -> list[str]:
    exclude = {"uniquepart_id", "result_timestamp", "pltProcessTime[s]",
               "LocalTypeSpecificDatFile"}
    return [c for c in columns if c not in exclude]


def hourly_nozzle(refresh: bool = False) -> pd.DataFrame:
    """Hourly moments (count/sum/sq/cube/quad) + min/max per nozzle signal.

    The raw moment sums (Σx, Σx², Σx³, Σx⁴) let the rolling-window features
    reconstruct mean/std/**skewness/kurtosis** over any window; min/max give the
    **range**. Hardware error sentinels are masked first (see ``SENTINEL_CAP``).
    Wide, hour-indexed.
    """
    out = INTERIM / "plasma_nozzle_hourly.parquet"
    if out.exists() and not refresh:
        cached = pd.read_parquet(out)
        maxcols = [c for c in cached.columns if c.endswith("__max")]
        # self-invalidate on schema change (higher moments) or a pre-sentinel-filter
        # cache (its __max still holds the ~1e10 error codes)
        if (any(c.endswith("__quad") for c in cached.columns) and maxcols
                and float(np.nanmax(np.abs(cached[maxcols].to_numpy()))) <= SENTINEL_CAP):
            return cached
    header = pd.read_csv(RAW_DIR / NOZZLE_FILE, nrows=0)
    signals = _nozzle_signals(header.columns)
    parts = []
    for chunk in pd.read_csv(RAW_DIR / NOZZLE_FILE,
                             usecols=["result_timestamp"] + signals,
                             chunksize=300_000):
        t = pd.to_datetime(chunk["result_timestamp"], utc=True, format="ISO8601")
        hour = t.dt.floor("h")
        vals = chunk[signals].apply(pd.to_numeric, errors="coerce")
        vals = vals.where(vals.abs() <= SENTINEL_CAP)   # drop hardware error sentinels
        agg = vals.groupby(hour).agg(["count", "sum", "min", "max"])
        agg.columns = [f"{s}__{stat}" for s, stat in agg.columns]
        moments = []
        for power, tag in ((2, "sq"), (3, "cube"), (4, "quad")):
            m = (vals ** power).groupby(hour).sum()
            m.columns = [f"{s}__{tag}" for s in m.columns]
            moments.append(m)
        parts.append(pd.concat([agg] + moments, axis=1))
    full = pd.concat(parts)
    # combine partial hourly aggregates across chunks (sums add; min/max reduce)
    agg_map = {}
    for s in signals:
        for suf in ("count", "sum", "sq", "cube", "quad"):
            agg_map[f"{s}__{suf}"] = "sum"
        agg_map[f"{s}__min"] = "min"
        agg_map[f"{s}__max"] = "max"
    combined = full.groupby(level=0).agg(agg_map)
    combined.index.name = "hour"
    INTERIM.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out)
    return combined


def hourly_mes(refresh: bool = False) -> pd.DataFrame:
    """Hourly MES error counts (event_type == 1): total + one per fault category.

    Columns: ``MES_total__count`` (all event_type==1) and ``MES_<category>__count``
    for each extracted fault category (servo_drive, sensor, power, …). The category
    breakdown only covers messages that carry an ``error_text``; the rest are still
    in the total.
    """
    out = INTERIM / "mes_error_hourly.parquet"
    expected = {f"{s}__count" for s in MES_SIGNALS}
    if out.exists() and not refresh:
        cached = pd.read_parquet(out)
        if expected.issubset(set(cached.columns)):   # self-invalidate on schema change
            return cached
    df = pd.read_csv(RAW_DIR / MES_FILE,
                     usecols=["result_timestamp", "event_type", "error_text"])
    df = df[df["event_type"] == 1].copy()
    df["hour"] = pd.to_datetime(df["result_timestamp"], utc=True,
                                format="ISO8601").dt.floor("h")
    df["cat"] = df["error_text"].map(_mes.categorize)

    res = pd.DataFrame(index=df.groupby("hour").size().index)
    res["MES_total__count"] = df.groupby("hour").size()
    cat_counts = (df[df["cat"] != "untyped"].groupby(["hour", "cat"]).size()
                  .unstack("cat", fill_value=0))
    for c in _mes.MES_FEATURE_CATEGORIES:
        res[f"MES_{c}__count"] = cat_counts[c] if c in cat_counts.columns else 0
    res = res.fillna(0).astype(int)
    res.index.name = "hour"
    INTERIM.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out)
    return res


# --------------------------------------------------------------------------- #
# Rolling window features
# --------------------------------------------------------------------------- #
def _continuous(hourly: pd.DataFrame) -> pd.DataFrame:
    # work in naive-UTC throughout so query end-hours (naive) align on reindex
    if getattr(hourly.index, "tz", None) is not None:
        hourly = hourly.copy()
        hourly.index = hourly.index.tz_localize(None)
    idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    return hourly.reindex(idx)


def _rolling_features(hourly: pd.DataFrame, signals: list[str], obs_days: int
                      ) -> dict[str, pd.DataFrame]:
    """For one observation length, the window stat per signal at every hour."""
    W = int(round(obs_days * 24))
    cont = _continuous(hourly)
    feats: dict[str, pd.DataFrame] = {}
    out = {stat: {} for stat in SIGNAL_STATS}
    for s in signals:
        n = (cont[f"{s}__count"].fillna(0).rolling(W, min_periods=1).sum()
             .replace(0, np.nan))
        # raw moments E[x^k] reconstructed from the per-hour moment sums
        m1 = cont[f"{s}__sum"].rolling(W, min_periods=1).sum() / n
        m2 = cont[f"{s}__sq"].rolling(W, min_periods=1).sum() / n
        m3 = cont[f"{s}__cube"].rolling(W, min_periods=1).sum() / n
        m4 = cont[f"{s}__quad"].rolling(W, min_periods=1).sum() / n
        var = (m2 - m1 ** 2).clip(lower=0)
        std = np.sqrt(var)
        # central moments -> skewness and excess kurtosis. Undefined (0/0) for a
        # constant window -> 0 by convention; genuinely-empty windows (no data)
        # stay NaN so they're dropped rather than faked as 0.
        mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        present, defined = n.notna(), var > 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            skew = (mu3 / var ** 1.5).where(defined, 0.0).where(present)
            kurt = (mu4 / var ** 2 - 3.0).where(defined, 0.0).where(present)
        rmin = cont[f"{s}__min"].rolling(W, min_periods=1).min()
        rmax = cont[f"{s}__max"].rolling(W, min_periods=1).max()
        out["mean"][s] = m1
        out["std"][s] = std
        out["skew"][s] = skew
        out["kurt"][s] = kurt
        out["min"][s] = rmin
        out["max"][s] = rmax
        out["range"][s] = rmax - rmin
    for stat in SIGNAL_STATS:
        feats[stat] = pd.DataFrame(out[stat])
    return feats


def _rolling_count(hourly: pd.DataFrame, col: str, obs_days: int) -> pd.Series:
    cont = _continuous(hourly)
    return cont[col].fillna(0).rolling(int(round(obs_days * 24)), min_periods=1).sum()


# --------------------------------------------------------------------------- #
# Normal samples
# --------------------------------------------------------------------------- #
def normal_points(events: pd.DataFrame, gran_h: int, cov_start, cov_end
                  ) -> pd.DatetimeIndex:
    """Grid points not in any defect run-up / recovery window."""
    pts = pd.date_range(cov_start, cov_end, freq=f"{gran_h}h")
    keep = np.ones(len(pts), dtype=bool)
    p = pts.values.astype("datetime64[ns]")
    for _, e in events.iterrows():
        lead_h = max(24.0, e["downtime_min"] / 60.0)
        lo = (e["start"].tz_convert("UTC").tz_localize(None)
              - pd.Timedelta(hours=lead_h)).to_datetime64()
        hi = (e["end"].tz_convert("UTC").tz_localize(None)
              + pd.Timedelta(hours=24)).to_datetime64()
        keep &= ~((p > lo) & (p < hi))
    pts = pts[keep]
    if len(pts) > NORMAL_CAP:
        rng = np.random.default_rng(7)
        pts = pts[np.sort(rng.choice(len(pts), NORMAL_CAP, replace=False))]
    return pts


# --------------------------------------------------------------------------- #
# AUC grid
# --------------------------------------------------------------------------- #
def _auc(defect: np.ndarray, normal: np.ndarray) -> tuple[float, int, int]:
    d = defect[np.isfinite(defect)]
    n = normal[np.isfinite(normal)]
    if len(d) < 3 or len(n) < 5 or (len(np.unique(np.concatenate([d, n]))) < 2):
        return np.nan, len(d), len(n)
    u, _ = mannwhitneyu(d, n, alternative="two-sided")
    return u / (len(d) * len(n)), len(d), len(n)


def _end_hours(times, advance_h: int) -> pd.DatetimeIndex:
    return (pd.DatetimeIndex(times) - pd.Timedelta(hours=advance_h)).floor("h")


def compute_auc_grid(group: str = "Plasma", refresh: bool = False) -> pd.DataFrame:
    """Tidy AUC table over the full (gran, advance, obs, signal, stat) grid."""
    out = INTERIM / f"early_factors_auc_{group.lower()}.parquet"
    if out.exists() and not refresh:
        cached = pd.read_parquet(out)
        if (_cache_covers(cached, advance_h=ADVANCE_HOURS, obs_days=OBS_DAYS,
                          gran_h=GRAN_HOURS, stat=SIGNAL_STATS)
                and set(MES_SIGNALS).issubset(set(cached["signal"].unique()))):
            return cached
        # stale cache (parameter grid or MES feature set changed) -> recompute below

    events = defect_events(group)
    nozzle = hourly_nozzle(refresh)
    mes = hourly_mes(refresh)
    signals = sorted({c.rsplit("__", 1)[0] for c in nozzle.columns})

    cov_start = nozzle.index.min().tz_localize(None)
    cov_end = nozzle.index.max().tz_localize(None)
    # back-date defect reference to the true onset (discovery − downtime-at-discovery)
    starts = defect_onsets(events)

    # precompute rolling features per observation length (shared across advances)
    mes_signals = [s for s in MES_SIGNALS if f"{s}__count" in mes.columns]
    roll_sig = {obs: _rolling_features(nozzle, signals, obs) for obs in OBS_DAYS}
    roll_mes = {obs: {s: _rolling_count(mes, f"{s}__count", obs)
                      for s in mes_signals} for obs in OBS_DAYS}

    rows = []
    for gran in GRAN_HOURS:
        npts = normal_points(events, gran, cov_start, cov_end)
        for advance in ADVANCE_HOURS:
            d_end = _end_hours(starts, advance)
            n_end = _end_hours(npts, advance)
            for obs in OBS_DAYS:
                for s in signals:
                    for stat in SIGNAL_STATS:
                        frame = roll_sig[obs][stat]
                        dv = frame[s].reindex(d_end).to_numpy()
                        nv = frame[s].reindex(n_end).to_numpy()
                        auc, nd, nn = _auc(dv, nv)
                        rows.append((gran, advance, obs, s, stat, auc, nd, nn))
                # MES error counts (total + per fault category)
                for s in mes_signals:
                    ser = roll_mes[obs][s]
                    dv = ser.reindex(d_end).to_numpy()
                    nv = ser.reindex(n_end).to_numpy()
                    auc, nd, nn = _auc(dv, nv)
                    rows.append((gran, advance, obs, s, "count", auc, nd, nn))

    grid = pd.DataFrame(rows, columns=["gran_h", "advance_h", "obs_days",
                                       "signal", "stat", "auc", "n_defect",
                                       "n_normal"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(out)
    return grid


# --------------------------------------------------------------------------- #
# Honest (leave-one-defect-out) AUC — same footing as the RUL grouped CV
# --------------------------------------------------------------------------- #
# The headline AUC in the heatmap is *in-sample* and is the **max over ~135
# (signal × stat) features**, so it is inflated by feature selection (with only 15
# events some feature scores high by chance). The honest analog selects the best
# feature *inside* each fold (on the other 14 defects) and scores it on the
# held-out defect — the direct parallel to RUL's leave-whole-event-out GroupKFold.
def _feature_panel(obs: int):
    """Every (signal·stat) + MES·count feature as an hourly series, for one obs."""
    nozzle, mes = hourly_nozzle(), hourly_mes()
    signals = sorted({c.rsplit("__", 1)[0] for c in nozzle.columns})
    roll = _rolling_features(nozzle, signals, obs)
    mes_sigs = [s for s in MES_SIGNALS if f"{s}__count" in mes.columns]
    feats = [(f"{s} · {stat}", roll[stat][s])
             for stat in SIGNAL_STATS for s in signals]
    feats += [(f"{s} · count", _rolling_count(mes, f"{s}__count", obs))
              for s in mes_sigs]
    return feats


def _normal_rank(D: np.ndarray, N: np.ndarray):
    """Rank of each defect value among the normal values, per feature (NaN-safe).

    ``R[d, f]`` = P(normal < defect_d) for feature f (ties counted at 0.5); the
    in-sample higher-orientation AUC of feature f is ``nanmean_d R[:, f]``.
    """
    Dn, F = D.shape
    R = np.full((Dn, F), np.nan)
    eligible = np.zeros(F, dtype=bool)
    for f in range(F):
        nv = np.sort(N[:, f][np.isfinite(N[:, f])])
        if len(nv) < 5:
            continue
        for d in range(Dn):
            x = D[d, f]
            if np.isfinite(x):
                lo = np.searchsorted(nv, x, "left")
                hi = np.searchsorted(nv, x, "right")
                R[d, f] = 0.5 * (lo + hi) / len(nv)
        if np.isfinite(R[:, f]).sum() >= 3:
            eligible[f] = True
    return R, eligible


def honest_auc_by_lead(group: str = "Plasma", obs: int = DEFAULT_OBS_DAYS,
                       gran: int = DEFAULT_GRAN_H, refresh: bool = False) -> pd.DataFrame:
    """In-sample best vs leave-one-defect-out AUC of the best-selected feature, per lead."""
    out = INTERIM / f"early_honest_auc_{group.lower()}_{obs}_{gran}.parquet"
    if out.exists() and not refresh:
        cached = pd.read_parquet(out)
        if (_cache_covers(cached, advance_h=ADVANCE_HOURS)
                and int(cached["n_features"].iloc[0]) == len(_feature_panel(obs))):
            return cached

    events = defect_events(group)
    starts = defect_onsets(events)
    nozzle = hourly_nozzle()
    cov_start = nozzle.index.min().tz_localize(None)
    cov_end = nozzle.index.max().tz_localize(None)
    npts = normal_points(events, gran, cov_start, cov_end)
    feats = _feature_panel(obs)
    names = [n for n, _ in feats]
    F = len(feats)

    rows = []
    for adv in ADVANCE_HOURS:
        d_end, n_end = _end_hours(starts, adv), _end_hours(npts, adv)
        D = np.column_stack([s.reindex(d_end).to_numpy() for _, s in feats])
        N = np.column_stack([s.reindex(n_end).to_numpy() for _, s in feats])
        R, eligible = _normal_rank(D, N)
        Dn = D.shape[0]

        with np.errstate(invalid="ignore"):
            auc = np.nanmean(R, axis=0)                     # higher-orientation AUC
        sep = np.where(eligible, np.abs(auc - 0.5), -1.0)
        bf = int(np.argmax(sep))
        in_best = 0.5 + sep[bf]
        in_feat = names[bf]

        # leave-one-defect-out: pick best feature on the other defects, score the held-out
        sumR, cntR = np.nansum(R, axis=0), np.isfinite(R).sum(axis=0)
        scores = []
        for i in range(Dn):
            ri = R[i]
            tr_sum = sumR - np.where(np.isfinite(ri), np.nan_to_num(ri), 0.0)
            tr_cnt = cntR - np.isfinite(ri).astype(int)
            with np.errstate(invalid="ignore"):
                auc_tr = tr_sum / np.where(tr_cnt > 0, tr_cnt, np.nan)
            sep_tr = np.where(eligible & (tr_cnt >= 2), np.abs(auc_tr - 0.5), -1.0)
            fstar = int(np.argmax(sep_tr))
            if sep_tr[fstar] < 0 or not np.isfinite(R[i, fstar]):
                continue
            s = R[i, fstar] if auc_tr[fstar] >= 0.5 else 1.0 - R[i, fstar]
            scores.append(s)
        lodo = float(np.mean(scores)) if len(scores) >= 5 else np.nan
        rows.append((adv, in_best, lodo, in_feat, len(scores)))

    df = pd.DataFrame(rows, columns=["advance_h", "in_sample_best", "lodo_best",
                                     "sel_feature", "n_eval"])
    df["n_features"] = F
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


# --------------------------------------------------------------------------- #
# Multivariate combo: do the top early factors together predict defects?
# --------------------------------------------------------------------------- #
def top_features(grid: pd.DataFrame, advance: int, obs: int, gran: int,
                 k: int = 6) -> list[tuple[str, str]]:
    """The k most discriminative (signal, stat) features for a grid cell."""
    sub = grid[(grid.advance_h == advance) & (grid.obs_days == obs)
               & (grid.gran_h == gran)].dropna(subset=["auc"]).copy()
    sub["sep"] = (sub.auc - 0.5).abs()
    sub = sub.sort_values("sep", ascending=False).head(k)
    return list(zip(sub.signal, sub.stat))


def feature_matrix(features: list[tuple[str, str]], advance: int, obs: int,
                   gran: int, group: str = "Plasma"):
    """Build (X, y) for the combo model: rows = pre-defect (1) and normal (0) windows."""
    events = defect_events(group)
    nozzle, mes = hourly_nozzle(), hourly_mes()
    cov_start = nozzle.index.min()
    cov_start = cov_start.tz_localize(None) if getattr(cov_start, "tz", None) else cov_start
    cov_end = nozzle.index.max()
    cov_end = cov_end.tz_localize(None) if getattr(cov_end, "tz", None) else cov_end
    starts = defect_onsets(events)  # downtime-adjusted onset
    npts = normal_points(events, gran, cov_start, cov_end)
    d_end, n_end = _end_hours(starts, advance), _end_hours(npts, advance)

    sigs_needed = [s for s, _ in features if not s.startswith("MES_")]
    roll = _rolling_features(nozzle, sigs_needed, obs) if sigs_needed else {}
    mes_roll = {s: _rolling_count(mes, f"{s}__count", obs)
                for s, _ in features if s.startswith("MES_")}

    cols_d, cols_n, names = {}, {}, []
    for sig, stat in features:
        name = f"{sig}·{stat}"
        names.append(name)
        ser = mes_roll[sig] if sig.startswith("MES_") else roll[stat][sig]
        cols_d[name] = ser.reindex(d_end).to_numpy()
        cols_n[name] = ser.reindex(n_end).to_numpy()
    Xd = pd.DataFrame(cols_d); Xn = pd.DataFrame(cols_n)
    Xd["y"] = 1; Xn["y"] = 0
    full = pd.concat([Xd, Xn], ignore_index=True).dropna()
    return full[names], full["y"].to_numpy(), names


def evaluate_combo(grid: pd.DataFrame, advance: int, obs: int, gran: int,
                   k: int = 6) -> dict:
    """Cross-validated logistic-regression combo of the top-k features at one cell."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    feats = top_features(grid, advance, obs, gran, k)
    X, y, names = feature_matrix(feats, advance, obs, gran)
    n_pos = int(y.sum())
    if n_pos < 5 or len(y) - n_pos < 10:
        return {"advance": advance, "ok": False, "n_pos": n_pos}

    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000,
                                            C=0.5))
    n_splits = min(5, n_pos)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                              n_jobs=-1)[:, 1]
    cv_auc = float(roc_auc_score(y, proba))
    fpr, tpr, _ = roc_curve(y, proba)

    # standardized coefficients from a full-data fit (for interpretation)
    pipe.fit(X, y)
    coef = pipe.named_steps["logisticregression"].coef_[0]
    coefs = sorted(zip(names, coef), key=lambda kv: -abs(kv[1]))
    # best single-feature AUC at this cell (oriented to >=0.5)
    sub = grid[(grid.advance_h == advance) & (grid.obs_days == obs)
               & (grid.gran_h == gran)].dropna(subset=["auc"])
    best_single = float(np.maximum(sub.auc, 1 - sub.auc).max()) if not sub.empty else np.nan

    return {"advance": advance, "ok": True, "cv_auc": cv_auc,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "coefs": coefs,
            "best_single": best_single, "n_pos": n_pos, "n_neg": int(len(y) - n_pos),
            "features": names}


def combo_case_trajectories(grid: pd.DataFrame, obs: int, gran: int, lead: int,
                            k: int = 6,
                            leads=(0, 6, 12, 24, 36, 48, 72, 96, 120, 168),
                            n_defect: int = 5, n_normal: int = 2,
                            group: str = "Plasma") -> dict:
    """Defect-probability trajectory of the combined model for selected cases.

    The classification analogue of :func:`rul.case_trajectories`. The top-``k`` combo
    (a standardized, balanced logistic regression selected at the reference ``lead``)
    is scored on windows ending at ``onset − L`` for a grid of durations ``L``. For a
    defect the probability should *rise* toward the onset; for a normal case it should
    stay low. Defect cases use **leave-one-defect-out** models (honest); normal cases
    use a model fit on all events.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    feats = top_features(grid, lead, obs, gran, k)
    if not feats:
        return {"lead": lead, "feats": [], "defects": [], "normals": []}

    events = defect_events(group)
    onsets = np.sort(defect_onsets(events).values.astype("datetime64[ns]"))
    nozzle, mes = hourly_nozzle(), hourly_mes()
    cov_start = nozzle.index.min().tz_localize(None)
    cov_end = nozzle.index.max().tz_localize(None)
    npts = pd.DatetimeIndex(normal_points(events, gran, cov_start, cov_end))

    sigs = [s for s, _ in feats if not s.startswith("MES_")]
    roll = _rolling_features(nozzle, sigs, obs) if sigs else {}
    mes_roll = {s: _rolling_count(mes, f"{s}__count", obs)
                for s, _ in feats if s.startswith("MES_")}

    def design(end_times) -> np.ndarray:
        idx = pd.DatetimeIndex(end_times).floor("h")
        cols = []
        for sig, stat in feats:
            ser = mes_roll[sig] if sig.startswith("MES_") else roll[stat][sig]
            cols.append(ser.reindex(idx).to_numpy())
        return np.column_stack(cols)

    def fit(Xtr, ytr):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", C=0.5,
                                                max_iter=1000))
        pipe.fit(Xtr, ytr)
        return pipe

    leads = np.asarray(leads)
    lead_td = pd.to_timedelta(leads, unit="h")
    Xpos = design(pd.DatetimeIndex(onsets) - pd.Timedelta(hours=lead))
    Xneg = design(npts - pd.Timedelta(hours=lead))
    Xneg = Xneg[np.isfinite(Xneg).all(axis=1)]
    pos_ok = np.isfinite(Xpos).all(axis=1)

    def traj_for(ref, model):
        """Probability + raw feature values + per-feature log-odds contribution."""
        ends = pd.DatetimeIndex([pd.Timestamp(ref)] * len(leads)) - lead_td
        Xt = design(ends)
        good = np.isfinite(Xt).all(axis=1)
        proba = np.full(len(leads), np.nan)
        value = np.full((len(leads), len(feats)), np.nan)
        contrib = np.full((len(leads), len(feats)), np.nan)
        if good.any():
            sc = model.named_steps["standardscaler"]
            lr = model.named_steps["logisticregression"]
            proba[good] = model.predict_proba(Xt[good])[:, 1]
            value[good] = Xt[good]
            contrib[good] = ((Xt[good] - sc.mean_) / sc.scale_) * lr.coef_[0]
        return proba, value, contrib

    # leave-one-defect-out trajectories
    defects = []
    for i in range(len(onsets)):
        others = [j for j in range(len(onsets)) if j != i and pos_ok[j]]
        if len(others) < 3:
            continue
        Xtr = np.vstack([Xpos[others], Xneg])
        ytr = np.concatenate([np.ones(len(others)), np.zeros(len(Xneg))])
        proba, value, contrib = traj_for(onsets[i], fit(Xtr, ytr))
        near = proba[leads <= lead]
        score = float(np.nanmean(near)) if np.isfinite(near).any() else np.nan
        if not np.isfinite(score):
            continue
        defects.append({"onset": pd.Timestamp(onsets[i]), "leads": leads.tolist(),
                        "proba": proba.tolist(), "value": value.tolist(),
                        "contrib": contrib.tolist(), "score": score})
    defects.sort(key=lambda d: -d["score"])
    defects = defects[:n_defect]

    # normal cases: full model, pick the steadiest-lowest healthy references
    full = fit(np.vstack([Xpos[pos_ok], Xneg]),
               np.concatenate([np.ones(int(pos_ok.sum())), np.zeros(len(Xneg))]))
    cand = npts[npts >= cov_start + pd.Timedelta(hours=int(leads.max()))]
    normals_all = []
    for ref in cand:
        proba, value, contrib = traj_for(ref, full)
        if np.isfinite(proba).mean() < 0.6:
            continue
        normals_all.append({"ref": pd.Timestamp(ref), "leads": leads.tolist(),
                            "proba": proba.tolist(), "value": value.tolist(),
                            "contrib": contrib.tolist(),
                            "score": float(np.nanmean(proba))})
    normals_all.sort(key=lambda d: d["score"])         # lowest = most confidently normal
    normals, used = [], []
    for d in normals_all:
        if all(abs((d["ref"] - r) / pd.Timedelta(days=1)) > 14 for r in used):
            normals.append(d)
            used.append(d["ref"])
        if len(normals) >= n_normal:
            break

    return {"lead": int(lead), "k": k,
            "feats": [f"{s} · {st}" for s, st in feats],
            "defects": defects, "normals": normals}


# --------------------------------------------------------------------------- #
# Robot feasibility (cheap span check; no 5.4 GB scan)
# --------------------------------------------------------------------------- #
def robot_signal_span() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Approximate robot-monitoring coverage from the file head and tail bytes."""
    cache = INTERIM / "robot_signal_span.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return pd.Timestamp(d["min"]), pd.Timestamp(d["max"])
    path = RAW_DIR / ROBOT_FILE
    with open(path, "rb") as f:
        f.readline()                      # header
        first = f.readline().split(b",", 1)[0].decode("utf-8", "ignore")
        f.seek(max(0, path.stat().st_size - 8000))
        tail = f.read().splitlines()
    last = tail[-1].split(b",", 1)[0].decode("utf-8", "ignore")
    tmin = pd.to_datetime(first, utc=True, errors="coerce")
    tmax = pd.to_datetime(last, utc=True, errors="coerce")
    INTERIM.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"min": str(tmin), "max": str(tmax)}))
    return tmin, tmax
