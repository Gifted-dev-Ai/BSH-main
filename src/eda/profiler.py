"""Dataset profiling engine.

Two execution paths share one set of partial-statistic primitives so their output
is identical in shape:

* small / medium files  -> loaded fully and profiled as a single chunk
* large files (>500 MB) -> split into byte ranges and profiled in parallel with a
  process pool (each worker reads its own slice, returns compact partial stats)

The partial stats are streaming-friendly (count / sum / sumsq / min / max,
capped value-counts, hourly time bins, a bounded random sample), so the 5.4 GB
robot file never has to live in memory all at once.
"""

from __future__ import annotations

import io
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import (ID_CARDINALITY, N_WORKERS, SAMPLE_ROWS, DatasetConfig)

_DISTINCT_CAP = 5_000          # stop growing a column's distinct-set past this
_TOPK_PER_CHUNK = 200          # value-counts kept per chunk before merging
_SAMPLE_PER_CHUNK = 8_000      # numeric rows sampled per chunk for charts
_GRP_PREFIX = "grp::"          # per-filter-group count columns in the ts frame


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class Schema:
    all_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_col: str | None
    highlight_cols: list[str]
    ts_regex: str | None = None
    ts_format: str | None = None
    ts_freq: str = "h"
    ts_filter_col: str | None = None
    ts_filter_values: tuple[str, ...] = ()


def infer_schema(cfg: DatasetConfig, probe_rows: int = 5_000) -> Schema:
    """Read a small head of the file to decide each column's role."""
    head = pd.read_csv(cfg.path, nrows=probe_rows, sep=cfg.sep,
                       encoding=cfg.encoding, low_memory=False)
    all_cols = [c for c in head.columns if c not in cfg.drop_cols]
    forced_cat = set(cfg.categorical_cols)
    ts = cfg.timestamp_col if cfg.timestamp_col in all_cols else None

    numeric, categorical = [], []
    for col in all_cols:
        if col == ts:
            continue
        if col in forced_cat:
            categorical.append(col)
            continue
        s = head[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            frac = coerced.notna().mean() if len(s) else 0.0
            (numeric if frac >= 0.9 else categorical).append(col)

    highlights = [c for c in cfg.highlight_cols if c in numeric]
    return Schema(all_cols, numeric, categorical, ts, highlights,
                  ts_regex=cfg.timestamp_regex, ts_format=cfg.timestamp_format,
                  ts_freq=cfg.ts_freq, ts_filter_col=cfg.ts_filter_col,
                  ts_filter_values=tuple(cfg.ts_filter_values))


# --------------------------------------------------------------------------- #
# Partial statistics from a DataFrame chunk
# --------------------------------------------------------------------------- #
def _profile_dataframe(df: pd.DataFrame, schema: Schema, rng: np.random.Generator) -> dict:
    n = len(df)
    numeric: dict[str, dict] = {}
    for col in schema.numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        vals = s.to_numpy(dtype="float64")
        finite = vals[np.isfinite(vals)]
        numeric[col] = {
            "count": int(finite.size),
            "missing": int(n - finite.size),
            "sum": float(finite.sum()) if finite.size else 0.0,
            "sumsq": float(np.square(finite).sum()) if finite.size else 0.0,
            "min": float(finite.min()) if finite.size else np.nan,
            "max": float(finite.max()) if finite.size else np.nan,
            "zeros": int((finite == 0).sum()),
            "negs": int((finite < 0).sum()),
        }

    categorical: dict[str, dict] = {}
    for col in schema.categorical_cols:
        s = df[col].astype("string")
        missing = int(s.isna().sum())
        vc = s.value_counts(dropna=True)
        distinct = set(vc.index[:_DISTINCT_CAP].tolist())
        categorical[col] = {
            "counts": Counter(vc.head(_TOPK_PER_CHUNK).to_dict()),
            "missing": missing,
            "distinct": distinct,
            "saturated": vc.size > _DISTINCT_CAP,
        }

    ts_part = None
    if schema.datetime_col:
        raw = df[schema.datetime_col]
        if schema.ts_format:
            if schema.ts_regex:
                raw = raw.astype("string").str.extract(
                    f"({schema.ts_regex})", expand=False)
            t = pd.to_datetime(raw, errors="coerce", format=schema.ts_format)
            if t.dt.tz is None:
                t = t.dt.tz_localize("UTC")
        else:
            t = pd.to_datetime(raw, errors="coerce", utc=True, format="ISO8601")
        valid = t.notna()
        # optional group filter — applied to the charted series only, not to the
        # column's genuine NaT missing-count
        keep = valid
        if schema.ts_filter_col and schema.ts_filter_col in df.columns:
            in_grp = df[schema.ts_filter_col].astype("string").isin(
                schema.ts_filter_values).to_numpy()
            keep = valid & in_grp
        if valid.any():
            hours = t[keep].dt.floor(schema.ts_freq)
            frame = pd.DataFrame({"hour": hours.values})
            for col in schema.highlight_cols:
                frame[col] = pd.to_numeric(df.loc[keep, col], errors="coerce").values
            # split the count into one column per filter group (e.g. Plasma /
            # Roboter) so the chart can colour each series separately
            group_vals = list(schema.ts_filter_values) if schema.ts_filter_col else []
            if group_vals:
                frame["_grp"] = df.loc[keep, schema.ts_filter_col].astype("string").values
            grouped = frame.groupby("hour")
            hour_bins: dict = {}
            counts = grouped.size()
            sums = grouped[schema.highlight_cols].sum() if schema.highlight_cols else None
            grp_counts = None
            if group_vals:
                grp_counts = (frame.groupby(["hour", "_grp"]).size()
                              .unstack("_grp").reindex(columns=group_vals).fillna(0))
            for hour, cnt in counts.items():
                row = [int(cnt)]
                if sums is not None:
                    row.extend(float(x) for x in sums.loc[hour].tolist())
                if grp_counts is not None:
                    row.extend(int(x) for x in grp_counts.loc[hour].tolist())
                hour_bins[pd.Timestamp(hour).value] = row
            tk = t[keep]
            ts_part = {
                "min": tk.min().value if keep.any() else t.min().value,
                "max": tk.max().value if keep.any() else t.max().value,
                "missing": int((~valid).sum()),
                "hours": hour_bins,
            }
        else:
            ts_part = {"min": None, "max": None, "missing": n, "hours": {}}

    # bounded numeric sample for histograms / correlation
    if schema.numeric_cols and n:
        take = min(_SAMPLE_PER_CHUNK, n)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        sample = df.iloc[idx][schema.numeric_cols].apply(pd.to_numeric, errors="coerce")
    else:
        sample = pd.DataFrame(columns=schema.numeric_cols)

    return {"n_rows": n, "numeric": numeric, "categorical": categorical,
            "ts": ts_part, "sample": sample}


# --------------------------------------------------------------------------- #
# Merge partials
# --------------------------------------------------------------------------- #
def _merge_partials(parts: list[dict], schema: Schema) -> dict:
    parts = [p for p in parts if p]
    merged_num: dict[str, dict] = {}
    for col in schema.numeric_cols:
        agg = {"count": 0, "missing": 0, "sum": 0.0, "sumsq": 0.0,
               "min": np.inf, "max": -np.inf, "zeros": 0, "negs": 0}
        for p in parts:
            d = p["numeric"][col]
            agg["count"] += d["count"]; agg["missing"] += d["missing"]
            agg["sum"] += d["sum"]; agg["sumsq"] += d["sumsq"]
            agg["zeros"] += d["zeros"]; agg["negs"] += d["negs"]
            if d["count"]:
                agg["min"] = min(agg["min"], d["min"])
                agg["max"] = max(agg["max"], d["max"])
        if agg["count"] == 0:
            agg["min"] = agg["max"] = np.nan
        merged_num[col] = agg

    merged_cat: dict[str, dict] = {}
    for col in schema.categorical_cols:
        counts: Counter = Counter()
        distinct: set = set()
        saturated = False
        missing = 0
        for p in parts:
            d = p["categorical"][col]
            counts.update(d["counts"]); missing += d["missing"]
            saturated = saturated or d["saturated"]
            if len(distinct) < _DISTINCT_CAP:
                distinct |= d["distinct"]
            if len(distinct) >= _DISTINCT_CAP:
                saturated = True
        merged_cat[col] = {"counts": counts, "missing": missing,
                           "distinct": distinct, "saturated": saturated}

    ts_merged = None
    if schema.datetime_col:
        mins = [p["ts"]["min"] for p in parts if p["ts"] and p["ts"]["min"] is not None]
        maxs = [p["ts"]["max"] for p in parts if p["ts"] and p["ts"]["max"] is not None]
        hours: dict = {}
        missing = 0
        for p in parts:
            if not p["ts"]:
                continue
            missing += p["ts"]["missing"]
            for h, row in p["ts"]["hours"].items():
                if h in hours:
                    hours[h] = [a + b for a, b in zip(hours[h], row)]
                else:
                    hours[h] = list(row)
        ts_merged = {"min": min(mins) if mins else None,
                     "max": max(maxs) if maxs else None,
                     "missing": missing, "hours": hours}

    samples = [p["sample"] for p in parts if len(p["sample"])]
    sample = pd.concat(samples, ignore_index=True) if samples else \
        pd.DataFrame(columns=schema.numeric_cols)
    if len(sample) > SAMPLE_ROWS:
        sample = sample.sample(SAMPLE_ROWS, random_state=7).reset_index(drop=True)

    n_rows = sum(p["n_rows"] for p in parts)
    return {"n_rows": n_rows, "numeric": merged_num, "categorical": merged_cat,
            "ts": ts_merged, "sample": sample}


# --------------------------------------------------------------------------- #
# Parallel byte-range reader
# --------------------------------------------------------------------------- #
def _byte_ranges(path, n_chunks: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    step = size // n_chunks
    bounds = [i * step for i in range(n_chunks)] + [size]
    return list(zip(bounds[:-1], bounds[1:]))


def _profile_byte_range(args) -> dict:
    """Worker: read one byte slice (aligned to line boundaries) and profile it."""
    (path, start, end, header_cols, schema, sep, encoding, seed) = args
    with open(path, "rb") as f:
        if start != 0:
            f.seek(start)
            f.readline()  # drop the partial line straddling `start`
        chunks = []
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            chunks.append(line)
        raw = b"".join(chunks)

    if not raw.strip():
        return None
    buf = io.BytesIO(raw)
    read_kw = dict(sep=sep, encoding=encoding, low_memory=False)
    if start == 0:
        df = pd.read_csv(buf, **read_kw)
        df = df[[c for c in header_cols if c in df.columns]]
    else:
        df = pd.read_csv(buf, header=None, names=header_cols, **read_kw)
    rng = np.random.default_rng(seed)
    return _profile_dataframe(df, schema, rng)


# --------------------------------------------------------------------------- #
# Finalize -> DatasetProfile
# --------------------------------------------------------------------------- #
@dataclass
class ColumnProfile:
    name: str
    kind: str                      # numeric | categorical | id | datetime | constant
    count: int
    missing: int
    missing_pct: float
    stats: dict = field(default_factory=dict)   # numeric summary
    top: list = field(default_factory=list)     # [(value, count)] for categorical
    cardinality: str = ""                       # textual ("12" or ">5000")


@dataclass
class DatasetProfile:
    cfg: DatasetConfig
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    numeric_sample: pd.DataFrame
    timeseries: pd.DataFrame | None
    highlight_cols: list[str]
    datetime_col: str | None
    ts_min: pd.Timestamp | None
    ts_max: pd.Timestamp | None
    engine: str
    n_workers: int
    load_seconds: float
    size_bytes: int
    # Optional dataset-specific report subsections: [{"id","label","html"}, ...]
    extra_sections: list = field(default_factory=list)


def _finalize(merged: dict, schema: Schema, cfg: DatasetConfig,
              engine: str, n_workers: int, load_seconds: float) -> DatasetProfile:
    n = merged["n_rows"]
    cols: list[ColumnProfile] = []

    for name in schema.all_cols:
        if name == schema.datetime_col:
            ts = merged["ts"]
            miss = ts["missing"] if ts else 0
            cols.append(ColumnProfile(name, "datetime", n - miss, miss,
                                      100.0 * miss / n if n else 0.0))
        elif name in merged["numeric"]:
            d = merged["numeric"][name]
            cnt = d["count"]
            mean = d["sum"] / cnt if cnt else np.nan
            var = (d["sumsq"] / cnt - mean ** 2) if cnt else np.nan
            std = float(np.sqrt(var)) if cnt and var > 0 else 0.0
            kind = "constant" if cnt and d["min"] == d["max"] else "numeric"
            cols.append(ColumnProfile(
                name, kind, cnt, d["missing"],
                100.0 * d["missing"] / n if n else 0.0,
                stats={"mean": mean, "std": std, "min": d["min"], "max": d["max"],
                       "zeros": d["zeros"], "negatives": d["negs"]}))
        elif name in merged["categorical"]:
            d = merged["categorical"][name]
            card = len(d["distinct"])
            card_txt = f">{_DISTINCT_CAP}" if d["saturated"] else str(card)
            kind = "id" if (d["saturated"] or card > ID_CARDINALITY) else \
                ("constant" if card <= 1 else "categorical")
            top = d["counts"].most_common(30)
            cols.append(ColumnProfile(
                name, kind, n - d["missing"], d["missing"],
                100.0 * d["missing"] / n if n else 0.0,
                top=top, cardinality=card_txt))

    # quantiles from sample, attached to numeric profiles
    sample = merged["sample"]
    for cp in cols:
        if cp.kind in ("numeric", "constant") and cp.name in sample.columns:
            s = pd.to_numeric(sample[cp.name], errors="coerce").dropna()
            if len(s):
                cp.stats["p25"] = float(s.quantile(0.25))
                cp.stats["p50"] = float(s.quantile(0.50))
                cp.stats["p75"] = float(s.quantile(0.75))

    # time-series frame
    timeseries = None
    ts_min = ts_max = None
    if schema.datetime_col and merged["ts"] and merged["ts"]["hours"]:
        ts = merged["ts"]
        group_vals = list(schema.ts_filter_values) if schema.ts_filter_col else []
        rows = []
        for h, vec in sorted(ts["hours"].items()):
            rec = {"hour": pd.Timestamp(h), "count": vec[0]}
            for i, col in enumerate(schema.highlight_cols):
                total, cnt = vec[1 + i], vec[0]
                rec[col] = total / cnt if cnt else np.nan
            base = 1 + len(schema.highlight_cols)
            for j, gv in enumerate(group_vals):
                rec[f"{_GRP_PREFIX}{gv}"] = vec[base + j]
            rows.append(rec)
        timeseries = pd.DataFrame(rows).set_index("hour").sort_index()
        if ts["min"] is not None:
            ts_min = pd.Timestamp(ts["min"])
            ts_max = pd.Timestamp(ts["max"])

    return DatasetProfile(
        cfg=cfg, n_rows=n, n_cols=len(schema.all_cols), columns=cols,
        numeric_sample=sample, timeseries=timeseries,
        highlight_cols=schema.highlight_cols, datetime_col=schema.datetime_col,
        ts_min=ts_min, ts_max=ts_max, engine=engine, n_workers=n_workers,
        load_seconds=load_seconds, size_bytes=cfg.size_bytes)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def profile_dataset(cfg: DatasetConfig) -> DatasetProfile:
    schema = infer_schema(cfg)
    t0 = time.perf_counter()

    if cfg.is_big:
        n_chunks = max(N_WORKERS * 4, N_WORKERS)
        ranges = _byte_ranges(cfg.path, n_chunks)
        tasks = [(cfg.path, s, e, schema.all_cols, schema, cfg.sep, cfg.encoding,
                  1000 + i) for i, (s, e) in enumerate(ranges)]
        with Pool(processes=N_WORKERS) as pool:
            parts = pool.map(_profile_byte_range, tasks)
        merged = _merge_partials(parts, schema)
        engine, workers = "parallel (byte-range)", N_WORKERS
    else:
        df = pd.read_csv(cfg.path, sep=cfg.sep, encoding=cfg.encoding,
                         low_memory=False)
        df = df[[c for c in schema.all_cols if c in df.columns]]
        rng = np.random.default_rng(42)
        merged = _merge_partials([_profile_dataframe(df, schema, rng)], schema)
        engine, workers = "single-process", 1

    load_seconds = time.perf_counter() - t0
    return _finalize(merged, schema, cfg, engine, workers, load_seconds)
