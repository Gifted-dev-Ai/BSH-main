"""Persist the trained models from every analysis to ``<project_root>/models/``.

The analyses fit thousands of throw-away cross-validation-fold estimators; those are
not artifacts worth keeping. What this module saves is, for each analysis, **every
candidate model refit on all available data** (the deployable form) plus the feature
list / target metadata needed to reuse it. Sklearn estimators are written with
``joblib``; the PyTorch sequence nets as ``state_dict`` + an architecture sidecar.

Layout::

    models/
      manifest.json                      # index of every saved artifact
      rul/        <model>.joblib + .json # RUL regressors (ramp, default obs/horizon)
      automl/     <model>.joblib + .json # early-warning classifiers (best lead)
      combined/   logistic.joblib + .json# combined top-k logistic (best lead)
      staging/    <model>.joblib + .json # 5-stage classifiers + focused alarm
      sequence/   <net>.pt + .json       # best LSTM/GRU/CNN classifier & RUL net
      survival/   <model>.joblib + .json # Random Survival Forest + Cox PH (run-to-failure)

Run with ``python -m src.eda.model_export`` (or ``python -m src.eda --export-models``).
"""

from __future__ import annotations

import json
import re

import numpy as np

from .config import PROJECT_ROOT
from . import early_factors as ef

MODELS_DIR = PROJECT_ROOT / "models"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _safe(X) -> np.ndarray:
    """Float matrix with no NaN/inf and no float32-overflowing magnitudes."""
    A = np.asarray(X, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(A, -1e30, 1e30)


def _dump_sklearn(model, subdir: str, name: str, meta: dict) -> dict:
    import joblib
    d = MODELS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{_slug(name)}.joblib"
    # compress: a 150-tree Random Survival Forest is ~1 GB raw; level 3 cuts it ~20×
    # with transparent loading (and shrinks every other artifact too)
    joblib.dump(model, d / fname, compress=3)
    meta = {"file": f"{subdir}/{fname}", "framework": "sklearn", **meta}
    (d / f"{_slug(name)}.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


# --------------------------------------------------------------------------- #
# Per-analysis exporters (each returns a list of manifest entries)
# --------------------------------------------------------------------------- #
def _export_rul() -> list[dict]:
    from . import rul
    obs, horizon = rul.DEFAULT_OBS, rul.DEFAULT_HORIZON
    _, feat_cols = rul.build_xy(obs)
    Xf, y, groups, _ = rul._ramp(obs, horizon, rul.RUL_GRAN_H)
    Xf = _safe(Xf)
    try:
        sw = rul.sweep().dropna(subset=["mae"])
        d = sw[(sw.obs_days == obs) & (sw.horizon_h == horizon)]
        best = d.loc[d["mae"].idxmin(), "model"]
        maes = dict(zip(d["model"], d["mae"]))
    except Exception:
        best, maes = None, {}
    out = []
    for name, model in rul.candidate_regressors().items():
        try:
            model.fit(Xf, y)
        except Exception as e:
            out.append({"analysis": "rul", "model": name, "error": str(e)[:160]})
            continue
        out.append(_dump_sklearn(model, "rul", name, {
            "analysis": "rul", "task": "regression (hours-to-onset, clamped)",
            "model": name, "is_best": name == best,
            "cv_mae_h": round(float(maes.get(name, float("nan"))), 2),
            "obs_days": obs, "horizon_h": horizon, "n_train": int(len(y)),
            "n_features": len(feat_cols), "features": feat_cols}))
    return out


def _export_automl() -> list[dict]:
    from . import automl
    grid = ef.compute_auc_grid("Plasma")
    obs, gran = ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H
    sweep = automl.automl_sweep(grid, obs, gran).dropna(subset=["auc"])
    best_row = sweep.loc[sweep["auc"].idxmax()]
    advance = int(best_row["advance_h"])
    aucs = dict(zip(zip(sweep["advance_h"], sweep["model"]), sweep["auc"]))
    feats = ef.top_features(grid, advance, obs, gran, automl.AUTOML_K)
    X, y, names = ef.feature_matrix(feats, advance, obs, gran)
    Xn = _safe(X)
    n_pos = int(y.sum())
    out = []
    for name, model in automl.candidate_models(n_pos, len(y) - n_pos).items():
        try:
            model.fit(Xn, y)
        except Exception as e:
            out.append({"analysis": "automl", "model": name, "error": str(e)[:160]})
            continue
        out.append(_dump_sklearn(model, "automl", name, {
            "analysis": "automl", "task": "binary early-warning (pre-defect vs normal)",
            "model": name, "lead_h": advance, "is_best": name == best_row["model"],
            "cv_roc_auc": round(float(aucs.get((advance, name), float("nan"))), 3),
            "obs_days": obs, "gran_h": gran, "n_pos": n_pos,
            "n_neg": int(len(y) - n_pos), "features": names}))
    return out


def _export_combined() -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    grid = ef.compute_auc_grid("Plasma")
    obs, gran, k = ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H, 6
    combos = {a: ef.evaluate_combo(grid, a, obs, gran, k) for a in ef.ADVANCE_HOURS}
    ok = {a: r for a, r in combos.items() if r.get("ok")}
    if not ok:
        return [{"analysis": "combined", "error": "no fittable combo"}]
    best_a = max(ok, key=lambda a: ok[a]["cv_auc"])
    feats = ef.top_features(grid, best_a, obs, gran, k)
    X, y, names = ef.feature_matrix(feats, best_a, obs, gran)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=0.5,
                                            max_iter=1000))
    pipe.fit(_safe(X), y)
    return [_dump_sklearn(pipe, "combined", "logistic", {
        "analysis": "combined", "task": "binary early-warning (top-k logistic)",
        "model": "Logistic (L2)", "lead_h": int(best_a), "is_best": True,
        "cv_roc_auc": round(float(ok[best_a]["cv_auc"]), 3), "obs_days": obs,
        "gran_h": gran, "k": k, "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()), "features": names})]


def _export_staging() -> list[dict]:
    from . import staging as st
    out = []
    res = st.compute()
    best5 = res["best"]["model"]
    qwk = {m["model"]: m["qwk"] for m in res["models"]}
    X, feat_cols = st.build_stage_xy()
    Xf, y = _safe(X[feat_cols].to_numpy()), X["__y"].to_numpy()
    for name, model in st.candidate_classifiers().items():
        try:
            model.fit(Xf, y)
        except Exception as e:
            out.append({"analysis": "staging", "model": name, "error": str(e)[:160]})
            continue
        out.append(_dump_sklearn(model, "staging", f"stage5_{name}", {
            "analysis": "staging", "task": "ordinal 5-stage classification",
            "model": name, "is_best": name == best5,
            "cv_qwk": round(float(qwk.get(name, float("nan"))), 3),
            "obs_days": res["obs_days"], "gran_h": res["gran_h"],
            "stages": st.STAGES, "n_train": int(len(y)), "features": feat_cols}))
    # focused short-lead alarm (the deployable win)
    fa = res.get("focused_alarm", {})
    foc = fa.get("focused", {})
    if foc.get("model"):
        Xs, fcs = st.build_stage_xy(obs_days=st.ALARM_OBS_D)
        acols = [c for c in fcs if any(key in c for key in st.ALARM_FEATURE_KEYS)]
        bs = (Xs["__y"].to_numpy() >= st.ALARM_THR_STAGE).astype(int)
        model = st.candidate_classifiers()[foc["model"]]
        try:
            model.fit(_safe(Xs[acols].to_numpy()), bs)
            out.append(_dump_sklearn(model, "staging", "focused_alarm", {
                "analysis": "staging", "task": "binary act-now alarm (<=24h or down)",
                "model": foc["model"], "is_best": True,
                "cv_roc_auc": round(float(foc.get("auc", float("nan"))), 3),
                "cv_avg_precision": round(float(foc.get("ap", float("nan"))), 3),
                "obs_h": st.ALARM_OBS_D * 24, "n_pos": int(bs.sum()),
                "n_neg": int((bs == 0).sum()), "features": acols}))
        except Exception as e:
            out.append({"analysis": "staging", "model": "focused_alarm",
                        "error": str(e)[:160]})
    return out


def _export_sequence() -> list[dict]:
    out = []
    try:
        import torch
        from . import seq_models as sm
    except Exception as e:
        return [{"analysis": "sequence", "error": f"torch unavailable: {e}"}]

    d = MODELS_DIR / "sequence"
    d.mkdir(parents=True, exist_ok=True)
    res = sm.compute()

    def _save_net(model, name, meta):
        torch.save(model.state_dict(), d / f"{name}.pt")
        meta = {"file": f"sequence/{name}.pt", "framework": "torch",
                "reload": "rebuild via seq_models._make_model(kind, C); load_state_dict",
                **meta}
        (d / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    # best classifier (the one used for explainability)
    exp = res.get("explain") or {}
    if exp.get("model"):
        lead = exp["lead_h"]
        X, y, _, signals = sm.build_classification(lead)
        pw = float((y == 0).sum() / max(1, (y == 1).sum()))
        model = sm._fit(sm._KIND[exp["model"]], X, y, X.shape[2], "clf", pos_weight=pw)
        out.append(_save_net(model, "best_classifier", {
            "analysis": "sequence", "task": "binary early-warning (raw sequence)",
            "kind": exp["model"], "lead_h": lead, "C": int(X.shape[2]),
            "hidden": sm._HIDDEN, "seq_len_h": sm.SEQ_LEN_H,
            "cv_roc_auc": round(float(exp.get("auc", float("nan"))), 3),
            "is_best": True, "channels": signals}))
    # best RUL net (lowest MAE)
    rul_rows = [r for r in res.get("rul", []) if r.get("mae") == r.get("mae")]
    if rul_rows:
        br = min(rul_rows, key=lambda r: r["mae"])
        Xr, yr, _, signals = sm.build_rul()
        model = sm._fit(sm._KIND[br["model"]], Xr, yr / sm.RUL_HORIZON_H,
                        Xr.shape[2], "reg")
        out.append(_save_net(model, "best_rul", {
            "analysis": "sequence", "task": "RUL regression (raw sequence, y/horizon)",
            "kind": br["model"], "C": int(Xr.shape[2]), "hidden": sm._HIDDEN,
            "seq_len_h": sm.SEQ_LEN_H, "horizon_h": sm.RUL_HORIZON_H,
            "cv_mae_h": round(float(br["mae"]), 2), "is_best": True,
            "channels": signals}))
    return out


def _export_survival() -> list[dict]:
    from . import survival as sv
    if not sv.have_sksurv():
        return [{"analysis": "survival", "error": "scikit-survival not installed"}]
    res = sv.compute()
    cidx = {m["model"]: m["c_index"] for m in res.get("models", [])}
    best = (res.get("best") or {}).get("model")
    data = sv.build_surv()
    F = _safe(data["F"])
    y = sv._structured_y(data["event"], data["time"])
    out = []
    for name, model in sv.candidate_models().items():
        try:
            model.fit(F, y)
        except Exception as e:
            out.append({"analysis": "survival", "model": name, "error": str(e)[:160]})
            continue
        out.append(_dump_sklearn(model, "survival", name, {
            "analysis": "survival",
            "task": "right-censored time-to-onset (survival; predict() = risk score)",
            "model": name, "is_best": name == best,
            "cv_c_index": round(float(cidx.get(name, float("nan"))), 3),
            "obs_days": data["obs_days"], "gran_h": data["gran_h"],
            "n_train": int(len(data["time"])), "n_events": int(data["event"].sum()),
            "n_censored": int((~data["event"]).sum()),
            "n_features": len(data["feat_cols"]), "features": data["feat_cols"]}))
    return out


def export_all() -> dict:
    """Train (on all data) and persist every analysis's models; write a manifest."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for fn in (_export_rul, _export_automl, _export_combined, _export_staging,
               _export_sequence, _export_survival):
        try:
            entries.extend(fn())
        except Exception as e:  # never let one analysis abort the rest
            entries.append({"analysis": fn.__name__, "error": str(e)[:200]})
    saved = [e for e in entries if "file" in e]
    manifest = {"n_saved": len(saved),
                "n_errors": len(entries) - len(saved), "models": entries}
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    m = export_all()
    print(f"Saved {m['n_saved']} models to {MODELS_DIR} "
          f"({m['n_errors']} skipped). Manifest: {MODELS_DIR / 'manifest.json'}")
    by = {}
    for e in m["models"]:
        if "file" in e:
            by.setdefault(e["analysis"], []).append(e.get("model", e.get("kind", "?")))
    for a, names in by.items():
        print(f"  {a:10s} {len(names):2d}: {', '.join(names)}")
    for e in m["models"]:
        if "error" in e:
            print(f"  ! {e.get('analysis')}/{e.get('model','?')}: {e['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
