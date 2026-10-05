"""Automated model search for early Plasma-defect prediction.

For each lead time we select the top-k discriminative features (from the cached AUC
grid), then evaluate a panel of candidate learners — the families an AutoML pipeline
chooses among — with stratified cross-validated ROC-AUC. This reveals which model
predicts best and how performance trades off against how *early* the prediction is
made. Results are cached to ``data/interim``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import early_factors as ef

INTERIM = ef.INTERIM
AUTOML_K = 15            # candidate features per lead (model does its own selection)


def candidate_models(n_pos: int, n_neg: int) -> dict:
    """Panel of classifiers (balanced for the heavy class imbalance)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                   HistGradientBoostingClassifier)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def mlp(layers):
        # standardized MLP; small + L2-regularized for the tiny positive class
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=layers, activation="relu", alpha=1e-2,
            max_iter=2000, random_state=7))

    models = {
        "Logistic (L2)": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000)),
        "Random forest": RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=3,
            class_weight="balanced", random_state=7, n_jobs=1),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=3,
            class_weight="balanced", random_state=7, n_jobs=1),
        "Hist GBM": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            class_weight="balanced", random_state=7),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "ANN (1 layer)": mlp((16,)),
        "ANN (2 layers)": mlp((32, 16)),
        "ANN (3 layers)": mlp((32, 16, 8)),
    }
    # (XGBoost is omitted: the installed build is incompatible with sklearn 1.6's
    #  __sklearn_tags__ API, which breaks cross-validation. LightGBM + Hist GBM
    #  already cover gradient boosting.)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=7,
            min_child_samples=5, class_weight="balanced", subsample=0.8,
            colsample_bytree=0.8, random_state=7, n_jobs=1, verbose=-1)
    except Exception:
        pass
    return models


def _cv_auc(model, X, y) -> float:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    n_pos = int(y.sum())
    cv = StratifiedKFold(n_splits=min(5, n_pos), shuffle=True, random_state=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba",
                                  n_jobs=1)[:, 1]
    return float(roc_auc_score(y, proba))


def _safe_cv_auc(model, X, y) -> float:
    """Top-level (picklable) wrapper for parallel model evaluation."""
    try:
        return _cv_auc(model, X, y)
    except Exception:
        return float("nan")


def automl_sweep(grid: pd.DataFrame, obs: int, gran: int, k: int = AUTOML_K,
                 refresh: bool = False) -> pd.DataFrame:
    """CV ROC-AUC of every candidate model at every lead time (cached)."""
    out = INTERIM / "automl_plasma.parquet"
    model_names = set(candidate_models(20, 100).keys())
    feat_sig = ",".join(sorted(ef.MES_SIGNALS) + sorted(ef.SIGNAL_STATS))  # feature-schema fingerprint
    if out.exists() and not refresh:
        cached = pd.read_parquet(out)
        # invalidate if lead-time grid, model panel, or feature schema changed
        if (ef._cache_covers(cached, advance_h=ef.ADVANCE_HOURS)
                and model_names.issubset(set(cached["model"].unique()))
                and str(cached.get("feat_sig", pd.Series(["?"])).iloc[0]) == feat_sig):
            return cached
    from joblib import Parallel, delayed
    rows = []
    for advance in ef.ADVANCE_HOURS:
        feats = ef.top_features(grid, advance, obs, gran, k)
        X, y, names = ef.feature_matrix(feats, advance, obs, gran)
        n_pos = int(y.sum())
        if n_pos < 5 or len(y) - n_pos < 10:
            continue
        models = candidate_models(n_pos, len(y) - n_pos)
        Xn = X.to_numpy()
        # evaluate the model panel in parallel (each model is single-threaded)
        aucs = Parallel(n_jobs=-1)(
            delayed(_safe_cv_auc)(model, Xn, y) for model in models.values())
        for (mname, _), auc in zip(models.items(), aucs):
            rows.append((advance, mname, auc, n_pos, int(len(y) - n_pos)))
    df = pd.DataFrame(rows, columns=["advance_h", "model", "auc",
                                     "n_pos", "n_neg"])
    df["feat_sig"] = feat_sig
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def best_factors(grid: pd.DataFrame, advance: int, obs: int, gran: int,
                 model_name: str, k: int = AUTOML_K) -> list[tuple[str, float]]:
    """Model-agnostic feature importance via permutation (ROC-AUC drop when shuffled).

    Permutation importance works for every learner — including k-NN and the ANNs,
    which expose no native importances — and measures real predictive contribution.
    """
    from sklearn.inspection import permutation_importance

    feats = ef.top_features(grid, advance, obs, gran, k)
    X, y, names = ef.feature_matrix(feats, advance, obs, gran)
    model = candidate_models(int(y.sum()), int(len(y) - y.sum()))[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X.to_numpy(), y)
        r = permutation_importance(model, X.to_numpy(), y, scoring="roc_auc",
                                   n_repeats=12, random_state=7, n_jobs=-1)
    imp = np.clip(r.importances_mean, 0, None)
    if imp.sum() == 0:
        imp = np.ones(len(names))
    pairs = sorted(zip(names, imp / imp.sum()), key=lambda kv: -kv[1])
    return pairs
