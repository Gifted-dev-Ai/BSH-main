"""Report section for the AutoML early-prediction model search.

Renders the lead-time vs model-performance trade-off (one line per candidate model
plus the best-achievable envelope), the best model per lead, and the best predictive
factors (feature importances of the overall-best model).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .charts import PRIMARY, _div
from . import early_factors as ef
from . import automl


def _tradeoff_chart(sweep: pd.DataFrame) -> str:
    sw = sweep.dropna(subset=["auc"])
    fig = go.Figure()
    for model, g in sw.groupby("model"):
        g = g.sort_values("advance_h")
        fig.add_trace(go.Scatter(x=g.advance_h, y=g.auc, mode="lines+markers",
                                 name=model, line=dict(width=1.6)))
    # best-achievable envelope (max over models at each lead)
    env = sw.groupby("advance_h")["auc"].max()
    fig.add_trace(go.Scatter(x=env.index, y=env.values, mode="lines",
                             name="best of all models", line=dict(color="#11161f",
                             width=3)))
    fig.add_hline(y=0.5, line=dict(color="#9fb3c8", dash="dash"),
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_layout(
        title="Lead time vs model performance (AutoML search, CV ROC-AUC)",
        xaxis_title="lead time before defect (hours) — earlier is harder",
        yaxis_title="ROC-AUC (cross-validated)", yaxis_range=[0.3, 1.0],
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    return _div(fig, height=420, margin=dict(l=60, r=20, t=44, b=96))


def _best_per_lead_table(sweep: pd.DataFrame) -> str:
    sw = sweep.dropna(subset=["auc"])
    best = sw.loc[sw.groupby("advance_h")["auc"].idxmax()].sort_values("advance_h")
    head = ("<tr><th>Lead time</th><th>best model</th><th>CV ROC-AUC</th>"
            "<th>n defect / normal</th></tr>")
    rows = []
    for _, r in best.iterrows():
        rows.append(f'<tr><td>{int(r.advance_h)} h</td><td class="cname">{r.model}</td>'
                    f'<td>{r.auc:.2f}</td><td>{int(r.n_pos)} / {int(r.n_neg)}</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _factor_bar(grid: pd.DataFrame, sweep: pd.DataFrame, obs: int, gran: int) -> str:
    sw = sweep.dropna(subset=["auc"])
    best = sw.loc[sw["auc"].idxmax()]
    factors = automl.best_factors(grid, int(best.advance_h), obs, gran, best.model)[:12]
    labels = [n for n, _ in factors][::-1]
    vals = [v * 100 for _, v in factors][::-1]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=PRIMARY,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(
        title=f"Best predictive factors — {best.model} @ {int(best.advance_h)} h "
              f"lead (AUC {best.auc:.2f})",
        xaxis_title="relative importance (%)")
    fig.update_yaxes(automargin=True)
    return _div(fig, height=max(360, 26 * len(labels) + 130))


def build_section(refresh: bool = False) -> dict:
    grid = ef.compute_auc_grid("Plasma")
    sweep = automl.automl_sweep(grid, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H,
                                refresh=refresh)
    if sweep.dropna(subset=["auc"]).empty:
        html = '<p class="muted">AutoML search produced no valid models.</p>'
        return {"id": "automl", "label": "AutoML early prediction", "html": html}

    n_models = sweep["model"].nunique()
    intro = (
        f'<p class="desc">An automated model search evaluates <b>{n_models} learner '
        'families</b> (logistic, random forest, extra-trees, hist-gradient-boosting, '
        'LightGBM) at every lead time, each scored by <b>stratified cross-validated '
        'ROC-AUC</b> on the top-15 candidate features (the model does its own '
        'selection). The black line is the <b>best achievable</b> performance at each '
        'lead — the core <i>early-vs-accurate</i> trade-off. Exploratory: 15 defect '
        'events, so AUC estimates are noisy.</p>')

    html = (
        intro
        + f'<div class="card">{_tradeoff_chart(sweep)}</div>'
        + '<div class="grid">'
        + f'<div class="half"><h5 style="margin:8px 0;color:#52606d">Best model per lead time</h5>'
        + f'{_best_per_lead_table(sweep)}</div>'
        + f'<div class="half">{_factor_bar(grid, sweep, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)}</div>'
        + '</div>')
    return {"id": "automl", "label": "AutoML early prediction", "html": html}
