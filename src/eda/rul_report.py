"""Report section for the Remaining-Useful-Life (RUL) analysis.

Presents the leakage-free model/parameter comparison, the temporal-leakage contrast
(naive vs grouped CV), a predicted-vs-actual view, and the useful-prediction-horizon
estimate — i.e. how far ahead the RUL model is still informative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts import PRIMARY, ACCENT, _div
from . import rul


def _model_comparison(sweep: pd.DataFrame, obs: int, horizon: int) -> str:
    s = sweep[(sweep.obs_days == obs) & (sweep.horizon_h == horizon)].dropna(subset=["mae"])
    s = s.sort_values("mae")
    base = s["baseline_mae"].iloc[0]
    colors = [ACCENT if m > base else "#00b3a4" for m in s["mae"]]
    fig = go.Figure(go.Bar(x=s["model"], y=s["mae"], marker_color=colors,
                           hovertemplate="%{x}: MAE %{y:.1f} h<extra></extra>"))
    fig.add_hline(y=base, line=dict(color="#11161f", dash="dash"),
                  annotation_text=f"predict-the-mean baseline ({base:.0f} h)",
                  annotation_position="top left")
    fig.update_layout(title=f"RUL error by model (grouped CV, obs {obs} d, horizon {horizon} h)",
                      yaxis_title="MAE (hours) — lower is better")
    return _div(fig, height=360)


def _param_sweep(sweep: pd.DataFrame) -> str:
    fig = go.Figure()
    for model, g in sweep.dropna(subset=["mae"]).groupby("model"):
        gg = g.groupby("horizon_h")["mae"].mean().sort_index()
        fig.add_trace(go.Scatter(x=gg.index, y=gg.values, mode="lines+markers",
                                 name=model, line=dict(width=1.6)))
    fig.update_layout(title="RUL error vs horizon (averaged over observation lengths)",
                      xaxis_title="RUL horizon ceiling (hours)",
                      yaxis_title="MAE (hours)",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    return _div(fig, height=380, margin=dict(l=60, r=20, t=44, b=110))


def _obs_sweep(sweep: pd.DataFrame, horizon: int) -> str:
    s = sweep[sweep.horizon_h == horizon].dropna(subset=["mae"])
    fig = go.Figure()
    for model, g in s.groupby("model"):
        gg = g.groupby("obs_days")["mae"].mean().sort_index()
        fig.add_trace(go.Scatter(x=gg.index, y=gg.values, mode="lines+markers",
                                 name=model, line=dict(width=1.6)))
    fig.update_layout(title=f"RUL error vs observation length (horizon {horizon} h)",
                      xaxis_title="observation window (days)", yaxis_title="MAE (hours)",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    return _div(fig, height=380, margin=dict(l=60, r=20, t=44, b=110))


def _scatter(pred: pd.DataFrame, model: str, horizon: int) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["true"], y=pred["pred"], mode="markers",
                             marker=dict(color=PRIMARY, size=4, opacity=0.35),
                             name="windows", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, horizon], y=[0, horizon], mode="lines",
                             line=dict(color="#11161f", dash="dash"),
                             name="perfect (y = x)"))
    fig.update_layout(title=f"Predicted vs actual RUL — {model} (grouped CV)",
                      xaxis_title="actual RUL (hours)", yaxis_title="predicted RUL (hours)",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return _div(fig, height=400, margin=dict(l=60, r=20, t=44, b=88))


def _error_vs_rul(ph: dict) -> str:
    bins = ph["bins"]
    if bins.empty:
        return '<p class="muted">Not enough ramp points to chart error vs RUL.</p>'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bins["center"], y=bins["mae"], mode="lines+markers",
                             name="model MAE", line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=bins["center"], y=bins["baseline_mae"], mode="lines+markers",
                             name="predict-the-mean baseline", line=dict(color="#9fb3c8",
                             width=2, dash="dot")))
    fig.update_layout(
        title="Error vs time-to-failure (where the model beats guessing)",
        xaxis_title="actual time before failure (hours)", yaxis_title="MAE (hours)",
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    return _div(fig, height=380, margin=dict(l=60, r=20, t=44, b=104))


def _case_feat_hover(case: dict, feat_names: list[str]):
    """Per-point hover text: the model's top-importance input features and values."""
    vals = case.get("feat_vals")
    if not vals or not feat_names:
        return None
    rows = []
    for row in vals:
        lines = [f"{n} = {v:.3g}" for n, v in zip(feat_names, row)]
        rows.append("<br>".join(lines))
    return [[s] for s in rows]


def _defect_trajectories_chart(ct: dict) -> str:
    """Small multiples: predicted RUL vs true hours-to-onset for each defect case."""
    cases, hz = ct["defects"], ct["horizon"]
    if not cases:
        return '<p class="muted">No defect case had enough ramp points.</p>'
    cols = min(3, len(cases))
    rows = (len(cases) + cols - 1) // cols
    titles = [f"{c['onset'].strftime('%Y-%m-%d')} · ramp-MAE {c['mae']:.0f} h"
              for c in cases]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.18)
    for i, c in enumerate(cases):
        r, cc = i // cols + 1, i % cols + 1
        mx = max(c["tt"]) if c["tt"] else hz
        fig.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode="lines", legendgroup="ideal",
            name="ideal (predicted = true RUL)", showlegend=(i == 0),
            line=dict(color="#9fb3c8", dash="dash")), row=r, col=cc)
        hov = _case_feat_hover(c, ct.get("feat_names", []))
        fig.add_trace(go.Scatter(
            x=c["tt"], y=c["pred"], mode="lines+markers", legendgroup="pred",
            name="predicted RUL", showlegend=(i == 0),
            line=dict(color=ACCENT, width=2), marker=dict(size=5),
            customdata=hov,
            hovertemplate=("true %{x:.0f} h → pred %{y:.0f} h"
                           + ("<br>──── top inputs ────<br>%{customdata[0]}"
                              if hov else "") + "<extra></extra>")),
            row=r, col=cc)
    fig.update_xaxes(autorange="reversed")          # onset (0 h) on the right
    fig.update_yaxes(range=[0, hz * 1.03])
    fig.update_xaxes(title_text="true hours to onset (→ failure)", row=rows)
    fig.update_yaxes(title_text="predicted RUL (h)", col=1)
    fig.update_layout(
        title="Predicted-RUL countdown — 5 best-tracked defect cases "
              "(leave-one-defect-out)",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
    return _div(fig, height=210 * rows + 150, margin=dict(l=64, r=20, t=96, b=56))


def _normal_trajectories_chart(ct: dict) -> str:
    """Small multiples: predicted RUL over time for healthy (normal) cases."""
    cases, hz = ct["normals"], ct["horizon"]
    if not cases:
        return '<p class="muted">No healthy stretch long enough to plot.</p>'
    cols = min(2, len(cases))
    fig = make_subplots(rows=1, cols=cols,
                        subplot_titles=[f"healthy stretch ending {c['ref'].strftime('%Y-%m-%d')}"
                                        for c in cases],
                        horizontal_spacing=0.09)
    for i, c in enumerate(cases):
        cc = i + 1
        xmax = max(c["hours_before"]) if c["hours_before"] else hz
        fig.add_trace(go.Scatter(
            x=[0, xmax], y=[hz, hz], mode="lines", legendgroup="hz",
            name=f"horizon ({hz} h) — 'no failure ahead'", showlegend=(i == 0),
            line=dict(color="#9fb3c8", dash="dash")), row=1, col=cc)
        hov = _case_feat_hover(c, ct.get("feat_names", []))
        fig.add_trace(go.Scatter(
            x=c["hours_before"], y=c["pred"], mode="lines+markers", legendgroup="pred",
            name="predicted RUL", showlegend=(i == 0),
            line=dict(color="#00b3a4", width=2), marker=dict(size=5),
            customdata=hov,
            hovertemplate=("%{x:.0f} h before ref → pred %{y:.0f} h"
                           + ("<br>──── top inputs ────<br>%{customdata[0]}"
                              if hov else "") + "<extra></extra>")),
            row=1, col=cc)
    fig.update_xaxes(autorange="reversed")          # 'now' (0 h) on the right
    fig.update_yaxes(range=[0, hz * 1.05])
    fig.update_xaxes(title_text="hours before reference")
    fig.update_yaxes(title_text="predicted RUL (h)", col=1)
    fig.update_layout(
        title="Predicted RUL on healthy cases — should stay near the horizon (no alarm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
    return _div(fig, height=380, margin=dict(l=64, r=20, t=96, b=56))


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')


def build_section(refresh: bool = False) -> dict:
    sweep = rul.sweep(refresh=refresh)
    ok = sweep.dropna(subset=["mae"])
    if ok.empty:
        return {"id": "rul", "label": "Remaining useful life",
                "html": '<p class="muted">RUL sweep produced no valid models.</p>'}

    obs, horizon = rul.DEFAULT_OBS, rul.DEFAULT_HORIZON
    dflt = ok[(ok.obs_days == obs) & (ok.horizon_h == horizon)]
    best = dflt.loc[dflt["mae"].idxmin()]
    base = float(dflt["baseline_mae"].iloc[0])

    pred = rul.predictions(obs, horizon, best.model)
    ph = rul.prognostic_horizon(pred, horizon)
    demo = rul.leakage_demo(obs, horizon, best.model)
    cases = rul.case_trajectories(obs, horizon, best.model)

    useful = ph["horizon"]
    useful_txt = (f"{useful:.0f} h" if useful and not np.isnan(useful) and useful > 0
                  else "≈ 0 h")
    kpis = "".join([
        _kpi("Best model (grouped CV)", best.model, f"MAE {best.mae:.0f} h"),
        _kpi("Mean baseline", f"{base:.0f} h", "MAE if you always guess the mean"),
        _kpi("Useful horizon", useful_txt, "lead where model beats baseline ≤24 h"),
        _kpi("Leakage gap (R²)", f"{demo['naive_r2']:+.2f} → {demo['grouped_r2']:+.2f}",
             "naive KFold → leakage-free GroupKFold"),
    ])

    intro = (
        '<p class="desc"><b>Remaining Useful Life</b> reframes early prediction as '
        'regression: at each time point, predict the <b>hours until the next defect '
        'onset</b> (downtime-adjusted), using the same windowed Nozzle/MES features. '
        'Evaluated <i>within the degradation ramp</i> (RUL &lt; horizon) with '
        '<b>GroupKFold</b> — whole run-to-failure segments are held out, so adjacent '
        'autocorrelated windows cannot leak between train and test.</p>'
        '<div class="card" style="padding:14px 18px"><b>Key terms</b>'
        '<ul style="margin:8px 0 0;line-height:1.7">'
        '<li><b>RUL</b> (the predicted value, hours) — remaining useful life = time '
        'until the next defect onset. A prediction of 40 means "expect the next defect '
        'in ≈ 40 h"; it counts down toward 0 as a failure approaches.</li>'
        '<li><b>Onset</b> — the true defect start (<code>discovery − downtime</code>), '
        'the same reference used for the early factors. RUL is the time from "now" to '
        'the next onset.</li>'
        '<li><b>Horizon</b> — the ceiling applied to RUL. Far from any failure the '
        'exact time-to-failure is unknowable, so RUL is capped (censored) at the '
        'horizon (e.g. 168 h = 7 d); the model only learns the <i>degradation ramp</i> '
        'within that horizon. Larger horizon = trying to see further ahead.</li>'
        '<li><b>Observation window</b> — same as the early factors: the span of past '
        'signal data summarised into features at each time point.</li>'
        '<li><b>MAE</b> (the value compared, hours) — mean absolute error between '
        'predicted and actual RUL. <b>Lower is better</b>; the bar to clear is the '
        '<i>predict-the-mean baseline</i>. <b>R²</b> &gt; 0 means the model beats that '
        'baseline; ≤ 0 means it does not.</li>'
        '</ul></div>')

    warn = (
        '<p class="desc" style="border-left:3px solid #e2001a;padding-left:10px">'
        '<b>Temporal-leakage warning.</b> With ordinary shuffled CV the same model '
        f'scores R² {demo["naive_r2"]:+.2f} (MAE looks excellent) — but that is an '
        'artifact: neighbouring points from one failure ramp end up in both train and '
        f'test. Under leakage-free GroupKFold it collapses to R² {demo["grouped_r2"]:+.2f}, '
        'i.e. <b>no better than predicting the mean</b>. The honest conclusion: with 15 '
        'sparse defects and weak upstream signal, RUL is <b>not yet reliably '
        'predictable</b>, and the useful prediction horizon is very short. More defect '
        'history (or stronger condition signals) is the lever — not a fancier model.</p>')

    html = (
        intro
        + f'<div class="kpi-row">{kpis}</div>'
        + warn
        + '<div class="grid">'
        + f'<div class="half">{_model_comparison(sweep, obs, horizon)}</div>'
        + f'<div class="half">{_error_vs_rul(ph)}</div>'
        + '</div>'
        + '<div class="grid">'
        + f'<div class="half">{_param_sweep(sweep)}</div>'
        + f'<div class="half">{_obs_sweep(sweep, horizon)}</div>'
        + '</div>'
        + f'<div class="card">{_scatter(pred, best.model, horizon)}</div>'
        + '<h5 style="margin:20px 0 8px;color:#52606d">Per-case predicted-RUL '
        'trajectories</h5>'
        + '<p class="desc">For the best model '
        f'(<b>{best.model}</b>) we trace the predicted RUL <i>as a case approaches its '
        'defect</i>. Each defect uses a model trained on the <b>other</b> onsets '
        '(leave-one-defect-out), then predicts its own run-up — read right-to-left, a '
        'good prognostic should <b>count down toward 0</b> at the onset and hug the '
        'dashed <i>ideal</i> line. The 5 best-tracked onsets are shown. <b>Hover any '
        'point</b> to see the model\'s most important input features and their values '
        'at that moment, so you can see which signals shifted as the prediction '
        'changed.</p>'
        + f'<div class="card">{_defect_trajectories_chart(cases)}</div>'
        + '<p class="desc">The two healthy (normal) cases are the contrast: with no '
        'failure ahead, a trustworthy model should keep the predicted RUL <b>pinned '
        'near the horizon</b> — any deep dip would be a false alarm.</p>'
        + f'<div class="card">{_normal_trajectories_chart(cases)}</div>'
        + '<p class="muted">Honest read: even the best-tracked cases only loosely '
        'follow the ideal countdown (they rarely reach 0 at onset), and the healthy '
        'cases sit somewhat below the horizon — consistent with the modest grouped-CV '
        'accuracy. The trajectories show <i>where</i> the model is and isn\'t '
        'informative, case by case.</p>')
    return {"id": "rul", "label": "Remaining useful life", "html": html}
