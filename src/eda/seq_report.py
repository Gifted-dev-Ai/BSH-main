"""Report section for the deep **sequence** models on raw nozzle signals.

Renders the raw-sequence pipeline, the early-warning ROC-AUC vs lead time (overlaid
on the AutoML summary-statistic envelope), the RUL error vs baseline, and the three
explainability views: channel permutation importance, input-gradient temporal
saliency, and the dominant pre-defect **motif** with its defect/normal lift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .charts import PRIMARY, ACCENT, _div
from . import seq_models as sm


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')


def _automl_envelope() -> pd.Series | None:
    """Best-of-all-models AUC per lead from the tabular AutoML sweep (cached)."""
    try:
        from . import early_factors as ef, automl
        grid = ef.compute_auc_grid("Plasma")
        sweep = automl.automl_sweep(grid, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)
        sw = sweep.dropna(subset=["auc"])
        return sw.groupby("advance_h")["auc"].max()
    except Exception:
        return None


def _clf_chart(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for model, g in df.groupby("model"):
        g = g.sort_values("lead_h")
        fig.add_trace(go.Scatter(x=g.lead_h, y=g.auc, mode="lines+markers",
                                 name=model, line=dict(width=2)))
    env = _automl_envelope()
    if env is not None:
        leads = sorted(df.lead_h.unique())
        ys = [float(env.get(l, np.nan)) for l in leads]
        fig.add_trace(go.Scatter(x=leads, y=ys, mode="lines+markers",
                                 name="AutoML summary-stats (best of all)",
                                 line=dict(color="#11161f", width=2.5, dash="dot")))
    fig.add_hline(y=0.5, line=dict(color="#9fb3c8", dash="dash"),
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_layout(
        title="Early-warning ROC-AUC vs lead time — raw-sequence nets vs summary-stat AutoML",
        xaxis_title="lead time before defect (hours) — earlier is harder",
        yaxis_title="ROC-AUC (grouped CV)", yaxis_range=[0.3, 1.0],
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return _div(fig, height=420, margin=dict(l=60, r=20, t=44, b=100))


def _rul_chart(rows: list[dict]) -> str:
    df = pd.DataFrame(rows).dropna(subset=["mae"]).sort_values("mae")
    base = float(df["baseline_mae"].iloc[0])
    colors = [ACCENT if m > base else "#00b3a4" for m in df["mae"]]
    fig = go.Figure(go.Bar(x=df["model"], y=df["mae"], marker_color=colors,
                           hovertemplate="%{x}: MAE %{y:.1f} h<extra></extra>"))
    fig.add_hline(y=base, line=dict(color="#11161f", dash="dash"),
                  annotation_text=f"predict-the-mean baseline ({base:.0f} h)",
                  annotation_position="top left")
    fig.update_layout(title="RUL error by sequence model (grouped CV)",
                      yaxis_title="MAE (hours) — lower is better")
    return _div(fig, height=360)


def _importance_chart(explain: dict, topn: int = 12) -> str:
    imp = explain["importance"][:topn][::-1]
    labels = [n for n, _ in imp]
    vals = [v for _, v in imp]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=PRIMARY,
                           hovertemplate="%{y}: %{x:.3f} AUC drop<extra></extra>"))
    fig.update_layout(
        title=f"Channel importance — {explain['model']} @ {explain['lead_h']} h lead",
        xaxis_title="ROC-AUC lost when the channel is shuffled")
    fig.update_yaxes(automargin=True)
    return _div(fig, height=max(360, 26 * len(labels) + 130))


def _saliency_chart(explain: dict, topn: int = 12) -> str:
    sal = np.asarray(explain["saliency"])               # [L, C]
    signals = explain["signals"]
    L = sal.shape[0]
    order = [signals.index(n) for n, _ in explain["importance"][:topn]]
    z = sal[:, order].T                                 # [topn, L]
    x = list(range(-(L - 1), 1))                        # hours before cutoff
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=[signals[i] for i in order], colorscale="Magma",
        colorbar=dict(title="|grad|"),
        hovertemplate="%{y}<br>%{x} h before cutoff<br>saliency %{z:.2e}<extra></extra>"))
    fig.update_layout(
        title="Temporal saliency — which hours × channels drive the alarm",
        xaxis_title="hours before prediction cutoff (0 = cutoff)")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _div(fig, height=max(360, 26 * len(order) + 150))


def _saliency_time_chart(explain: dict) -> str:
    sal = np.asarray(explain["saliency"])               # [L, C]
    L = sal.shape[0]
    per_t = sal.mean(axis=1)
    x = list(range(-(L - 1), 1))
    fig = go.Figure(go.Scatter(x=x, y=per_t, mode="lines", fill="tozeroy",
                               line=dict(color=ACCENT, width=2)))
    fig.update_layout(
        title="When in the window the model looks (mean saliency per hour)",
        xaxis_title="hours before prediction cutoff (0 = cutoff)",
        yaxis_title="mean |grad| over channels")
    return _div(fig, height=320)


def _motif_chart(explain: dict) -> str:
    motif = explain.get("motif")
    if not motif:
        return '<p class="muted">Not enough positive windows to extract a motif.</p>'
    c = np.asarray(motif["centroid"])
    m = motif["m"]
    x = list(range(-(m - 1), 1))
    fig = go.Figure(go.Scatter(x=x, y=c, mode="lines+markers",
                               line=dict(color=PRIMARY, width=2.5)))
    fig.update_layout(
        title=f"Dominant pre-defect motif on {explain['top_channel']} "
              f"({m} h, z-normalised)",
        xaxis_title="hours within the motif (0 = motif end)",
        yaxis_title="z-score")
    return _div(fig, height=320)


def build_section(refresh: bool = False) -> dict:
    res = sm.compute(refresh=refresh)
    clf, rul_rows, explain = res["classification"], res["rul"], res["explain"]
    clf_ok = [r for r in clf if r["auc"] == r["auc"]]   # drop NaN
    if not clf_ok:
        return {"id": "seqdl", "label": "Sequence deep learning",
                "html": '<p class="muted">Sequence models produced no valid scores '
                        '(torch unavailable or too few samples).</p>'}

    best = max(clf_ok, key=lambda r: r["auc"])
    rul_df = pd.DataFrame(rul_rows).dropna(subset=["mae"])
    best_rul = rul_df.sort_values("mae").iloc[0] if not rul_df.empty else None
    n_pos = max(r["n_pos"] for r in clf)

    intro = (
        '<p class="desc">A complementary take on the same problem that feeds the '
        'model the <b>raw time-ordered signal sequence</b> instead of window summary '
        f'statistics. Each sample is a <b>{res["seq_len_h"]} h</b> window of the '
        f'<b>{res["n_channels"]} nozzle signals</b> resampled to a regular hourly grid '
        '(one reading per step, robust-z-scored) and fed to recurrent / convolutional '
        'networks (<b>LSTM, GRU, 1-D CNN</b>, PyTorch). Two heads share the pipeline: '
        '<b>early-warning classification</b> (pre-defect vs normal, by lead time) and '
        '<b>RUL regression</b> (hours-to-next-onset). Validation holds out whole defect '
        'onsets / run-to-failure segments (no temporal leakage). The point is twofold: '
        'see whether temporal structure beats hand-built statistics, and — via '
        '<b>saliency and motif discovery</b> — read off <i>which moments and shapes</i> '
        'foreshadow a defect.</p>'
        '<div class="kpi-row">'
        + _kpi("best early-warning AUC", f"{best['auc']:.2f}",
               f"{best['model']} @ {best['lead_h']} h lead")
        + _kpi("raw-sequence inputs", f"{res['seq_len_h']}h × {res['n_channels']}",
               "hours × signals per window")
        + _kpi("pre-defect windows", f"{n_pos}",
               f"from 15 onsets (×{len(sm.POS_JITTER_H)} jitter)")
        + (_kpi("best RUL MAE", f"{best_rul['mae']:.0f} h",
                f"{best_rul['model']} (base {best_rul['baseline_mae']:.0f} h)")
           if best_rul is not None else "")
        + '</div>'
        '<p class="muted">Exploratory: with only 15 defect events deep nets are '
        'data-starved, so absolute scores are noisy and not expected to beat the '
        'tabular models — the value here is the method and its explanations.</p>')

    explain_html = ""
    if explain:
        explain_html = (
            '<h5 style="margin:18px 0 8px;color:#52606d">What the network keys on</h5>'
            '<p class="desc">Read from the best classifier '
            f'(<b>{explain["model"]}</b> @ {explain["lead_h"]} h lead). '
            '<b>Channel importance</b> is the ROC-AUC lost when a signal is shuffled; '
            '<b>temporal saliency</b> is the input-gradient magnitude, showing which '
            'hours and channels move the alarm; the <b>motif</b> is the recurring '
            'sub-sequence shape (on the top channel) that most marks the lead-up to a '
            'defect.</p>'
            + '<div class="grid">'
            + f'<div class="half">{_importance_chart(explain)}</div>'
            + f'<div class="half">{_saliency_chart(explain)}</div>'
            + '</div>'
            + '<div class="grid">'
            + f'<div class="half">{_saliency_time_chart(explain)}</div>'
            + f'<div class="half">{_motif_chart(explain)}</div>'
            + '</div>')
        motif = explain.get("motif")
        if motif:
            explain_html += (
                f'<p class="desc">The motif appears (within the matched-distance '
                f'threshold) in <b>{motif["p_pos"]*100:.0f}%</b> of pre-defect windows '
                f'vs <b>{motif["p_neg"]*100:.0f}%</b> of normal windows — a '
                f'<b>{motif["lift"]:.1f}×</b> lift toward the lead-up to a defect.</p>')

    html = (
        intro
        + '<h5 style="margin:18px 0 8px;color:#52606d">Early warning — sequence nets '
        'vs summary-statistic AutoML</h5>'
        + f'<div class="card">{_clf_chart(clf_ok)}</div>'
        + '<h5 style="margin:18px 0 8px;color:#52606d">Remaining useful life from raw '
        'sequences</h5>'
        + '<p class="desc">Same windows, regression head predicting hours to the next '
        'onset (clamped at the '
        f'{res["rul_horizon_h"]} h horizon). Bars below the dashed line beat a '
        'predict-the-mean baseline.</p>'
        + f'<div class="card">{_rul_chart(rul_rows)}</div>'
        + explain_html
        + '<p class="muted" style="margin-top:16px">Method notes: hourly-mean '
        'resampling reuses the cached aggregates; positives are augmented by jittering '
        'the window end (±8 h); classes are balanced via loss weighting; models use '
        'early stopping on an internal validation split. Motif distance is '
        'z-normalised Euclidean on the most important channel.</p>')

    return {"id": "seqdl", "label": "Sequence deep learning", "html": html}
