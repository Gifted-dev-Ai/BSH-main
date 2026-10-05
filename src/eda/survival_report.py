"""Report section for the survival-analysis (Random Survival Forest) reformulation.

Renders the run-to-failure-as-survival study from :mod:`survival`: the model
concordance comparison, the temporal-leakage contrast (naive vs grouped C-index),
risk-stratified Kaplan-Meier curves, in-sample feature importance, and the per-segment
predicted-failure-probability countdowns. Reads only the cached, render-ready dict.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts import PRIMARY, ACCENT, _div
from . import survival as sv

TEAL = "#00b3a4"
GREY = "#9fb3c8"


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _concordance_chart(res: dict) -> str:
    """Grouped-CV concordance per model vs the 0.5 (Kaplan-Meier) baseline."""
    rows = [m for m in res["models"] if m["c_index"] == m["c_index"]]
    names = [m["model"] for m in rows]
    cvals = [m["c_index"] for m in rows]
    colors = [TEAL if c > 0.5 else ACCENT for c in cvals]
    fig = go.Figure(go.Bar(
        x=names, y=cvals, marker_color=colors,
        text=[f"{c:.3f}" for c in cvals], textposition="outside",
        hovertemplate="%{x}: C-index %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color="#11161f", dash="dash"),
                  annotation_text="Kaplan-Meier baseline (C = 0.5, chance)",
                  annotation_position="top left")
    fig.update_layout(
        title="Discrimination by model (Harrell C-index, grouped CV)",
        yaxis_title="concordance — higher is better", yaxis_range=[0, 1])
    return _div(fig, height=360)


def _leakage_chart(res: dict) -> str:
    """Naive shuffled-CV vs leakage-free grouped-CV concordance for the RSF."""
    lk = res["leakage"]
    fig = go.Figure(go.Bar(
        x=["naive shuffled CV", "leakage-free grouped CV"],
        y=[lk["naive_c"], lk["grouped_c"]],
        marker_color=[ACCENT, PRIMARY],
        text=[f"{lk['naive_c']:.3f}", f"{lk['grouped_c']:.3f}"],
        textposition="outside",
        hovertemplate="%{x}: C-index %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=GREY, dash="dash"),
                  annotation_text="chance (0.5)", annotation_position="bottom right")
    fig.update_layout(
        title="Temporal-leakage contrast — Random Survival Forest",
        yaxis_title="concordance index", yaxis_range=[0, 1])
    return _div(fig, height=360)


def _km_terciles_chart(res: dict) -> str:
    """Population KM + KM per RSF out-of-fold risk tercile (does risk separate?)."""
    fig = go.Figure()
    km = res["km"]
    fig.add_trace(go.Scatter(
        x=km["time"], y=km["surv"], mode="lines", line_shape="hv",
        name="population (Kaplan-Meier)", line=dict(color="#11161f", width=2, dash="dot"),
        hovertemplate="t %{x:.0f} h<br>S(t) %{y:.2f}<extra></extra>"))
    colors = {"low risk": TEAL, "medium risk": "#ff9f1c", "high risk": ACCENT}
    for ter in res["terciles"]:
        med = ter["median_surv"]
        med_txt = f"{med:.0f} h" if med == med else "n/a"
        fig.add_trace(go.Scatter(
            x=ter["time"], y=ter["surv"], mode="lines", line_shape="hv",
            name=f"{ter['label']} (n={ter['n']}, median {med_txt})",
            line=dict(color=colors.get(ter["label"], PRIMARY), width=2.2),
            hovertemplate=f"{ter['label']}<br>t %{{x:.0f}} h<br>S(t) %{{y:.2f}}<extra></extra>"))
    fig.update_layout(
        title="Survival curves by RSF risk group — overlap ⇒ risk score is not separating",
        xaxis_title="hours to next defect onset", yaxis_title="survival probability S(t)",
        yaxis_range=[0, 1.02],
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return _div(fig, height=400, margin=dict(l=60, r=20, t=44, b=96))


def _importance_chart(res: dict) -> str:
    """Top in-sample permutation importances (concordance lost when a feature is shuffled)."""
    imp = res["importance"]
    feats, drops = imp["features"][::-1], imp["drops"][::-1]
    fig = go.Figure(go.Bar(
        x=drops, y=feats, orientation="h", marker_color=PRIMARY,
        hovertemplate="%{y}<br>ΔC when shuffled: %{x:.3f}<extra></extra>"))
    fig.update_layout(
        title=f"RSF permutation importance (in-sample, base C = {imp['base_c']:.2f})",
        xaxis_title="concordance lost when the feature is shuffled")
    return _div(fig, height=max(360, 26 * len(feats) + 130),
                margin=dict(l=220, r=20, t=44, b=50))


def _countdown_chart(res: dict) -> str:
    """Small multiples: OOF P(fail ≤ horizon) vs true hours-to-onset per defect segment."""
    cases = res["cases"]["defects"]
    hz = res["pfail_horizon_h"]
    if not cases:
        return '<p class="muted">No segment had enough out-of-fold points to chart.</p>'
    cols = min(3, len(cases))
    rows = (len(cases) + cols - 1) // cols

    def _title(c: dict) -> str:
        head = (c["onset"].strftime("%Y-%m-%d") if c["onset"] is not None
                else f"segment {c['seg']}")
        pbar = c.get("near_pfail")
        ptxt = f" · P̄ {pbar:.2f}" if pbar is not None and pbar == pbar else ""
        return f"{head} · ρ {c['rho']:+.2f}{ptxt}"

    titles = [_title(c) for c in cases]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.18)
    for i, c in enumerate(cases):
        r, cc = i // cols + 1, i % cols + 1
        fig.add_trace(go.Scatter(
            x=[hz, hz], y=[0, 1], mode="lines", legendgroup="hz",
            name=f"prediction horizon ({hz} h)", showlegend=(i == 0),
            line=dict(color=GREY, width=1, dash="dot")), row=r, col=cc)
        fig.add_trace(go.Scatter(
            x=c["tt"], y=c["pfail"], mode="lines+markers", legendgroup="pf",
            name=f"P(fail ≤ {hz} h)", showlegend=(i == 0),
            line=dict(color=ACCENT, width=2), marker=dict(size=5),
            hovertemplate="true %{x:.0f} h to onset<br>P(fail) %{y:.2f}<extra></extra>"),
            row=r, col=cc)
    fig.update_xaxes(autorange="reversed")          # onset (0 h) on the right
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title_text="true hours to onset (→ failure)", row=rows)
    fig.update_yaxes(title_text=f"P(fail ≤ {hz} h)", col=1)
    fig.update_layout(
        title=f"Predicted failure-probability countdown — {len(cases)} best-ranked "
              "segments (out-of-fold)",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
    return _div(fig, height=210 * rows + 150, margin=dict(l=64, r=20, t=96, b=56))


def _normal_countdown_chart(res: dict) -> str:
    """The right-censored healthy tail: P(fail ≤ horizon) should stay low (no alarm)."""
    normals = res["cases"]["normals"]
    hz = res["pfail_horizon_h"]
    if not normals:
        return '<p class="muted">No censored healthy stretch to chart.</p>'
    c = normals[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c["hours_before"], y=c["pfail"], mode="lines+markers",
        name=f"P(fail ≤ {hz} h)", line=dict(color=TEAL, width=2), marker=dict(size=5),
        hovertemplate="%{x:.0f} h before record end<br>P(fail) %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=GREY, dash="dash"),
                  annotation_text="0.5", annotation_position="top right")
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(
        title="Predicted failure probability on the censored healthy tail "
              "(should stay low)",
        xaxis_title="hours before end of record", yaxis_title=f"P(fail ≤ {hz} h)",
        yaxis_range=[0, 1])
    return _div(fig, height=360, margin=dict(l=60, r=20, t=44, b=50))


# --------------------------------------------------------------------------- #
# Section assembly
# --------------------------------------------------------------------------- #
def _intro(res: dict) -> str:
    return (
        '<p class="desc"><b>Survival analysis</b> reframes the run-to-failure data as '
        '<b>time-to-event with censoring</b>. At each grid point (the same windowed '
        'Nozzle/MES features as the RUL section) the target is the <b>time to the next '
        'defect onset</b>; the healthy stretch after the last onset is <b>right-'
        'censored</b> at the end of the record — we know it survived that long, not when '
        'it would fail. A <b>Random Survival Forest</b> (RSF) — a non-parametric '
        'ensemble of log-rank survival trees, after <i>arXiv&nbsp;2504.12156</i> — is '
        'the primary model; it captures complex survival curves <i>without</i> the Cox '
        'proportional-hazards assumption. Evaluated with <b>GroupKFold over whole run-to-'
        'failure segments</b> and scored by Harrell\'s concordance index.</p>'
        '<div class="card" style="padding:14px 18px"><b>Key terms</b>'
        '<ul style="margin:8px 0 0;line-height:1.7">'
        '<li><b>Event / censoring</b> — an observation is an <i>event</i> (δ&nbsp;=&nbsp;1) '
        'if a defect onset is actually reached, and <i>right-censored</i> '
        '(δ&nbsp;=&nbsp;0) if the record ends first. Censoring is what separates '
        'survival analysis from plain regression: censored rows are kept and modelled, '
        'not discarded or mislabelled.</li>'
        '<li><b>Survival function</b> S(t) — the probability the machine is still '
        'failure-free t hours from now. It starts at 1 and steps down at each observed '
        'onset.</li>'
        '<li><b>Random Survival Forest</b> — bagged log-rank trees; each leaf holds a '
        'Nelson-Aalen cumulative hazard, and the forest averages them into a per-row '
        '<i>risk score</i> (higher ⇒ expected to fail sooner) and a full survival curve. '
        'No proportional-hazards assumption.</li>'
        '<li><b>Cox PH</b> — the classical semi-parametric baseline that <i>does</i> '
        'assume a single proportional hazard ratio per feature; RSF is contrasted '
        'against it.</li>'
        '<li><b>Concordance index (C)</b> — the survival analogue of ROC-AUC: the '
        'fraction of comparable pairs the model ranks correctly (the one that fails '
        'sooner gets the higher risk). <b>0.5 = chance</b> (= the Kaplan-Meier '
        'population baseline); 1.0 = perfect ranking.</li>'
        '</ul></div>')


def _warn(res: dict) -> str:
    lk = res["leakage"]
    return (
        '<p class="desc" style="border-left:3px solid #e2001a;padding-left:10px">'
        '<b>Temporal-leakage warning.</b> With ordinary shuffled CV the RSF scores '
        f'C&nbsp;{lk["naive_c"]:.2f} (discrimination looks excellent) — but that is an '
        'artifact: neighbouring windows from one failure ramp land in both train and '
        f'test. Under leakage-free GroupKFold it collapses to C&nbsp;{lk["grouped_c"]:.2f}, '
        '<b>no better than the Kaplan-Meier baseline</b>. The honest conclusion matches '
        'the RUL and staging sections: with 15 sparse defects and weak upstream signal, '
        'time-to-failure is <b>not yet reliably predictable</b> — the non-parametric '
        'survival model is the right tool, but more failure history (or stronger '
        'condition signals) is the lever, not a fancier estimator.</p>')


def build_section(refresh: bool = False) -> dict:
    res = sv.compute(refresh=refresh)
    if not res.get("ok"):
        return {"id": "survival", "label": "Survival models (RSF)",
                "html": f'<p class="muted">Survival section skipped: '
                        f'{res.get("reason", "unavailable")}.</p>'}

    best, lk = res["best"], res["leakage"]
    kpis = "".join([
        _kpi("Best model (grouped CV)", best["model"] or "—",
             f"C-index {best['c_index']:.3f}"),
        _kpi("Kaplan-Meier baseline", "C 0.50", "chance — predict the population curve"),
        _kpi("Leakage gap (C)", f"{lk['naive_c']:.2f} → {lk['grouped_c']:.2f}",
             "naive shuffled → leakage-free grouped"),
        _kpi("Failure events (segments)", f"{res['n_segments']}",
             f"{res['n_samples']:,} at-risk windows · {res['n_censored']} "
             f"right-censored ({res['censoring_frac']*100:.0f}%)"),
    ])

    model_table_rows = "".join(
        f'<tr><td class="cname">{m["model"]}</td>'
        f'<td>{m["c_index"]:.3f}</td>'
        f'<td>{"beats" if m["c_index"] > 0.5 else "≤"} chance</td></tr>'
        for m in res["models"] if m["c_index"] == m["c_index"])
    model_table = (
        '<div class="table-wrap"><table><tr><th>Model</th><th>C-index (grouped CV)</th>'
        f'<th>vs 0.5</th></tr>{model_table_rows}</table></div>')

    if res["cases"]["normals"]:
        mean_pf = res["cases"]["normals"][0]["mean_pfail"]
        healthy_note = (
            '<p class="desc">The right-censored <b>healthy tail</b> is the contrast: '
            'with no failure ahead the predicted probability should stay <i>low</i>. In '
            f'practice it hovers near {mean_pf:.2f} — a moderate standing alarm even with '
            'nothing ahead, the false-positive face of the same weak signal.</p>')
    else:
        healthy_note = ('<p class="desc">The right-censored <b>healthy tail</b> is the '
                        'contrast — with no failure ahead, predicted risk should stay '
                        'low.</p>')

    html = (
        '<h4 style="margin:18px 0 8px">Plasma — run-to-failure as survival '
        '(Random Survival Forest)</h4>'
        + _intro(res)
        + f'<div class="kpi-row">{kpis}</div>'
        + _warn(res)
        + '<div class="grid">'
        + f'<div class="half">{_concordance_chart(res)}</div>'
        + f'<div class="half">{_leakage_chart(res)}</div>'
        + '</div>'
        + '<h5 style="margin:18px 0 8px;color:#52606d">Do the RSF risk groups actually '
        'fail at different rates?</h5>'
        + '<p class="desc">Splitting the windows into <b>risk terciles</b> by the RSF '
        'out-of-fold risk score, each group gets its own Kaplan-Meier curve. If the '
        'model discriminated, the <span style="color:#e2001a">high-risk</span> curve '
        'would drop first (fails soonest) and the <span style="color:#00b3a4">low-risk'
        '</span> curve last. Here the curves <b>overlap</b> — the visual counterpart of '
        f'the C-index sitting at {best["c_index"]:.2f}.</p>'
        + f'<div class="card">{_km_terciles_chart(res)}</div>'
        + model_table
        + '<h5 style="margin:20px 0 8px;color:#52606d">Which signals does the forest '
        'lean on?</h5>'
        + '<p class="desc">Permutation importance on an <i>in-sample</i> fit — the '
        'concordance lost when each feature is shuffled. It shows what the RSF keys on '
        '(nozzle distance-check channels, rod speed, current variability) but, given the '
        'near-chance held-out C-index, treat this as descriptive, not a validated driver '
        'list.</p>'
        + f'<div class="card">{_importance_chart(res)}</div>'
        + '<h5 style="margin:20px 0 8px;color:#52606d">Per-segment failure-probability '
        'countdown</h5>'
        + '<p class="desc">The survival counterpart of the RUL trajectories: for each '
        'run-to-failure segment the RSF (trained on the <b>other</b> segments) predicts '
        f'<b>P(fail within {res["pfail_horizon_h"]} h)</b> at windows ending closer and '
        'closer to the onset, over the ramp (≤ 2× the horizon). Read right-to-left, a '
        'good prognostic should <b>climb toward 1</b> as the defect approaches; ρ is the '
        'rank correlation between the predicted probability and closeness to onset. A '
        '<b>few</b> of the best-ranked segments do climb cleanly, but most stay flat or '
        'fall — the segment-to-segment inconsistency behind the near-chance overall '
        'C-index.</p>'
        + f'<div class="card">{_countdown_chart(res)}</div>'
        + healthy_note
        + f'<div class="card">{_normal_countdown_chart(res)}</div>'
        + '<p class="muted" style="margin-top:14px">Why no integrated-Brier / time-'
        'dependent-AUC: those IPCW metrics need a well-supported censoring distribution, '
        'but here censoring is sparse (one healthy tail) and concentrated at short times, '
        'so they are omitted in favour of the censoring-robust concordance index.</p>')
    return {"id": "survival", "label": "Survival models (RSF)", "html": html}
