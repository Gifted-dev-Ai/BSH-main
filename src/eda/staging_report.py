"""Report subsection for degradation-**stage** classification.

Reframes RUL as an ordinal 5-stage problem (Normal → Failed), shows the stage
definitions and class balance, the grouped-CV model panel vs a majority baseline,
the confusion matrix, and the coarse 'approaching-failure' alarm AUCs — reported
honestly (the pre-failure stages do not separate out-of-sample with 15 events).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .charts import _div, PRIMARY, ACCENT
from . import staging as st


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')


def _stage_table(res: dict) -> str:
    """Stage definition table — driven by the scheme stored in the result."""
    sch = res["scheme"]
    counts = res["stage_counts"]
    total = max(1, sum(counts))
    head = ("<tr><th>Stage</th><th>RUL (to next onset)</th><th>Meaning / action</th>"
            "<th>samples</th></tr>")
    rows = []
    for i, name in enumerate(sch["stages"]):
        rows.append(
            f'<tr><td><span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:2px;background:{sch["colors"][i]};margin-right:6px"></span>'
            f'<b>{name}</b></td><td>{sch["rul_labels"][i]}</td>'
            f'<td>{sch["meanings"][i]}</td><td>{counts[i]:,} '
            f'({100*counts[i]/total:.0f}%)</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _model_table(res: dict) -> str:
    head = ("<tr><th>Model (grouped CV)</th><th>accuracy</th>"
            "<th>within ±1</th><th>balanced acc.</th><th>macro-F1</th>"
            "<th>QWK</th><th>MCC</th><th>stage-MAE</th></tr>")

    def _row(name, m, style=""):
        return (f'<tr{style}><td class="cname">{name}</td><td>{m["acc"]:.2f}</td>'
                f'<td>{m["within1"]:.2f}</td><td>{m["bal_acc"]:.2f}</td>'
                f'<td>{m["macro_f1"]:.2f}</td><td><b>{m["qwk"]:+.2f}</b></td>'
                f'<td>{m["mcc"]:+.2f}</td><td>{m["stage_mae"]:.2f}</td></tr>')

    rows = [_row(m["model"], m) for m in sorted(res["models"], key=lambda r: -r["qwk"])]
    rows.append(_row("majority-class baseline", res["baseline"],
                     style=' style="color:#52606d"'))
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _confusion_chart(res: dict) -> str:
    stages = res["scheme"]["stages"]
    cm = np.asarray(res["best"]["confusion"], dtype=float)
    rown = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)   # recall per row
    text = [[f"{int(cm[i, j])}<br>{rown[i, j]*100:.0f}%" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=rown, x=stages, y=stages, zmin=0, zmax=1, colorscale="Blues",
        colorbar=dict(title="row %"), text=text, texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="actual %{y}<br>predicted %{x}<br>%{z:.0%} of actual<extra></extra>"))
    fig.update_layout(
        title=f"Confusion matrix — {res['best']['model']} (row-normalised)",
        xaxis_title="predicted stage", yaxis_title="actual stage")
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(automargin=True)
    return _div(fig, height=420, margin=dict(l=110, r=20, t=44, b=80))


def _recall_chart(res: dict) -> str:
    sch = res["scheme"]
    rc = res["best"]["recall"]
    fig = go.Figure(go.Bar(x=sch["stages"], y=rc, marker_color=sch["colors"],
                           hovertemplate="%{x}: recall %{y:.2f}<extra></extra>"))
    fig.update_layout(
        title=f"Per-stage recall — {res['best']['model']} (grouped CV)",
        yaxis_title="recall (caught / actual)", yaxis_range=[0, 1])
    return _div(fig, height=360)


def _alarm_roc_chart(fa: dict) -> str:
    f, b = fa["focused"], fa["baseline"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color="#9fb3c8", dash="dash")))
    fig.add_trace(go.Scatter(
        x=b["fpr"], y=b["tpr"], mode="lines",
        name=f"all features · 5 d window (AUC {b['auc']:.2f})",
        line=dict(color=PRIMARY, width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=f["fpr"], y=f["tpr"], mode="lines",
        name=f"MES sensor+total · 6 h window (AUC {f['auc']:.2f})",
        line=dict(color=ACCENT, width=2.5)))
    op = f["op"]
    fig.add_trace(go.Scatter(
        x=[op["fpr"]], y=[op["tpr"]], mode="markers",
        name=f"operating point — recall {op['recall']:.0%}",
        marker=dict(color=ACCENT, size=11, symbol="circle-open",
                    line=dict(width=2))))
    fig.update_layout(
        title="Act-now alarm (≤ 24 h to failure / down) — ROC, grouped CV",
        xaxis_title="false-positive rate", yaxis_title="true-positive rate (recall)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return _div(fig, height=420, margin=dict(l=60, r=20, t=44, b=96))


def build_section(refresh: bool = False) -> dict:
    res = st.compute(refresh=refresh)
    if not res.get("models"):
        return {"id": "staging", "label": "Degradation staging",
                "html": '<p class="muted">Stage classification produced no valid '
                        'models.</p>'}

    best, base, alarm = res["best"], res["baseline"], res.get("alarm", {})
    a24 = alarm.get("le24h", float("nan"))
    fa = res.get("focused_alarm", {})
    f_auc = fa.get("focused", {}).get("auc", float("nan"))
    b_auc = fa.get("baseline", {}).get("auc", float("nan"))
    f_rec = fa.get("focused", {}).get("op", {}).get("recall", float("nan"))

    intro = (
        '<p class="desc">Exact remaining-useful-life proved unreliable here (honest '
        'grouped-CV barely beats predict-the-mean). A coarser, more robust target is '
        'the <b>health stage</b>: bucket the hours-to-next-onset into five ordered '
        'maintenance categories and classify <i>those</i>. Same windowed Nozzle/MES '
        'features, same leakage-free <b>GroupKFold</b> over whole run-to-failure '
        'segments. The fine 5-way split turns out to be too much to ask of 15 events — '
        'but <b>coarsening to a binary "act-now" alarm and shortening the window to '
        'recent MES sensor errors</b> recovers a genuinely useful signal (below).</p>'
        '<div class="kpi-row">'
        + _kpi("Act-now alarm AUC", f"{f_auc:.2f}" if f_auc == f_auc else "—",
               f"vs {b_auc:.2f} kitchen-sink")
        + _kpi("Alarm recall", f"{f_rec:.0%}" if f_rec == f_rec else "—",
               "imminent failures caught")
        + _kpi("5-stage within ±1", f"{best['within1']:.2f}",
               f"exact-stage {best['acc']:.2f}")
        + _kpi("5-stage balanced acc.", f"{best['bal_acc']:.2f}",
               f"≈ majority baseline {base['bal_acc']:.2f}")
        + '</div>')

    focused = (
        '<h5 style="margin:20px 0 8px;color:#52606d">The one thing that works — a '
        'focused short-lead alarm</h5>'
        '<p class="desc">The 5-stage attempt fails because it asks the 15 events to '
        'pin a stage 1–7 days out. Two changes fix the question rather than the model: '
        '<b>(1) coarsen</b> the target to the single actionable call — <i>"act now: a '
        'defect within 24 h, or already down"</i> — and <b>(2) shorten</b> the window '
        f'to a <b>{fa.get("obs_d", 0.25)*24:.0f} h</b> look-back of just the '
        '<b>MES sensor + total error counts</b> (the only signal that survived the '
        'leave-one-defect-out test earlier). Under the same whole-event GroupKFold the '
        f'ROC-AUC rises to <b>{f_auc:.2f}</b> (from <b>{b_auc:.2f}</b> for the '
        'all-features / 5-day "kitchen sink" on the identical target) — modest, but '
        'real and leakage-free.</p>'
        + f'<div class="card">{_alarm_roc_chart(fa)}</div>'
        + '<p class="desc">At the marked operating point the alarm catches '
        f'<b>{f_rec:.0%}</b> of imminent failures at a '
        f'{fa.get("focused", {}).get("op", {}).get("fpr", float("nan")):.0%} '
        'false-positive rate. Under this heavy imbalance the imbalance-honest metric is '
        '<b>average precision</b> (PR-AUC): '
        f'<b>{fa.get("focused", {}).get("ap", float("nan")):.2f}</b> against a '
        f'no-skill baseline of {fa.get("focused", {}).get("prevalence", float("nan")):.2f} '
        '(the prevalence) — about a 3× lift, but still low in absolute terms, which is '
        'why precision is poor '
        f'({fa.get("focused", {}).get("op", {}).get("precision", float("nan")):.0%}): '
        'imminent points are only ~4% of the timeline. So it is best used as a '
        '<i>triage flag</i> — "look now" — not an unattended trip. This is the '
        'practical takeaway of the whole modelling study: don\'t predict a precise '
        'countdown; raise a coarse, short-lead, sensor-driven alert.</p>')

    verdict = (
        '<p class="desc" style="border-left:3px solid #00b3a4;padding-left:10px">'
        '<b>Honest verdict.</b> Fine 5-stage RUL classification still '
        '<b>collapses toward "Normal"</b> under whole-event hold-out (per-stage recall '
        'for the pre-failure stages near zero) — the same <b>15-event ceiling</b> the '
        'RUL, leave-one-defect-out and sequence-model sections hit. But the right '
        '<i>operational</i> reframing does pay off: coarsening to a binary act-now call '
        'and focusing on recent MES sensor errors lifts the alarm to a usable '
        f'<b>AUC {f_auc:.2f}</b> with <b>{f_rec:.0%}</b> recall. Granularity and '
        'feature scope, not model complexity, were the levers.</p>')

    html = (
        intro
        + focused
        + '<h5 style="margin:20px 0 8px;color:#52606d">For completeness — the full '
        '5-stage attempt</h5>'
        + '<p class="desc">"Failed" is the active down-window '
        '(<code>onset … end</code>); the rest split the lead-up by hours-to-onset. '
        'Note the heavy imbalance — most of the timeline is healthy, and under honest '
        'CV the models cannot resolve the minority pre-failure stages.</p>'
        + _stage_table(res)
        + '<p class="desc">Because the classes are <b>imbalanced and ordinal</b>, raw '
        'accuracy is misleading (predicting "Normal" everywhere already scores '
        f'{base["acc"]:.2f}). The trustworthy summaries are <b>quadratic-weighted '
        'kappa</b> (QWK — chance-corrected, penalises far-off stage errors), <b>MCC</b> '
        '(imbalance-robust) and <b>stage-MAE</b> (mean ordinal distance); models are '
        'selected by QWK. <i>(Synthetic oversampling like SMOTE is deliberately avoided '
        '— with autocorrelated windows and 15 events it would leak across the '
        'event-held-out folds and inflate the scores.)</i></p>'
        + _model_table(res)
        + f'<p class="muted">Even the QWK-best model (<b>{best["model"]}</b>) lands at '
        f'<b>QWK {best["qwk"]:+.2f}</b> / MCC {best["mcc"]:+.2f} — i.e. barely above '
        'chance once the metric stops rewarding the all-"Normal" collapse that '
        'accuracy and within-±1 flatter.</p>'
        + '<div class="grid">'
        + f'<div class="half">{_confusion_chart(res)}</div>'
        + f'<div class="half">{_recall_chart(res)}</div>'
        + '</div>'
        + verdict
        + '<p class="muted" style="margin-top:12px">Setup: '
        f'{res["n_samples"]:,} sample points on a {res["gran_h"]} h grid; the 5-stage '
        f'model uses {res["n_features"]} features over a {res["obs_days"]} d window, '
        f'the focused alarm just {fa.get("focused", {}).get("model", "?")} on '
        f'{fa.get("n_features", "?")} MES features over '
        f'{fa.get("obs_d", 0.25)*24:.0f} h. Classes balanced via loss weighting; '
        'features winsorised at the 0.1/99.9 percentiles.</p>')

    return {"id": "staging", "label": "Degradation staging", "html": html}


# --------------------------------------------------------------------------- #
# Coarser 3-stage variant (Normal / Degrading / Critical)
# --------------------------------------------------------------------------- #
def build_section_3stage(refresh: bool = False) -> dict:
    """Same pipeline as the 5-stage section, but with the coarse 3-stage scheme."""
    res = st.compute(refresh=refresh, scheme=st.SCHEME_3)
    if not res.get("models"):
        return {"id": "staging3", "label": "Degradation staging (3-stage)",
                "html": '<p class="muted">3-stage classification produced no valid '
                        'models.</p>'}

    sch = res["scheme"]
    best, base = res["best"], res["baseline"]
    alarm = res.get("alarm", {})
    a24 = alarm.get("le24h", float("nan"))      # ≤24 h or down (Critical) — the act-now AUC
    a168 = alarm.get("le168h", float("nan"))    # ≤168 h (Degrading or worse)
    crit_recall = best["recall"][-1]            # recall of the top (Critical) stage
    # 5-stage result (cached) for a like-for-like QWK comparison
    try:
        qwk5 = st.compute(scheme=st.SCHEME_5)["best"]["qwk"]
    except Exception:
        qwk5 = float("nan")

    def _f(x, fmt="{:.2f}"):
        return fmt.format(x) if x == x else "—"

    kpis = (
        '<div class="kpi-row">'
        + _kpi("Act-now AUC (Critical)", _f(a24),
               "stage model · ≤24 h or down")
        + _kpi("Critical recall", _f(crit_recall, "{:.0%}"),
               "≤24 h / down caught")
        + _kpi("3-stage QWK", f"{_f(best['qwk'], '{:+.2f}')}",
               f"5-stage was {_f(qwk5, '{:+.2f}')}")
        + _kpi("Balanced acc.", _f(best["bal_acc"]),
               f"baseline {_f(base['bal_acc'])}")
        + '</div>')

    intro = (
        '<h4 style="margin:18px 0 8px">Plasma — coarse 3-stage health staging</h4>'
        '<p class="desc">The same degradation-staging pipeline as above, but collapsed '
        'from five stages to <b>three</b>: '
        f'<b>{sch["stages"][0]}</b> (&gt; 168 h to onset), '
        f'<b>{sch["stages"][1]}</b> (24–168 h) and '
        f'<b>{sch["stages"][2]}</b> (≤ 24 h <i>or</i> already in the down-window — the '
        'former "Failure&nbsp;Imminent" and "Failed" stages merged into one actionable '
        '"act&nbsp;now" class). Fewer, larger classes and one fewer ordinal boundary '
        'make the target coarser and easier to read (a single actionable class); '
        'everything else is identical — the same windowed Nozzle/MES features, the same '
        'leakage-free <b>GroupKFold</b> over whole run-to-failure segments, and the same '
        'QWK-based model selection.</p>'
        + kpis)

    stage_block = (
        '<p class="desc">The three stages and their class balance (still '
        'heavily skewed toward Normal):</p>'
        + _stage_table(res)
        + '<p class="desc">As with the 5-stage version the classes are <b>imbalanced '
        'and ordinal</b>, so the trustworthy summaries are <b>QWK</b>, <b>MCC</b> and '
        f'<b>stage-MAE</b> (predicting "Normal" everywhere already scores '
        f'{base["acc"]:.2f} raw accuracy). Models are selected by QWK.</p>'
        + _model_table(res))

    alarm_line = (
        '<p class="desc">Collapsing the 3-stage model to the two binary '
        '"approaching-failure" questions (best model, grouped CV): '
        f'<b>≤ 24 h or down</b> (Critical) reaches ROC-AUC <b>{_f(a24)}</b>, while the '
        f'softer <b>≤ 168 h</b> (Degrading or worse) sits at <b>{_f(a168)}</b>. The '
        'dedicated short-window <b>MES-sensor act-now alarm</b> in the 5-stage section '
        'above targets this exact <i>Critical</i> class and pushes the AUC higher still '
        '— that focused alarm is the deployable signal; the multi-class model here is '
        'the diagnostic view.</p>')

    better = best["qwk"] > qwk5 if (best["qwk"] == best["qwk"] and qwk5 == qwk5) else False
    verdict = (
        '<p class="desc" style="border-left:3px solid #00b3a4;padding-left:10px">'
        '<b>Honest verdict.</b> '
        + ('Coarsening to three stages helps a little, but does not break the '
           '<b>15-event ceiling</b>.' if better else
           'Coarsening to three stages does <b>not</b> break the '
           '<b>15-event ceiling</b>.')
        + ' The QWK-best model still collapses onto Normal — QWK '
        f'<b>{_f(best["qwk"], "{:+.2f}")}</b> (≈ chance'
        + ('' if better else f', no better than the {_f(qwk5, "{:+.2f}")} at five stages')
        + f'), Critical (≤ 24 h / down) recall only <b>{_f(crit_recall, "{:.0%}")}</b>, '
        f'and the Critical-vs-rest collapse sits at AUC <b>{_f(a24)}</b> (≈ chance). The '
        f'high within-±1 ({_f(best["within1"])}) is mostly an artifact of having fewer '
        'stages, not new skill. As in the RUL, sequence and survival sections, the lever '
        'is <i>target granularity and feature scope</i> — the short-lead MES act-now '
        'alarm in the 5-stage section above (AUC ≈ 0.70) — not the number of stages or '
        'the model.</p>')

    html = (
        intro
        + stage_block
        + '<div class="grid">'
        + f'<div class="half">{_confusion_chart(res)}</div>'
        + f'<div class="half">{_recall_chart(res)}</div>'
        + '</div>'
        + alarm_line
        + verdict
        + '<p class="muted" style="margin-top:12px">Setup: '
        f'{res["n_samples"]:,} sample points on a {res["gran_h"]} h grid; '
        f'{res["n_features"]} features over a {res["obs_days"]} d window; 3 stages with '
        'RUL edges at 168 h and 24 h. Classes balanced via loss weighting; features '
        'winsorised at the 0.1/99.9 percentiles. Same data and CV as the 5-stage '
        'section — only the label granularity differs.</p>')

    return {"id": "staging3", "label": "Degradation staging (3-stage)", "html": html}
