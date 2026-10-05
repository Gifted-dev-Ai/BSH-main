"""Report charts + section assembly for the early-predictive-factor analysis.

Reads the cached AUC grid and hourly aggregates from :mod:`early_factors` and renders
the *Early predictive factors* subsection. **Plasma defects only** — robot defects are
excluded because the robot-monitoring stream does not overlap any robot defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts import ACCENT, PRIMARY, _div
from . import early_factors as ef

TOPK = 5  # number of features averaged for the "separation" summaries


def _feat_label(signal: str, stat: str) -> str:
    return f"{signal} · {stat}"


def _obs_label(obs) -> str:
    """Format an observation-window length (days) as e.g. ``12 h`` or ``3 d``."""
    return f"{int(round(obs * 24))} h" if obs < 1 else f"{int(obs)} d"


def _terms_timeline() -> str:
    """Inline-SVG illustration placing each key term on a single time axis.

    Geometry mirrors the definitions: observation window → lead time → onset →
    downtime → discovery, so ``onset = discovery − downtime`` and the window ends
    at ``onset − lead`` (the prediction cutoff) read directly off the picture.
    """
    return (
        '<div style="margin:12px 0 4px;overflow-x:auto">'
        '<svg viewBox="0 0 880 196" width="100%" '
        'style="min-width:660px;font-family:inherit" role="img" '
        'aria-label="Timeline showing observation window, lead time, onset, '
        'downtime and discovery">'
        '<defs><marker id="ef-ah" markerWidth="9" markerHeight="9" refX="7" '
        'refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#52606d"/>'
        '</marker></defs>'
        # title
        '<text x="40" y="20" font-size="13" font-weight="600" fill="#1f2933">'
        'How the key terms line up on the time axis</text>'
        # bands: observation window / lead / downtime
        '<rect x="70" y="66" width="340" height="48" rx="5" fill="#e7f0ff" '
        'stroke="#0a7cff"/>'
        '<rect x="410" y="66" width="210" height="48" rx="5" fill="#fdecd0" '
        'stroke="#e0a800"/>'
        '<rect x="620" y="66" width="160" height="48" rx="5" fill="#fbe0e3" '
        'stroke="#e2001a"/>'
        '<text x="240" y="89" text-anchor="middle" font-size="12.5" '
        'font-weight="600" fill="#0a4ea8">Observation window</text>'
        '<text x="240" y="105" text-anchor="middle" font-size="10.5" '
        'fill="#52606d">1–10 d · signals summarised</text>'
        '<text x="515" y="89" text-anchor="middle" font-size="12.5" '
        'font-weight="600" fill="#9a6b00">Lead time / advance</text>'
        '<text x="515" y="105" text-anchor="middle" font-size="10.5" '
        'fill="#52606d">1–96 h ahead</text>'
        '<text x="700" y="89" text-anchor="middle" font-size="12.5" '
        'font-weight="600" fill="#b3001a">Downtime</text>'
        '<text x="700" y="105" text-anchor="middle" font-size="10.5" '
        'fill="#52606d">machine down</text>'
        # time axis
        '<line x1="40" y1="142" x2="846" y2="142" stroke="#52606d" '
        'stroke-width="1.5" marker-end="url(#ef-ah)"/>'
        '<text x="838" y="134" text-anchor="end" font-size="10.5" '
        'fill="#52606d">time</text>'
        # connectors from band bottom (114) to axis (142)
        '<line x1="70" y1="114" x2="70" y2="142" stroke="#98a2b3" '
        'stroke-dasharray="3 3"/>'
        '<line x1="410" y1="114" x2="410" y2="142" stroke="#52606d"/>'
        '<line x1="620" y1="114" x2="620" y2="142" stroke="#98a2b3" '
        'stroke-dasharray="3 3"/>'
        '<line x1="780" y1="114" x2="780" y2="142" stroke="#98a2b3" '
        'stroke-dasharray="3 3"/>'
        # critical points on the axis
        '<circle cx="70" cy="142" r="3.5" fill="#0a7cff"/>'
        '<circle cx="410" cy="142" r="3.8" fill="#52606d"/>'
        '<circle cx="620" cy="142" r="3.8" fill="#111827"/>'
        '<circle cx="780" cy="142" r="3.8" fill="#e2001a"/>'
        # point labels below the axis
        '<text x="70" y="160" text-anchor="middle" font-size="10.5" '
        'fill="#52606d">obs start</text>'
        '<text x="410" y="160" text-anchor="middle" font-size="11" '
        'font-weight="600" fill="#364152">onset − lead</text>'
        '<text x="410" y="174" text-anchor="middle" font-size="10" '
        'fill="#52606d">prediction cutoff</text>'
        '<text x="620" y="160" text-anchor="middle" font-size="11" '
        'font-weight="600" fill="#111827">onset</text>'
        '<text x="620" y="174" text-anchor="middle" font-size="10" '
        'fill="#52606d">= discovery − downtime</text>'
        '<text x="780" y="160" text-anchor="middle" font-size="11" '
        'font-weight="600" fill="#b3001a">discovery</text>'
        '<text x="780" y="174" text-anchor="middle" font-size="10" '
        'fill="#52606d">logged time</text>'
        '</svg></div>'
        '<p class="muted" style="margin:2px 0 0">Features come only from the '
        '<b>observation window</b> — data available no later than the '
        '<b>prediction cutoff</b> (<code>onset − lead</code>); nothing to the '
        'right of it is ever seen by the model.</p>')


def _auc_heatmap(grid: pd.DataFrame, obs: int, gran: int, topn: int = 15) -> str:
    sub = grid[(grid.obs_days == obs) & (grid.gran_h == gran)].dropna(subset=["auc"]).copy()
    sub["feat"] = [_feat_label(s, st) for s, st in zip(sub.signal, sub.stat)]
    sub["sep"] = (sub.auc - 0.5).abs()
    top = (sub.groupby("feat")["sep"].max().sort_values(ascending=False)
           .head(topn).index.tolist())
    piv = (sub[sub.feat.isin(top)].pivot_table(index="feat", columns="advance_h",
                                               values="auc")
           .reindex(index=top, columns=list(ef.ADVANCE_HOURS)))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[f"{a} h" for a in piv.columns], y=piv.index,
        zmin=0.0, zmax=1.0, colorscale="Viridis",
        colorbar=dict(title="AUC"), text=piv.values, texttemplate="%{text:.2f}",
        textfont=dict(size=9, color="white"),
        hovertemplate="%{y}<br>lead %{x}<br>AUC = %{z:.2f}<extra></extra>"))
    fig.update_layout(title=f"Defect-vs-normal separation (AUC) — obs {_obs_label(obs)}, "
                            f"normal grid {gran} h")
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(automargin=True, title="lead time before defect")
    return _div(fig, height=max(360, 28 * len(top) + 150))


def _honest_auc_chart(obs: int, gran: int) -> str:
    """In-sample best AUC vs leave-one-defect-out AUC of the selected feature, by lead."""
    df = ef.honest_auc_by_lead(obs=obs, gran=gran)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.advance_h, y=df.in_sample_best, mode="lines+markers",
        name="in-sample best feature (what the heatmap shows)",
        line=dict(color=PRIMARY, width=2, dash="dot"),
        customdata=df.sel_feature,
        hovertemplate="lead %{x} h<br>in-sample AUC %{y:.2f}"
                      "<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df.advance_h, y=df.lodo_best, mode="lines+markers",
        name="leave-one-defect-out (honest, feature re-selected per fold)",
        line=dict(color=ACCENT, width=2.5),
        hovertemplate="lead %{x} h<br>held-out AUC %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color="#9fb3c8", dash="dash"),
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_layout(
        title="Honest check — in-sample vs leave-one-defect-out AUC, by lead time",
        xaxis_title="lead time before defect (hours)",
        yaxis_title="best-feature AUC (oriented ≥ 0.5)", yaxis_range=[0.2, 0.95],
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    return _div(fig, height=380, margin=dict(l=60, r=20, t=44, b=104))


def _mes_category_chart(grid: pd.DataFrame, obs: int, gran: int) -> str:
    """AUC of each MES error-category count, by lead time."""
    sub = grid[(grid.obs_days == obs) & (grid.gran_h == gran)
               & grid.signal.str.startswith("MES_")].dropna(subset=["auc"])
    if sub.empty:
        return ""
    # order legend by how discriminative each category is overall
    order = (sub.assign(sep=(sub.auc - 0.5).abs()).groupby("signal")["sep"].max()
             .sort_values(ascending=False).index)
    fig = go.Figure()
    for sig in order:
        g = sub[sub.signal == sig].sort_values("advance_h")
        fig.add_trace(go.Scatter(x=g.advance_h, y=g.auc, mode="lines+markers",
                                 name=sig.replace("MES_", ""), line=dict(width=1.6)))
    fig.add_hline(y=0.5, line=dict(color="#9fb3c8", dash="dash"),
                  annotation_text="no separation", annotation_position="bottom right")
    fig.update_layout(
        title=f"MES error categories — separation (AUC) by lead time "
              f"(obs {obs} d, grid {gran} h)",
        xaxis_title="lead time before defect (hours)", yaxis_title="AUC",
        yaxis_range=[0.3, 0.75],
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return _div(fig, height=400, margin=dict(l=60, r=20, t=44, b=96))


def _leadtime_sensitivity(grid: pd.DataFrame, gran: int) -> str:
    sub = grid[grid.gran_h == gran].dropna(subset=["auc"]).copy()
    sub["sep"] = (sub.auc - 0.5).abs()
    fig = go.Figure()
    for obs in ef.OBS_DAYS:
        s = sub[sub.obs_days == obs]
        ys = [s[s.advance_h == a]["sep"].nlargest(TOPK).mean() for a in ef.ADVANCE_HOURS]
        fig.add_trace(go.Scatter(x=list(ef.ADVANCE_HOURS), y=ys, mode="lines+markers",
                                 name=f"{_obs_label(obs)} obs"))
    fig.update_layout(
        title=f"Separation vs lead time (mean top-{TOPK} |AUC−0.5|, grid {gran} h)",
        xaxis_title="lead time / advance (hours)",
        yaxis_title=f"mean top-{TOPK} separation",
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    return _div(fig, height=380, margin=dict(l=60, r=20, t=44, b=96))


def _gran_sensitivity(grid: pd.DataFrame, obs: int, advance: int) -> str:
    sub = grid[(grid.obs_days == obs) & (grid.advance_h == advance)].dropna(subset=["auc"]).copy()
    sub["sep"] = (sub.auc - 0.5).abs()
    ys = [sub[sub.gran_h == g]["sep"].nlargest(TOPK).mean() for g in ef.GRAN_HOURS]
    fig = go.Figure(go.Bar(x=[f"{g} h" for g in ef.GRAN_HOURS], y=ys, marker_color=PRIMARY))
    fig.update_layout(title=f"Separation vs normal-grid granularity (obs {obs} d, lead {advance} h)",
                      xaxis_title="normal-sample granularity",
                      yaxis_title=f"mean top-{TOPK} sep")
    return _div(fig, height=320)


def _top_table(grid: pd.DataFrame, advance: int, obs: int, gran: int, topn: int = 8) -> str:
    sub = grid[(grid.advance_h == advance) & (grid.obs_days == obs)
               & (grid.gran_h == gran)].dropna(subset=["auc"]).copy()
    sub["sep"] = (sub.auc - 0.5).abs()
    sub = sub.sort_values("sep", ascending=False).head(topn)
    head = ("<tr><th>State variable</th><th>stat</th><th>AUC</th><th>direction</th>"
            "<th>n def / norm</th></tr>")
    rows = []
    for _, r in sub.iterrows():
        d = "higher before defect" if r.auc > 0.5 else "lower before defect"
        rows.append(f'<tr><td class="cname">{r.signal}</td><td>{r.stat}</td>'
                    f'<td>{r.auc:.2f}</td><td>{d}</td>'
                    f'<td>{int(r.n_defect)} / {int(r.n_normal)}</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _best_feature_box(grid, nozzle, mes, events, obs, gran, advance) -> str:
    sub = grid[(grid.advance_h == advance) & (grid.obs_days == obs)
               & (grid.gran_h == gran)].dropna(subset=["auc"]).copy()
    if sub.empty:
        return ""
    sub["sep"] = (sub.auc - 0.5).abs()
    best = sub.sort_values("sep", ascending=False).iloc[0]
    signal, stat = best.signal, best.stat

    starts = ef.defect_onsets(events)  # downtime-adjusted onset
    cmin, cmax = nozzle.index.min(), nozzle.index.max()
    cov_start = cmin.tz_localize(None) if getattr(cmin, "tz", None) else cmin
    cov_end = cmax.tz_localize(None) if getattr(cmax, "tz", None) else cmax
    npts = ef.normal_points(events, gran, cov_start, cov_end)
    d_end = ef._end_hours(starts, advance)
    n_end = ef._end_hours(npts, advance)
    if signal.startswith("MES_"):
        ser = ef._rolling_count(mes, f"{signal}__count", obs)
        dv, nv = ser.reindex(d_end).to_numpy(), ser.reindex(n_end).to_numpy()
    else:
        frame = ef._rolling_features(nozzle, [signal], obs)[stat]
        dv = frame[signal].reindex(d_end).to_numpy()
        nv = frame[signal].reindex(n_end).to_numpy()
    dv, nv = dv[np.isfinite(dv)], nv[np.isfinite(nv)]
    fig = go.Figure()
    fig.add_trace(go.Box(y=nv, name="normal", marker_color="#9fb3c8",
                         boxpoints="outliers"))
    fig.add_trace(go.Box(y=dv, name="pre-defect", marker_color=ACCENT,
                         boxpoints="all", jitter=0.4, pointpos=0))
    fig.update_layout(title=f"Best separator: {signal} · {stat} "
                            f"(lead {advance} h, obs {obs} d, AUC {best.auc:.2f})",
                      yaxis_title=f"{signal} [{stat}]", showlegend=False)
    return _div(fig, height=380)


def _combo_point_hover(case: dict, feats: list[str], topn: int = 6):
    """Per-point hover text: top input features by |log-odds contribution|."""
    val = np.array(case["value"], dtype=float)
    con = np.array(case["contrib"], dtype=float)
    rows = []
    for p in range(len(case["leads"])):
        v, c = val[p], con[p]
        if not np.isfinite(v).any():
            rows.append("no input data in this window")
            continue
        order = np.argsort(-np.abs(np.nan_to_num(c)))
        lines = []
        for j in order[:topn]:
            if not np.isfinite(v[j]):
                continue
            arrow = "▲ raises risk" if c[j] > 0 else "▼ lowers risk"
            lines.append(f"{feats[j]} = {v[j]:.3g}  ({arrow} {c[j]:+.2f})")
        rows.append("<br>".join(lines) or "—")
    return [[s] for s in rows]


_COMBO_HOVER = ("%{x:.0f} h before onset · P(defect) %{y:.2f}"
                "<br>───── top inputs (Δlog-odds) ─────<br>%{customdata[0]}<extra></extra>")


def _combo_defect_traj_chart(tr: dict) -> str:
    """Small multiples: combined-model defect probability vs hours-before-onset."""
    cases, lead = tr["defects"], tr["lead"]
    if not cases:
        return '<p class="muted">Not enough samples for case trajectories.</p>'
    cols = min(3, len(cases))
    rows = (len(cases) + cols - 1) // cols
    titles = [f"{c['onset'].strftime('%Y-%m-%d')} · P̄ {c['score']:.2f}" for c in cases]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.18)
    for i, c in enumerate(cases):
        r, cc = i // cols + 1, i % cols + 1
        xmax = max(c["leads"])
        fig.add_trace(go.Scatter(
            x=[0, xmax], y=[0.5, 0.5], mode="lines", legendgroup="thr",
            name="0.5 decision threshold", showlegend=(i == 0),
            line=dict(color="#9fb3c8", dash="dash")), row=r, col=cc)
        fig.add_trace(go.Scatter(
            x=[lead, lead], y=[0, 1], mode="lines", legendgroup="ref",
            name=f"reference lead ({lead} h)", showlegend=(i == 0),
            line=dict(color="#cbd2d9", width=1, dash="dot")), row=r, col=cc)
        fig.add_trace(go.Scatter(
            x=c["leads"], y=c["proba"], mode="lines+markers", legendgroup="p",
            name="P(defect)", showlegend=(i == 0),
            line=dict(color=ACCENT, width=2), marker=dict(size=5),
            customdata=_combo_point_hover(c, tr["feats"]),
            hovertemplate=_COMBO_HOVER), row=r, col=cc)
    fig.update_xaxes(autorange="reversed")          # onset (0 h) on the right
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title_text="hours before onset (→ failure)", row=rows)
    fig.update_yaxes(title_text="P(defect)", col=1)
    fig.update_layout(
        title="Combined-model defect probability — 5 best-detected onsets "
              "(leave-one-defect-out)",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
    return _div(fig, height=210 * rows + 150, margin=dict(l=60, r=20, t=96, b=56))


def _combo_normal_traj_chart(tr: dict) -> str:
    """Small multiples: combined-model defect probability over healthy stretches."""
    cases, lead = tr["normals"], tr["lead"]
    if not cases:
        return '<p class="muted">No healthy reference long enough to plot.</p>'
    cols = min(2, len(cases))
    fig = make_subplots(rows=1, cols=cols, horizontal_spacing=0.09,
                        subplot_titles=[f"healthy reference {c['ref'].strftime('%Y-%m-%d')}"
                                        for c in cases])
    for i, c in enumerate(cases):
        cc = i + 1
        xmax = max(c["leads"])
        fig.add_trace(go.Scatter(
            x=[0, xmax], y=[0.5, 0.5], mode="lines", legendgroup="thr",
            name="0.5 decision threshold", showlegend=(i == 0),
            line=dict(color="#9fb3c8", dash="dash")), row=1, col=cc)
        fig.add_trace(go.Scatter(
            x=c["leads"], y=c["proba"], mode="lines+markers", legendgroup="p",
            name="P(defect)", showlegend=(i == 0),
            line=dict(color="#00b3a4", width=2), marker=dict(size=5),
            customdata=_combo_point_hover(c, tr["feats"]),
            hovertemplate=_COMBO_HOVER.replace("before onset", "before ref")),
            row=1, col=cc)
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title_text="hours before reference")
    fig.update_yaxes(title_text="P(defect)", col=1)
    fig.update_layout(
        title="Defect probability on healthy cases — should stay low (no alarm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
    return _div(fig, height=380, margin=dict(l=60, r=20, t=96, b=56))


def _combo_block(grid: pd.DataFrame, obs: int, gran: int, k: int = 6) -> str:
    """Multivariate logistic combo of the top-k features: CV-ROC vs lead time."""
    combos = {a: ef.evaluate_combo(grid, a, obs, gran, k) for a in ef.ADVANCE_HOURS}
    ok = {a: r for a, r in combos.items() if r.get("ok")}
    if not ok:
        return '<p class="muted">Not enough samples to fit the combined model.</p>'

    # 1) combined CV-AUC vs best single feature across lead time
    adv = list(ok.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=adv, y=[ok[a]["cv_auc"] for a in adv],
                             mode="lines+markers", name=f"combined top-{k} (CV ROC-AUC)",
                             line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=adv, y=[ok[a]["best_single"] for a in adv],
                             mode="lines+markers", name="best single feature (AUC)",
                             line=dict(color=PRIMARY, width=2, dash="dot")))
    fig.add_hline(y=0.5, line=dict(color="#9fb3c8", dash="dash"),
                  annotation_text="chance", annotation_position="bottom right")
    fig.update_layout(title="Combined model vs best single factor — by lead time",
                      xaxis_title="lead time before defect (hours)",
                      yaxis_title="ROC-AUC (cross-validated)",
                      yaxis_range=[0.4, 1.0],
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    lead_chart = _div(fig, height=380, margin=dict(l=60, r=20, t=44, b=96))

    # 2) ROC curves at a few lead times
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line=dict(color="#9fb3c8", dash="dash"), showlegend=False))
    for a in [x for x in (1, 6, 12, 24, 48, 72, 96) if x in ok]:
        r = ok[a]
        roc.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                 name=f"{a} h lead (AUC {r['cv_auc']:.2f})"))
    roc.update_layout(title="ROC of combined model at different lead times",
                      xaxis_title="false-positive rate", yaxis_title="true-positive rate",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    roc_chart = _div(roc, height=400, margin=dict(l=60, r=20, t=44, b=92))

    # 3) coefficients at the best-AUC lead time
    best_a = max(ok, key=lambda a: ok[a]["cv_auc"])
    r = ok[best_a]
    head = "<tr><th>Feature</th><th>std. coefficient</th><th>effect</th></tr>"
    rows = []
    for name, c in r["coefs"]:
        eff = "↑ raises defect risk" if c > 0 else "↓ lowers defect risk"
        rows.append(f'<tr><td class="cname">{name}</td><td>{c:+.2f}</td><td>{eff}</td></tr>')
    coef_table = (f'<h5 style="margin:8px 0;color:#52606d">Model weights at {best_a} h '
                  f'lead (CV ROC-AUC {r["cv_auc"]:.2f}, {r["n_pos"]} defect / '
                  f'{r["n_neg"]} normal)</h5>'
                  f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>')

    note = ('<p class="desc">A standardized, L2-regularized <b>logistic regression</b> '
            f'over the top-{k} factors, scored by <b>stratified cross-validation</b> '
            '(out-of-sample ROC-AUC, balanced classes). It tests whether the factors '
            '<i>together</i> predict a defect better than any one alone, and how early. '
            'Still exploratory — 15 positives.</p>')

    # per-case defect-probability trajectories for the best combined model
    tr = ef.combo_case_trajectories(grid, obs, gran, best_a, k)
    traj_html = (
        '<h5 style="margin:20px 0 8px;color:#52606d">Per-case defect-probability '
        'trajectories</h5>'
        + '<p class="desc">The classification counterpart of the RUL trajectories: the '
        f'best combined model (<b>{best_a} h</b> lead, CV ROC-AUC '
        f'{ok[best_a]["cv_auc"]:.2f}) is scored on windows ending at increasing '
        'durations before each onset. Read right-to-left, a good early warning should '
        '<b>rise toward 1</b> as the defect approaches and cross the 0.5 threshold; the '
        'dotted line marks the reference lead where the AUC is measured. Defect models '
        'are leave-one-defect-out (the case is never in training); the 5 best-detected '
        'onsets are shown. <b>Hover any point</b> to see the input features that drove '
        'it — each top factor\'s value and its signed push on the log-odds '
        '(▲ raises / ▼ lowers defect risk), so you can read <i>why</i> the probability '
        'moved between leads.</p>'
        + f'<div class="card">{_combo_defect_traj_chart(tr)}</div>'
        + '<p class="desc">For the two healthy (normal) references the probability '
        'should stay <b>well below 0.5</b> throughout — any sustained rise would be a '
        'false alarm.</p>'
        + f'<div class="card">{_combo_normal_traj_chart(tr)}</div>'
        + '<p class="muted">Honest read: the best onsets do sit above the threshold '
        'near failure and the healthy cases stay low, but the defect trajectories are '
        'noisy and non-monotone — consistent with the ~0.7 in-sample / lower honest '
        'AUC. Treat as a triage signal, not a calibrated probability.</p>')

    return (note
            + '<div class="grid">'
            + f'<div class="half">{lead_chart}</div>'
            + f'<div class="half">{roc_chart}</div></div>'
            + coef_table
            + traj_html)


def build_sections(refresh: bool = False) -> list[dict]:
    """Build the 'Early predictive factors' subsection (Plasma defects only)."""
    group = "Plasma"
    grid = ef.compute_auc_grid(group, refresh=refresh)
    nozzle, mes, events = ef.hourly_nozzle(), ef.hourly_mes(), ef.defect_events(group)
    n_def = len(events)
    nn = grid[grid.gran_h == ef.DEFAULT_GRAN_H]["n_normal"].dropna()
    n_norm = int(nn.iloc[0]) if not nn.empty else 0

    intro = (
        '<p class="desc">Searching for <b>early predictive factors</b>: state '
        'variables whose behaviour <i>ahead of</i> a defect differs from normal '
        'operation. Defect events are bursts of incidents &lt; 1 day apart (first '
        'incident = defect discovery). Because the machine is already down at '
        'discovery, the true fault <b>onset</b> is back-dated by the downtime '
        '(<code>onset = discovery − downtime</code>). For each <b>lead time</b> '
        '(advance, 1–96 h) an <b>observation window</b> (1–10 days) ending at '
        '<code>onset − advance</code> is summarised (min/max/mean/std; MES '
        '<code>event_type=1</code> error counts, <b>total and per fault category</b> '
        'extracted from the error text) and compared — via Mann-Whitney <b>AUC</b> — '
        'against the same windows before '
        '<b>normal</b> time points (grid 2–24 h, kept away from any defect). '
        'AUC 0.5 = no separation; →1 higher / →0 lower before defects.</p>'
        f'<div class="kpi-row">'
        f'<div class="kpi"><div class="kpi-val">{n_def}</div>'
        f'<div class="kpi-label">Plasma defect events</div></div>'
        f'<div class="kpi"><div class="kpi-val">{n_norm:,}</div>'
        f'<div class="kpi-label">normal samples (grid {ef.DEFAULT_GRAN_H} h)</div></div>'
        f'<div class="kpi"><div class="kpi-val">'
        f'{len(ef.ADVANCE_HOURS)}×{len(ef.OBS_DAYS)}×{len(ef.GRAN_HOURS)}</div>'
        f'<div class="kpi-label">advance × obs × granularity grid</div></div></div>'
        '<p class="muted">Exploratory: only 15 defect events; incident times are '
        'treated as UTC to align with the signal streams.</p>')

    terms = (
        '<div class="card" style="padding:14px 18px"><b>Key terms</b>'
        + _terms_timeline() +
        '<ul style="margin:8px 0 0;line-height:1.7">'
        '<li><b>Onset</b> — the true start of a defect. The logged time is when the '
        'defect is <i>discovered</i>, but the machine is already down by then, so the '
        'onset is back-dated by the downtime: <code>onset = discovery − downtime</code>. '
        'All windows are measured from the onset.</li>'
        '<li><b>Lead time</b> (a.k.a. <i>advance</i>, the heatmap columns / x-axes) — '
        'how far <i>before</i> the onset the prediction is made. A 24 h lead uses only '
        'data available 24 h before the failure; the observation window ends at '
        '<code>onset − lead</code>. Larger lead = earlier (more useful, but harder) '
        'warning.</li>'
        '<li><b>Observation window</b> (observation duration, 1–10 days) — the span of '
        'time the signals are summarised over (min / max / mean / std, plus MES error '
        'counts by <b>fault category</b>) to form the features. It ends at '
        '<code>onset − lead</code>.</li>'
        '<li><b>MES fault categories</b> — the MES <code>error_text</code> is parsed '
        'into fault types (servo_drive, sensor, power, motion_position, gripper, '
        'status, robot_drive, other); each becomes an hourly error-count feature, '
        'alongside the total. See <code>data/processed/mes_error_extracted.csv</code>.</li>'
        '<li><b>Normal-grid granularity</b> (2–24 h) — how densely the <i>normal</i> '
        '(non-defect) reference time points are sampled for comparison.</li>'
        '<li><b>AUC</b> (the value plotted, 0–1) — probability that a random '
        'pre-defect window scores higher than a random normal window. '
        '<b>0.5 = no separation</b> (the signal looks like normal operation); '
        '<b>&gt;0.5</b> = the feature is typically <i>higher</i> before defects; '
        '<b>&lt;0.5</b> = typically <i>lower</i>. The further from 0.5, the more the '
        'feature distinguishes the lead-up to a defect.</li>'
        '</ul></div>')

    html = (
        '<h4 style="margin:18px 0 8px">Plasma — Nozzle-head &amp; MES early factors</h4>'
        + intro
        + terms
        + '<div class="grid">'
        + "".join(
            f'<div class="half">{_auc_heatmap(grid, obs, gran)}</div>'
            for gran in (12, 6) for obs in (3, 1, 0.5))
        + '</div>'
        + '<h5 style="margin:18px 0 8px;color:#52606d">Honest check — does the AUC '
        'survive holding out whole defects?</h5>'
        + '<p class="desc">The heatmap AUC is <b>in-sample</b> and is the '
        '<b>best of ~135 (signal × statistic) features</b>, so it is inflated by '
        'feature selection — with only 15 events some feature scores ~0.7 by chance. '
        'The honest curve below pays that cost the same way the RUL section does: for '
        'each defect, the best feature is <b>re-selected on the other 14 events</b> and '
        'scored on the <b>held-out</b> one (leave-one-defect-out). Where the red line '
        'falls back toward — or below — 0.5, the apparent early signal does <i>not</i> '
        'generalise to unseen defects. This is the same optimism the RUL leakage demo '
        'shows (naive R² +0.94 → leakage-free −0.18), and why a ~0.7 in-sample AUC can '
        'coexist with a 0 h useful prediction horizon.</p>'
        + f'<div class="card">{_honest_auc_chart(ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)}</div>'
        + '<div class="grid">'
        + f'<div class="half">{_leadtime_sensitivity(grid, ef.DEFAULT_GRAN_H)}</div>'
        + f'<div class="half">{_gran_sensitivity(grid, ef.DEFAULT_OBS_DAYS, 24)}</div>'
        + '</div>'
        + f'<div class="card">{_best_feature_box(grid, nozzle, mes, events, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H, 24)}</div>'
        + '<div class="grid">'
        + f'<div class="half"><h5 style="margin:8px 0;color:#52606d">Top features at 6 h lead</h5>'
        + f'{_top_table(grid, 6, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)}</div>'
        + f'<div class="half"><h5 style="margin:8px 0;color:#52606d">Top features at 96 h lead</h5>'
        + f'{_top_table(grid, 96, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)}</div>'
        + '</div>'
        + '<h5 style="margin:18px 0 8px;color:#52606d">MES error categories as '
        'predictors</h5>'
        + '<p class="desc">The MES <code>error_text</code> is parsed into fault '
        'categories (see Key terms); each category\'s hourly error count is an extra '
        'feature. This shows how well each separates the lead-up to a defect.</p>'
        + f'<div class="card">{_mes_category_chart(grid, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)}</div>'
        + '<h5 style="margin:18px 0 8px;color:#52606d">Combined early-warning model '
        '(top factors together)</h5>'
        + _combo_block(grid, ef.DEFAULT_OBS_DAYS, ef.DEFAULT_GRAN_H)
        + '<p class="muted" style="margin-top:16px">This analysis covers <b>Plasma '
        'defects only</b>. Robot defects are excluded: the robot-monitoring stream '
        'covers only ~3 days (early Mar 2025) and does not overlap any robot defect '
        'event, so no robot predictors are available.</p>')
    sections = [{"id": "earlyfactors", "label": "Early predictive factors", "html": html}]
    try:
        from .automl_report import build_section as build_automl
        sections.append(build_automl(refresh=refresh))
    except Exception as e:  # never let AutoML break the rest of the report
        sections.append({"id": "automl", "label": "AutoML early prediction",
                         "html": f'<p class="muted">AutoML skipped: {e}</p>'})
    try:
        from .rul_report import build_section as build_rul
        sections.append(build_rul(refresh=refresh))
    except Exception as e:  # never let RUL break the rest of the report
        sections.append({"id": "rul", "label": "Remaining useful life",
                         "html": f'<p class="muted">RUL skipped: {e}</p>'})
    try:
        from .staging_report import build_section as build_staging
        sections.append(build_staging(refresh=refresh))
    except Exception as e:  # never let staging break the rest
        sections.append({"id": "staging", "label": "Degradation staging",
                         "html": f'<p class="muted">Staging skipped: {e}</p>'})
    try:
        from .staging_report import build_section_3stage as build_staging3
        sections.append(build_staging3(refresh=refresh))
    except Exception as e:  # never let the 3-stage variant break the rest
        sections.append({"id": "staging3", "label": "Degradation staging (3-stage)",
                         "html": f'<p class="muted">3-stage staging skipped: {e}</p>'})
    try:
        from .seq_report import build_section as build_seq
        sections.append(build_seq(refresh=refresh))
    except Exception as e:  # never let the sequence models break the rest
        sections.append({"id": "seqdl", "label": "Sequence deep learning",
                         "html": f'<p class="muted">Sequence models skipped: {e}</p>'})
    try:
        from .survival_report import build_section as build_survival
        sections.append(build_survival(refresh=refresh))
    except Exception as e:  # never let the survival models break the rest
        sections.append({"id": "survival", "label": "Survival models (RSF)",
                         "html": f'<p class="muted">Survival models skipped: {e}</p>'})
    return sections
