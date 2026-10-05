"""Interactive Plotly chart builders.

Every function returns an HTML ``<div>`` string (no inline plotly.js — the library
is loaded once in the report ``<head>`` via CDN). Charts are kept lightweight:
distributions and correlations are drawn from the bounded sample, not the full data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .profiler import DatasetProfile, _GRP_PREFIX

# Shared Bosch-ish palette / template.
PRIMARY = "#0a7cff"
ACCENT = "#e2001a"   # Bosch red
GRID = "#e6e9ef"
# distinct colours for the per-group time-series series (Plasma, Roboter, …)
_GROUP_COLORS = ["#0a7cff", "#ff9f1c", "#00b3a4", "#7b61ff", "#e2001a"]
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12,
              color="#1f2733"),
    margin=dict(l=60, r=20, t=40, b=40),
    colorway=[PRIMARY, ACCENT, "#00b3a4", "#ff9f1c", "#7b61ff", "#3d4756"],
    paper_bgcolor="white",
    plot_bgcolor="white",
)
_CONFIG = {"displayModeBar": "hover", "responsive": True,
           "displaylogo": False}


def _div(fig: go.Figure, height: int = 360, margin: dict | None = None) -> str:
    fig.update_layout(**_LAYOUT, height=height)
    if margin:  # per-chart override (e.g. extra room for a bottom legend)
        fig.update_layout(margin=margin)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CONFIG)


def missingness_chart(profile: DatasetProfile) -> str:
    cols = [c for c in profile.columns]
    names = [c.name for c in cols]
    miss = [c.missing_pct for c in cols]
    color = [ACCENT if m > 20 else (PRIMARY if m > 0 else "#9fb3c8") for m in miss]
    fig = go.Figure(go.Bar(x=miss, y=names, orientation="h",
                           marker_color=color,
                           hovertemplate="%{y}: %{x:.2f}%% missing<extra></extra>"))
    fig.update_layout(title="Missing values by column (%)",
                      xaxis_title="% missing")
    fig.update_yaxes(autorange="reversed")
    return _div(fig, height=max(320, 22 * len(names) + 80))


def timeseries_chart(profile: DatasetProfile, events: pd.DataFrame | None = None
                     ) -> str | None:
    ts = profile.timeseries
    if ts is None or ts.empty:
        return None
    freq = getattr(profile.cfg, "ts_freq", "h")
    bucket = {"h": "hour", "D": "day", "W": "week"}.get(freq, freq)
    adverb = {"hour": "hourly", "day": "daily", "week": "weekly"}.get(bucket, bucket)
    has_means = len(profile.highlight_cols) > 0
    # sparse logs read better as bars; dense streams as a filled line
    sparse = len(ts) <= 400
    fig = make_subplots(specs=[[{"secondary_y": has_means}]])
    count_name = f"records / {bucket}"
    # per-filter-group columns (e.g. Plasma / Roboter) -> one coloured series each
    group_cols = [c for c in ts.columns if c.startswith(_GRP_PREFIX)]
    if group_cols:
        for i, col in enumerate(group_cols):
            label = col[len(_GRP_PREFIX):]
            color = _GROUP_COLORS[i % len(_GROUP_COLORS)]
            if sparse:
                fig.add_trace(go.Bar(x=ts.index, y=ts[col], name=label,
                                     marker_color=color), secondary_y=False)
            else:
                fig.add_trace(go.Scatter(
                    x=ts.index, y=ts[col], name=label, mode="lines",
                    line=dict(color=color, width=1.2)), secondary_y=False)
    elif sparse:
        fig.add_trace(go.Bar(x=ts.index, y=ts["count"], name=count_name,
                             marker_color="#9fb3c8"), secondary_y=False)
    else:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["count"], name=count_name,
                                 mode="lines", line=dict(color="#9fb3c8", width=1),
                                 fill="tozeroy", fillcolor="rgba(159,179,200,0.25)"),
                      secondary_y=False)
    for col in profile.highlight_cols:
        if col in ts.columns:
            fig.add_trace(go.Scatter(x=ts.index, y=ts[col], name=col, mode="lines",
                                     line=dict(width=1.5)), secondary_y=True)
    # overlay tagged defect events (e.g. the 15 Plasma failure events)
    if events is not None and not events.empty:
        ymax = float(ts["count"].max()) or 1.0
        xs, ys = [], []
        for t in events["discovery"]:
            xs += [t, t, None]
            ys += [0, ymax, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", legendgroup="evt", hoverinfo="skip",
            name=f"Plasma defect events ({len(events)})",
            line=dict(color=ACCENT, width=1, dash="dot")), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=events["discovery"], y=[ymax] * len(events),
            mode="markers+text", text=events["n"].astype(str),
            textposition="top center", textfont=dict(size=9, color=ACCENT),
            marker=dict(symbol="triangle-down", color=ACCENT, size=9),
            legendgroup="evt", showlegend=False,
            customdata=np.c_[events["onset_s"], events["downtime_h"]],
            hovertemplate=("Plasma defect #%{text}<br>discovery %{x|%Y-%m-%d %H:%M}"
                           "<br>onset %{customdata[0]}"
                           "<br>downtime %{customdata[1]:.0f} h<extra></extra>")),
            secondary_y=False)

    fig.update_layout(title=f"Activity over time ({adverb})",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0))
    fig.update_yaxes(title_text=count_name, secondary_y=False)
    if has_means:
        fig.update_yaxes(title_text="signal mean", secondary_y=True)
    return _div(fig, height=380)


def numeric_distributions(profile: DatasetProfile, max_cols: int = 12) -> str | None:
    sample = profile.numeric_sample
    num_cols = [c.name for c in profile.columns
                if c.kind == "numeric" and c.name in sample.columns]
    if not num_cols:
        return None
    num_cols = num_cols[:max_cols]
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=num_cols,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    for i, col in enumerate(num_cols):
        r, c = i // ncols + 1, i % ncols + 1
        vals = pd.to_numeric(sample[col], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=vals, nbinsx=40, marker_color=PRIMARY,
                                   showlegend=False), row=r, col=c)
    fig.update_layout(title=f"Numeric distributions (sample, n={len(sample):,})",
                      bargap=0.03)
    return _div(fig, height=260 * nrows + 60)


def correlation_heatmap(profile: DatasetProfile, max_cols: int = 25) -> str | None:
    sample = profile.numeric_sample
    num_cols = [c.name for c in profile.columns
                if c.kind == "numeric" and c.name in sample.columns][:max_cols]
    if len(num_cols) < 2:
        return None
    corr = sample[num_cols].apply(pd.to_numeric, errors="coerce").corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True, zmid=0,
        hovertemplate="%{x}<br>%{y}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlation matrix (sample, Pearson)")
    fig.update_yaxes(autorange="reversed")
    side = max(420, 26 * len(num_cols) + 120)
    return _div(fig, height=side)


def categorical_charts(profile: DatasetProfile, max_charts: int = 6) -> str | None:
    cats = [c for c in profile.columns if c.kind == "categorical" and c.top]
    if not cats:
        return None
    cats = cats[:max_charts]
    parts = []
    for cp in cats:
        labels = [str(v) for v, _ in cp.top][:15]
        values = [n for _, n in cp.top][:15]
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                               marker_color=PRIMARY,
                               hovertemplate="%{y}: %{x:,}<extra></extra>"))
        fig.update_layout(title=f"{cp.name} (top {len(labels)} of {cp.cardinality})",
                          xaxis_title="count")
        fig.update_yaxes(autorange="reversed")
        parts.append(_div(fig, height=max(260, 26 * len(labels) + 90)))
    return "".join(f'<div class="half">{p}</div>' for p in parts)
