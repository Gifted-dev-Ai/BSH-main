"""Incident deep-dive: Plasma vs Roboter (filtered by ``Groupdescr``).

Consumes the **processed** extraction output
(``data/processed/incident_extracted.csv``) rather than re-running the extractor —
the four structured bilingual columns ``component_de_en`` / ``phenomenon_de_en`` /
``action_de_en`` / ``reason_de_en`` are the interesting signals. Within each cell
multiple labels are ``; ``-joined, so they are split back into lists here.

For the **Plasma** and **Roboter** incident groups we build a *time distribution*
(monthly counts + mean downtime) and a *correlation* heatmap of co-occurring
defect categories.

The output is a list of report subsections (id, label, html) appended under the
Incident-Management dataset section.
"""

from __future__ import annotations

import re
from statistics import median

import pandas as pd

from .charts import ACCENT, PRIMARY, _div
from .config import DATASETS
from .extraction.__main__ import PROCESSED_DIR
from .extraction.__main__ import run as run_extraction
from .patterns import association_rules, recency_ngrams

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Groups to compare (values of the Groupdescr column) and their plot colour.
FOCUS_GROUPS = [("Plasma", PRIMARY), ("Roboter", ACCENT)]

# Structured columns from incident_extracted.csv to analyse:
#   (category key, source column, plot colour, short tag for correlation labels)
ANALYSIS_CATEGORIES = [
    ("component", "component_de_en", PRIMARY, "comp"),
    ("phenomenon", "phenomenon_de_en", "#00b3a4", "phen"),
    ("action", "action_de_en", "#ff9f1c", "act"),
    ("reason", "reason_de_en", "#7b61ff", "reason"),
]
PROCESSED_FILE = "incident_extracted.csv"
MIN_LABEL_COUNT = 2          # drop singletons from the correlation matrix

# --- pattern-mining parameters ---------------------------------------------
RULE_MIN_CONFIDENCE = 0.6    # association-rule confidence threshold
RECENCY_LEVELS = (1, 3, 7, 14, 28)  # recency windows (days) compared side by side
SEQ_MAX_N = 3               # longest sub-sequence to mine
# Category order used to pick one representative label per incident for sequencing.
SEQ_PRIMARY_ORDER = ("phenomenon", "component", "action")


def _tagged_labels(row) -> set[str]:
    """All structured labels of one incident, tagged by category (for Apriori)."""
    out: set[str] = set()
    for cat, _, _, tag in ANALYSIS_CATEGORIES:
        out.update(f"{tag}: {lab}" for lab in row[f"_lab_{cat}"])
    return out


def _primary_label(row) -> str | None:
    """One representative label per incident for the sequence stream."""
    for cat in SEQ_PRIMARY_ORDER:
        labs = row[f"_lab_{cat}"]
        if labs:
            return labs[0]
    return None


def _split_labels(cell) -> list[str]:
    """Split a '; '-joined label cell into individual labels (handles NaN)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [x.strip() for x in re.split(r"\s*;\s*", str(cell)) if x.strip()]


def load_enriched_incidents() -> pd.DataFrame:
    """Processed incident rows + parsed timestamp + per-category label lists."""
    path = PROCESSED_DIR / PROCESSED_FILE
    if not path.exists():
        run_extraction()  # generate the processed file if missing
    # utf-8-sig: the file is written with a BOM; comma-separated with the German /
    # English labels (which contain ';' only as the in-cell label separator) safely
    # quoted by the CSV writer, so the default parser reads them back intact.
    df = pd.read_csv(path, encoding="utf-8-sig")

    cfg = next(d for d in DATASETS if d.key == "incident_mgmt")
    ts_raw = df[cfg.timestamp_col].astype("string").str.extract(
        f"({cfg.timestamp_regex})", expand=False)
    df["_ts"] = pd.to_datetime(ts_raw, format=cfg.timestamp_format, errors="coerce")

    for cat, col, _, _ in ANALYSIS_CATEGORIES:
        df[f"_lab_{cat}"] = df[col].apply(_split_labels)
    return df


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _time_distribution(df: pd.DataFrame) -> str:
    """Monthly incident counts (bars) + mean downtime (lines) for both groups."""
    sub = df[df["_ts"].notna()].copy()
    sub["month"] = sub["_ts"].dt.to_period("M").dt.to_timestamp()
    months = pd.date_range(sub["month"].min(), sub["month"].max(), freq="MS")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for name, colour in FOCUS_GROUPS:
        g = sub[sub["Groupdescr"] == name]
        counts = g.groupby("month").size().reindex(months, fill_value=0)
        downtime = g.groupby("month")["Calcdowntime"].mean().reindex(months)
        fig.add_trace(go.Bar(x=months, y=counts.values, name=f"{name} — count",
                             marker_color=colour, opacity=0.85), secondary_y=False)
        fig.add_trace(go.Scatter(x=months, y=downtime.values, name=f"{name} — mean downtime",
                                 mode="lines+markers", line=dict(color=colour, width=2,
                                 dash="dot")), secondary_y=True)
    fig.update_layout(title="Monthly incident count & mean downtime — Plasma vs Roboter",
                      barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0))
    fig.update_yaxes(title_text="incidents / month", secondary_y=False)
    fig.update_yaxes(title_text="mean downtime (min)", secondary_y=True)
    return _div(fig, height=400)


def _correlation_heatmap(df: pd.DataFrame, group: str) -> str | None:
    """Co-occurrence correlation across all four structured categories in one group.

    Labels keep their full bilingual ``German (English)`` form and are tagged by
    category (``comp:`` / ``phen:`` / ``act:`` / ``reason:``) so they stay distinct.
    """
    g = df[df["Groupdescr"] == group]
    if len(g) < 3:
        return None
    cols: list[str] = []
    mat = []
    # discover the tagged label vocabulary present in this group, preserving order
    seen = set()
    for cat, _, _, tag in ANALYSIS_CATEGORIES:
        for labels in g[f"_lab_{cat}"]:
            for lab in labels:
                key = f"{tag}: {lab}"
                if key not in seen:
                    seen.add(key)
                    cols.append(key)
    for _, row in g.iterrows():
        present = set()
        for cat, _, _, tag in ANALYSIS_CATEGORIES:
            present.update(f"{tag}: {x}" for x in row[f"_lab_{cat}"])
        mat.append([1 if c in present else 0 for c in cols])
    oh = pd.DataFrame(mat, columns=cols)
    # keep only labels seen at least MIN_LABEL_COUNT times (need variance to correlate)
    keep = [c for c in cols if oh[c].sum() >= MIN_LABEL_COUNT and oh[c].nunique() > 1]
    if len(keep) < 2:
        return None
    corr = oh[keep].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=keep, y=keep, zmin=-1, zmax=1, zmid=0,
        colorscale="RdBu", reversescale=True,
        hovertemplate="%{x}<br>%{y}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(title=f"{group}: category co-occurrence (n={len(g)})")
    # automargin so the long bilingual tick labels are not clipped
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(automargin=True)
    side = max(380, 30 * len(keep) + 170)
    return _div(fig, height=side)


def _top_categories(df: pd.DataFrame, group: str) -> str:
    """Top labels per structured category for one group (horizontal bars)."""
    g = df[df["Groupdescr"] == group]
    parts = []
    for cat, _, colour, _ in ANALYSIS_CATEGORIES:
        s = g[f"_lab_{cat}"].explode().dropna()
        vc = s.value_counts().head(8)
        if vc.empty:
            continue
        fig = go.Figure(go.Bar(x=vc.values, y=list(vc.index),
                               orientation="h", marker_color=colour,
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_layout(title=f"{group}: top {cat}s", xaxis_title="incidents")
        # automargin so the full bilingual labels are not clipped
        fig.update_yaxes(autorange="reversed", automargin=True)
        parts.append(_div(fig, height=max(240, 26 * len(vc) + 90)))
    return "".join(f'<div class="half">{p}</div>' for p in parts)


def _kpis(df: pd.DataFrame) -> str:
    cards = []
    for name, _ in FOCUS_GROUPS:
        g = df[df["Groupdescr"] == name]
        n = len(g)
        total = int(g["Calcdowntime"].sum())
        mean = g["Calcdowntime"].mean()
        cards.append(
            f'<div class="kpi"><div class="kpi-val">{n}</div>'
            f'<div class="kpi-label">{name} incidents</div>'
            f'<div class="kpi-sub">{total:,} min total · {mean:.0f} min avg</div></div>')
    return f'<div class="kpi-row">{"".join(cards)}</div>'


def _defect_gap_analysis() -> str:
    """Time between consecutive defect events (onset-to-onset) per group.

    Reuses the project defect-event definition: incidents < 1 day apart form one
    event, whose reference time is the onset = first-incident time − downtime
    (see early_factors.defect_events / defect_onsets). The gap between consecutive
    events is the difference of successive onsets — i.e. the time between failures.
    """
    from . import early_factors as ef  # local import avoids a circular dependency

    per_group = {}
    for group, colour in FOCUS_GROUPS:
        ev = ef.defect_events(group)
        onsets = ef.defect_onsets(ev).sort_values().reset_index(drop=True)
        gaps = onsets.diff().dropna().dt.total_seconds() / 86400.0  # days
        later = onsets.iloc[1:].reset_index(drop=True)              # date of 2nd event
        per_group[group] = (onsets, gaps, later, colour)

    # KPI cards: events + median/mean time between defects
    cards = []
    for group, (onsets, gaps, _later, _c) in per_group.items():
        cards.append(
            f'<div class="kpi"><div class="kpi-val">{gaps.median():.0f} d</div>'
            f'<div class="kpi-label">{group}: median time between defects</div>'
            f'<div class="kpi-sub">{len(onsets)} events · mean {gaps.mean():.0f} d · '
            f'range {gaps.min():.0f}–{gaps.max():.0f} d</div></div>')
    kpi_row = f'<div class="kpi-row">{"".join(cards)}</div>'

    # timeline: gap vs the date of the later event
    tl = go.Figure()
    for group, (_o, gaps, later, colour) in per_group.items():
        tl.add_trace(go.Scatter(x=later, y=gaps, mode="lines+markers", name=group,
                                line=dict(color=colour, width=1.8),
                                hovertemplate="%{x|%Y-%m-%d}<br>gap %{y:.1f} d<extra></extra>"))
    tl.update_layout(title="Time between consecutive defect events (onset → onset)",
                     xaxis_title="date of the later defect event",
                     yaxis_title="gap to previous defect (days)",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    timeline = _div(tl, height=360)

    # distribution: box of gaps per group
    box = go.Figure()
    for group, (_o, gaps, _l, colour) in per_group.items():
        box.add_trace(go.Box(y=gaps, name=group, marker_color=colour,
                             boxpoints="all", jitter=0.4, pointpos=0))
    box.update_layout(title="Distribution of time between defects",
                      yaxis_title="gap (days)", showlegend=False)
    distribution = _div(box, height=360)

    return (
        '<h4 style="margin:20px 0 8px">Gap between consecutive defect events</h4>'
        '<p class="desc">Defect events cluster incidents &lt; 1 day apart; each event '
        'is anchored at its <b>onset</b> (first incident − downtime). The <b>gap</b> is '
        'the time from one onset to the next — the time between failures.</p>'
        + kpi_row
        + '<div class="grid">'
        + f'<div class="half">{timeline}</div>'
        + f'<div class="half">{distribution}</div></div>')


def _defect_weekday_analysis() -> str:
    """Weekday distribution of defect events (by onset) per group."""
    from . import early_factors as ef  # local import avoids a circular dependency

    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure()
    for group, colour in FOCUS_GROUPS:
        ev = ef.defect_events(group)
        wd = ef.defect_onsets(ev).dt.dayofweek
        counts = wd.value_counts().reindex(range(7), fill_value=0).sort_index()
        fig.add_trace(go.Bar(x=names, y=counts.values, name=group, marker_color=colour,
                             hovertemplate="%{x}: %{y} defect events<extra></extra>"))
    # shade the weekend
    fig.add_vrect(x0=4.5, x1=6.5, fillcolor="#9fb3c8", opacity=0.12, line_width=0,
                  annotation_text="weekend", annotation_position="top right")
    fig.update_layout(title="Defect events by weekday (onset)", barmode="group",
                      xaxis_title="weekday", yaxis_title="number of defect events",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return (
        '<h4 style="margin:20px 0 8px">Defect events by weekday</h4>'
        '<p class="desc">Which day of the week each defect <b>onset</b> falls on — '
        'reveals weekly / shift patterns in when failures occur.</p>'
        f'<div class="card">{_div(fig, height=340)}</div>')


# --------------------------------------------------------------------------- #
# Pattern mining: association rules + recency-aware sequences
# --------------------------------------------------------------------------- #
def _rules_table(df: pd.DataFrame, group: str) -> str:
    """Apriori association rules among the tagged category labels of one group."""
    g = df[df["Groupdescr"] == group]
    transactions = [_tagged_labels(r) for _, r in g.iterrows()]
    transactions = [t for t in transactions if t]
    min_count = max(2, round(0.12 * len(transactions)))
    rules = association_rules(transactions, min_count=min_count,
                              min_confidence=RULE_MIN_CONFIDENCE, max_len=3)
    if not rules:
        return ('<p class="muted">No association rules above the support/confidence '
                f'threshold (min {min_count} incidents, conf ≥ '
                f'{int(RULE_MIN_CONFIDENCE*100)}%).</p>')
    head = ("<tr><th>If (antecedent)</th><th>→ then (consequent)</th>"
            "<th>incidents</th><th>confidence</th><th>lift</th></tr>")
    rows = []
    for r in rules[:12]:
        ante = " + ".join(sorted(r.antecedent))
        cons = " + ".join(sorted(r.consequent))
        lift = "—" if r.lift != r.lift else f"{r.lift:.2f}"
        rows.append(
            f'<tr><td>{ante}</td><td class="cname">{cons}</td>'
            f'<td>{r.count}</td><td>{r.confidence*100:.0f}%</td><td>{lift}</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _flow_sunburst(bigrams: dict, title: str) -> str:
    """2-level sunburst of source-event -> following-event transitions."""
    if not bigrams:
        return f'<p class="muted" style="text-align:center">{title}: none</p>'
    ids, labels, parents, values = [], [], [], []
    src_total: dict[str, int] = {}
    for (a, b), c in bigrams.items():
        src_total[a] = src_total.get(a, 0) + c
    for a, tot in src_total.items():
        ids.append(f"S|{a}"); labels.append(a); parents.append(""); values.append(tot)
    for (a, b), c in sorted(bigrams.items(), key=lambda kv: -kv[1])[:14]:
        ids.append(f"S|{a}|{b}"); labels.append(b)
        parents.append(f"S|{a}"); values.append(c)
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", insidetextorientation="radial",
        hovertemplate="%{label}<br>%{value} transition(s)<extra></extra>"))
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      margin=dict(l=6, r=6, t=36, b=6), height=300,
                      font=dict(family="Segoe UI, Roboto, Arial, sans-serif", size=11))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", default_height="300px",
                       config={"displaylogo": False, "responsive": True})


def _sequence_analysis(df: pd.DataFrame, group: str) -> str:
    """Frequent consecutive sub-sequences compared across recency windows."""
    g = df[df["Groupdescr"] == group]
    events = [(r["_ts"], _primary_label(r)) for _, r in g.iterrows()
              if pd.notna(r["_ts"]) and _primary_label(r)]
    events.sort(key=lambda e: e[0])
    if len(events) < 3:
        return '<p class="muted">Too few time-ordered events to mine sequences.</p>'

    # mine once per recency window
    per_level: dict[int, dict] = {}
    spans_wide: dict = {}
    widest = max(RECENCY_LEVELS)
    for w in RECENCY_LEVELS:
        grams, spans = recency_ngrams(events, max_gap_days=w, max_n=SEQ_MAX_N)
        per_level[w] = grams
        if w == widest:
            spans_wide = spans

    # one sunburst per window, laid out in an adaptive, centered, wrapping row
    sb = []
    for w in RECENCY_LEVELS:
        bigrams = {k: v for k, v in per_level[w].items() if len(k) == 2}
        sb.append(f'<div class="seqcol"><div class="card">'
                  f'{_flow_sunburst(bigrams, f"≤ {w} days")}</div></div>')
    sun_row = f'<div class="seqrow">{"".join(sb)}</div>'

    # combined table: counts of each sub-sequence at every window (monotonic),
    # ranked by the widest-window count; median gap from the widest window.
    seqs = [k for k in per_level[widest] if len(k) >= 2]
    seqs.sort(key=lambda k: (-per_level[widest][k], len(k)))
    head = ('<tr><th>Frequent sub-sequence (recency-ordered)</th>'
            + "".join(f"<th>≤ {w} d</th>" for w in RECENCY_LEVELS)
            + "<th>median gap</th></tr>")
    rows = []
    for seq in seqs[:12]:
        arrow = "  →  ".join(seq)
        counts = "".join(f"<td>{per_level[w].get(seq, 0)}</td>" for w in RECENCY_LEVELS)
        gap = median(spans_wide[seq]) if spans_wide.get(seq) else 0.0
        rows.append(f'<tr><td class="cname">{arrow}</td>{counts}'
                    f'<td>{gap:.0f} d</td></tr>')
    table = f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'
    note = ('<p class="muted" style="margin-top:6px">Counts are cumulative in the '
            'window — a transition within ≤7 d is also counted at ≤14 d and ≤30 d — '
            'so widening the window reveals more (but more loosely-coupled) '
            'sequences. Sunbursts show the 2-step event flow at each level.</p>')
    return sun_row + table + note


def _pattern_section_html(df: pd.DataFrame) -> str:
    levels = " / ".join(f"{w}" for w in RECENCY_LEVELS)
    intro = ('<p class="desc"><b>Dependence</b> (Apriori association rules over the '
             'tagged category labels within each incident) and <b>frequent '
             'sub-sequences</b> of consecutive incidents, chained only when they '
             f'occur within a recency window. The window is varied — <b>{levels} '
             'days</b> — to see which progressions are tightly vs loosely coupled in '
             'time. Both views are exploratory given the small samples.</p>')
    blocks = []
    for group, _ in FOCUS_GROUPS:
        blocks.append(
            f'<h4 style="margin:20px 0 8px">{group}</h4>'
            '<h5 style="margin:8px 0;color:#52606d">Association rules (within incident)</h5>'
            f'{_rules_table(df, group)}'
            '<h5 style="margin:16px 0 8px;color:#52606d">Frequent event sub-sequences '
            f'(recency {levels} days)</h5>'
            f'{_sequence_analysis(df, group)}')
    return intro + "".join(blocks)


def build_deepdive_sections() -> list[dict]:
    """Return report subsections (id/label/html) for the incident deep-dive."""
    df = load_enriched_incidents()

    intro = ('<p class="desc">Incidents filtered by <code>Groupdescr</code> into the '
             '<b>Plasma</b> and <b>Roboter</b> groups, analysed via the structured '
             'fields from <code>data/processed/incident_extracted.csv</code> '
             '(<code>component</code>, <code>phenomenon</code>, <code>action</code>, '
             '<code>reason</code>). Correlation matrices show how these categories '
             '<i>co-occur</i> within each group (exploratory — small samples).</p>')

    cards = []
    for group, _ in FOCUS_GROUPS:
        corr = _correlation_heatmap(df, group)
        corr_html = f'<div class="card">{corr}</div>' if corr else \
            '<p class="muted">Not enough repeated categories to correlate.</p>'
        cards.append(
            f'<h4 style="margin:18px 0 8px">{group}</h4>'
            f'<div class="grid">{_top_categories(df, group)}</div>'
            f'<div style="margin-top:14px">{corr_html}</div>')

    html = (intro + _kpis(df)
            + f'<div class="card">{_time_distribution(df)}</div>'
            + _defect_gap_analysis()
            + _defect_weekday_analysis()
            + "".join(cards))
    return [
        {"id": "deepdive", "label": "Plasma vs Robot", "html": html},
        {"id": "patterns", "label": "Patterns & sequences",
         "html": _pattern_section_html(df)},
    ]
