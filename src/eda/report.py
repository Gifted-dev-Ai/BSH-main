"""Assemble the self-contained HTML analysis report (EDA + modelling + KG).

Layout: a fixed left navigation sidebar (datasets + per-dataset subsections) and a
scrollable content pane. Plotly.js is loaded once from CDN; every chart is an inline
``<div>``. The output is a single ``.html`` file that opens in any browser.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

from . import charts
from .config import DatasetConfig
from .profiler import DatasetProfile

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

# Subsections rendered for every dataset (id-suffix, label).
SUBSECTIONS = [
    ("overview", "Overview"),
    ("columns", "Column profile"),
    ("missing", "Missingness"),
    ("timeseries", "Time series"),
    ("distributions", "Distributions"),
    ("correlation", "Correlation"),
    ("categorical", "Categories"),
]


def _fmt_int(x) -> str:
    return f"{int(x):,}" if x is not None and pd.notna(x) else "—"


def _fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    ax = abs(x)
    if ax != 0 and (ax >= 1e6 or ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:,.3f}"


def _fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{html.escape(value)}</div>'
            f'<div class="kpi-label">{html.escape(label)}</div>{sub_html}</div>')


def _overview(p: DatasetProfile) -> str:
    cfg = p.cfg
    span = "—"
    if p.ts_min is not None and p.ts_max is not None:
        span = (f'{p.ts_min:%Y-%m-%d %H:%M} → {p.ts_max:%Y-%m-%d %H:%M} UTC')
    n_num = sum(1 for c in p.columns if c.kind in ("numeric", "constant"))
    n_cat = sum(1 for c in p.columns if c.kind in ("categorical", "id"))
    kpis = "".join([
        _kpi("Rows", _fmt_int(p.n_rows)),
        _kpi("Columns", _fmt_int(p.n_cols), f"{n_num} numeric · {n_cat} categorical"),
        _kpi("File size", _fmt_bytes(p.size_bytes)),
        _kpi("Time span", span if span != "—" else "—",
             p.datetime_col or ""),
        _kpi("Profiled in", f"{p.load_seconds:.1f} s",
             f"{p.engine}" + (f" · {p.n_workers}w" if p.n_workers > 1 else "")),
    ])
    return (f'<p class="desc">{html.escape(cfg.description)}</p>'
            f'<div class="kpi-row">{kpis}</div>')


def _column_table(p: DatasetProfile) -> str:
    head = ("<tr><th>Column</th><th>Type</th><th>Missing %</th><th>Mean</th>"
            "<th>Std</th><th>Min</th><th>Max</th><th>Card. / Top</th></tr>")
    rows = []
    for c in p.columns:
        miss_cls = "warn" if c.missing_pct > 20 else ("muted" if c.missing_pct == 0 else "")
        if c.kind in ("numeric", "constant"):
            s = c.stats
            extra = f'p50 {_fmt_num(s.get("p50"))}'
            cells = (f'<td>{_fmt_num(s.get("mean"))}</td><td>{_fmt_num(s.get("std"))}</td>'
                     f'<td>{_fmt_num(s.get("min"))}</td><td>{_fmt_num(s.get("max"))}</td>'
                     f'<td class="muted">{extra}</td>')
        elif c.kind == "datetime":
            cells = ('<td colspan="4" class="muted">timestamp (UTC)</td>'
                     '<td class="muted">primary time index</td>')
        else:
            top = c.top[0][0] if c.top else "—"
            cells = (f'<td colspan="4" class="muted">—</td>'
                     f'<td class="muted">{html.escape(str(c.cardinality))} uniq · '
                     f'top: {html.escape(str(top))[:24]}</td>')
        badge = f'<span class="badge b-{c.kind}">{c.kind}</span>'
        rows.append(f'<tr><td class="cname">{html.escape(c.name)}</td><td>{badge}</td>'
                    f'<td class="{miss_cls}">{c.missing_pct:.2f}</td>{cells}</tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _plasma_defect_events() -> tuple[pd.DataFrame | None, str]:
    """The 15 Plasma defect events (clustering rule), as overlay markers + a caption.

    discovery = first incident of the cluster; onset = discovery − downtime-at-
    discovery (the back-dated true fault start used by the modelling sections).
    """
    try:
        from . import early_factors as ef
        ev = ef.defect_events("Plasma").reset_index(drop=True)
        if ev.empty:
            return None, ""
        onsets = ef.defect_onsets(ev)
        events_df = pd.DataFrame({
            "n": range(1, len(ev) + 1),
            "discovery": ev["start"].dt.tz_convert("UTC").dt.tz_localize(None),
            "onset_s": onsets.dt.strftime("%Y-%m-%d %H:%M"),
            "downtime_h": ev["downtime_min"] / 60.0,
        })
        note = (
            f'<p class="muted">Red markers tag the <b>{len(ev)} Plasma defect '
            'events</b> from the incident-clustering rule (incidents &lt; 1 day apart '
            '= one event; the first incident = <i>discovery</i>). Lines sit at the '
            'discovery time; hover shows the back-dated <i>onset</i> '
            '(<code>discovery − downtime</code>) and the cluster downtime — the same '
            'events used in the Plasma modelling section.</p>')
        return events_df, note
    except Exception:
        return None, ""


def _section(p: DatasetProfile) -> str:
    key = p.cfg.key
    blocks = [f'<section id="{key}" class="dataset">',
              f'<h2>{html.escape(p.cfg.title)}</h2>']

    blocks.append(f'<div id="{key}-overview" class="sub"><h3>Overview</h3>'
                  f'{_overview(p)}</div>')
    blocks.append(f'<div id="{key}-columns" class="sub"><h3>Column profile</h3>'
                  f'{_column_table(p)}</div>')

    miss = charts.missingness_chart(p)
    blocks.append(f'<div id="{key}-missing" class="sub"><h3>Missingness</h3>'
                  f'<div class="card">{miss}</div></div>')

    events_df, events_note = _plasma_defect_events() if key == "repair_activities" \
        else (None, "")
    filt_note = ""
    if p.cfg.ts_filter_col and p.cfg.ts_filter_values:
        vals = ", ".join(html.escape(v) for v in p.cfg.ts_filter_values)
        filt_note = (f'<p class="muted">Time series restricted to '
                     f'<code>{html.escape(p.cfg.ts_filter_col)}</code> ∈ '
                     f'{{{vals}}} — the rest of this profile still covers all rows.</p>')
    ts = charts.timeseries_chart(p, events=events_df)
    ts_html = f'{filt_note}<div class="card">{ts}</div>{events_note}' if ts else \
        '<p class="muted">No usable timestamp column.</p>'
    blocks.append(f'<div id="{key}-timeseries" class="sub"><h3>Time series</h3>'
                  f'{ts_html}</div>')

    dist = charts.numeric_distributions(p)
    dist_html = f'<div class="card">{dist}</div>' if dist else \
        '<p class="muted">No numeric columns to plot.</p>'
    blocks.append(f'<div id="{key}-distributions" class="sub"><h3>Distributions</h3>'
                  f'{dist_html}</div>')

    corr = charts.correlation_heatmap(p)
    corr_html = f'<div class="card">{corr}</div>' if corr else \
        '<p class="muted">Need ≥2 numeric columns for a correlation matrix.</p>'
    blocks.append(f'<div id="{key}-correlation" class="sub"><h3>Correlation</h3>'
                  f'{corr_html}</div>')

    cat = charts.categorical_charts(p)
    cat_html = f'<div class="grid">{cat}</div>' if cat else \
        '<p class="muted">No low-cardinality categorical columns.</p>'
    blocks.append(f'<div id="{key}-categorical" class="sub"><h3>Categories</h3>'
                  f'{cat_html}</div>')

    for extra in p.extra_sections:
        blocks.append(f'<div id="{key}-{extra["id"]}" class="sub">'
                      f'<h3>{html.escape(extra["label"])}</h3>{extra["html"]}</div>')

    blocks.append("</section>")
    return "".join(blocks)


def _standalone_section(sec: dict) -> str:
    """A top-level section (not tied to a dataset) containing labelled subsections."""
    sid = sec["id"]
    blocks = [f'<section id="{sid}" class="dataset">',
              f'<h2>{html.escape(sec["title"])}</h2>']
    for sub in sec["subs"]:
        blocks.append(f'<div id="{sid}-{sub["id"]}" class="sub">'
                      f'<h3>{html.escape(sub["label"])}</h3>{sub["html"]}</div>')
    blocks.append("</section>")
    return "".join(blocks)


def _sidebar(profiles: list[DatasetProfile],
             standalone_sections: list[dict] | None = None) -> str:
    items = ['<div class="brand">BSH · PdM</div>',
             '<a class="nav-top" href="#summary">Summary</a>']
    for p in profiles:
        items.append(f'<div class="nav-group"><a class="nav-ds" href="#{p.cfg.key}">'
                     f'{html.escape(p.cfg.title)}</a><div class="nav-subs">')
        for sid, label in SUBSECTIONS:
            items.append(f'<a href="#{p.cfg.key}-{sid}">{label}</a>')
        for extra in p.extra_sections:
            items.append(f'<a href="#{p.cfg.key}-{extra["id"]}">{html.escape(extra["label"])}</a>')
        items.append("</div></div>")
    for sec in (standalone_sections or []):
        items.append(f'<div class="nav-group"><a class="nav-ds" href="#{sec["id"]}">'
                     f'{html.escape(sec["title"])}</a><div class="nav-subs">')
        for sub in sec["subs"]:
            items.append(f'<a href="#{sec["id"]}-{sub["id"]}">{html.escape(sub["label"])}</a>')
        items.append("</div></div>")
    return f'<nav id="sidebar">{"".join(items)}</nav>'


def _summary(profiles: list[DatasetProfile]) -> str:
    rows = []
    for p in profiles:
        span = "—"
        if p.ts_min is not None:
            span = f"{p.ts_min:%Y-%m-%d} → {p.ts_max:%Y-%m-%d}"
        rows.append(
            f'<tr><td class="cname"><a href="#{p.cfg.key}">'
            f'{html.escape(p.cfg.title)}</a></td>'
            f'<td>{_fmt_int(p.n_rows)}</td><td>{_fmt_int(p.n_cols)}</td>'
            f'<td>{_fmt_bytes(p.size_bytes)}</td><td>{span}</td>'
            f'<td class="muted">{p.engine}</td></tr>')
    table = ('<div class="table-wrap"><table><tr><th>Dataset</th><th>Rows</th>'
             '<th>Cols</th><th>Size</th><th>Time span</th><th>Engine</th></tr>'
             f'{"".join(rows)}</table></div>')
    total_rows = sum(p.n_rows for p in profiles)
    total_bytes = sum(p.size_bytes for p in profiles)
    kpis = "".join([
        _kpi("Datasets", str(len(profiles))),
        _kpi("Total rows", _fmt_int(total_rows)),
        _kpi("Total volume", _fmt_bytes(total_bytes)),
    ])
    return (f'<section id="summary" class="dataset"><h2>Summary</h2>'
            f'<p class="desc">Exploratory data analysis of the raw predictive-'
            f'maintenance datasets in <code>data/raw</code>. Charts are interactive '
            f'(hover, zoom, pan); distributions and correlations are computed on a '
            f'bounded random sample of each dataset.</p>'
            f'<div class="kpi-row">{kpis}</div>{table}</section>')


def build_report(profiles: list[DatasetProfile],
                 standalone_sections: list[dict] | None = None) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sidebar = _sidebar(profiles, standalone_sections)
    body = (_summary(profiles)
            + "".join(_section(p) for p in profiles)
            + "".join(_standalone_section(s) for s in (standalone_sections or [])))
    return _TEMPLATE.format(plotly=PLOTLY_CDN, css=_CSS, js=_JS,
                            sidebar=sidebar, body=body, generated=generated)


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#f4f6fa;--card:#fff;--ink:#1f2733;--muted:#7b8794;--line:#e6e9ef;
--primary:#0a7cff;--accent:#e2001a;--side:#11161f;--side-ink:#c7d0dc;}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);}
a{color:var(--primary);text-decoration:none}
#sidebar{position:fixed;top:0;left:0;width:288px;height:100vh;overflow-y:auto;
background:var(--side);color:var(--side-ink);padding:18px 14px 60px;font-size:13.5px;}
.brand{font-weight:700;font-size:17px;color:#fff;letter-spacing:.3px;
padding:6px 8px 16px;border-bottom:1px solid #232c3a;margin-bottom:12px}
.nav-top{display:block;color:#fff;font-weight:600;padding:8px;border-radius:7px}
.nav-top:hover{background:#1c2533}
.nav-group{margin-top:10px}
.nav-ds{display:block;color:#fff;font-weight:600;padding:8px;border-radius:7px;
line-height:1.3}
.nav-ds:hover{background:#1c2533}
.nav-subs{display:flex;flex-direction:column;margin:4px 0 6px 6px;
border-left:1px solid #2a3444;padding-left:8px}
.nav-subs a{color:var(--side-ink);padding:4px 8px;border-radius:6px;font-size:12.5px}
.nav-subs a:hover{background:#1c2533;color:#fff}
.nav-subs a.active{color:#fff;background:#0a7cff22;border-left:2px solid var(--primary);margin-left:-9px;padding-left:14px}
#content{margin-left:288px;padding:30px 40px 80px;max-width:1280px}
header.top{margin-bottom:8px}
header.top h1{margin:0;font-size:26px}
header.top .gen{color:var(--muted);font-size:13px;margin-top:4px}
.dataset{padding-top:26px;margin-top:14px;border-top:1px solid var(--line)}
.dataset>h2{font-size:21px;margin:0 0 6px;color:#0d1b2a}
.sub{margin:22px 0}
.sub>h3{font-size:15px;text-transform:uppercase;letter-spacing:.6px;
color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:6px;margin:0 0 14px}
.desc{color:#3d4756;max-width:900px;line-height:1.55}
code{background:#eef1f6;padding:1px 6px;border-radius:5px;font-size:12.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:10px 8px;box-shadow:0 1px 3px rgba(16,24,40,.04);overflow:hidden}
.kpi-row{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:14px 18px;min-width:150px;flex:1;box-shadow:0 1px 3px rgba(16,24,40,.04)}
.kpi-val{font-size:22px;font-weight:700;color:#0d1b2a}
.kpi-label{color:var(--muted);font-size:12px;margin-top:3px;text-transform:uppercase;
letter-spacing:.5px}
.kpi-sub{color:#52606d;font-size:12px;margin-top:5px}
.grid{display:flex;flex-wrap:wrap;gap:16px}
.grid .half{flex:1 1 calc(50% - 8px);min-width:340px;background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:8px;
box-shadow:0 1px 3px rgba(16,24,40,.04)}
.seqrow{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin:6px 0}
.seqcol{flex:1 1 210px;min-width:190px;max-width:340px}
.seqcol .card{height:100%}
.table-wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);
border-radius:12px;box-shadow:0 1px 3px rgba(16,24,40,.04)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);
white-space:nowrap}
th{background:#f7f9fc;color:#52606d;font-weight:600;position:sticky;top:0}
tr:hover td{background:#f7faff}
td.cname{font-weight:600;color:#0d1b2a}
td.muted,.muted{color:var(--muted)}
td.warn{color:var(--accent);font-weight:600}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.b-numeric{background:#e3f0ff;color:#0a5bcc}
.b-categorical{background:#e8f7f0;color:#0a8a5a}
.b-datetime{background:#f0e9ff;color:#6b3fd1}
.b-id{background:#fdeef0;color:#c2304a}
.b-constant{background:#eef1f6;color:#52606d}
.kg-toggle{display:inline-flex;align-items:center;gap:6px;margin:4px 0 12px}
.kg-lab{color:#7b8794;font-size:11px;text-transform:uppercase;letter-spacing:.6px;margin-right:2px}
.kg-btn{border:1px solid #cdd5e0;background:#fff;color:#3d4756;padding:4px 13px;
border-radius:7px;cursor:pointer;font-size:12.5px;font-weight:600}
.kg-btn.active{background:var(--primary);color:#fff;border-color:var(--primary)}
.kg-btn:hover{border-color:var(--primary)}
@media(max-width:900px){#sidebar{display:none}#content{margin-left:0;padding:20px}}
"""

_JS = """
const subs=[...document.querySelectorAll('.sub, .dataset')];
const links=new Map([...document.querySelectorAll('.nav-subs a, .nav-ds')]
  .map(a=>[a.getAttribute('href').slice(1),a]));
const obs=new IntersectionObserver((es)=>{
  es.forEach(e=>{if(e.isIntersecting){
    links.forEach(l=>l.classList.remove('active'));
    const l=links.get(e.target.id); if(l)l.classList.add('active');
  }});
},{rootMargin:'-10% 0px -80% 0px',threshold:0});
subs.forEach(s=>obs.observe(s));

// Responsive Plotly charts inside flex/grid columns can mis-size on first paint
// (they fall back to the default 700px). Nudge them once the layout has settled
// so each chart fits — and centers in — its container.
function resizePlots(){window.dispatchEvent(new Event('resize'));}
window.addEventListener('load',function(){setTimeout(resizePlots,60);});

// DE/EN label toggle for the knowledge-graph / ontology visuals
function kgLang(btn,lang){
  document.querySelectorAll('.kg-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.lang===lang);});
  document.querySelectorAll('.kg-de').forEach(function(e){
    e.style.display = lang==='de'?'':'none';});
  document.querySelectorAll('.kg-en').forEach(function(e){
    e.style.display = lang==='en'?'':'none';});
  setTimeout(resizePlots,30);
}
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BSH Predictive Maintenance — Data Analysis & Modelling</title>
<script src="{plotly}" charset="utf-8"></script>
<style>{css}</style>
</head>
<body>
{sidebar}
<main id="content">
<header class="top">
<h1>Predictive Maintenance — Data Analysis &amp; Modelling</h1>
<div class="gen">Generated {generated} · Bosch / BSH · EDA · predictive modelling · knowledge graph</div>
</header>
{body}
</main>
<script>{js}</script>
</body>
</html>"""
