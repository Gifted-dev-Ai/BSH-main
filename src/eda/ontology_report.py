"""Report section for the maintenance ontology + knowledge graph.

Renders: the ontology schema (classes + relations), the populated knowledge graph
as a concept network, the phenomenon -> action -> result process flow (Sankey), the
outcome distribution, and a bilingual instance table.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .charts import _div
from . import ontology as onto

# colour per ontology class
CLASS_COLOR = {
    "MaintenanceEvent": "#0a7cff", "RepairActivity": "#3d4756",
    "Phenomenon": "#e2001a", "Action": "#ff9f1c", "Result": "#00b3a4",
    "Component": "#7b61ff", "FaultGroup": "#0d8a5a", "Incident": "#9fb3c8",
}
RESULT_PRIORITY = ["behoben", "getauscht", "eskaliert", "erfolglos", "in Bearbeitung"]


def _de(label: str) -> str:
    return onto._split_de_en(label)[0]


def _en(label: str) -> str:
    return onto._split_de_en(label)[1]


def _build_de2en(ext: pd.DataFrame) -> dict:
    """Map each canonical German concept label -> its English gloss."""
    m = {}
    for _, r in ext.iterrows():
        for cat in ("phenomenon", "action", "result", "component"):
            for lab in r["_labels"][cat]:
                de, en = onto._split_de_en(lab)
                m[de] = en
    return m


def _disp(de: str, lang: str, de2en: dict) -> str:
    """Display a German canonical label in the requested language."""
    return de if lang == "de" else de2en.get(de, de)


# DE/EN toggle control + wrappers (a global kgLang() in report.py JS switches them)
_TOGGLE = (
    '<div class="kg-toggle"><span class="kg-lab">Labels</span>'
    '<button class="kg-btn active" data-lang="de" onclick="kgLang(this,\'de\')">'
    'Deutsch</button>'
    '<button class="kg-btn" data-lang="en" onclick="kgLang(this,\'en\')">'
    'English</button></div>')


def _bilingual(de_html: str, en_html: str) -> str:
    return (_TOGGLE
            + f'<div class="kg-de">{de_html}</div>'
            + f'<div class="kg-en" style="display:none">{en_html}</div>')


def _primary(labels: list[str], priority=None) -> str | None:
    if not labels:
        return None
    if priority:
        for p in priority:
            for lab in labels:
                if _de(lab) == p:
                    return lab
    return labels[0]


# --------------------------------------------------------------------------- #
# 1. Ontology schema (T-Box)
# --------------------------------------------------------------------------- #
def _schema_diagram() -> str:
    pos = {
        "Incident": (0.0, 1.6), "MaintenanceEvent": (1.3, 1.6),
        "FaultGroup": (1.3, 2.7), "RepairActivity": (2.7, 1.6),
        "Phenomenon": (4.2, 2.7), "Action": (4.2, 2.0),
        "Result": (4.2, 1.2), "Component": (4.2, 0.5),
    }
    fig = go.Figure()
    for s, rel, o in onto.RELATIONS:
        x0, y0 = pos[s]; x1, y1 = pos[o]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2,
                           arrowsize=1.1, arrowwidth=1.2, arrowcolor="#9fb3c8")
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.06, text=rel,
                           showarrow=False, font=dict(size=9, color="#52606d"))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()],
        mode="markers+text", text=list(pos.keys()), textposition="middle center",
        textfont=dict(size=11, color="white"),
        marker=dict(size=46, color=[CLASS_COLOR[c] for c in pos],
                    line=dict(color="white", width=1.5), symbol="square"),
        hovertext=[onto.CLASSES[c] for c in pos], hoverinfo="text"))
    fig.update_layout(title="Maintenance ontology — classes & relations",
                      showlegend=False, plot_bgcolor="white",
                      xaxis=dict(visible=False, range=[-0.5, 5.2]),
                      yaxis=dict(visible=False, range=[0.0, 3.1]),
                      margin=dict(l=10, r=10, t=40, b=10), height=380)
    return _div(fig, height=380)


# --------------------------------------------------------------------------- #
# 2. Knowledge graph — concept network (A-Box)
# --------------------------------------------------------------------------- #
def _concept_network(ext: pd.DataFrame, de2en: dict, lang: str,
                     per_class: int = 8) -> str:
    cats = ["phenomenon", "action", "result", "component"]
    cls_of = {"phenomenon": "Phenomenon", "action": "Action",
              "result": "Result", "component": "Component"}
    freq = {c: Counter() for c in cats}
    for _, r in ext.iterrows():
        for c in cats:
            for lab in r["_labels"][c]:
                freq[c][_de(lab)] += 1
    keep = set()
    for c in cats:
        keep |= {f"{c}:{de}" for de, _ in freq[c].most_common(per_class)}

    co = Counter()
    node_freq = Counter()
    for _, r in ext.iterrows():
        present = [f"{c}:{_de(lab)}" for c in cats for lab in r["_labels"][c]
                   if f"{c}:{_de(lab)}" in keep]
        for n in present:
            node_freq[n] += 1
        for a, b in combinations(sorted(set(present)), 2):
            co[(a, b)] += 1

    G = nx.Graph()
    for n in keep:
        if node_freq[n]:
            G.add_node(n)
    for (a, b), w in co.items():
        if w >= 2 and a in G and b in G:
            G.add_edge(a, b, weight=w)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    if G.number_of_nodes() == 0:
        return '<p class="muted">Not enough co-occurring concepts to graph.</p>'

    layout = nx.spring_layout(G, seed=7, k=0.9)
    ex, ey = [], []
    for a, b in G.edges():
        ex += [layout[a][0], layout[b][0], None]
        ey += [layout[a][1], layout[b][1], None]
    edges = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none",
                       line=dict(color="#d7dde6", width=1))
    traces = [edges]
    for c in cats:
        ns = [n for n in G if n.startswith(f"{c}:")]
        if not ns:
            continue
        traces.append(go.Scatter(
            x=[layout[n][0] for n in ns], y=[layout[n][1] for n in ns],
            mode="markers+text", name=cls_of[c],
            text=[_disp(n.split(":", 1)[1], lang, de2en) for n in ns],
            textposition="top center", textfont=dict(size=9),
            marker=dict(size=[10 + 2.2 * node_freq[n] for n in ns],
                        color=CLASS_COLOR[cls_of[c]], line=dict(color="white", width=1)),
            hovertext=[f"{_disp(n.split(':',1)[1], lang, de2en)} — {node_freq[n]} "
                       "activities" for n in ns],
            hoverinfo="text"))
    fig = go.Figure(traces)
    fig.update_layout(
        title="Knowledge graph — concept instances (size = frequency, edges = co-occurrence)",
        plot_bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10), height=480)
    return _div(fig, height=480)


# --------------------------------------------------------------------------- #
# 3. Phenomenon -> Action -> Result flow (Sankey)
# --------------------------------------------------------------------------- #
def _flow_sankey(ext: pd.DataFrame, lang: str) -> str:
    show = _de if lang == "de" else _en
    pa, ar = Counter(), Counter()
    for _, r in ext.iterrows():
        ph = _primary(r["_labels"]["phenomenon"])
        ac = _primary(r["_labels"]["action"])
        rs = _primary(r["_labels"]["result"], RESULT_PRIORITY)
        ph, ac, rs = (show(ph) if ph else None, show(ac) if ac else None,
                      show(rs) if rs else None)
        if ph and ac:
            pa[(ph, ac)] += 1
        if ac and rs:
            ar[(ac, rs)] += 1
    if not pa and not ar:
        return '<p class="muted">Not enough phenomenon→action→result chains.</p>'
    labels, idx = [], {}

    def node(lab):
        if lab not in idx:
            idx[lab] = len(labels)
            labels.append(lab)
        return idx[lab]

    src, tgt, val = [], [], []
    for (ph, ac), w in pa.most_common(14):
        src.append(node("P: " + ph)); tgt.append(node("A: " + ac)); val.append(w)
    for (ac, rs), w in ar.most_common(14):
        src.append(node("A: " + ac)); tgt.append(node("R: " + rs)); val.append(w)
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=14, thickness=14,
                  line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val),
        textfont=dict(size=10)))
    fig.update_layout(title="Repair reasoning flow: phenomenon → action → result",
                      margin=dict(l=8, r=8, t=40, b=8), height=460)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", config={"displaylogo": False,
                                                     "responsive": True})


# --------------------------------------------------------------------------- #
# 4. Outcome distribution
# --------------------------------------------------------------------------- #
def _outcome_bar(ext: pd.DataFrame, lang: str) -> str:
    show = _de if lang == "de" else _en
    c = Counter()
    for _, r in ext.iterrows():
        for lab in r["_labels"]["result"]:
            c[show(lab)] += 1
    if not c:
        return ""
    items = c.most_common()
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items],
                           orientation="h", marker_color=CLASS_COLOR["Result"]))
    fig.update_layout(title="Outcomes recorded in repair activities",
                      xaxis_title="activities")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _div(fig, height=300)


# --------------------------------------------------------------------------- #
# 5. Instances table
# --------------------------------------------------------------------------- #
def _instance_table(ext: pd.DataFrame, lang: str, n: int = 12) -> str:
    show = _de if lang == "de" else _en
    sub = ext[ext["phenomenon_de_en"].str.len() > 0].head(n)
    desc_col = "Description (DE)" if lang == "de" else "Summary (EN)"
    head = (f"<tr><th>Brid</th><th>{desc_col}</th><th>Phenomenon</th>"
            "<th>Action</th><th>Result</th></tr>")

    def cell(labels):
        return "; ".join(show(x) for x in labels)

    rows = []
    for _, r in sub.iterrows():
        if lang == "de":
            txt = (r["description_de"][:90] + "…") if len(r["description_de"]) > 90 \
                else r["description_de"]
        else:
            txt = r["description_en_summary"] or "—"
        rows.append(
            f'<tr><td class="cname">{r["Brid"]}</td><td>{txt}</td>'
            f'<td>{cell(r["_labels"]["phenomenon"])}</td>'
            f'<td>{cell(r["_labels"]["action"])}</td>'
            f'<td>{cell(r["_labels"]["result"])}</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


# --------------------------------------------------------------------------- #
# Fault Tree Analysis (derived from component -> phenomenon causation)
# --------------------------------------------------------------------------- #
def _component_phenomena(ext: pd.DataFrame):
    """comp -> Counter(phenomenon) and comp frequency, from co-occurring labels."""
    comp_freq = Counter()
    comp_phen = defaultdict(Counter)
    for _, r in ext.iterrows():
        comps = {_de(l) for l in r["_labels"]["component"]}
        phens = {_de(l) for l in r["_labels"]["phenomenon"]}
        for c in comps:
            comp_freq[c] += 1
            for p in phens:
                comp_phen[c][p] += 1
    return comp_freq, comp_phen


def _fault_tree(ext: pd.DataFrame, de2en: dict, lang: str,
                n_comp: int = 5, n_phen: int = 3) -> str:
    comp_freq, comp_phen = _component_phenomena(ext)
    groups = []
    for c, _ in comp_freq.most_common(n_comp):
        phens = [p for p, _ in comp_phen[c].most_common(n_phen)]
        if phens:
            groups.append((c, phens))
    if not groups:
        return '<p class="muted">Not enough component→phenomenon links for a tree.</p>'

    # layout: leaves (phenomena) evenly spaced, components centred over their leaves
    leafgap, groupgap = 1.9, 1.2
    x = 0.0
    comp_x, phen_x = {}, {}
    for c, phens in groups:
        xs = []
        for p in phens:
            phen_x[(c, p)] = x; xs.append(x); x += leafgap
        comp_x[c] = sum(xs) / len(xs)
        x += groupgap
    top_x = sum(comp_x.values()) / len(comp_x)
    TOP_Y, TG_Y, C_Y, CG_Y, P_Y = 5.0, 4.3, 3.3, 2.6, 1.3

    def short(s, n=20):
        return s if len(s) <= n else s[:n - 1] + "…"

    ex, ey = [], []

    def edge(x0, y0, x1, y1):
        ex.extend([x0, x1, None]); ey.extend([y0, y1, None])

    edge(top_x, TOP_Y, top_x, TG_Y)
    for c, phens in groups:
        edge(top_x, TG_Y, comp_x[c], C_Y)
        edge(comp_x[c], C_Y, comp_x[c], CG_Y)
        for p in phens:
            edge(comp_x[c], CG_Y, phen_x[(c, p)], P_Y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none",
                             line=dict(color="#9fb3c8", width=1), showlegend=False))
    # gates (OR / ≥1)
    gx = [top_x] + [comp_x[c] for c, _ in groups]
    gy = [TG_Y] + [CG_Y for _ in groups]
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers+text", name="OR gate (≥1)",
                             text=["≥1"] * len(gx), textposition="middle center",
                             textfont=dict(size=8, color="#52606d"),
                             marker=dict(symbol="diamond-wide", size=24,
                                         color="#eef1f6", line=dict(color="#9fb3c8", width=1))))
    # top + component events (rectangles)
    fig.add_trace(go.Scatter(
        x=[top_x] + [comp_x[c] for c, _ in groups],
        y=[TOP_Y] + [C_Y for _ in groups], mode="markers+text",
        name="event (intermediate)",
        text=["STATION<br>maintenance event"]
        + [short(_disp(c, lang, de2en)) for c, _ in groups],
        textposition="middle center", textfont=dict(size=8, color="white"),
        marker=dict(symbol="square", size=46,
                    color=["#0d1b2a"] + [CLASS_COLOR["Component"]] * len(groups),
                    line=dict(color="white", width=1)),
        hovertext=["Top event"]
        + [f"{_disp(c, lang, de2en)} — {comp_freq[c]} activities" for c, _ in groups],
        hoverinfo="text"))
    # phenomena (basic events / circles)
    px = [phen_x[(c, p)] for c, phens in groups for p in phens]
    py = [P_Y] * len(px)
    plab = [short(_disp(p, lang, de2en), 16) for c, phens in groups for p in phens]
    phov = [f"{_disp(p, lang, de2en)} (in {_disp(c, lang, de2en)}): {comp_phen[c][p]}×"
            for c, phens in groups for p in phens]
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", name="basic cause (mode)",
                             text=plab, textposition="bottom center",
                             textfont=dict(size=8.5),
                             marker=dict(symbol="circle", size=15,
                                         color=CLASS_COLOR["Phenomenon"],
                                         line=dict(color="white", width=1)),
                             hovertext=phov, hoverinfo="text"))
    fig.update_layout(
        title="Fault tree — station maintenance event (OR-decomposition)",
        plot_bgcolor="white", xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0.4, 5.6]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10), height=460)
    return _div(fig, height=460)


def _cut_sets_table(ext: pd.DataFrame, de2en: dict, lang: str, n: int = 12) -> str:
    """Single-event minimal cut sets = component·failure-mode pairs, by frequency."""
    pairs = Counter()
    for _, r in ext.iterrows():
        comps = {_de(l) for l in r["_labels"]["component"]}
        phens = {_de(l) for l in r["_labels"]["phenomenon"]}
        for c in comps:
            for p in phens:
                pairs[(c, p)] += 1
    if not pairs:
        return ""
    total = sum(pairs.values())
    head = ("<tr><th>Component (subsystem)</th><th>Failure mode (phenomenon)</th>"
            "<th>count</th><th>share</th></tr>")
    rows = []
    for (c, p), w in pairs.most_common(n):
        rows.append(f'<tr><td class="cname">{_disp(c, lang, de2en)}</td>'
                    f'<td>{_disp(p, lang, de2en)}</td><td>{w}</td>'
                    f'<td>{100*w/total:.0f}%</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


# --------------------------------------------------------------------------- #
# Section
# --------------------------------------------------------------------------- #
def build_standalone_section(refresh: bool = False) -> dict:
    ext, G, triples = onto.run()
    de2en = _build_de2en(ext)
    n_ev = sum(1 for _, d in G.nodes(data=True) if d["ntype"] == "MaintenanceEvent")

    intro = (
        '<p class="desc">The repair-activity log '
        '(<code>description_repair_activities_data.csv</code>, linked to incidents by '
        '<code>Brid</code>) is a German free-text narrative of each maintenance step. '
        'Its <b>phenomenon</b>, <b>action</b> and <b>result</b> are extracted (reusing '
        'the bilingual incident lexicon + a result lexicon) and populated into a '
        '<b>maintenance ontology</b>. Every concept node carries both the original '
        'German label and an English gloss. The full graph is exported to '
        '<code>data/processed/</code> (extracted CSV, RDF-style triples, ontology JSON).</p>'
        f'<div class="kpi-row">'
        f'<div class="kpi"><div class="kpi-val">{n_ev}</div>'
        f'<div class="kpi-label">maintenance events (Brid)</div></div>'
        f'<div class="kpi"><div class="kpi-val">{len(ext):,}</div>'
        f'<div class="kpi-label">repair activities</div></div>'
        f'<div class="kpi"><div class="kpi-val">{len(triples):,}</div>'
        f'<div class="kpi-label">knowledge-graph triples</div></div>'
        f'<div class="kpi"><div class="kpi-val">{len(onto.CLASSES)}</div>'
        f'<div class="kpi-label">ontology classes</div></div></div>')

    schema_html = (intro + f'<div class="card">{_schema_diagram()}</div>'
                   + '<p class="muted">Classes: '
                   + ", ".join(f"<b>{c}</b> ({d})" for c, d in onto.CLASSES.items())
                   + ".</p>")

    graph_html = (
        _bilingual(f'<div class="card">{_concept_network(ext, de2en, "de")}</div>',
                   f'<div class="card">{_concept_network(ext, de2en, "en")}</div>')
        + '<p class="muted">Concept instances coloured by ontology class; '
        'node size = number of activities, edges = co-occurrence within a step (≥ 2).</p>')

    def _flow(lang):
        return ('<div class="grid">'
                f'<div class="half"><div class="card">{_flow_sankey(ext, lang)}</div></div>'
                f'<div class="half">{_outcome_bar(ext, lang)}</div></div>')
    flow_html = _bilingual(_flow("de"), _flow("en"))

    fta_intro = (
        '<p class="desc">A <b>fault tree</b> derived from the knowledge graph: the top '
        'event (a station maintenance event) is decomposed — via <b>OR gates</b> '
        '(≥1, any cause suffices) — into the most-involved <b>components</b> '
        '(intermediate events), then into their observed <b>failure modes</b> '
        '(phenomena = basic causes). Frequencies come from how often each '
        'component·mode co-occurs in the repair logs.</p>')
    cut_hdr = (
        '<h5 style="margin:14px 0 8px;color:#52606d">Minimal cut sets — most frequent '
        'single-event causes (component · failure mode)</h5>'
        '<p class="muted">In this OR-decomposition each row is a size-1 cut set; the '
        'share approximates each cause\'s contribution to maintenance events.</p>')

    def _fta(lang):
        return (f'<div class="card">{_fault_tree(ext, de2en, lang)}</div>'
                + cut_hdr + _cut_sets_table(ext, de2en, lang))
    fta_html = fta_intro + _bilingual(_fta("de"), _fta("en"))

    inst_intro = ('<p class="desc">Sample populated instances — each repair activity '
                  'with its extracted concepts (toggle DE / EN). Full export: '
                  '<code>data/processed/repair_descriptions_extracted.csv</code> and '
                  '<code>maintenance_kg_triples.csv</code>.</p>')
    inst_html = inst_intro + _bilingual(_instance_table(ext, "de"),
                                        _instance_table(ext, "en"))

    return {
        "id": "ontology",
        "title": "Maintenance ontology & knowledge graph",
        "subs": [
            {"id": "schema", "label": "Ontology schema", "html": schema_html},
            {"id": "graph", "label": "Knowledge graph", "html": graph_html},
            {"id": "flow", "label": "Phenomenon → action → result", "html": flow_html},
            {"id": "faulttree", "label": "Fault tree analysis", "html": fta_html},
            {"id": "instances", "label": "Instances (DE / EN)", "html": inst_html},
        ],
    }
