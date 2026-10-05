"""Report subsection for the MES ``error_text`` extraction.

Explains *how* the free-text plasma-station error messages are turned into the fault
categories used by the models, shows the matching lexicon, and plots the distribution
of the extracted categories. Attached to the ``plasma_errors`` dataset section.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .charts import PRIMARY, _div
from . import mes_text as mt
from .config import RAW_DIR

# plain-language gloss for each lexicon category (keeps the table readable)
_DESC = {
    "gripper": "gripper / clamping faults",
    "motion_position": "positioning, travel-path & movement-order errors",
    "power": "supply-voltage / power-on faults",
    "servo_drive": "servo & EtherCAT drive terminals (EP7211/EP7041), NC diag, decoupling",
    "robot_drive": "robot drive axis (90LX)",
    "sensor": "sensor terminal addresses (M-Bx, S-Ax, -Bxxx)",
    "status": "machine state / handshake / program / temperature",
    "other": "text present but no pattern matched (residual bucket)",
}
_AMBER = "#ff9f1c"


def _load() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "Plasma_Station__MES_error_event.csv",
                     usecols=["event_type", "error_text"])
    ev = df[df["event_type"] == 1].copy()
    return mt.extract_frame(ev)


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>')


def _lexicon_table() -> str:
    head = ("<tr><th>Priority</th><th>Fault category</th><th>What it captures</th>"
            "<th>Match patterns (regex, case-insensitive)</th></tr>")
    rows = []
    for i, (cat, pats) in enumerate(mt.LEXICON, 1):
        pat = ", ".join(f"<code>{p}</code>" for p in pats)
        rows.append(f'<tr><td>{i}</td><td class="cname">{cat}</td>'
                    f'<td>{_DESC.get(cat, "")}</td><td>{pat}</td></tr>')
    rows.append(f'<tr style="color:#52606d"><td>fallback</td>'
                f'<td class="cname">other</td><td>{_DESC["other"]}</td>'
                f'<td>— (no pattern matched)</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


def _category_chart(ext: pd.DataFrame) -> str:
    typed = ext[ext["error_text"].notna()]
    counts = typed["mes_category"].value_counts()
    cats = [c for c in mt.MES_FEATURE_CATEGORIES if c in counts.index]
    cats = sorted(cats, key=lambda c: counts[c])          # ascending → biggest on top
    vals = [int(counts[c]) for c in cats]
    total = int(counts.sum()) or 1
    colors = [_AMBER if c == "other" else PRIMARY for c in cats]
    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h", marker_color=colors,
        text=[f"{v:,} ({v / total * 100:.1f}%)" for v in vals], textposition="auto",
        hovertemplate="%{y}: %{x:,} messages<extra></extra>"))
    fig.update_layout(
        title=f"Extracted MES fault-category distribution ({total:,} typed messages)",
        xaxis_title="error messages")
    fig.update_yaxes(automargin=True)
    return _div(fig, height=max(320, 38 * len(cats) + 120))


def build_section() -> dict:
    ext = _load()
    n_ev = len(ext)
    has_text = ext["error_text"].notna()
    n_text = int(has_text.sum())
    typed = ext[has_text]
    n_matched = int((typed["mes_category"] != "other").sum())
    n_term = int((ext["mes_terminal"] != "").sum())
    n_code = int((ext["mes_code"] != "").sum())

    intro = (
        '<p class="desc">The plasma station raises a free-text <code>error_text</code> '
        'on every <b>error event</b> (<code>event_type = 1</code>). The raw messages '
        'are <i>not</i> a clean code field — each one mixes up to three things:</p>'
        '<ol class="desc" style="margin:6px 0 10px;line-height:1.7">'
        '<li><b>Beckhoff / EtherCAT terminal addresses</b> — e.g. '
        '<code>=101+E-K018A15 (EP7041-1002)</code>, identifying the physical I/O '
        'terminal that faulted;</li>'
        '<li><b>TwinCAT / diagnostic codes</b> — e.g. <code>NC DiagNo.: 18000</code>, '
        '<code>ErrorCode: 0x0100</code>;</li>'
        '<li><b>human-readable faults</b> — e.g. <i>"Actuator voltage is too low"</i>, '
        '<i>"minimum travel path not reached"</i>, <i>"ParCmd.GripForce out of '
        'range"</i>.</li></ol>'
        '<p class="desc"><b>Processing logic</b> '
        '(<code>src/eda/mes_text.py</code>): each message is run through a '
        '<b>priority-ordered lexicon</b> of regular expressions; the <b>first</b> '
        'category whose patterns match wins, mapping the message to a single '
        '<b>fault type</b>. In parallel a regex pulls the <b>terminal</b> '
        '(<code>EP\\d{4}</code>) and the <b>diagnostic code</b> '
        '(<code>NC DiagNo.</code> / <code>0x…</code>) when present. Messages with no '
        'text are <b>untyped</b> (counted only in the total); typed messages that '
        'match nothing land in <b>other</b>. The per-category hourly counts become the '
        'MES features used by the early-prediction / AutoML / RUL / staging models — a '
        'richer signal than the single total error count.</p>'
        '<div class="kpi-row">'
        + _kpi("error events", f"{n_ev:,}", "event_type = 1")
        + _kpi("carry free text", f"{n_text / max(n_ev,1)*100:.0f}%",
               f"{n_text:,} typed · {n_ev - n_text:,} untyped (null)")
        + _kpi("matched a fault type", f"{n_matched / max(n_text,1)*100:.0f}%",
               f"{n_text - n_matched:,} fell to 'other'")
        + _kpi("terminal / code pulled", f"{n_term:,} / {n_code:,}",
               "EP-terminal · NC-diag / 0x code")
        + '</div>')

    html = (
        intro
        + '<h4 style="margin:18px 0 8px">Categorisation lexicon (first match wins)</h4>'
        '<p class="desc">Priority matters: a drive-terminal message that also mentions '
        'a voltage is classed as <code>servo_drive</code> only if that pattern sits '
        'higher in the list. Extend by adding a <code>(category, [regex])</code> row to '
        '<code>LEXICON</code>.</p>'
        + _lexicon_table()
        + '<h4 style="margin:18px 0 8px">Distribution of extracted fault categories</h4>'
        '<p class="desc">Across the typed messages, <b>servo-drive</b> and '
        '<b>sensor</b> faults dominate; the small <b>other</b> bar (the residual that '
        'matched no pattern) is the coverage gap to watch when extending the '
        'lexicon.</p>'
        + f'<div class="card">{_category_chart(ext)}</div>'
        + '<p class="muted">Note: the ~70% of error events with <i>no</i> error_text '
        '(untyped) carry no fault detail and are excluded from the chart above; they '
        'still contribute to the <code>MES_total</code> count.</p>')

    return {"id": "mes_extraction",
            "label": "Error-text extraction & fault categories", "html": html}
