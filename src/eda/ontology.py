"""Maintenance ontology + knowledge graph from the repair-activity descriptions.

``description_repair_activities_data.csv`` is a per-incident repair log (keyed by
``Brid``, linking to the incident table). Each row is a German free-text narrative
of a maintenance step — the **phenomenon** observed, the **action** taken and the
**result**. This module:

1. extracts those facts (reusing the bilingual incident lexicon for component /
   phenomenon / action, plus a small **result** lexicon defined here),
2. defines a small **maintenance ontology** (classes + relations), and
3. populates it into a **knowledge graph** of instances — every concept node carries
   both the original German label and an English gloss.

Outputs (data/processed):
* ``repair_descriptions_extracted.csv`` — each activity with bilingual phenomenon /
  action / result / component and an English summary of the German text.
* ``maintenance_kg_triples.csv`` — the populated knowledge graph as RDF-style triples.
* ``maintenance_ontology.json`` — the schema (classes + relations) and node labels.
"""

from __future__ import annotations

import json
import re

import networkx as nx
import pandas as pd

from .config import PROJECT_ROOT, RAW_DIR
from .extraction.extractor import LexiconExtractor
from .extraction.lexicon import Term
from .extraction.normalize import normalize

PROCESSED = PROJECT_ROOT / "data" / "processed"
REPAIR_FILE = "description_repair_activities_data.csv"
TS_RE = r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})"
TS_FMT = "%d.%m.%Y %H:%M"

# --- result lexicon (bilingual), priority order for the "outcome" of a step ------
RESULT_TERMS: list[Term] = [
    Term("behoben", "resolved",
         (r"\blief\b", r"lauft wieder", r"lauft wieder", r"behoben", r"gelost",
          r"funktioniert", r"in betrieb", r"freigabe", r"erfolgreich", r"\bi\.?o\b",
          r"wieder ok", r"problem.*gelost", r"anlage lief")),
    Term("getauscht", "replaced",
         (r"getauscht", r"ersetzt", r"\btausch", r"erneuert", r"gewechselt")),
    Term("erfolglos", "unsuccessful",
         (r"vergebens", r"vergeblich", r"ohne erfolg", r"nicht behoben",
          r"weiterhin", r"erneut", r"leider", r"kein erfolg", r"keine veranderung",
          r"keine anderung")),
    Term("eskaliert", "escalated",
         (r"eskal", r"\bafa\b", r"hersteller", r"lieferant", r"techniker",
          r"\bticket\b", r"service", r"support", r"steuerung informiert",
          r"pool.?steu", r"informiert")),
    Term("in Bearbeitung", "in progress",
         (r"fehlersuche", r"beobacht", r"in bearbeitung", r"kontrolle",
          r"abwarten", r"prufen")),
]
_RESULT_COMPILED = [(t, [re.compile(p) for p in t.patterns]) for t in RESULT_TERMS]

# --- ontology schema (T-Box) -----------------------------------------------------
CLASSES = {
    "MaintenanceEvent": "a defect/incident being worked on (one Brid)",
    "RepairActivity": "one logged repair/diagnosis step (one description)",
    "Phenomenon": "the observed symptom / fault",
    "Action": "the corrective or diagnostic action taken",
    "Result": "the outcome of the step",
    "Component": "the machine module / part involved",
    "FaultGroup": "the incident group (Roboter, Plasma, …)",
    "Incident": "the originating incident record (Brid) in the incident log",
}
RELATIONS = [
    ("MaintenanceEvent", "linkedToIncident", "Incident"),
    ("MaintenanceEvent", "hasActivity", "RepairActivity"),
    ("MaintenanceEvent", "belongsToGroup", "FaultGroup"),
    ("RepairActivity", "observesPhenomenon", "Phenomenon"),
    ("RepairActivity", "performsAction", "Action"),
    ("RepairActivity", "yieldsResult", "Result"),
    ("RepairActivity", "involvesComponent", "Component"),
]
# extracted category -> (ontology class, relation from RepairActivity)
CAT_TO_CLASS = {
    "phenomenon": ("Phenomenon", "observesPhenomenon"),
    "action": ("Action", "performsAction"),
    "result": ("Result", "yieldsResult"),
    "component": ("Component", "involvesComponent"),
}


def _result_labels(text: str) -> list[str]:
    norm = normalize(text)
    return [t.label for t, rxs in _RESULT_COMPILED if any(r.search(norm) for r in rxs)]


def _split_de_en(label: str) -> tuple[str, str]:
    """'Messung fehlerhaft (measurement faulty)' -> ('Messung fehlerhaft', '…')."""
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", label)
    return (m.group(1), m.group(2)) if m else (label, label)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_descriptions() -> pd.DataFrame:
    """Repair rows + extracted bilingual phenomenon / action / result / component."""
    df = pd.read_csv(RAW_DIR / REPAIR_FILE, encoding="utf-8-sig")
    df = df[df["Description"].notna() & (df["Description"].astype(str).str.strip()
                                         != "-")].copy()
    df["_ts"] = pd.to_datetime(
        df["Histdate"].astype("string").str.extract(TS_RE, expand=False),
        format=TS_FMT, errors="coerce")

    extractor = LexiconExtractor(source_cols=("Description",))
    recs = []
    for _, r in df.iterrows():
        res = extractor.extract({"Description": str(r["Description"])})
        labels = {
            "phenomenon": res.labels.get("phenomenon", []),
            "action": res.labels.get("action", []),
            "component": res.labels.get("component", []),
            "result": _result_labels(str(r["Description"])),
        }
        en_bits = []
        for cat in ("phenomenon", "action", "result"):
            for lab in labels[cat]:
                en_bits.append(f"{cat[:4]}: {_split_de_en(lab)[1]}")
        recs.append({
            "Brid": r["Brid"], "Histdate": r["Histdate"], "ts": r["_ts"],
            "Groupdescr": r.get("Groupdescr", "-"),
            "description_de": str(r["Description"]).replace("\n", " ").strip(),
            "phenomenon_de_en": "; ".join(labels["phenomenon"]),
            "action_de_en": "; ".join(labels["action"]),
            "result_de_en": "; ".join(labels["result"]),
            "component_de_en": "; ".join(labels["component"]),
            "description_en_summary": "; ".join(en_bits),
            "_labels": labels,
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
def build_graph(ext: pd.DataFrame) -> tuple[nx.DiGraph, list[tuple]]:
    """Populate the ontology with instances -> (graph, triples)."""
    G = nx.DiGraph()
    triples: list[tuple] = []

    def add_node(nid, ntype, de, en=""):
        if nid not in G:
            G.add_node(nid, ntype=ntype, de=de, en=en or de)

    def add_edge(s, p, o):
        G.add_edge(s, o, rel=p)
        triples.append((s, p, o))

    for brid, rows in ext.groupby("Brid"):
        ev = f"event:{brid}"
        add_node(ev, "MaintenanceEvent", str(brid), str(brid))
        add_node(f"incident:{brid}", "Incident", str(brid), str(brid))
        add_edge(ev, "linkedToIncident", f"incident:{brid}")
        groups = {g for g in rows["Groupdescr"] if isinstance(g, str) and g not in ("-", "")}
        for g in groups:
            gid = f"group:{g}"
            add_node(gid, "FaultGroup", g, g)
            add_edge(ev, "belongsToGroup", gid)
        for i, (_, r) in enumerate(rows.iterrows()):
            act = f"act:{brid}:{i}"
            add_node(act, "RepairActivity",
                     str(r["Histdate"]), r["description_en_summary"])
            G.nodes[act]["desc_de"] = r["description_de"]
            add_edge(ev, "hasActivity", act)
            for cat, (cls, rel) in CAT_TO_CLASS.items():
                for lab in r["_labels"][cat]:
                    de, en = _split_de_en(lab)
                    nid = f"{cat}:{de}"
                    add_node(nid, cls, de, en)
                    add_edge(act, rel, nid)
    return G, triples


# --------------------------------------------------------------------------- #
# Persist
# --------------------------------------------------------------------------- #
def run(out_dir=PROCESSED) -> tuple[pd.DataFrame, nx.DiGraph, list[tuple]]:
    ext = extract_descriptions()
    G, triples = build_graph(ext)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["Brid", "Histdate", "Groupdescr", "description_de",
            "phenomenon_de_en", "action_de_en", "result_de_en",
            "component_de_en", "description_en_summary"]
    ext[cols].to_csv(out_dir / "repair_descriptions_extracted.csv",
                     index=False, encoding="utf-8-sig")

    trip = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
    trip.to_csv(out_dir / "maintenance_kg_triples.csv",
                index=False, encoding="utf-8-sig")

    nodes = [{"id": n, "type": d["ntype"], "de": d.get("de"), "en": d.get("en")}
             for n, d in G.nodes(data=True)]
    schema = {"classes": CLASSES,
              "relations": [{"from": s, "rel": p, "to": o} for s, p, o in RELATIONS],
              "nodes": nodes,
              "counts": {"events": sum(1 for _, d in G.nodes(data=True)
                                       if d["ntype"] == "MaintenanceEvent"),
                         "activities": sum(1 for _, d in G.nodes(data=True)
                                           if d["ntype"] == "RepairActivity"),
                         "triples": len(triples)}}
    (out_dir / "maintenance_ontology.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ev = schema["counts"]["events"]
    print(f"Maintenance KG: {n_ev} events, {len(ext)} activities, "
          f"{len(triples)} triples -> {out_dir}")
    return ext, G, triples


if __name__ == "__main__":
    run()
