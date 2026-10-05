# `src/eda` — Exploratory Data Analysis pipeline

Profiles the raw datasets in [`data/raw`](../../data/raw) and renders a single,
self-contained interactive HTML report into [`reports/`](../../reports).

## Run

```bash
# from the project root
python -m src.eda                                   # all datasets -> reports/pdm_report.html
python -m src.eda --only plasma_errors incident_mgmt
python -m src.eda --out reports/custom.html
```

Requires `pandas`, `numpy`, `plotly`, `pyarrow` (see [requirements.txt](../../requirements.txt)).
The report loads `plotly.js` from a CDN, so an internet connection is needed to
*view* the charts (the file itself is fully portable).

## How it works

| Module        | Responsibility                                                            |
| ------------- | ------------------------------------------------------------------------- |
| `config.py`   | Declarative description of each dataset (paths, timestamp col, categoricals, highlight signals). Single source of truth. |
| `profiler.py` | Computes per-column statistics, time bins, correlations and a bounded sample. |
| `charts.py`   | Builds interactive Plotly figures (returned as HTML `<div>`s).             |
| `report.py`   | Assembles the HTML page with a fixed left navigation sidebar.              |
| `incident_analysis.py` | Incident deep-dive: enriches the incident log with the [extraction](extraction/README.md) fields and builds the *Plasma vs Robot* (time distribution + defect-category correlations) and *Patterns & sequences* subsections, filtered by `Groupdescr`. |
| `patterns.py` | Dependency-free pattern mining: Apriori frequent itemsets / association rules (support·confidence·lift) and recency-aware consecutive n-gram sequences (compared across several recency windows, set by `RECENCY_LEVELS`). |
| `mes_text.py` | Extracts critical info from the MES `error_text` (fault category, terminal, error code) via a lexicon; the per-category hourly error counts become model features. Writes `data/processed/mes_error_extracted.csv`. |
| `ontology.py` | Builds a bilingual **maintenance ontology + knowledge graph** from the repair-activity descriptions: extracts phenomenon/action/result/component, defines classes + relations, and populates instances (networkx). Writes `repair_descriptions_extracted.csv`, `maintenance_kg_triples.csv`, `maintenance_ontology.json`. |
| `ontology_report.py` | Report section: ontology schema diagram, concept knowledge-graph network, phenomenon→action→result Sankey, outcome distribution, bilingual instance table. |
| `early_factors.py` | Early-predictive-factor analysis: clusters incidents into defect events, aggregates Nozzle-head + MES signals (total **and per fault category**) to hourly buckets (cached in `data/interim`), and scores how well each state variable separates pre-defect vs normal windows (Mann-Whitney AUC) over an advance × observation × granularity grid. |
| `early_factors_report.py` | Charts + report subsection for the early-factor analysis (AUC heatmap, lead-time / granularity sensitivity, top-feature tables, best-separator box plot, and a cross-validated logistic **combined early-warning model** with ROC vs lead time). |
| `automl.py` | Automated model search: evaluates a panel of classifiers (logistic, random forest, extra-trees, hist-GBM, LightGBM) per lead time with stratified CV ROC-AUC; extracts best-model feature importances. Cached to `data/interim`. |
| `automl_report.py` | Report subsection: lead-time vs model-performance trade-off (per model + best envelope), best model per lead, and best predictive factors. |
| `rul.py` | Remaining-Useful-Life regression: labels each time point with hours-to-next-defect, builds windowed features, and evaluates regressors with **leakage-free GroupKFold** over an observation × horizon × model grid (cached). Includes the naive-vs-grouped leakage demo and a useful-prediction-horizon estimate. |
| `rul_report.py` | Report subsection: model/parameter comparison, temporal-leakage contrast, predicted-vs-actual, and error-vs-time-to-failure (predictive horizon). |
| `extraction/` | Free-text → structured bilingual fields. Runs automatically for the incident log (see its [README](extraction/README.md)). |
| `__main__.py` | CLI entry point / orchestration (profiling + incident extraction + deep-dive). |

### Big-data handling (multiprocessing)

Files larger than 500 MB (the 5.4 GB robot telemetry file) are profiled with a
**parallel byte-range engine**: the file is split into line-aligned byte slices, a
process pool profiles each slice independently, and the compact partial statistics
(count / sum / sum-of-squares / min / max, capped value-counts, hourly time bins,
a bounded random sample) are merged. This streams the data and never holds the full
5.4 GB in memory — the whole file profiles in ~20 s on 18 workers.

Smaller files are loaded directly and profiled in a single process. Both paths share
the exact same partial-stat primitives, so their output is identical in shape.

### What the report contains

For each dataset: an overview (row/column counts, time span, profiling engine), a
full column profile table, missingness, an hourly time-series view, numeric
distributions, a correlation matrix, and top-category bar charts. Distributions and
correlations are computed on a bounded random sample (configurable via
`SAMPLE_ROWS` in `config.py`) to keep the charts responsive.
