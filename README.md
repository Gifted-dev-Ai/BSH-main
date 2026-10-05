# BSH — Bosch Predictive Maintenance

Data analysis and predictive-maintenance (PdM) for the **Plasma coating station** and
its handling **Stäubli robot** on production line `0206` / station `0370`.

The goal is to turn raw machine telemetry, process parameters and MES fault logs into
**early-warning health indicators and failure predictions** — so maintenance can be
planned before unplanned downtime occurs.

---

## What's in here

| Signal source            | Dataset                                | Role in PdM                          |
| ------------------------ | -------------------------------------- | ------------------------------------ |
| Stäubli robot telemetry  | `Stäubli_Robot_monitoring_Data.csv`    | Condition monitoring (wide, ~50 ms)  |
| Plasma process & quality | `Plasma_Nozzle_Head_Data.csv`          | Process drift / product quality      |
| Plasma MES faults        | `Plasma_Station__MES_error_event.csv`  | Failure labels / downtime            |
| Maintenance incidents    | `Maintenance_report_Incident_Management_data.csv` | Curated downtime / root-cause labels |
| Repair activity logs     | `description_repair_activities_data.csv` | Repair narratives → maintenance knowledge graph |

Full column-level documentation lives in [data/README.md](data/README.md). The
incident log's German free-text columns are mined into structured bilingual fields
by [src/eda/extraction](src/eda/extraction/README.md) → `data/processed/`.

---

## Repository structure

```
BSH/
├── data/                  # All datasets (git-ignored; see data/README.md)
│   ├── raw/               # Immutable original extracts — read-only ground truth
│   ├── interim/           # Partially cleaned, intermediate data (rebuildable)
│   ├── processed/         # Analysis-ready feature tables / model inputs
│   └── external/          # Reference data (asset master, maintenance logs, specs)
│
├── notebooks/             # Exploratory analysis & prototyping (EDA, experiments)
│
├── src/                   # Reusable, production pipeline code
│   ├── data/              # Loading, cleaning, validation, raw → interim/processed
│   ├── features/          # Feature engineering (signal stats, lags, health indices)
│   ├── models/            # Training, evaluation, inference (PdM / anomaly / RUL)
│   └── visualization/     # Plotting & reporting helpers
│
├── models/                # Serialized trained models & artifacts (git-ignored)
├── reports/               # Generated analysis output
│   └── figures/           # Generated charts & figures
│
├── config/                # Configuration (paths, hyper-params, pipeline settings)
├── tests/                 # Unit / data-quality tests
└── README.md
```

---

## Running the analysis

All commands are run **from the project root** (`c:\Code\BSH`).

### 1. Set up the environment (once)

> Requires Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 2. Add the raw data

Place the source CSVs in [`data/raw/`](data/raw/) (they are **not** stored in Git).
Expected files are listed in the table above and in [data/README.md](data/README.md).

### 3. Generate the analysis report

Profiles every dataset and writes a single interactive HTML report —
**Predictive Maintenance — Data Analysis & Modelling** (EDA + predictive modelling +
maintenance knowledge graph) — to `reports/pdm_report.html` (open it in a browser).

```bash
python -m src.eda                                   # all datasets
python -m src.eda --out reports/custom.html         # custom output path
```

