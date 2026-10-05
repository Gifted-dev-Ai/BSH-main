# Data

This folder holds all datasets for the Bosch / BSH Predictive Maintenance project.
It follows the common data-science layering convention — data flows **left to right**
and is **never edited in place**:

| Sub-folder        | Purpose                                                                                  | Tracked in Git? |
| ----------------- | ---------------------------------------------------------------------------------------- | --------------- |
| `raw/`            | Immutable, original extracts exactly as pulled from the line / MES / Kafka. Read-only.   | No (see below)  |
| `interim/`        | Intermediate, partially-cleaned data produced by pipeline steps. Safe to delete/rebuild. | No              |
| `processed/`      | Final, analysis-ready feature tables / model inputs.                                     | No              |
| `external/`       | Third-party or reference data (asset master data, maintenance logs, specs, lookups).     | No              |

> **Note on version control:** the whole `data/` tree is ignored via `.gitignore`
> (`/data/*`). The CSVs are large (the robot monitoring file alone is ~15M rows),
> so they are kept out of the repository. Only this `README.md` and `.gitkeep`
> placeholders are committed, so the folder structure travels with the repo while
> the data is shared out-of-band (shared drive / data lake / Kafka topic).

---

## Files in `raw/`

All four datasets cover the **Plasma coating station** and its handling **Stäubli robot**
on the same production line (MES line `0206`, station `0370`). Together they let you
correlate robot health, process parameters, product quality and machine faults.

### 1. `Stäubli_Robot_monitoring_Data.csv`  — high-frequency robot telemetry
- **Shape:** ~15.3M rows, **wide** format (one column per channel).
- **Frequency:** ~50 ms cadence (bursts of identical rows at each sample).
- **What it is:** Raw condition-monitoring stream of the Stäubli `tx2_90l-S2` arm —
  the core signal for robot predictive maintenance.
- **Key columns:**
  - `result_timestamp`, `result_date` — sample time (UTC).
  - `Robot_Manufacturer`, `Arm_Type` — asset identity.
  - `Payload_CH01[kg]`, `Operation_Counter_CH01[s]` — load and cumulative run-time.
  - `CPU_Temperature_CH01[°C]`, `Joint_Temperature_CH01..06[°C]` — thermal signals.
  - `Joint_Positions_CH01..06[°]`, `Joint_Velocity_CH01..06[°/s]` — kinematics.
  - `Joint_Torques_CH01..06[nm]` — load on each axis (wear / friction indicator).
  - `Position_Error_CH01..06` — tracking error per joint (early fault indicator).
  - `TCP_Position_CH01..06[mm;°]`, `TCP_Velocity_CH01[mm/s]` — tool-centre-point motion.

### 2. `Plasma_Nozzle_Head_Data.csv`  — plasma process & quality measurements
- **Shape:** ~1.13M rows, one row per produced part.
- **What it is:** Per-part process parameters and in-line geometric checks of the
  plasma nozzle head — the **product-quality / process-drift** signal used to detect
  nozzle wear and process degradation.
- **Key columns:**
  - `uniquepart_id`, `result_timestamp` — part traceability.
  - `CheckTheCheckerDistance1..12[mm]`, `CheckTheCheckerSpeed[m/min]` — in-line
    dimensional checks (quality / drift indicators).
  - `pltCurrent1[A]`, `pltVoltage1[V]`, `pltFrequency1[kHz]`,
    `pltBkPressure1[mBar]`, `pltRodSpeed1[1/min]`, `pltPCT1[%]`,
    `pltProcessTime[s]` — plasma process parameters.
  - `LocalTypeSpecificDatFile` — product/recipe type identifier.

### 3. `Plasma_Station__MES_error_event.csv`  — MES fault & event log
- **Shape:** ~213k rows, one row per event.
- **What it is:** The plasma station's MES error/event journal — the **label source**
  for failures and downtime. Pair it with the sensor data above to build
  supervised PdM targets (e.g. "fault within next N hours").
- **Key columns:**
  - `result_timestamp` — event time.
  - `process_number`, `type_number` — process / product type the event belongs to.
  - `event_type` — event category (`1`, `2`, `3`).
  - `error_state` — `1` = error raised, `0` = cleared (pair rows to get fault durations).
  - `error_text` — human-readable description (often `null`).

### 4. `Maintenance_report_Incident_Management_data.csv`  — maintenance / downtime incident log
- **Shape:** ~178 rows, one row per logged incident. **`;`-separated, `latin-1`
  encoded, German column names.**
- **What it is:** Manually-curated maintenance incidents for the
  *LP umpacken + Plasma (Station 10)* — the human counterpart to the automatic MES
  error log. Best source for **downtime accounting** and **root-cause labelling**.
- **Key columns:**
  - `Brid` — unique incident id.
  - `Störungsbeginn` — incident start (format `DD.MM.YY HH:MM:SS` + German weekday).
  - `Calcdowntime` — calculated downtime in **minutes** (numeric).
  - `Equipdescr` — equipment/station, `Bam` — flag.
  - `Groupdescr`, `Moduldescr` — affected group / module (Moduldescr also encodes
    `Reaktion: …` corrective actions).
  - `Problemdes`, `Errdescr` — free-text problem / error description.
  - `T class descr` — fault class (e.g. `Technik`), `Stoertyp` — `Störung` vs
    `Kleinstörung`, `Anz. Bilder` — number of attached images.
- **Derived:** the free-text columns are mined into structured bilingual fields
  (`component` / `phenomenon` / `action` / `reason`) by
  [`src/eda/extraction`](../src/eda/extraction/README.md) →
  `data/processed/incident_extracted.csv`.

### 5. `description_repair_activities_data.csv`  — repair-activity log
- **Shape:** ~387 rows (one per repair step), ~174 unique `Brid`. **`,`-separated,
  UTF-8 (BOM), German free text** (multi-line, quoted).
- **What it is:** A per-incident **repair thread** — chronological German narratives
  of each maintenance step (phenomenon → diagnosis/action → result), keyed by `Brid`
  (links to the incident table).
- **Key columns:**
  - `Brid` — incident id (foreign key to the incident log).
  - `Histdate` — step timestamp (`DD.MM.YYYY HH:MM` + German weekday).
  - `Description` — the free-text repair narrative.
  - `Groupdescr`, `Moduldescr`, `Errdescr` — group / module / error (often `-`).
- **Derived:** mined into a bilingual **maintenance ontology / knowledge graph**
  (`src/eda/ontology`) → `data/processed/repair_descriptions_extracted.csv`,
  `maintenance_kg_triples.csv`, `maintenance_ontology.json`.

---

## Conventions
- **Never modify files in `raw/`.** Treat them as read-only ground truth.
- All cleaning/feature code reads from `raw/` and writes to `interim/` or `processed/`.
- Timestamps are **UTC** (ISO-8601, `Z` suffix). Convert to local time only for display.
- Re-generatable artifacts (`interim/`, `processed/`, `models/`) should be reproducible
  by re-running the pipeline in `src/`.
