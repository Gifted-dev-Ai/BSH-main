"""Static configuration for the EDA pipeline.

Each dataset is described once here so the profiler, chart builder and report all
share a single source of truth (paths, the timestamp column, columns to treat as
categorical, and which signals to highlight in the time-series view).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# config.py lives in <root>/src/eda/, so the project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Files larger than this are profiled with the parallel byte-range engine.
BIG_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

# Number of worker processes for the parallel profiler.
N_WORKERS = max(1, (os.cpu_count() or 2) - 2)

# Rows kept (sampled) for distribution/correlation charts on large datasets.
SAMPLE_ROWS = 60_000

# Cardinality above which a column is treated as an identifier (no bar chart).
ID_CARDINALITY = 2_000


@dataclass
class DatasetConfig:
    """Declarative description of one raw dataset."""

    key: str                       # short id used in anchors/filenames
    title: str                     # human-readable section title
    filename: str                  # file in data/raw
    description: str               # one-paragraph context for the report
    timestamp_col: str | None = None
    categorical_cols: list[str] = field(default_factory=list)
    highlight_cols: list[str] = field(default_factory=list)  # time-series means
    drop_cols: list[str] = field(default_factory=list)       # ignored entirely
    sep: str = ","
    encoding: str = "utf-8"
    # Custom timestamp parsing (defaults: ISO-8601). `timestamp_regex` extracts the
    # parseable substring first (e.g. to drop a trailing weekday); `timestamp_format`
    # is a strptime pattern. `ts_freq` is the resample bucket for the time-series view.
    timestamp_regex: str | None = None
    timestamp_format: str | None = None
    ts_freq: str = "h"
    # Restrict the time-series view to rows whose `ts_filter_col` is in
    # `ts_filter_values` (the rest of the profile still covers all rows).
    ts_filter_col: str | None = None
    ts_filter_values: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename

    @property
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    @property
    def is_big(self) -> bool:
        return self.size_bytes >= BIG_FILE_BYTES


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        key="robot_monitoring",
        title="Stäubli Robot Monitoring (high-frequency telemetry)",
        filename="Stäubli_Robot_monitoring_Data.csv",
        description=(
            "High-frequency condition-monitoring stream of the Stäubli tx2_90l-S2 arm "
            "(~50 ms cadence, wide format). Core signal for robot predictive "
            "maintenance: joint temperatures, positions, velocities, torques, "
            "position errors and TCP motion."
        ),
        timestamp_col="result_timestamp",
        categorical_cols=["Robot_Manufacturer", "Arm_Type", "result_date"],
        highlight_cols=[
            "CPU_Temperature_CH01[°C]",
            "Joint_Temperature_CH01[°C]",
            "Joint_Torques_CH01[nm]",
            "Payload_CH01[kg]",
        ],
    ),
    DatasetConfig(
        key="plasma_nozzle",
        title="Plasma Nozzle Head — Process & Quality",
        filename="Plasma_Nozzle_Head_Data.csv",
        description=(
            "Per-part plasma process parameters and in-line geometric checks of the "
            "nozzle head. Used to detect nozzle wear and process drift "
            "(CheckTheChecker distances + plasma current/voltage/pressure)."
        ),
        timestamp_col="result_timestamp",
        categorical_cols=["LocalTypeSpecificDatFile"],
        highlight_cols=[
            "pltCurrent1[A]",
            "pltVoltage1[V]",
            "pltBkPressure1[mBar]",
            "CheckTheCheckerSpeed[m/min]",
        ],
        drop_cols=["pltProcessTime[s]"],  # stored as "T#3s786ms" text, not numeric
    ),
    DatasetConfig(
        key="plasma_errors",
        title="Plasma Station — MES Error / Event Log",
        filename="Plasma_Station__MES_error_event.csv",
        description=(
            "MES error/event journal for the plasma station — the label source for "
            "failures and downtime. error_state 1 = raised, 0 = cleared."
        ),
        timestamp_col="result_timestamp",
        categorical_cols=["process_number", "type_number", "event_type",
                          "error_state", "error_text"],
    ),
    DatasetConfig(
        key="incident_mgmt",
        title="Incident Management (maintenance / downtime log)",
        filename="Maintenance_report_Incident_Management_data.csv",
        description=(
            "Manually-logged maintenance incidents for the LP-umpacken + Plasma "
            "station (Station 10): incident id (Brid), start time, calculated "
            "downtime (minutes), the affected module/group, free-text problem and "
            "error descriptions and the fault class. A human-curated counterpart to "
            "the automatic MES error log — useful for downtime accounting and for "
            "labelling failure root causes. The free-text columns are mined into "
            "structured fields by src/eda/extraction (-> data/processed)."
        ),
        sep=";",
        encoding="latin-1",
        timestamp_col="Störungsbeginn",
        timestamp_regex=r"\d{2}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}",
        timestamp_format="%d.%m.%y %H:%M:%S",
        ts_freq="D",
        drop_cols=["Brid"],  # unique incident id — not a signal to chart
        categorical_cols=[
            "Equipdescr", "Bam", "Groupdescr", "Moduldescr",
            "T class descr", "Stoertyp",
        ],
        highlight_cols=["Calcdowntime"],
    ),
    DatasetConfig(
        key="repair_activities",
        title="Repair Activities (maintenance descriptions)",
        filename="description_repair_activities_data.csv",
        description=(
            "Per-incident repair log (linked to the incident table by Brid): German "
            "free-text narratives of each maintenance step — the phenomenon observed, "
            "the action taken and the result. Mined into a bilingual maintenance "
            "ontology / knowledge graph (see the Maintenance ontology section)."
        ),
        sep=",",
        encoding="utf-8-sig",
        timestamp_col="Histdate",
        timestamp_regex=r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}",
        timestamp_format="%d.%m.%Y %H:%M",
        ts_freq="D",
        drop_cols=["Brid", "Description"],  # id + free text (analysed in the ontology)
        categorical_cols=["Groupdescr", "Moduldescr", "Errdescr"],
        # the modelling focuses on the Plasma & Roboter stations — chart only those
        ts_filter_col="Groupdescr",
        ts_filter_values=["Plasma", "Roboter"],
    ),
]
