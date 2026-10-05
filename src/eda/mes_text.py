"""Critical-information extraction from the MES ``error_text`` field.

The Plasma-station MES emits free-text error messages on ``event_type == 1`` events.
They mix three things: Beckhoff/EtherCAT terminal addresses (``=101+E-K018A15
(EP7041-1002)``), TwinCAT/diagnostic codes (``NC DiagNo.: 18000``, ``ErrorCode:
0x0100``) and human-readable faults (``Actuator voltage is too low``, ``minimum
travel path not reached``, ``ParCmd.GripForce out of range``).

This module categorises each message into a **fault type**, and pulls out the
**terminal** (``EP7041``…) and **error code** when present. The fault type then drives
per-category hourly error counts that feed the early-prediction / AutoML / RUL models
(previously only a single total error count was used).

Extending: add a ``(category, [regex])`` entry to :data:`LEXICON` (first match wins,
by priority order).
"""

from __future__ import annotations

import re

import pandas as pd

# Fault categories used as model features, in priority order (first match wins).
# Patterns are matched case-insensitively against the raw text.
LEXICON: list[tuple[str, tuple[str, ...]]] = [
    ("gripper", (r"grip",)),
    ("motion_position", (r"travel path", r"actpos", r"targetpos", r"workposition",
                         r"movement order", r"minimum range", r"\bpos\b",
                         r"drehposition")),
    ("power", (r"voltage", r"supplystate", r"armpoweron", r"ep9214",
               r"\bpwron\b")),
    ("servo_drive", (r"ep7211", r"ep7041", r"nc diagno", r"decoupled",
                     r"\bm-m\d", r"-xm\d")),
    ("robot_drive", (r"90lx",)),
    ("sensor", (r"m-b\d", r"\+m-b", r"s-a\d", r"-b\d{3}")),
    ("status", (r"startorder", r"stable", r"\bwait\b", r"initrobot", r"\babort\b",
                r"program number", r"devicemode", r"set tool", r"temperature",
                r"handshake", r"deviceMode")),
]

# The category columns produced (in a stable order) — "untyped" (null text) is not a
# feature, only counted in the total.
MES_FEATURE_CATEGORIES: list[str] = [c for c, _ in LEXICON] + ["other"]

_COMPILED = [(cat, [re.compile(p, re.I) for p in pats]) for cat, pats in LEXICON]
_TERMINAL_RE = re.compile(r"(EP\d{4})", re.I)
_CODE_RE = re.compile(r"(NC DiagNo\.: \d+|0x[0-9A-Fa-f]+)")


def categorize(text) -> str:
    """Map one error_text to a fault category ('untyped' for missing text)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "untyped"
    s = str(text)
    for cat, regexes in _COMPILED:
        if any(rx.search(s) for rx in regexes):
            return cat
    return "other"


def terminal(text) -> str:
    m = _TERMINAL_RE.search(str(text)) if pd.notna(text) else None
    return m.group(1).upper() if m else ""


def error_code(text) -> str:
    m = _CODE_RE.search(str(text)) if pd.notna(text) else None
    return m.group(1) if m else ""


def extract_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add mes_category / mes_terminal / mes_code columns to a raw MES frame."""
    out = df.copy()
    out["mes_category"] = out["error_text"].map(categorize)
    out["mes_terminal"] = out["error_text"].map(terminal)
    out["mes_code"] = out["error_text"].map(error_code)
    return out


def run(out_dir=None) -> pd.DataFrame:
    """Write the categorised MES error messages to data/processed (typed rows only)."""
    from .config import PROJECT_ROOT, RAW_DIR
    out_dir = out_dir or (PROJECT_ROOT / "data" / "processed")
    df = pd.read_csv(RAW_DIR / "Plasma_Station__MES_error_event.csv")
    typed = df[(df["event_type"] == 1) & df["error_text"].notna()].copy()
    ext = extract_frame(typed)[
        ["result_timestamp", "type_number", "error_text",
         "mes_category", "mes_terminal", "mes_code"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mes_error_extracted.csv"
    ext.to_csv(out_path, index=False, encoding="utf-8-sig")

    counts = ext["mes_category"].value_counts()
    print(f"MES error_text categorised: {len(ext):,} typed messages -> {out_path}")
    for cat, c in counts.items():
        print(f"  {cat:16s} {c:6,d}")
    return ext


if __name__ == "__main__":
    run()
