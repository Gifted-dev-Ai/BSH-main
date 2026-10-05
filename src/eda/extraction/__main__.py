"""CLI: mine the incident free-text columns into structured bilingual fields.

    python -m src.eda.extraction                 # -> data/processed/incident_extracted.csv
    python -m src.eda.extraction --coverage      # also print per-category coverage

Outputs (in data/processed/):
* ``incident_extracted.csv``           original key columns + *_de_en structured columns
* ``incident_extraction_unmatched.csv``rows where no category matched (to extend the lexicon)
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..config import DatasetConfig, PROJECT_ROOT, RAW_DIR
from .extractor import LexiconExtractor
from .lexicon import CATEGORIES

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_FILE = "Maintenance_report_Incident_Management_data.csv"
SEP = ";"
ENCODING = "latin-1"

# Columns carried through to the output for traceability.
KEY_COLS = ["Brid", "Störungsbeginn", "Calcdowntime", "Groupdescr",
            "Problemdes", "Moduldescr", "Errdescr"]
SOURCE_COLS = ("Problemdes", "Moduldescr", "Errdescr")


def run(coverage: bool = False, out_dir=PROCESSED_DIR) -> pd.DataFrame:
    path = RAW_DIR / SOURCE_FILE
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")
    df = pd.read_csv(path, sep=SEP, encoding=ENCODING)

    extractor = LexiconExtractor(source_cols=SOURCE_COLS)
    rows = []
    for _, r in df.iterrows():
        texts = {c: r.get(c) for c in SOURCE_COLS}
        res = extractor.extract({k: ("" if pd.isna(v) else v)
                                 for k, v in texts.items()})
        rows.append({f"{cat}_de_en": res.joined(cat) for cat in CATEGORIES}
                    | {"_n_matched": res.n_matched_categories})
    extracted = pd.DataFrame(rows, index=df.index)

    keep = [c for c in KEY_COLS if c in df.columns]
    out = pd.concat([df[keep], extracted], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "incident_extracted.csv"
    # utf-8-sig so Excel shows the German/English text correctly.
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    unmatched = out[out["_n_matched"] == 0]
    unmatched_path = out_dir / "incident_extraction_unmatched.csv"
    unmatched[keep].to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    # --- console report -----------------------------------------------------
    n = len(out)
    print(f"Extracted {n} incidents from {SOURCE_FILE}")
    print(f"  -> {out_path}")
    print(f"  -> {unmatched_path} ({len(unmatched)} rows with no match)\n")
    print("Per-category coverage:")
    for cat in CATEGORIES:
        col = f"{cat}_de_en"
        filled = (out[col].str.len() > 0).sum()
        print(f"  {cat:11s} {filled:3d}/{n}  ({100*filled/n:4.1f}%)")
    any_match = (out["_n_matched"] > 0).sum()
    print(f"\n  any category {any_match:3d}/{n}  ({100*any_match/n:4.1f}%)")

    if coverage:
        print("\nMost frequent labels per category:")
        for cat in CATEGORIES:
            s = out[f"{cat}_de_en"].str.split("; ").explode()
            s = s[s.str.len() > 0]
            print(f"\n  [{cat}]")
            for label, cnt in s.value_counts().head(10).items():
                print(f"    {cnt:3d}  {label}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract structured fields from the "
                                             "incident free-text columns.")
    ap.add_argument("--coverage", action="store_true",
                    help="print most frequent labels per category")
    args = ap.parse_args(argv)
    try:
        run(coverage=args.coverage)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
