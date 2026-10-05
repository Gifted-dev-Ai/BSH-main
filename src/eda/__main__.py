"""CLI entry point: profile every configured dataset and write the HTML report.

Usage
-----
    python -m src.eda                 # build the report -> reports/pdm_report.html
    python -m src.eda --only plasma_errors incident_mgmt
    python -m src.eda --out reports/custom.html
"""

from __future__ import annotations

import argparse
import sys
import time

from .config import DATASETS, REPORTS_DIR
from .profiler import profile_dataset
from .report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the BSH EDA HTML report.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these dataset keys")
    parser.add_argument("--out", default=str(REPORTS_DIR / "pdm_report.html"),
                        help="output HTML path")
    parser.add_argument("--export-models", action="store_true",
                        help="also train+save every analysis model to models/")
    args = parser.parse_args(argv)

    datasets = DATASETS
    if args.only:
        wanted = set(args.only)
        datasets = [d for d in DATASETS if d.key in wanted]
        if not datasets:
            print(f"No datasets match {args.only}. "
                  f"Known keys: {[d.key for d in DATASETS]}", file=sys.stderr)
            return 2

    profiles = []
    standalone_sections = []   # top-level sections not tied to a dataset
    for cfg in datasets:
        if not cfg.path.exists():
            print(f"  ! skipping {cfg.key}: file not found ({cfg.path})")
            continue
        size_mb = cfg.size_bytes / 1024 / 1024
        engine = "parallel" if cfg.is_big else "single-process"
        print(f"-> profiling {cfg.key:18s} ({size_mb:8.1f} MB, {engine}) ...",
              end="", flush=True)
        t0 = time.perf_counter()
        prof = profile_dataset(cfg)
        print(f" {prof.n_rows:>12,} rows in {time.perf_counter() - t0:5.1f}s")

        # The incident log gets the free-text extraction + Plasma/Roboter deep-dive
        # folded into the EDA workflow.
        if cfg.key == "incident_mgmt":
            try:
                from .extraction.__main__ import run as run_extraction
                from .incident_analysis import build_deepdive_sections
                from .mes_text import run as run_mes_extraction
                run_extraction()      # writes data/processed/incident_extracted.csv
                run_mes_extraction()  # writes data/processed/mes_error_extracted.csv
                prof.extra_sections = build_deepdive_sections()
            except Exception as e:  # never let the deep-dive break the whole report
                print(f"  ! incident deep-dive skipped: {e}")
            try:
                from .early_factors_report import build_sections as build_early
                print("   early predictive factors ...", end="", flush=True)
                t1 = time.perf_counter()
                # these go into their own top-level section, not under the dataset
                standalone_sections.append({
                    "id": "modelling",
                    "title": "Predictive maintenance modelling (Plasma)",
                    "subs": build_early()})
                print(f" done in {time.perf_counter() - t1:4.1f}s")
            except Exception as e:  # never let early-factors break the whole report
                print(f"\n  ! early factors skipped: {e}")

        # The MES error log gets the error-text extraction logic + category distribution.
        if cfg.key == "plasma_errors":
            try:
                from .mes_report import build_section as build_mes
                prof.extra_sections = [build_mes()]
            except Exception as e:  # never let it break the whole report
                print(f"  ! MES extraction section skipped: {e}")

        profiles.append(prof)

    # Maintenance ontology / knowledge graph (its own top-level section)
    try:
        from .ontology_report import build_standalone_section as build_ontology
        print("-> maintenance ontology ...", end="", flush=True)
        t1 = time.perf_counter()
        standalone_sections.append(build_ontology())
        print(f" done in {time.perf_counter() - t1:4.1f}s")
    except Exception as e:  # never let the ontology break the whole report
        print(f"\n  ! ontology skipped: {e}")

    if not profiles:
        print("No datasets profiled; nothing to write.", file=sys.stderr)
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = build_report(profiles, standalone_sections)
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nReport written to {out_path} ({len(html) / 1024:.0f} KB)")

    if args.export_models:
        try:
            from .model_export import export_all
            print("-> exporting trained models ...", end="", flush=True)
            t1 = time.perf_counter()
            m = export_all()
            print(f" {m['n_saved']} saved ({m['n_errors']} skipped) in "
                  f"{time.perf_counter() - t1:4.1f}s")
        except Exception as e:
            print(f"\n  ! model export skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
