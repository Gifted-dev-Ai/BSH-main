"""Exploratory Data Analysis (EDA) package for the BSH Predictive Maintenance project.

Profiles the raw datasets in ``data/raw`` and renders a single, self-contained
interactive HTML report into ``reports/``.

Entry point:  ``python -m src.eda``
"""

from .config import DATASETS, PROJECT_ROOT, RAW_DIR, REPORTS_DIR

__all__ = ["DATASETS", "PROJECT_ROOT", "RAW_DIR", "REPORTS_DIR"]
