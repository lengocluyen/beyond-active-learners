from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return project root (parent of src/)."""

    return Path(__file__).resolve().parents[1]


def get_data_path(relative: str | Path) -> Path:
    """
    Resolve a data path relative to project root.

    Example: get_data_path("raw/studentInfo.csv") -> <root>/data/raw/studentInfo.csv

    If PCG_DATASET is set, processed/results paths are namespaced under the dataset:
      processed/x.csv -> processed/<dataset>/x.csv
      results/x.csv   -> results/<dataset>/x.csv

    Raw-data compatibility:
    - Some checkouts store OULAD raw files under data/raw/Oulab/.
      When PCG_DATASET=oulab (or when the flat path is missing), we fall back to that folder.
    """

    rel = Path(relative)
    dataset = os.getenv("PCG_DATASET")

    # Namespace derived artifacts by dataset.
    if dataset and rel.parts and rel.parts[0] in {"processed", "results"}:
        rel = Path(rel.parts[0]) / dataset / Path(*rel.parts[1:])

    base = project_root() / "data"
    path = base / rel

    # Backwards/alternate layout for OULAD raw data.
    if rel.parts and rel.parts[0] == "raw" and (dataset == "oulab" or not path.exists()):
        alt = base / "raw" / "Oulab" / Path(*rel.parts[1:])
        if alt.exists():
            return alt

    return path
