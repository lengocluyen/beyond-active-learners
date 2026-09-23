"""Input fingerprints and fail-closed checks for reproducible benchmark runs.

This module uses only the standard library so preprocessing and server preflight
commands can validate evidence without importing or fitting an estimator.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Iterable

from .paths import project_root


TEMPORAL_POLICY = "submission_time_v1"
MANIFEST_RELATIVE_PATH = "data/processed/oulab/temporal_preprocessing.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"manifest path is outside the project: {relative!r}")
    return path


def validate_temporal_manifest(
    root: Path | None = None, dataset: str = "oulab"
) -> dict[str, object] | None:
    """Require rebuilt OULAD evidence and verify every recorded artifact hash."""
    if dataset != "oulab":
        return None
    root = (project_root() if root is None else Path(root)).resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    remedy = "Run python3 scripts/rebuild_oulad_temporal.py before fitting."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("policy") != TEMPORAL_POLICY:
            raise ValueError("unsupported temporal preprocessing schema or policy")
        for section in ("source_files", "output_files"):
            files = manifest.get(section)
            if not isinstance(files, dict) or not files:
                raise ValueError(f"manifest requires nonempty {section}")
            for relative, expected in files.items():
                path = _within_root(root, relative)
                if not path.is_file() or sha256_file(path) != expected:
                    raise ValueError(f"missing or changed {section} artifact: {relative}")
        required_outputs = {
            "data/processed/oulab/assess_weekly_evidence.csv",
            "data/processed/oulab/vle_weekly_evidence.csv",
        }
        if not required_outputs.issubset(manifest["output_files"]):
            raise ValueError("manifest does not cover both weekly evidence tables")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"OULAD temporal preprocessing is unverified: {error}. {remedy}") from error
    return manifest


def collect_run_provenance(
    dataset: str, weeks: Iterable[int], root: Path | None = None
) -> dict[str, object]:
    """Fingerprint inputs actually eligible for use, including absent blocks.

    A run captures this once and stores it in both the checkpoint manifest and
    final protocol. Cached estimator files and unrelated result tables are not
    inputs. Missing optional blocks are recorded explicitly, so adding one
    invalidates earlier checkpoints too.
    """
    root = (project_root() if root is None else Path(root)).resolve()
    manifest = validate_temporal_manifest(root, dataset)
    processed = root / "data" / "processed" / dataset
    paths = {
        processed / name
        for name in (
            "labels.csv", "static_features.csv", "vle_weekly_evidence.csv",
            "assess_weekly_evidence.csv",
        )
    }
    for week in weeks:
        for name in (f"behavioral_week{week}.csv", f"traversal_week{week}.csv"):
            path = processed / name
            paths.add(path)
            if (manifest is not None and path.exists()
                    and path.relative_to(root).as_posix() not in manifest["output_files"]):
                raise ValueError(
                    f"Temporal preprocessing manifest does not cover {path.name}. "
                    "Run python3 scripts/rebuild_oulad_temporal.py before fitting "
                    "and use only the validated landmarks."
                )
    if manifest is not None:
        paths.add(root / MANIFEST_RELATIVE_PATH)
        paths.update(root / relative for relative in manifest["source_files"])
        paths.update(root / relative for relative in manifest["output_files"])
        # Match get_data_path's alternate-layout preference for OULAD.
        for name in ("studentInfo.csv", "studentRegistration.csv"):
            alternate = root / "data" / "raw" / "Oulab" / name
            paths.add(alternate if alternate.exists() else root / "data" / "raw" / name)
    verified = {} if manifest is None else {
        **manifest["source_files"], **manifest["output_files"]
    }
    inputs = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        inputs[relative] = (verified[relative] if relative in verified
                            else sha256_file(path) if path.is_file() else None)
    code = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "schema": 1,
        "temporal_policy": manifest["policy"] if manifest is not None else None,
        "input_files": inputs,
        "source_code": code,
        "software_versions": versions,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
