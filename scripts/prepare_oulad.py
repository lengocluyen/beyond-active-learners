"""Initialize missing OULAD inputs and validate temporal features without fitting.

Place the source CSVs in data/raw/Oulab/, then run:
    python3 -u scripts/prepare_oulad.py
Existing labels, static features, and traversal tables are retained. Use the
experiment launcher's --skip-rebuild option when resuming an existing run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEEKS = (2, 4, 6, 8, 10, 12, 14, 16)
RAW_NAMES = ("studentAssessment", "assessments", "studentVle", "vle", "studentInfo", "studentRegistration")


def initialization_commands(processed: Path, python: str = sys.executable) -> list[list[str]]:
    commands = []
    for filename, module in (
        ("competencies.csv", "src.build_competency_graph"),
        ("labels.csv", "src.make_labels"),
        ("static_features.csv", "src.features_static"),
    ):
        if not (processed / filename).is_file():
            commands.append([python, "-u", "-m", module])
    missing_weeks = [week for week in WEEKS if not (processed / f"traversal_week{week}.csv").is_file()]
    if missing_weeks:
        commands.append([python, "-u", "scripts/build_traversal_features.py", "--dataset", "oulab",
                         "--weeks", *map(str, missing_weeks)])
    return commands


def prepare(output_root: Path | None = None) -> None:
    os.environ["PCG_DATASET"] = "oulab"
    from src.paths import get_data_path

    required = [get_data_path(f"raw/{name}.csv") for name in RAW_NAMES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing OULAD source files; no builders were run:\n  " + "\n  ".join(missing))
    commands = initialization_commands(get_data_path("processed"))
    validation = [sys.executable, "-u", "scripts/run_experiments.py", "--prepare-only"]
    if output_root is not None:
        validation.extend(["--output-root", str(output_root)])
    commands.append(validation)
    for command in commands:
        print("[prepare] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy(), check=True)
    print("[done] OULAD inputs prepared and checked. No models were fitted.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, help="Optional location for preparation logs and future runs")
    args = parser.parse_args()
    try:
        prepare(args.output_root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[stopped] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
