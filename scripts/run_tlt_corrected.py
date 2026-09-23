"""One-command TLT preprocessing, guarded fitting, and numerical reports.

The launcher exits on a failed subprocess, leaving the interactive tmux shell
open. It never enables shell-wide strict-mode options.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEEKS = (2, 4, 6, 8, 10, 12, 14, 16)
LEARNER_MODELS = ("stack_7_full", "temporal_hgb", "base_lr_profile", "base_rf_gini")


def commands(output_root: Path, jobs: int, python: str = sys.executable) -> list[tuple[str, list[str]]]:
    common = [python, "-u", "scripts/run_cohort_exchange.py", "--dataset", "oulab",
              "--weeks", *map(str, WEEKS), "--folds", "5", "--repeats", "5",
              "--seed", "42", "--jobs", str(jobs), "--bootstrap", "2000",
              "--cluster", "presentation", "--verbose"]
    main = output_root / "main"
    learner = output_root / "learner_disjoint"
    return [
        ("main", [*common, "--reference-baselines", "--split-unit", "presentation",
                   "--output", str(main)]),
        ("learner_disjoint", [*common, "--models", *LEARNER_MODELS,
                               "--split-unit", "learner", "--no-hazard", "--output", str(learner)]),
        ("paper_numbers", [python, "-u", "scripts/report_paper_numbers.py",
                           "--results", str(main / "oulab")]),
        ("aggregate_ci", [python, "-u", "scripts/aggregate_decomposition_ci.py",
                           "--results", str(main / "oulab"), "--n-boot", "2000", "--seed", "42"]),
    ]


def run_logged(command: list[str], log_path: Path) -> None:
    print("[launch] " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PCG_DATASET"] = "oulab"
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_pipeline(args: argparse.Namespace) -> None:
    os.environ["PCG_DATASET"] = "oulab"
    if args.jobs < 1:
        raise ValueError("--jobs must be a positive number of concurrent folds")
    root = args.output_root.expanduser().resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not args.skip_rebuild:
        run_logged([sys.executable, "-u", "scripts/rebuild_oulad_temporal.py"],
                   logs / f"prepare_{timestamp}.log")
    from src.run_provenance import validate_temporal_manifest
    validate_temporal_manifest(root=PROJECT_ROOT, dataset="oulab")
    print("[preflight] Submission-time manifest and input hashes verified.", flush=True)
    if args.prepare_only:
        print("[done] Prepared and verified; no models fitted.", flush=True)
        return
    for name, command in commands(root, args.jobs):
        run_logged(command, logs / f"{name}_{timestamp}.log")
    print(f"[done] Main, learner-disjoint, and numerical reports: {root}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=5, help="Concurrent CV folds (default: 5)")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/tlt_submission_time_v1")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Use existing evidence only if its corrected manifest and hashes validate")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Rebuild and independently validate features, without fitting models")
    return parser.parse_args(argv)


def main() -> int:
    try:
        run_pipeline(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[stopped] {error}\nLater stages were not launched. Your tmux shell remains open.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[stopped] Interrupted; rerun with --skip-rebuild to resume verified checkpoints.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
