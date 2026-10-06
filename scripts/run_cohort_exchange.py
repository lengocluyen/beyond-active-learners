"""Run the cohort-exchange benchmark.

Smoke test:
    python scripts/run_cohort_exchange.py --dataset oulab --weeks 2 4 \
        --folds 3 --repeats 1 --bootstrap 100 --no-hazard --verbose

Full protocol:
    python scripts/run_cohort_exchange.py --dataset both --folds 5 --repeats 5 --jobs 5 \
        --bootstrap 2000 --reference-baselines \
        --output results/cohort_exchange_decomposed --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fold-level parallelism is the only parallel layer used by this runner.  Give
# every fold one BLAS/OpenMP thread so five concurrent folds cannot each create
# an 80- or 128-thread native pool.  PCG_BLAS_THREADS is an explicit escape
# hatch for controlled benchmarking; the safe default is one.
_INNER_THREADS = os.environ.get("PCG_BLAS_THREADS", "1")
for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = _INNER_THREADS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort_exchange import (  # noqa: E402
    BenchmarkConfig,
    DEFAULT_MODELS,
    DEFAULT_WEEKS,
    run_cohort_exchange,
)
from src.full_evaluation import REFERENCE_BASELINES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("oulab", "kdd", "both"), default="both")
    parser.add_argument("--weeks", type=int, nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--reference-baselines",
        action="store_true",
        help=(
            "Append the DOI-traced LR, RF-gini, RF-entropy, SVM, and DFFNN "
            "literature-family baselines (plus k-NN on OULAD; exact k-NN is "
            "excluded from the 200k-row KDD default because prediction is "
            "prohibitively expensive)."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="CV folds fitted concurrently; 5 is suitable for a five-fold server run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--hazard-days", type=int, default=7,
        help="Withdrawal-hazard interval; 7 gives the person-week formulation.",
    )
    parser.add_argument("--cluster", choices=("presentation", "module"), default="presentation")
    parser.add_argument(
        "--split-unit", choices=("presentation", "learner"), default="presentation",
        help="Cross-validation grouping unit; 'learner' gives the "
             "learner-disjoint sensitivity arm.",
    )
    parser.add_argument("--output", type=Path, default=Path("results") / "cohort_exchange")
    parser.add_argument(
        "--augment-from",
        type=Path,
        help=(
            "Reuse a completed OULAD run's within-protocol predictions and fit "
            "only the added activity-trained/cutoff-valid evaluation arm."
        ),
    )
    parser.add_argument("--no-hazard", action="store_true")
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help=(
            "Stop after checkpointed predictions and cohort files are written; "
            "a later identical command without this flag resumes analysis."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed week/model checkpoints and recompute them.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs == 0:
        raise SystemExit("--jobs must be non-zero (use -1 for all CPUs)")
    datasets = ("oulab", "kdd") if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        weeks = tuple(args.weeks or DEFAULT_WEEKS[dataset])
        requested = list(args.models or DEFAULT_MODELS[dataset])
        if args.reference_baselines:
            requested.extend(REFERENCE_BASELINES[dataset])
        models = tuple(dict.fromkeys(requested))
        config = BenchmarkConfig(
            dataset=dataset,
            weeks=weeks,
            models=models,
            folds=args.folds,
            repeats=args.repeats,
            seed=args.seed,
            bootstrap_iterations=args.bootstrap,
            hazard_days=args.hazard_days,
            cluster=args.cluster,
            split_unit=args.split_unit,
            jobs=args.jobs,
        )
        print(
            f"[start] {dataset}: weeks={list(weeks)} models={list(models)} "
            f"folds={args.folds} repeats={args.repeats}",
            flush=True,
        )
        outputs = run_cohort_exchange(
            config,
            output_root=args.output,
            run_hazard=not args.no_hazard,
            verbose=args.verbose,
            resume=not args.no_resume,
            augment_from=args.augment_from,
            fit_only=args.fit_only,
        )
        print(f"[done] {dataset}: {len(outputs)} artifacts -> {args.output / dataset}")


if __name__ == "__main__":
    main()
