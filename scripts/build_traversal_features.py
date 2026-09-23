"""Precompute per-learner traversal features once per snapshot week.

Loading and ordering ~10M resource-level events costs ~40s per week, so the
evaluation harness reads cached tables rather than recomputing per fold.

    python scripts/build_traversal_features.py --weeks 2 4 8 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import get_data_path  # noqa: E402
from src.traversal import load_events, traversal_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oulab")
    parser.add_argument("--weeks", type=int, nargs="+", default=[2, 4, 8, 16])
    args = parser.parse_args()

    os.environ["PCG_DATASET"] = args.dataset
    events = load_events(None)
    print(f"loaded {len(events):,} resource-level events")

    for week in args.weeks:
        features = traversal_features(week, events=events)
        path = get_data_path(f"processed/traversal_week{week}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(path, index=False)
        print(f"week {week:2d}: {len(features):,} learners -> {path}")


if __name__ == "__main__":
    main()
