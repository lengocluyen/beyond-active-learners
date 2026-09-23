from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.aggregate_decomposition_ci import AA, AV, VA, VV, COMPONENTS, METRICS, aggregate_draws


def predictions() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for week, order in ((2, ("B", "A", "C")), (4, ("C", "B", "A"))):
        for protocol in (AA, AV, VA, VV):
            for cluster in order:
                for y in (0, 1):
                    rows.append(dict(week=week, model="example", protocol=protocol,
                                     cluster_id=cluster, y=y, p_success=float(rng.uniform(.1, .9))))
    return pd.DataFrame(rows)


class AggregateDecompositionTests(unittest.TestCase):
    def run_aggregate(self, frame: pd.DataFrame, n_boot: int = 29, points: bool = False):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame.to_csv(folder / "predictions.csv.gz", index=False)
            with redirect_stdout(io.StringIO()):
                return aggregate_draws(folder, n_boot, 7, return_points=points)

    def test_named_cluster_draws_invariant_to_heterogeneous_cell_order(self):
        frame = predictions()
        initial, n_initial = self.run_aggregate(frame)
        reordered, n_reordered = self.run_aggregate(frame.sample(frac=1, random_state=90))
        self.assertEqual(n_initial, 2)
        self.assertEqual(n_initial, n_reordered)
        for metric in METRICS:
            for component in COMPONENTS:
                np.testing.assert_allclose(initial[metric][component], reordered[metric][component],
                                           rtol=0, atol=1e-15)

    def test_observed_mean_is_not_mean_of_bootstrap_draws(self):
        frame = predictions()
        stacks, n_cells, observed = self.run_aggregate(frame, n_boot=1, points=True)
        expected_cells = []
        for _, cell in frame.groupby(["week", "model"]):
            errors = {}
            for protocol in (AA, VV):
                rows = cell[cell.protocol.eq(protocol)]
                errors[protocol] = float(np.mean((rows.y-rows.p_success)**2))
            expected_cells.append(errors[AA]-errors[VV])
        self.assertEqual(n_cells, 2)
        self.assertAlmostEqual(observed["brier"]["joint"], float(np.mean(expected_cells)), places=15)
        self.assertGreater(abs(observed["brier"]["joint"]-float(np.mean(stacks["brier"]["joint"]))), 1e-5)

    def test_round_trip_reader_preserves_adjacent_probabilities(self):
        high = 0.30666666666666664
        low = float(np.nextafter(high, 0))
        self.assertLess(low, high)
        frame = pd.DataFrame([
            dict(week=2, model="ties", protocol=protocol, cluster_id=cluster,
                 y=y, p_success=(low if y == 0 else high) if protocol == AA else .5)
            for protocol in (AA, AV, VA, VV) for cluster in ("A", "B") for y in (0, 1)
        ])
        draws, _, observed = self.run_aggregate(frame, points=True)
        # AA perfectly ranks the two adjacent scores; VV ties all learners.
        self.assertEqual(observed["auc"]["joint"], .5)
        # Subtraction merges these adjacent success probabilities into one
        # risk-score tie. AP must use that tie in observed and bootstrap values.
        self.assertEqual(observed["pr_auc_risk"]["joint"], 0.0)
        np.testing.assert_allclose(draws["pr_auc_risk"]["joint"], 0.0, atol=1e-15)

    def test_missing_protocol_is_rejected_instead_of_silently_skipping_cell(self):
        frame = predictions()
        incomplete = frame[~(frame.week.eq(4)&frame.protocol.eq(VA))]
        with self.assertRaisesRegex(ValueError, "Incomplete 2x2 cell week=4"):
            self.run_aggregate(incomplete)

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No prediction cells"):
            self.run_aggregate(predictions().iloc[:0])


if __name__ == "__main__":
    unittest.main()
