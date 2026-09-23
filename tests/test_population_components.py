import unittest

import numpy as np
import pandas as pd

from scripts.analyze_population_components import AA, AV, VA, VV, cell_components


class PopulationComponentTests(unittest.TestCase):
    def test_opposing_population_restrictions_and_fixed_capacity(self):
        rows = []
        for protocol in (AA, AV, VA, VV):
            ids = (1, 2, 3) if protocol in (AA, VA) else (1, 2, 4)
            scores = {1: .8, 2: .2, 3: .05, 4: .25} if protocol in (AA, AV) else {1: .7, 2: .3, 3: .1, 4: .15}
            for learner in ids:
                y = int(learner == 1)
                rows.append(dict(week=2, model="example", protocol=protocol, cluster_id="A::P",
                    id_student=learner, code_module="A", code_presentation="P", y=y, risk=1-y,
                    p_success=scores[learner], risk_score=1-scores[learner], cutoff_valid=learner != 3))
        metric, budgets, gain = cell_components(pd.DataFrame(rows), .5)
        for row in metric:
            self.assertAlmostEqual(row["nonelig"]+row["silence"], row["evaluation"], places=15)
        budget = {row["arm"]: row for row in budgets}
        self.assertEqual(budget[AA]["wasted_budget_rate"], .5)
        self.assertEqual(budget["AI"]["wasted_budget_rate"], 0)
        self.assertEqual(budget[AV]["coverage_adjusted_recall"], 1)
        self.assertEqual(gain["total"], gain["eligibility"]+gain["silence"])
        self.assertEqual(gain["silence"], .5)


if __name__ == "__main__":
    unittest.main()
