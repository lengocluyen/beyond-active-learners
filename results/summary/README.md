# Aggregate inference and exploratory population contrasts

These files contain aggregate statistics only; no learner records are included.

| File | Quantity |
|---|---|
| `aggregate_main_final.json` | Observed mean components and canonical paired-cluster intervals across 88 main cells |
| `aggregate_main_matched4_final.json` | Same inference restricted to the 32 main cells matched to the learner-disjoint analysis |
| `aggregate_learner_disjoint_final.json` | Mean components and intervals across the 32 learner-disjoint cells |
| `aggregate_learner_minus_matched_main_final.json` | Paired differences between the learner-disjoint and matched presentation-grouped aggregate components |
| `secondary_metric_components.csv` | Eligibility-first noneligibility and silence terms, averaged across the two fitted models |
| `active_eligible_budget.csv` | Fixed-capacity decision summaries for A→A, A→(A∩V), A→V, and V→V |
| `paired_recall_gains.csv` | Within-cell eligibility and silent-restoration gains in coverage-adjusted recall |
| `class_composition.csv` | Aggregate adverse-outcome counts separated into eventual withdrawal and failure at the reported landmarks |

Aggregate intervals use 2,000 bootstrap draws with seed 42. A single canonical
ordering of presentation names assigns the same cluster multiplicities to every
configuration and landmark in a draw. Average precision uses ties in the actual
adverse-outcome score `1 - p_success`. `point_mean` is the observed arithmetic
mean; `bootstrap_mean`, where present, is a resampling diagnostic rather than
the point estimate reported in the paper.

`absolute_evaluation_minus_absolute_training` compares the magnitudes of the
two aggregate mean components within each draw. It is not the mean of
cellwise absolute differences. The matched-four file retains the arm label
`main`, with its 32-cell count identifying the matched subset; the paired
between-grouping file reports the three original contrasts.

The secondary metric split follows the fixed eligibility-first path A → A∩V → V.
It averages across activity-trained and valid-trained models, not across both
possible orders of correcting the population errors. The recall gains are
paired within cells before taking their medians. Separately calculated medians
need not add, although the corresponding identities hold within each cell.
