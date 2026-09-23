"""Regenerate risk-score AP intervals from saved predictions without fitting.

Example (from the project root)::

    python3 scripts/postprocess_ap_intervals.py \
      --source results/tlt_submission_time_v1/main/oulab \
      --output results/tlt_submission_time_v1_publication/main/oulab

Copies aggregate CSVs into a separate publication snapshot, replacing only AP
point estimates/intervals with the exact 1-p score convention. Other metrics,
predictions and fitted-run provenance remain untouched. Each statistic retains
the original seeded cluster order, isolating the AP tie correction from Monte
Carlo changes. A manifest records inputs, code and changed values.

When the p and 1-p tie partitions are identical, the original AP computation is
unchanged. Those intervals are retained exactly, with a recorded partition
check; only statistics involving a changed partition are recomputed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cohort_exchange import (
    DECOMPOSITION_CONTRASTS, KEY,
    _bootstrap_cluster_counts,
    _weighted_cluster_average_precision_draws,
    _weighted_cluster_metric_draws,
)

REQUIRED = ('activity_conditioned', 'activity_to_cutoff_valid',
            'cutoff_valid_to_activity', 'cutoff_valid')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def interval(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        raise ValueError('No finite AP bootstrap draws')
    return tuple(float(value) for value in np.quantile(finite, [.025, .975]))


def ap_point(frame: pd.DataFrame) -> float:
    return float(average_precision_score(1 - frame.y, 1 - frame.p_success))


def first_order(frame: pd.DataFrame) -> np.ndarray:
    return frame.cluster_id.dropna().astype(str).unique()


def merged_risk_thresholds(frame: pd.DataFrame) -> int:
    probabilities = np.unique(frame.p_success.to_numpy(dtype=float))
    risk = 1.0 - probabilities
    return int(np.count_nonzero(risk[1:] == risk[:-1]))


class Postprocessor:
    def __init__(self, source: Path, output: Path, n_boot: int, seed: int):
        self.source, self.output = source, output
        self.n_boot, self.seed = n_boot, seed
        self.tables: dict[str, pd.DataFrame] = {}
        self.counts: dict[int, np.ndarray] = {}
        self.changes: list[dict] = []
        self.partitions: list[dict] = []
        self.audit = dict(status='running', source=str(source), output=str(output),
                          n_boot=n_boot, seed=seed, model_refits=0,
                          risk_score='1 - p_success',
                          cluster_order='original first-occurrence order per statistic',
                          max_fast_reference_error=0.0, max_ap_point_change=0.0,
                          validated_draws=0, landmark_cells=0, hazard_cells=0,
                          landmark_cells_retained=0, hazard_cells_retained=0,
                          input_files={}, output_files={})
        self.audit['code_sha256'] = {
            str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path)
            for path in (Path(__file__), ROOT / 'src/cohort_exchange.py')
        }

    def cluster_counts(self, n: int) -> np.ndarray:
        if n not in self.counts:
            self.counts[n] = _bootstrap_cluster_counts(
                self.n_boot, n, np.random.default_rng(self.seed))
        return self.counts[n]

    def draws(self, frame: pd.DataFrame, order: np.ndarray) -> np.ndarray:
        counts = self.cluster_counts(len(order))
        result = _weighted_cluster_average_precision_draws(frame, counts, order)
        check_indices = np.unique([0, self.n_boot // 2, self.n_boot - 1])
        reference = _weighted_cluster_metric_draws(
            frame, counts[check_indices], order)['pr_auc_risk']
        if not np.allclose(result[check_indices], reference, rtol=0, atol=5e-14, equal_nan=True):
            raise ArithmeticError('AP-only helper differs from row-weight metric helper')
        finite = np.isfinite(reference) & np.isfinite(result[check_indices])
        discrepancy = float(np.max(np.abs(reference[finite] - result[check_indices][finite]))) if finite.any() else 0.
        self.audit['max_fast_reference_error'] = max(self.audit['max_fast_reference_error'], discrepancy)
        self.audit['validated_draws'] += len(check_indices)
        return result

    def replace(self, table: str, mask: pd.Series, values: dict, detail: dict) -> None:
        frame = self.tables[table]
        if int(mask.sum()) != 1:
            raise ValueError(f'Expected one output row in {table}: {detail}')
        index = frame.index[mask][0]
        for column, value in values.items():
            previous = float(frame.loc[index, column])
            frame.loc[index, column] = value
            if column in ('delta', 'pr_auc_risk_mean'):
                self.audit['max_ap_point_change'] = max(self.audit['max_ap_point_change'], abs(value - previous))
            self.changes.append(dict(table=table, column=column, previous=previous,
                                     current=float(value), **detail))

    def save(self, status: str) -> None:
        for name, table in self.tables.items():
            table.to_csv(self.output / name, index=False)
        counts = []
        for filename, unit in [('training_evaluation_decomposition.csv', 'presentation'),
                               ('training_evaluation_decomposition_module.csv', 'module')]:
            table = self.tables[filename]
            for (metric, contrast), group in table.groupby(['metric', 'contrast']):
                excluded = group.ci_low.gt(0) | group.ci_high.lt(0)
                counts.append(dict(cluster_unit=unit, metric=metric, contrast=contrast,
                                   cells=len(group), excludes_zero=int(excluded.sum()),
                                   percentage=100 * float(excluded.mean()),
                                   median=float(group.delta.median()), mean=float(group.delta.mean())))
        pd.DataFrame(counts).to_csv(self.output / 'interval_exclusion_counts.csv', index=False)
        pd.DataFrame(self.changes).to_csv(self.output / 'ap_interval_changes.csv', index=False)
        pd.DataFrame(self.partitions).to_csv(self.output / 'ap_tie_partition_audit.csv', index=False)
        self.audit['status'] = status
        self.audit['updated_utc'] = datetime.now(timezone.utc).isoformat()
        self.audit['output_files'] = {path.name: sha256(path) for path in sorted(self.output.glob('*.csv'))}
        (self.output / 'postprocessing_manifest.json').write_text(
            json.dumps(self.audit, indent=2), encoding='utf-8')

    def landmarks(self) -> None:
        protocol = json.loads((self.source / 'protocol.json').read_text())
        expected = {(int(week), model) for week in protocol['weeks'] for model in protocol['models']}
        seen = set()
        for checkpoint in sorted((self.source / 'checkpoints').glob('week_*__*.csv.gz')):
            cell = pd.read_csv(checkpoint, float_precision='round_trip')
            week, model = int(cell.week.iloc[0]), str(cell.model.iloc[0])
            if (week, model) not in expected or (week, model) in seen:
                raise ValueError(f'Unexpected checkpoint cell: {checkpoint}')
            seen.add((week, model))
            self.audit['input_files'][f'checkpoints/{checkpoint.name}'] = sha256(checkpoint)
            frames = {name: group for name, group in cell.groupby('protocol', sort=False)}
            if not set(REQUIRED).issubset(frames):
                raise ValueError(f'Incomplete 2x2 predictions: {checkpoint}')
            changed = set()
            for name, frame in frames.items():
                merged = merged_risk_thresholds(frame)
                self.partitions.append(dict(week=week, model=model, protocol=name,
                                            merged_risk_thresholds=merged))
                if merged:
                    changed.add(name)
            if not changed:
                self.audit['landmark_cells'] += 1
                self.audit['landmark_cells_retained'] += 1
                print(f'[AP] {self.audit["landmark_cells"]}/{len(expected)} week={week} model={model}: partitions unchanged', flush=True)
                continue
            points = {name: ap_point(group) for name, group in frames.items()}
            cache: dict[tuple, np.ndarray] = {}

            def get(protocol_name, order, unit='presentation'):
                key = (unit, protocol_name, tuple(order))
                if key not in cache:
                    frame = frames[protocol_name]
                    if unit == 'module':
                        frame = frame.assign(cluster_id=frame.code_module.astype(str))
                    cache[key] = self.draws(frame, order)
                return cache[key]

            detail = dict(week=week, model=model)
            for filename, unit in [('training_evaluation_decomposition.csv', 'presentation'),
                                   ('training_evaluation_decomposition_module.csv', 'module')]:
                clustered = cell if unit == 'presentation' else cell.assign(cluster_id=cell.code_module.astype(str))
                order = first_order(clustered)
                table = self.tables[filename]
                for contrast, coefficients in DECOMPOSITION_CONTRASTS:
                    if not changed.intersection(coefficients):
                        continue
                    draws = sum(coef * get(name, order, unit) for name, coef in coefficients.items())
                    point = sum(coef * points[name] for name, coef in coefficients.items())
                    lo, hi = interval(draws)
                    mask = table.week.eq(week) & table.model.eq(model) & table.metric.eq('pr_auc_risk') & table.contrast.eq(contrast)
                    self.replace(filename, mask, dict(delta=point, ci_low=lo, ci_high=hi),
                                 dict(contrast=contrast, cluster_unit=unit, **detail))
            summary = self.tables['clustered_summary.csv']
            for name, frame in frames.items():
                if name not in changed:
                    continue
                lo, hi = interval(get(name, first_order(frame)))
                mask = summary.week.eq(week) & summary.model.eq(model) & summary.protocol.eq(name)
                self.replace('clustered_summary.csv', mask,
                             dict(pr_auc_risk_mean=points[name], pr_auc_risk_ci_low=lo, pr_auc_risk_ci_high=hi),
                             dict(protocol=name, **detail))
            distortion = self.tables['metric_distortion.csv']
            reference = 'cutoff_valid'
            for name, frame in frames.items():
                if name == reference:
                    continue
                if not changed.intersection((name, reference)):
                    continue
                order = pd.concat([frames[reference], frame], ignore_index=True).cluster_id.dropna().astype(str).unique()
                lo, hi = interval(get(name, order) - get(reference, order))
                mask = distortion.week.eq(week) & distortion.model.eq(model) & distortion.metric.eq('pr_auc_risk') & distortion.protocol.eq(name)
                self.replace('metric_distortion.csv', mask,
                             dict(delta=points[name] - points[reference], ci_low=lo, ci_high=hi),
                             dict(protocol=name, **detail))
            self.audit['landmark_cells'] += 1
            print(f'[AP] {self.audit["landmark_cells"]}/{len(expected)} week={week} model={model}', flush=True)
        if seen != expected:
            raise ValueError(f'Missing checkpoint cells: {sorted(expected - seen)}')
        self.save('landmarks_complete')
        print('[AP] Landmark summaries and interval-exclusion counts saved', flush=True)

    def hazards(self) -> None:
        path = self.source / 'discrete_hazard_predictions.csv.gz'
        if not path.exists():
            return
        self.audit['input_files'][path.name] = sha256(path)
        columns = ['dataset', 'model', 'landmark_week', *KEY, 'hazard_event', 'cluster_id', 'cluster_unit', 'p_event']
        current, parts, finished = None, [], set()

        def consume(model, pieces):
            frame = pd.concat(pieces, ignore_index=True)
            groups = [column for column in columns if column != 'p_event']
            averaged = frame.groupby(groups, as_index=False).agg(p_event=('p_event', 'mean'))
            averaged['y'] = 1 - averaged.hazard_event
            averaged['p_success'] = 1 - averaged.p_event
            table = self.tables['discrete_hazard_summary.csv']
            for week, group in averaged.groupby('landmark_week'):
                merged = merged_risk_thresholds(group)
                self.partitions.append(dict(week=int(week), model=str(model), protocol='discrete_hazard',
                                            merged_risk_thresholds=merged))
                self.audit['hazard_cells'] += 1
                if not merged:
                    self.audit['hazard_cells_retained'] += 1
                    continue
                point = ap_point(group)
                lo, hi = interval(self.draws(group, first_order(group)))
                mask = table.week.eq(week) & table.model.eq(model)
                self.replace('discrete_hazard_summary.csv', mask,
                             dict(pr_auc_risk_mean=point, pr_auc_risk_ci_low=lo, pr_auc_risk_ci_high=hi),
                             dict(week=int(week), model=str(model), protocol='discrete_hazard'))
            print(f'[AP hazard] model={model} complete', flush=True)

        for chunk in pd.read_csv(path, usecols=columns, float_precision='round_trip', chunksize=200000):
            for model, group in chunk.groupby('model', sort=False):
                if current is not None and model != current:
                    consume(current, parts)
                    finished.add(current)
                    parts = []
                if model in finished:
                    raise ValueError('Hazard input must preserve the runner model-block order')
                current = model
                parts.append(group)
        if parts:
            consume(current, parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--aggregate-json', type=Path, action='append', default=[])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Use a separate output directory, outside the downloaded input directory')
    if args.n_boot < 1 or args.threads < 1:
        raise ValueError('n-boot and threads must be positive')
    output.mkdir(parents=True, exist_ok=True)
    runner = Postprocessor(source, output, args.n_boot, args.seed)
    protocol = json.loads((source / 'protocol.json').read_text())
    if args.n_boot != protocol['bootstrap_iterations'] or args.seed != protocol['seed']:
        raise ValueError('Postprocessing must use the original bootstrap count and seed')
    for path in sorted(source.glob('*.csv')):
        shutil.copy2(path, output / path.name)
        runner.audit['input_files'][path.name] = sha256(path)
    shutil.copy2(source / 'protocol.json', output / 'fit_protocol_original.json')
    runner.audit['input_files']['protocol.json'] = sha256(source / 'protocol.json')
    # Preserve the original protocol separately: this postprocessed directory
    # must not masquerade as a new fitted run or forge its code/input hashes.
    for name in ['clustered_summary.csv', 'metric_distortion.csv',
                 'training_evaluation_decomposition.csv',
                 'training_evaluation_decomposition_module.csv', 'discrete_hazard_summary.csv']:
        if (source / name).exists():
            runner.tables[name] = pd.read_csv(source / name, float_precision='round_trip')
    aggregate_rows = []
    for path in args.aggregate_json:
        rows = json.loads(path.read_text())
        aggregate_rows.extend(dict(aggregate_source=path.name, **row) for row in rows)
        runner.audit.setdefault('aggregate_inputs', {})[str(path)] = sha256(path)
    if aggregate_rows:
        pd.DataFrame(aggregate_rows).to_csv(output / 'aggregate_decomposition_summary.csv', index=False)
    (output / 'README.md').write_text(
        '# Publication postprocessing snapshot\n\n'
        'Contains aggregate CSVs derived from the saved fitted predictions. '
        'Risk-score AP intervals use the corrected 1-p tie convention. '
        'No models were refitted and no downloaded predictions were modified.\n\n'
        f'Original fitted results: `{source}`.\n\n'
        '`fit_protocol_original.json` is archived fitted-run provenance; '
        '`postprocessing_manifest.json` describes these derived summaries. '
        'Large prediction files, checkpoints and original figures remain in the input directory.\n',
        encoding='utf-8')
    runner.save('running')
    with threadpool_limits(limits=args.threads):
        runner.landmarks()
        runner.hazards()
    runner.save('complete')
    print(f'[done] AP postprocessing -> {output}', flush=True)


if __name__ == '__main__':
    main()
