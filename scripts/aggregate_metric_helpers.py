"""Compatibility helpers for risk-score AP and archived bootstrap draws.

The cohort helper now uses separate ties for ROC scores ``p`` and AP scores
``1-p``. The standalone delta function is retained only for correcting archived
draws produced by the former p-threshold implementation. Do not add its output
to draws returned by the corrected cohort helper.
"""
from __future__ import annotations

import numpy as np

from src.cohort_exchange import (
    _cluster_metric_components,
    _weighted_cluster_metric_draws,
)


def average_precision_tie_correction(
    frame,
    cluster_counts: np.ndarray,
    cluster_order: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Return AP(``1-p``) minus historical p-threshold bootstrap AP.

    Columns of ``cluster_counts`` correspond to the supplied named cluster
    order. Only thresholds that collapse after ``1-p`` require matrix work.
    Empty or zero-adverse-case draws receive a zero adjustment; the original
    helper's undefined/single-class draw policy is retained by the wrapper.
    """
    counts = np.asarray(cluster_counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != len(cluster_order):
        raise ValueError("cluster_counts columns must match cluster_order")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    correction = np.zeros(len(counts), dtype=float)
    if frame.empty or not len(counts) or not len(cluster_order):
        return correction

    comp = _cluster_metric_components(frame, cluster_order)
    starts = comp["score_starts"].astype(np.int64)
    probabilities = np.sort(frame["p_success"].to_numpy(dtype=float), kind="mergesort")
    risk_thresholds = 1.0 - probabilities[starts]
    risk_starts = np.r_[0, 1 + np.flatnonzero(risk_thresholds[1:] != risk_thresholds[:-1])]
    risk_ends = np.r_[risk_starts[1:], len(starts)]
    merged = [(int(start), int(end)) for start, end in zip(risk_starts, risk_ends)
              if end - start > 1]
    if not merged:
        return correction

    # Within a merged block, the last threshold's contribution cancels. For
    # each earlier group g, add F_g * (precision_at_block_end - precision_g).
    selected = np.concatenate([np.arange(start, end - 1) for start, end in merged])
    ends = np.array([end - 1 for _, end in merged])
    end_for_selected = np.repeat(np.arange(len(merged)), [end - start - 1 for start, end in merged])
    cluster_codes = comp["sorted_cluster"].astype(np.int64)
    threshold_codes = np.searchsorted(starts, np.arange(len(cluster_codes)), side="right") - 1
    shape = (len(starts), len(cluster_order))
    total = np.zeros(shape, dtype=float)
    failure = np.zeros(shape, dtype=float)
    np.add.at(total, (threshold_codes, cluster_codes), 1.0)
    np.add.at(failure, (threshold_codes, cluster_codes), 1.0 - comp["sorted_y"])
    cumulative_total = np.cumsum(total, axis=0)
    cumulative_failure = np.cumsum(failure, axis=0)
    failure_totals = cumulative_failure[-1]

    for start in range(0, len(counts), batch_size):
        stop = min(start + batch_size, len(counts))
        batch = counts[start:stop]
        current_numerator = batch @ cumulative_failure[selected].T
        current_denominator = batch @ cumulative_total[selected].T
        current_precision = np.divide(
            current_numerator, current_denominator,
            out=np.zeros_like(current_numerator), where=current_denominator > 0,
        )
        end_numerator = batch @ cumulative_failure[ends].T
        end_denominator = batch @ cumulative_total[ends].T
        end_precision = np.divide(
            end_numerator, end_denominator,
            out=np.zeros_like(end_numerator), where=end_denominator > 0,
        )
        group_failure = batch @ failure[selected].T
        delta_numerator = np.sum(
            group_failure * (end_precision[:, end_for_selected] - current_precision), axis=1
        )
        n_failure = batch @ failure_totals
        np.divide(delta_numerator, n_failure,
                  out=correction[start:stop], where=n_failure > 0)
    return correction


def weighted_cluster_metric_draws_exact(
    frame, cluster_counts: np.ndarray, cluster_order: np.ndarray,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Compatibility alias for the corrected cohort metric helper.

    The source helper already handles risk-score ties; no extra delta is added.
    """
    return _weighted_cluster_metric_draws(frame, cluster_counts, cluster_order, batch_size)
