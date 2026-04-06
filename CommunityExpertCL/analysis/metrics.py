"""
Metrics and aggregation helpers for Exp1.
"""

import numpy as np


def lower_triangle_row_mean(acc_matrix):
    """Return the mean of each lower-triangular row."""
    row_means = []
    for row_id, row in enumerate(acc_matrix):
        valid = row[:row_id + 1]
        if not valid:
            row_means.append(float('nan'))
            continue
        row_means.append(float(np.mean(valid)))
    return row_means


def llm4gcl_metrics(joint_acc):
    """
    Metrics aligned with LLM4GCL-style reporting.

    joint_acc is the session-wise micro accuracy on the accumulated test set.
    """
    if not joint_acc:
        return {
            'avg_micro': float('nan'),
            'last_micro': float('nan'),
        }

    joint_acc = np.asarray(joint_acc, dtype=float)
    return {
        'avg_micro': float(np.mean(joint_acc)),
        'last_micro': float(joint_acc[-1]),
    }


def square_lower_triangle(acc_matrix):
    """Pad a lower-triangular nested list into an NxN matrix with NaNs."""
    num_sessions = len(acc_matrix)
    square = np.full((num_sessions, num_sessions), np.nan, dtype=float)
    for row_id, row in enumerate(acc_matrix):
        if row:
            square[row_id, :len(row)] = np.asarray(row, dtype=float)
    return square


def _pad_curves(curves):
    if not curves:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(curve) for curve in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=float)
    for idx, curve in enumerate(curves):
        padded[idx, :len(curve)] = np.asarray(curve, dtype=float)
    return padded


def _nan_stats(array):
    if array.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.nanmean(array, axis=0), np.nanstd(array, axis=0)


def aggregate_trial_records(trial_records):
    """Aggregate multiple trial records into report-ready statistics."""
    if not trial_records:
        raise ValueError("trial_records must not be empty")

    trial_records = sorted(
        trial_records,
        key=lambda record: record['trial_index'],
    )

    matrices = [square_lower_triangle(record['results']['acc_matrix'])
                for record in trial_records]
    max_sessions = max(matrix.shape[0] for matrix in matrices)
    padded_matrices = np.full(
        (len(matrices), max_sessions, max_sessions),
        np.nan,
        dtype=float,
    )
    for idx, matrix in enumerate(matrices):
        h, w = matrix.shape
        padded_matrices[idx, :h, :w] = matrix

    mean_matrix = np.nanmean(padded_matrices, axis=0)
    std_matrix = np.nanstd(padded_matrices, axis=0)

    joint_curves = _pad_curves([
        record['results'].get('joint_acc', [])
        for record in trial_records
        if record['results'].get('joint_acc')
    ])
    joint_mean, joint_std = _nan_stats(joint_curves)

    row_mean_curves = _pad_curves([
        record['results']['lower_triangle_row_mean'] for record in trial_records
    ])
    row_mean_mean, row_mean_std = _nan_stats(row_mean_curves)

    llm4gcl_avg = np.asarray([
        record['results']['llm4gcl_avg_micro'] for record in trial_records
    ], dtype=float)
    llm4gcl_last = np.asarray([
        record['results']['llm4gcl_last_micro'] for record in trial_records
    ], dtype=float)

    row_avg = np.asarray([
        record['results']['lower_triangle_avg_macro'] for record in trial_records
    ], dtype=float)
    row_last = np.asarray([
        record['results']['lower_triangle_last_macro'] for record in trial_records
    ], dtype=float)

    first = trial_records[0]
    summary = {
        'dataset': first['dataset'],
        'model': first['model'],
        'ntrials': len(trial_records),
        'num_sessions': int(max_sessions),
        'class_splits': first['config']['class_splits'],
        'session_strategy': first['config']['session_strategy'],
        'mean_matrix': mean_matrix.tolist(),
        'std_matrix': std_matrix.tolist(),
        'mean_joint_acc': joint_mean.tolist(),
        'std_joint_acc': joint_std.tolist(),
        'mean_lower_triangle_row_mean': row_mean_mean.tolist(),
        'std_lower_triangle_row_mean': row_mean_std.tolist(),
        'metrics': {
            'llm4gcl_avg_micro_mean': float(np.mean(llm4gcl_avg)),
            'llm4gcl_avg_micro_std': float(np.std(llm4gcl_avg)),
            'llm4gcl_last_micro_mean': float(np.mean(llm4gcl_last)),
            'llm4gcl_last_micro_std': float(np.std(llm4gcl_last)),
            'lower_triangle_avg_macro_mean': float(np.mean(row_avg)),
            'lower_triangle_avg_macro_std': float(np.std(row_avg)),
            'lower_triangle_last_macro_mean': float(np.mean(row_last)),
            'lower_triangle_last_macro_std': float(np.std(row_last)),
        },
    }

    return summary
