"""
Import official CGLB/DeLoMe NCGL result pickles into the local Exp1 format.
"""

import argparse
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from analysis.io import save_aggregate_summary, save_trial_record
from analysis.metrics import aggregate_trial_records, lower_triangle_row_mean


def _to_lower_triangle_rows(matrix):
    array = np.asarray(matrix, dtype=float)
    return [array[row_id, :row_id + 1].tolist()
            for row_id in range(array.shape[0])]


def _parse_class_splits(raw_value):
    if raw_value is None:
        return None
    import json
    return json.loads(raw_value)


def _load_class_splits(path):
    if path is None:
        return None
    import json
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get('class_splits')
    return payload


def main():
    parser = argparse.ArgumentParser(
        description='Import official NCGL result pickles into Exp1 JSON format',
    )
    parser.add_argument('--result_pkl', type=str, required=True,
                        help='Path to the official result pickle')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name used in the local Exp1 reports')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Display/model identifier used locally')
    parser.add_argument('--output_dir', type=str, default='./results/exp1')
    parser.add_argument('--source', type=str, default='cglb',
                        choices=['cglb', 'delome'],
                        help='Which official pipeline produced this pickle')
    parser.add_argument('--class_splits_json', type=str, default=None,
                        help='Optional JSON string with explicit class splits')
    parser.add_argument('--class_splits_file', type=str, default=None,
                        help='Optional JSON file with explicit class splits')
    parser.add_argument('--session_strategy', type=str, default='external_official')
    parser.add_argument('--seed_base', type=int, default=0)
    args = parser.parse_args()

    with open(args.result_pkl, 'rb') as f:
        performance_matrices = pickle.load(f)

    if not isinstance(performance_matrices, list) or not performance_matrices:
        raise ValueError('Expected a non-empty list of performance matrices')

    class_splits = _load_class_splits(args.class_splits_file)
    if class_splits is None:
        class_splits = _parse_class_splits(args.class_splits_json)
    trial_records = []
    for trial_index, matrix in enumerate(performance_matrices):
        lower_rows = _to_lower_triangle_rows(matrix)
        row_mean = lower_triangle_row_mean(lower_rows)
        trial_record = {
            'dataset': args.dataset,
            'model': args.model_name,
            'trial_index': trial_index,
            'seed': args.seed_base + trial_index,
            'config': {
                'class_splits': class_splits,
                'split_S': None,
                'split_t': None,
                'split_v': None,
                'svd_dim': None,
                'session_strategy': args.session_strategy,
                'requested_sessions': len(lower_rows),
                'effective_sessions': len(lower_rows),
                'max_experts': None,
                'reorder_by_class_size': None,
                'external_source': args.source,
                'joint_metric_available': False,
            },
            'results': {
                'acc_matrix': lower_rows,
                'joint_acc': [],
                'lower_triangle_row_mean': row_mean,
                'llm4gcl_avg_micro': float('nan'),
                'llm4gcl_last_micro': float('nan'),
                'lower_triangle_avg_macro': float(np.mean(row_mean)),
                'lower_triangle_last_macro': float(row_mean[-1]),
            },
        }
        trial_records.append(trial_record)
        save_trial_record(args.output_dir, trial_record)

    summary = aggregate_trial_records(trial_records)
    save_aggregate_summary(args.output_dir, summary)


if __name__ == '__main__':
    main()
