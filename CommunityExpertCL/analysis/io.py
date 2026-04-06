"""
I/O helpers for experiment records.
"""

import json
from pathlib import Path

from .metrics import aggregate_trial_records


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path, payload):
    _ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_trial_record(results_root, trial_record):
    """Save a single trial record under dataset/model folders."""
    output_dir = (Path(results_root)
                  / trial_record['dataset']
                  / trial_record['model'])
    filename = (
        f"trial_{trial_record['trial_index']:02d}"
        f"_seed{trial_record['seed']}.json"
    )
    output_path = output_dir / filename
    _write_json(output_path, trial_record)
    return output_path


def save_aggregate_summary(results_root, aggregate_summary):
    """Save aggregated summary for a dataset/model pair."""
    output_path = (Path(results_root)
                   / aggregate_summary['dataset']
                   / aggregate_summary['model']
                   / 'summary.json')
    _write_json(output_path, aggregate_summary)
    return output_path


def discover_trial_records(results_root, datasets=None, models=None):
    """Load all saved trial records grouped by dataset/model."""
    results_root = Path(results_root)
    dataset_filter = set(datasets) if datasets else None
    model_filter = set(models) if models else None

    grouped = {}
    for record_path in results_root.glob('*/*/trial_*.json'):
        dataset = record_path.parent.parent.name
        model = record_path.parent.name
        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model not in model_filter:
            continue
        with open(record_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        grouped.setdefault(dataset, {}).setdefault(model, []).append(payload)
    return grouped


def load_and_aggregate_results(results_root, datasets=None, models=None):
    """Load trial records and aggregate them into report-ready summaries."""
    grouped = discover_trial_records(
        results_root,
        datasets=datasets,
        models=models,
    )
    aggregated = {}
    for dataset, model_records in grouped.items():
        aggregated[dataset] = {}
        for model, records in model_records.items():
            aggregated[dataset][model] = aggregate_trial_records(records)
    return aggregated
