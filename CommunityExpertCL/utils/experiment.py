"""
Experiment helpers for session construction and dataset settings.
"""

import json
from copy import deepcopy
from pathlib import Path


def _contiguous_group_splits(num_classes, group_size=2):
    """Build contiguous class groups, keeping the final remainder group."""
    return [
        list(range(start, min(start + group_size, num_classes)))
        for start in range(0, num_classes, group_size)
    ]


DATASET_SETTINGS = {
    'cora': {
        'legacy_class_splits': [[0, 1], [2, 3], [4, 5, 6]],
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'citeseer': {
        'legacy_class_splits': [[0, 1], [2, 3], [4, 5]],
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'cora-full': {
        'legacy_class_splits': _contiguous_group_splits(70, group_size=2),
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'coauthor-cs': {
        'legacy_class_splits': _contiguous_group_splits(15, group_size=2),
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'amazon-computers': {
        'legacy_class_splits': _contiguous_group_splits(10, group_size=2),
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'wikics': {
        'legacy_class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'ogbn-arxiv': {
        'legacy_class_splits': _contiguous_group_splits(40, group_size=2),
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'ogbn-products': {
        'legacy_class_splits': _contiguous_group_splits(47, group_size=2),
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'reddit': {
        'legacy_class_splits': _contiguous_group_splits(41, group_size=2),
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
}

KNOWN_CLASS_COUNTS = {
    'cora': 7,
    'citeseer': 6,
    'cora-full': 70,
    'coauthor-cs': 15,
    'amazon-computers': 10,
    'wikics': 10,
    'ogbn-arxiv': 40,
    'ogbn-products': 47,
    'reddit': 41,
}

EXP1_FIXED_GROUP_SIZES = {
    'cora-full': 4,
    'reddit': 4,
    'ogbn-arxiv': 4,
    'ogbn-products': 4,
    'coauthor-cs': 2,
    'amazon-computers': 2,
}


def get_dataset_setting(dataset):
    """Return a deep copy so callers can mutate safely."""
    if dataset not in DATASET_SETTINGS:
        raise ValueError(f"Unknown dataset setting: {dataset}")
    return deepcopy(DATASET_SETTINGS[dataset])


def get_known_class_ids(dataset, num_classes=None):
    """Return a contiguous class id list when the class count is known."""
    if num_classes is None:
        if dataset not in KNOWN_CLASS_COUNTS:
            raise ValueError(
                f"Unknown class count for dataset: {dataset}. "
                "Pass num_classes explicitly."
            )
        num_classes = KNOWN_CLASS_COUNTS[dataset]
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return list(range(int(num_classes)))


def build_balanced_class_splits(class_ids, num_sessions):
    """
    Partition class IDs into contiguous balanced sessions.

    The class order is preserved to stay close to the CGLB default order.
    """
    if not class_ids:
        raise ValueError("class_ids must not be empty")
    if num_sessions <= 0:
        raise ValueError("num_sessions must be positive")

    ordered = list(class_ids)
    num_sessions = min(num_sessions, len(ordered))

    base = len(ordered) // num_sessions
    remainder = len(ordered) % num_sessions

    splits = []
    start = 0
    for session_id in range(num_sessions):
        size = base + (1 if session_id < remainder else 0)
        end = start + size
        splits.append(ordered[start:end])
        start = end

    return [split for split in splits if split]


def build_fixed_class_splits(dataset, class_ids):
    """Build fixed contiguous sessions for the Exp1 benchmark."""
    if dataset not in EXP1_FIXED_GROUP_SIZES:
        raise ValueError(
            f"Dataset '{dataset}' does not have an Exp1 fixed session rule"
        )
    group_size = EXP1_FIXED_GROUP_SIZES[dataset]
    ordered = list(sorted(class_ids))
    splits = [
        ordered[start:start + group_size]
        for start in range(0, len(ordered), group_size)
    ]
    return [split for split in splits if split], {
        'strategy': 'exp1_fixed',
        'group_size': group_size,
        'requested_sessions': len(splits),
        'effective_sessions': len(splits),
    }


def load_task_sequence_file(task_seq_file, class_ids=None):
    """Load and validate an explicit task sequence JSON file."""
    with Path(task_seq_file).open('r', encoding='utf-8') as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        class_splits = payload.get('class_splits')
        meta = payload.get('meta', {})
    else:
        class_splits = payload
        meta = {}

    if not isinstance(class_splits, list) or not class_splits:
        raise ValueError('task_seq_file must contain a non-empty list of class splits')

    normalized = []
    seen = set()
    allowed = set(class_ids) if class_ids is not None else None
    for task_id, task_cls in enumerate(class_splits):
        if not isinstance(task_cls, list) or not task_cls:
            raise ValueError(f'task_seq[{task_id}] must be a non-empty list')
        current = [int(cls_id) for cls_id in task_cls]
        for cls_id in current:
            if allowed is not None and cls_id not in allowed:
                raise ValueError(f'class id {cls_id} is not valid for this dataset')
            if cls_id in seen:
                raise ValueError(f'class id {cls_id} is duplicated in task_seq_file')
            seen.add(cls_id)
        normalized.append(current)

    if allowed is not None:
        missing = sorted(allowed - seen)
        if missing:
            raise ValueError(
                f'task_seq_file does not cover all classes, missing: {missing}'
            )

    meta = {
        **meta,
        'strategy': meta.get('strategy', 'task_seq_file'),
        'requested_sessions': meta.get('requested_sessions', len(normalized)),
        'effective_sessions': meta.get('effective_sessions', len(normalized)),
    }
    return normalized, meta


def resolve_class_splits(dataset, class_ids, strategy,
                         num_sessions=None,
                         session_multiplier=2.0,
                         max_experts=None):
    """
    Resolve class splits for a dataset.

    strategy:
      - legacy: use the repository's hard-coded splits unless num_sessions
        overrides them
      - balanced: build balanced contiguous splits
      - exp1_fixed: use the agreed fixed group size per dataset
    """
    ordered_class_ids = list(sorted(class_ids))

    if strategy == 'exp1_fixed':
        return build_fixed_class_splits(dataset, ordered_class_ids)

    if strategy == 'legacy':
        dataset_setting = get_dataset_setting(dataset)
        legacy_splits = dataset_setting['legacy_class_splits']
    else:
        legacy_splits = None

    if strategy == 'legacy' and num_sessions is None:
        return deepcopy(legacy_splits), {
            'strategy': 'legacy',
            'requested_sessions': len(legacy_splits),
            'effective_sessions': len(legacy_splits),
        }

    if num_sessions is None:
        if max_experts is None:
            raise ValueError(
                "max_experts is required when num_sessions is not specified"
            )
        num_sessions = int(round(max_experts * session_multiplier))

    effective_sessions = min(max(1, num_sessions), len(ordered_class_ids))
    return build_balanced_class_splits(ordered_class_ids, effective_sessions), {
        'strategy': strategy,
        'requested_sessions': num_sessions,
        'effective_sessions': effective_sessions,
    }


def resolve_trial_seeds(seed_values, ntrials):
    """Extend configured seeds deterministically when more trials are requested."""
    seeds = list(seed_values)
    if not seeds:
        seeds = [0]
    if len(seeds) >= ntrials:
        return seeds[:ntrials]

    next_seed = seeds[-1] + 1
    while len(seeds) < ntrials:
        seeds.append(next_seed)
        next_seed += 1
    return seeds
