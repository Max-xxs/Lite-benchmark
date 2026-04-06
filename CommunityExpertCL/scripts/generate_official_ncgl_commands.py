"""
Generate official CGLB/DeLoMe run commands for Exp1 with explicit task splits.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_known_class_ids, resolve_class_splits


OFFICIAL_DATASET_MAP = {
    'cora-full': 'CoraFull-CL',
    'coauthor-cs': 'coauthor-cs',
    'amazon-computers': 'amazon-computers',
    'ogbn-arxiv': 'Arxiv-CL',
    'reddit': 'Reddit-CL',
    'ogbn-products': 'Products-CL',
}

CGLB_BASELINES = ['bare', 'ewc', 'mas', 'lwf', 'gem', 'twp', 'ergnn', 'joint']
DELOME_BASELINES = ['DeLoMe']


def write_task_seq_file(task_seq_dir, dataset, class_splits, meta):
    task_seq_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{meta['strategy']}_s{meta['effective_sessions']}"
        f"_e{meta.get('max_experts', 'na')}"
    )
    output_path = task_seq_dir / f'{dataset}_{suffix}.json'
    payload = {
        'dataset': dataset,
        'class_splits': class_splits,
        'meta': meta,
    }
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return output_path


def build_task_seq(dataset, session_strategy, num_sessions,
                   session_multiplier, max_experts):
    class_ids = get_known_class_ids(dataset)
    class_splits, meta = resolve_class_splits(
        dataset=dataset,
        class_ids=class_ids,
        strategy=session_strategy,
        num_sessions=num_sessions,
        session_multiplier=session_multiplier,
        max_experts=max_experts,
    )
    meta = {
        **meta,
        'dataset': dataset,
        'num_classes': len(class_ids),
        'max_experts': max_experts,
    }
    return class_splits, meta


def main():
    parser = argparse.ArgumentParser(
        description='Print official run commands for CGLB/DeLoMe baselines',
    )
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--ori_data_path', type=str, default='/store/data')
    parser.add_argument('--backbone', type=str, default='GCN')
    parser.add_argument('--delome_backbone', type=str, default='GCN')
    parser.add_argument('--task_seq_dir', type=str,
                        default='./results/exp1/task_sequences')
    parser.add_argument('--session_strategy', type=str, default='balanced',
                        choices=['legacy', 'balanced'])
    parser.add_argument('--num_sessions', type=int, default=None)
    parser.add_argument('--session_multiplier', type=float, default=2.0)
    parser.add_argument('--max_experts', type=int, default=8)
    args = parser.parse_args()

    cglb_root = Path('/Users/max/Documents/Expert/external/CGLB/NCGL')
    delome_root = Path('/Users/max/Documents/Expert/external/DeLoMe')
    task_seq_dir = Path(args.task_seq_dir)

    dataset_task_files = {}
    for dataset in OFFICIAL_DATASET_MAP:
        class_splits, meta = build_task_seq(
            dataset=dataset,
            session_strategy=args.session_strategy,
            num_sessions=args.num_sessions,
            session_multiplier=args.session_multiplier,
            max_experts=args.max_experts,
        )
        task_seq_file = write_task_seq_file(task_seq_dir, dataset, class_splits, meta)
        dataset_task_files[dataset] = (task_seq_file, class_splits, meta)

    print('# CGLB baselines')
    for dataset, official_name in OFFICIAL_DATASET_MAP.items():
        task_seq_file, class_splits, meta = dataset_task_files[dataset]
        task_seq_name = Path(task_seq_file).stem
        print(
            f'# {dataset}: {meta["effective_sessions"]} sessions, '
            f'sizes {[len(split) for split in class_splits]}'
        )
        for method in CGLB_BASELINES:
            cmd = (
                f'cd {cglb_root} && '
                f'python train.py '
                f'--dataset {official_name} '
                f'--method {method} '
                f'--backbone {args.backbone} '
                f'--gpu {args.gpu} '
                f'--ILmode classIL '
                f'--inter-task-edges True '
                f'--minibatch True '
                f'--batch_size 2000 '
                f'--sample_nbs True '
                f'--n_nbs_sample 10,25 '
                f'--n_cls_per_task 2 '
                f'--task_seq_file {task_seq_file} '
                f'--task_seq_name {task_seq_name} '
                f'--epochs {args.epochs} '
                f'--repeats {args.repeats} '
                f'--ori_data_path {args.ori_data_path} '
                f'--data_path {args.data_path} '
                f'--result_path {args.result_path}'
            )
            print(cmd)

    print('\n# DeLoMe baseline')
    for dataset, official_name in OFFICIAL_DATASET_MAP.items():
        task_seq_file, class_splits, meta = dataset_task_files[dataset]
        task_seq_name = Path(task_seq_file).stem
        print(
            f'# {dataset}: {meta["effective_sessions"]} sessions, '
            f'sizes {[len(split) for split in class_splits]}'
        )
        for method in DELOME_BASELINES:
            cmd = (
                f'cd {delome_root} && '
                f'python train.py '
                f'--dataset {official_name} '
                f'--method {method} '
                f'--backbone {args.delome_backbone} '
                f'--gpu {args.gpu} '
                f'--ILmode classIL '
                f'--inter-task-edges True '
                f'--minibatch True '
                f'--batch_size 2000 '
                f'--sample_nbs True '
                f'--n_nbs_sample 10,25 '
                f'--n_cls_per_task 2 '
                f'--task_seq_file {task_seq_file} '
                f'--task_seq_name {task_seq_name} '
                f'--epochs {args.epochs} '
                f'--repeats {args.repeats} '
                f'--ori_data_path {args.ori_data_path} '
                f'--data_path {args.data_path} '
                f'--result_path {args.result_path}'
            )
            print(cmd)

    print('\n# Notes')
    print('# The generated commands keep official training logic and only add an explicit task sequence input.')
    print('# coauthor-cs and amazon-computers are now routed through a dataset-loader adapter in the official codebase.')
    print('# task_seq_file avoids the official n_cls_per_task rounding/distortion when class counts do not divide cleanly.')


if __name__ == '__main__':
    main()
