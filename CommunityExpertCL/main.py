"""
CommunityExpertCL - Main entry point.

Usage:
  python main.py --dataset cora --gpu 0
  python main.py --dataset coauthor-cs --gpu 0 --amp
  python main.py --dataset cora-full --ntrials 5 --output_dir ./results/exp1
"""

import os
import argparse
from copy import deepcopy
import yaml
import numpy as np

import torch

from analysis.io import save_aggregate_summary, save_trial_record
from analysis.metrics import llm4gcl_metrics, lower_triangle_row_mean
from data import GraphDataset, TaskLoader
from models import LiteExpertCL
from utils import (
    DATASET_SETTINGS,
    get_dataset_setting,
    resolve_class_splits,
    resolve_trial_seeds,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser(description='CommunityExpertCL')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=list(DATASET_SETTINGS.keys()))
    parser.add_argument('--data_path', type=str, default='./data_files/')
    parser.add_argument('--ntrials', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training (AMP)')
    parser.add_argument('--svd_dim', type=int, default=0,
                        help='Truncated SVD target dim (0 = disabled)')
    parser.add_argument('--data_protocol', type=str, default='native',
                        choices=['native', 'cglb'],
                        help='Dataset loading protocol for LiteExpert')
    parser.add_argument('--output_dir', type=str, default='./results/exp1',
                        help='Directory for per-trial results and summaries')
    parser.add_argument('--model_name', type=str, default='liteexpert',
                        help='Model identifier used in saved results')
    parser.add_argument('--session_strategy', type=str, default='legacy',
                        choices=['legacy', 'balanced'],
                        help='How to construct class sessions')
    parser.add_argument('--num_sessions', type=int, default=None,
                        help='Override the number of sessions')
    parser.add_argument('--session_multiplier', type=float, default=2.0,
                        help='When num_sessions is omitted, use max_experts * multiplier')
    parser.add_argument('--reorder_by_class_size', action='store_true',
                        help='Use the legacy class reordering by class size')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config_lite.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    config = deepcopy(full_config['default'])

    dataset_setting = get_dataset_setting(args.dataset)
    config['split_S'] = dataset_setting.get('split_S', config.get('split_S', 5))
    config['split_t'] = dataset_setting.get('split_t', config.get('split_t', 3))
    config['split_v'] = dataset_setting.get('split_v', config.get('split_v', 1))
    config['use_amp'] = args.amp

    # Device
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )

    graph_dataset = GraphDataset(
        args.dataset,
        args.data_path,
        svd_dim=args.svd_dim,
        reorder_by_class_size=args.reorder_by_class_size,
        data_protocol=args.data_protocol,
    )
    class_splits, split_meta = resolve_class_splits(
        dataset=args.dataset,
        class_ids=sorted(graph_dataset.id_by_class.keys()),
        strategy=args.session_strategy,
        num_sessions=args.num_sessions,
        session_multiplier=args.session_multiplier,
        max_experts=config.get('max_experts', 8),
    )
    config['class_splits'] = class_splits

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Data protocol: {args.data_protocol}")
    print(f"Model: {args.model_name}")
    print(f"Session strategy: {args.session_strategy}")
    print(f"Requested sessions: {split_meta['requested_sessions']}")
    print(f"Effective sessions: {split_meta['effective_sessions']}")
    print(f"Class splits: {config['class_splits']}")
    print(f"Split ratio: t/S={config['split_t']}/{config['split_S']}, "
          f"v/S={config['split_v']}/{config['split_S']}")
    print(f"AMP: {'enabled' if args.amp else 'disabled'}")
    print(f"Class reorder by size: {'enabled' if args.reorder_by_class_size else 'disabled'}")
    if args.svd_dim > 0:
        print(f"SVD dim: {args.svd_dim}")

    seeds = resolve_trial_seeds(config.get('seed', [0, 1, 2, 3, 4]), args.ntrials)
    trial_records = []

    for trial, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"Trial {trial + 1}/{len(seeds)}, Seed: {seed}")
        print(f"{'#'*60}")

        seed_everything(seed)

        task_loader = TaskLoader(
            batch_size=config.get('batch_size', 256),
            graph_dataset=graph_dataset,
            class_splits=config['class_splits'],
            split_S=config['split_S'],
            split_t=config['split_t'],
            split_v=config['split_v'],
        )

        model = LiteExpertCL(
            task_loader=task_loader,
            config=config,
            device=device,
        )

        results = model.fit(trial)
        row_mean = lower_triangle_row_mean(results['acc_matrix'])
        llm_metrics = llm4gcl_metrics(results['joint_acc'])
        trial_record = {
            'dataset': args.dataset,
            'model': args.model_name,
            'trial_index': trial,
            'seed': seed,
            'config': {
                'class_splits': config['class_splits'],
                'split_S': config['split_S'],
                'split_t': config['split_t'],
                'split_v': config['split_v'],
                'svd_dim': args.svd_dim,
                'data_protocol': args.data_protocol,
                'session_strategy': args.session_strategy,
                'requested_sessions': split_meta['requested_sessions'],
                'effective_sessions': split_meta['effective_sessions'],
                'max_experts': config.get('max_experts', None),
                'reorder_by_class_size': args.reorder_by_class_size,
            },
            'results': {
                'acc_matrix': results['acc_matrix'],
                'joint_acc': results['joint_acc'],
                'lower_triangle_row_mean': row_mean,
                'llm4gcl_avg_micro': llm_metrics['avg_micro'],
                'llm4gcl_last_micro': llm_metrics['last_micro'],
                'lower_triangle_avg_macro': float(np.mean(row_mean)),
                'lower_triangle_last_macro': float(row_mean[-1]),
            },
        }
        trial_records.append(trial_record)
        save_trial_record(args.output_dir, trial_record)

        print(f"\nTrial {trial + 1} Summary:")
        print(f"  Joint Abar: {llm_metrics['avg_micro']:.4f}")
        print(f"  Joint Afinal: {llm_metrics['last_micro']:.4f}")
        print(f"  RowMean Afinal: {row_mean[-1]:.4f}")

    from analysis.metrics import aggregate_trial_records

    summary = aggregate_trial_records(trial_records)
    save_aggregate_summary(args.output_dir, summary)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY ({len(seeds)} trials)")
    print(f"{'='*60}")
    print(
        "LLM4GCL Abar (micro): "
        f"{summary['metrics']['llm4gcl_avg_micro_mean']:.4f} "
        f"\u00b1 {summary['metrics']['llm4gcl_avg_micro_std']:.4f}"
    )
    print(
        "LLM4GCL Afinal (micro): "
        f"{summary['metrics']['llm4gcl_last_micro_mean']:.4f} "
        f"\u00b1 {summary['metrics']['llm4gcl_last_micro_std']:.4f}"
    )
    print(
        "Lower-triangle RowMean Afinal: "
        f"{summary['metrics']['lower_triangle_last_macro_mean']:.4f} "
        f"\u00b1 {summary['metrics']['lower_triangle_last_macro_std']:.4f}"
    )


if __name__ == '__main__':
    main()
