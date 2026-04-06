"""
Run Exp1 end-to-end:
1. Write task sequences
2. Train LiteExpert
3. Train official CGLB baselines and DeLoMe
4. Import external results into the local report format
5. Generate report figures and summary tables
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

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

DEFAULT_DATASETS = list(OFFICIAL_DATASET_MAP.keys())
CGLB_BASELINES = ['bare', 'ewc', 'mas', 'lwf', 'gem', 'twp', 'ergnn', 'joint']
DELOME_BASELINES = ['DeLoMe']

DISPLAY_NAME_MAP = {
    'bare': 'BARE',
    'ewc': 'EWC',
    'mas': 'MAS',
    'lwf': 'LWF',
    'gem': 'GEM',
    'twp': 'TWP',
    'ergnn': 'ER-GNN',
    'joint': 'JOINT',
    'DeLoMe': 'DeLoMe',
}

DEFAULT_LITEEXPERT_SVD = {
    'cora-full': 512,
    'coauthor-cs': 256,
    'ogbn-arxiv': 0,
}


def parse_json_mapping(raw_value):
    if raw_value is None:
        return {}
    return json.loads(raw_value)


def run_command(cmd, cwd, dry_run=False):
    rendered = shlex.join(str(part) for part in cmd)
    print(f'\n[RUN] {rendered}\n  cwd={cwd}')
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def build_task_sequence_file(dataset, task_seq_dir, session_strategy,
                             num_sessions, session_multiplier, max_experts):
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
    suffix = (
        f"{meta['strategy']}_s{meta['effective_sessions']}"
        f"_e{meta['max_experts']}"
    )
    path = task_seq_dir / f'{dataset}_{suffix}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'dataset': dataset,
                'class_splits': class_splits,
                'meta': meta,
            },
            f,
            indent=2,
        )
    return path, class_splits, meta


def find_latest_result_pkl(result_root, official_dataset_name, task_seq_name, method):
    pattern = f'te_{official_dataset_name}_{task_seq_name}_{method}_*.pkl'
    matches = list(Path(result_root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'No result pickle found for pattern {pattern} under {result_root}'
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def import_external_result(python_bin, project_root, result_pkl, dataset,
                           model_name, output_dir, source, class_splits_file,
                           dry_run=False):
    cmd = [
        python_bin,
        project_root / 'scripts' / 'import_external_ncgl_results.py',
        '--result_pkl', result_pkl,
        '--dataset', dataset,
        '--model_name', model_name,
        '--output_dir', output_dir,
        '--source', source,
        '--class_splits_file', class_splits_file,
        '--session_strategy', 'external_official_exp1',
    ]
    run_command(cmd, project_root, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description='Run the Exp1 benchmark end-to-end')
    parser.add_argument('--datasets', nargs='*', default=DEFAULT_DATASETS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ntrials', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--session_strategy', type=str, default='balanced',
                        choices=['legacy', 'balanced'])
    parser.add_argument('--num_sessions', type=int, default=None)
    parser.add_argument('--session_multiplier', type=float, default=2.0)
    parser.add_argument('--max_experts', type=int, default=8)
    parser.add_argument('--liteexpert_data_path', type=str,
                        default=str(PROJECT_ROOT.parent / 'external_runs' / 'raw'))
    parser.add_argument('--liteexpert_data_protocol', type=str, default='cglb',
                        choices=['native', 'cglb'])
    parser.add_argument('--results_root', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'exp1'))
    parser.add_argument('--report_root', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'exp1' / 'reports'))
    parser.add_argument('--external_raw_path', type=str,
                        default=str(PROJECT_ROOT.parent / 'external_runs' / 'raw'))
    parser.add_argument('--external_data_path', type=str,
                        default=str(PROJECT_ROOT.parent / 'external_runs' / 'data'))
    parser.add_argument('--external_result_path', type=str,
                        default=str(PROJECT_ROOT.parent / 'external_runs' / 'results'))
    parser.add_argument('--cglb_root', type=str,
                        default=str(PROJECT_ROOT.parent / 'external' / 'CGLB' / 'NCGL'))
    parser.add_argument('--delome_root', type=str,
                        default=str(PROJECT_ROOT.parent / 'external' / 'DeLoMe'))
    parser.add_argument('--delome_backbone', type=str, default='GCN',
                        choices=['GCN', 'SGC', 'GAT', 'GIN'])
    parser.add_argument('--svd_override_json', type=str, default=None,
                        help='JSON object of dataset -> svd_dim overrides')
    parser.add_argument('--skip_liteexpert', action='store_true')
    parser.add_argument('--skip_external', action='store_true')
    parser.add_argument('--skip_report', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    datasets = args.datasets or DEFAULT_DATASETS
    unknown = sorted(set(datasets) - set(OFFICIAL_DATASET_MAP))
    if unknown:
        raise ValueError(f'Unsupported datasets for this runner: {unknown}')

    python_bin = Path(sys.executable)
    results_root = Path(args.results_root)
    report_root = Path(args.report_root)
    task_seq_dir = results_root / 'task_sequences'
    results_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    svd_dims = dict(DEFAULT_LITEEXPERT_SVD)
    svd_dims.update(parse_json_mapping(args.svd_override_json))

    task_seq_files = {}
    for dataset in datasets:
        task_seq_file, class_splits, meta = build_task_sequence_file(
            dataset=dataset,
            task_seq_dir=task_seq_dir,
            session_strategy=args.session_strategy,
            num_sessions=args.num_sessions,
            session_multiplier=args.session_multiplier,
            max_experts=args.max_experts,
        )
        task_seq_files[dataset] = {
            'path': task_seq_file,
            'class_splits': class_splits,
            'meta': meta,
            'name': task_seq_file.stem,
        }

    if not args.skip_liteexpert:
        for dataset in datasets:
            svd_dim = int(svd_dims.get(dataset, 0))
            cmd = [
                python_bin,
                PROJECT_ROOT / 'main.py',
                '--dataset', dataset,
                '--data_path', args.liteexpert_data_path,
                '--ntrials', str(args.ntrials),
                '--gpu', str(args.gpu),
                '--svd_dim', str(svd_dim),
                '--data_protocol', args.liteexpert_data_protocol,
                '--output_dir', results_root,
                '--model_name', 'LiteExpertCL',
                '--session_strategy', args.session_strategy,
                '--session_multiplier', str(args.session_multiplier),
            ]
            if args.num_sessions is not None:
                cmd.extend(['--num_sessions', str(args.num_sessions)])
            run_command(cmd, PROJECT_ROOT, dry_run=args.dry_run)

    if not args.skip_external:
        cglb_root = Path(args.cglb_root)
        delome_root = Path(args.delome_root)

        for dataset in datasets:
            official_dataset_name = OFFICIAL_DATASET_MAP[dataset]
            task_seq = task_seq_files[dataset]
            common_tail = [
                '--gpu', str(args.gpu),
                '--ILmode', 'classIL',
                '--inter-task-edges', 'True',
                '--minibatch', 'True',
                '--batch_size', '2000',
                '--sample_nbs', 'True',
                '--n_nbs_sample', '10,25',
                '--n_cls_per_task', '2',
                '--task_seq_file', task_seq['path'],
                '--task_seq_name', task_seq['name'],
                '--epochs', str(args.epochs),
                '--repeats', str(args.ntrials),
                '--ori_data_path', args.external_raw_path,
                '--data_path', args.external_data_path,
                '--result_path', args.external_result_path,
            ]

            for method in CGLB_BASELINES:
                cmd = [
                    python_bin,
                    cglb_root / 'train.py',
                    '--dataset', official_dataset_name,
                    '--method', method,
                    '--backbone', 'GCN',
                ] + common_tail
                run_command(cmd, cglb_root, dry_run=args.dry_run)
                if not args.dry_run:
                    result_pkl = find_latest_result_pkl(
                        args.external_result_path,
                        official_dataset_name,
                        task_seq['name'],
                        method,
                    )
                    import_external_result(
                        python_bin,
                        PROJECT_ROOT,
                        result_pkl,
                        dataset,
                        DISPLAY_NAME_MAP[method],
                        results_root,
                        'cglb',
                        task_seq['path'],
                        dry_run=False,
                    )

            for method in DELOME_BASELINES:
                cmd = [
                    python_bin,
                    delome_root / 'train.py',
                    '--dataset', official_dataset_name,
                    '--method', method,
                    '--backbone', args.delome_backbone,
                ] + common_tail
                run_command(cmd, delome_root, dry_run=args.dry_run)
                if not args.dry_run:
                    result_pkl = find_latest_result_pkl(
                        args.external_result_path,
                        official_dataset_name,
                        task_seq['name'],
                        method,
                    )
                    import_external_result(
                        python_bin,
                        PROJECT_ROOT,
                        result_pkl,
                        dataset,
                        DISPLAY_NAME_MAP[method],
                        results_root,
                        'delome',
                        task_seq['path'],
                        dry_run=False,
                    )

    if not args.skip_report:
        cmd = [
            python_bin,
            PROJECT_ROOT / 'scripts' / 'report_exp1.py',
            '--results_root', results_root,
            '--report_root', report_root,
            '--datasets',
            *datasets,
        ]
        run_command(cmd, PROJECT_ROOT, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
