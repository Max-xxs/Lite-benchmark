"""
Visualization helpers for Exp1 reports.
"""

import csv
import math
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


METRIC_LABELS = {
    'llm4gcl_avg_micro_mean': 'Abar',
    'llm4gcl_last_micro_mean': 'Afinal',
    'lower_triangle_avg_macro_mean': 'RowAvg',
    'lower_triangle_last_macro_mean': 'RowFinal',
}


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _display_name(model_name):
    return model_name.replace('_', ' ').upper()


def plot_comparison_heatmaps(dataset, aggregated_results, output_path):
    """Plot one heatmap per model for a dataset."""
    model_names = list(sorted(aggregated_results))
    if not model_names:
        raise ValueError("aggregated_results must not be empty")

    n_models = len(model_names)
    ncols = min(3, n_models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad(color='white')

    image = None
    for ax, model_name in zip(axes.flatten(), model_names):
        matrix = np.asarray(
            aggregated_results[model_name]['mean_matrix'],
            dtype=float,
        ) * 100.0
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
        num_sessions = matrix.shape[0]
        ax.set_title(_display_name(model_name), fontsize=12, fontweight='bold')
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Tasks')
        ax.set_xticks(np.arange(num_sessions))
        ax.set_yticks(np.arange(num_sessions))
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, num_sessions, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_sessions, 1), minor=True)
        ax.tick_params(which='minor', bottom=False, left=False)

    for ax in axes.flatten()[n_models:]:
        ax.axis('off')

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    ntrials = next(iter(aggregated_results.values()))['ntrials']
    fig.suptitle(
        f'Comparison Heatmaps on {dataset} (Avg over {ntrials} trials)',
        fontsize=16,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_curves(dataset, aggregated_results, output_path,
                           curve_key, ylabel, title_prefix):
    """Plot one line per model for a dataset."""
    if not aggregated_results:
        raise ValueError("aggregated_results must not be empty")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    style_cycle = [
        ('o', '-'),
        ('s', '--'),
        ('^', '-.'),
        ('v', ':'),
        ('D', '--'),
        ('p', '-.'),
        ('*', '--'),
        ('X', '-'),
        ('h', '-.'),
    ]

    for idx, model_name in enumerate(sorted(aggregated_results)):
        summary = aggregated_results[model_name]
        mean_curve = np.asarray(summary[curve_key], dtype=float) * 100.0
        std_curve = np.asarray(
            summary[curve_key.replace('mean_', 'std_')],
            dtype=float,
        ) * 100.0

        valid = ~np.isnan(mean_curve)
        if valid.sum() == 0:
            continue
        x = np.arange(1, valid.sum() + 1)
        y = mean_curve[valid]
        y_std = std_curve[valid]
        marker, linestyle = style_cycle[idx % len(style_cycle)]

        ax.plot(
            x,
            y,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=_display_name(model_name),
        )
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.12)

    ntrials = next(iter(aggregated_results.values()))['ntrials']
    ax.set_title(
        f'{title_prefix} on {dataset} (Avg over {ntrials} trials)',
        fontsize=16,
    )
    ax.set_xlabel('Tasks Learned', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(bottom=0)

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_summary_markdown(aggregated, output_path,
                           dataset_order=None, model_order=None,
                           metric_keys=None):
    """Write a CGLB-style markdown summary table."""
    dataset_names = dataset_order or sorted(aggregated)
    if metric_keys is None:
        metric_keys = [
            'llm4gcl_avg_micro_mean',
            'llm4gcl_last_micro_mean',
            'lower_triangle_avg_macro_mean',
            'lower_triangle_last_macro_mean',
        ]

    if model_order is None:
        model_names = sorted({
            model
            for dataset in dataset_names
            for model in aggregated.get(dataset, {})
        })
    else:
        model_names = model_order

    header = ['Model']
    for dataset in dataset_names:
        for metric_key in metric_keys:
            header.append(f'{dataset}:{METRIC_LABELS[metric_key]}')

    rows = [header]
    for model_name in model_names:
        row = [model_name]
        for dataset in dataset_names:
            summary = aggregated.get(dataset, {}).get(model_name)
            for metric_key in metric_keys:
                if summary is None:
                    row.append('-')
                    continue
                std_key = metric_key.replace('_mean', '_std')
                mean_val = summary['metrics'][metric_key] * 100.0
                std_val = summary['metrics'][std_key] * 100.0
                if np.isnan(mean_val) or np.isnan(std_val):
                    row.append('-')
                else:
                    row.append(f'{mean_val:.2f}±{std_val:.2f}')
        rows.append(row)

    lines = []
    lines.append('| ' + ' | '.join(rows[0]) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
    for row in rows[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_summary_csv(aggregated, output_path,
                      dataset_order=None, model_order=None):
    """Write a long-format CSV with all summary metrics."""
    dataset_names = dataset_order or sorted(aggregated)
    if model_order is None:
        model_names = sorted({
            model
            for dataset in dataset_names
            for model in aggregated.get(dataset, {})
        })
    else:
        model_names = model_order

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    fieldnames = [
        'dataset',
        'model',
        'ntrials',
        'num_sessions',
        'llm4gcl_avg_micro_mean',
        'llm4gcl_avg_micro_std',
        'llm4gcl_last_micro_mean',
        'llm4gcl_last_micro_std',
        'lower_triangle_avg_macro_mean',
        'lower_triangle_avg_macro_std',
        'lower_triangle_last_macro_mean',
        'lower_triangle_last_macro_std',
    ]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in dataset_names:
            for model in model_names:
                summary = aggregated.get(dataset, {}).get(model)
                if summary is None:
                    continue
                row = {
                    'dataset': dataset,
                    'model': model,
                    'ntrials': summary['ntrials'],
                    'num_sessions': summary['num_sessions'],
                }
                row.update(summary['metrics'])
                writer.writerow(row)
