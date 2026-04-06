"""
Generate Exp1 figures and summary tables from saved trial results.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.io import load_and_aggregate_results
from analysis.plotting import (
    plot_comparison_curves,
    plot_comparison_heatmaps,
    write_summary_csv,
    write_summary_markdown,
)


def main():
    parser = argparse.ArgumentParser(description='Generate Exp1 report assets')
    parser.add_argument('--results_root', type=str, default='./results/exp1')
    parser.add_argument('--report_root', type=str, default='./results/exp1/reports')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--models', nargs='*', default=None)
    args = parser.parse_args()

    aggregated = load_and_aggregate_results(
        args.results_root,
        datasets=args.datasets,
        models=args.models,
    )

    if not aggregated:
        raise ValueError(
            f'No trial records found under {args.results_root}. '
            'Run main.py first.'
        )

    report_root = Path(args.report_root)
    for dataset, model_summaries in aggregated.items():
        ntrials = next(iter(model_summaries.values()))['ntrials']
        plot_comparison_heatmaps(
            dataset,
            model_summaries,
            report_root / f'comparison_heatmaps_{dataset}_avg{ntrials}.png',
        )
        plot_comparison_curves(
            dataset,
            model_summaries,
            report_root / f'comparison_learning_dynamics_{dataset}_avg{ntrials}.png',
            curve_key='mean_lower_triangle_row_mean',
            ylabel='Average Accuracy (%)',
            title_prefix='Learning Dynamics',
        )
        plot_comparison_curves(
            dataset,
            model_summaries,
            report_root / f'comparison_joint_micro_{dataset}_avg{ntrials}.png',
            curve_key='mean_joint_acc',
            ylabel='Micro Accuracy (%)',
            title_prefix='Joint Micro Accuracy',
        )

    write_summary_markdown(
        aggregated,
        report_root / 'summary_table_llm4gcl.md',
    )
    write_summary_csv(
        aggregated,
        report_root / 'summary_metrics.csv',
    )


if __name__ == '__main__':
    main()
