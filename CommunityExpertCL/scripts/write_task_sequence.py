"""
Write an explicit class-session JSON file for Exp1.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_known_class_ids, resolve_class_splits


def main():
    parser = argparse.ArgumentParser(
        description='Write an explicit task/session sequence JSON file',
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Override the known class count for this dataset')
    parser.add_argument('--session_strategy', type=str, default='balanced',
                        choices=['legacy', 'balanced'])
    parser.add_argument('--num_sessions', type=int, default=None)
    parser.add_argument('--session_multiplier', type=float, default=2.0)
    parser.add_argument('--max_experts', type=int, default=8)
    args = parser.parse_args()

    class_ids = get_known_class_ids(args.dataset, num_classes=args.num_classes)
    class_splits, meta = resolve_class_splits(
        dataset=args.dataset,
        class_ids=class_ids,
        strategy=args.session_strategy,
        num_sessions=args.num_sessions,
        session_multiplier=args.session_multiplier,
        max_experts=args.max_experts,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset': args.dataset,
        'class_splits': class_splits,
        'meta': {
            **meta,
            'max_experts': args.max_experts,
            'num_classes': len(class_ids),
        },
    }
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'Wrote task sequence to {output_path}')
    print(json.dumps(payload['meta'], indent=2))


if __name__ == '__main__':
    main()
