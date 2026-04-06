"""
Download Exp1 raw datasets from official source websites and extract them into
loader-compatible directories for the local benchmark and the official baselines.
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    'cora-full': {
        'kind': 'dgl',
        'url': 'https://data.dgl.ai/dataset/cora_full.zip',
        'target_relpath': Path('cora_full'),
    },
    'coauthor-cs': {
        'kind': 'dgl',
        'url': 'https://data.dgl.ai/dataset/coauthor_cs.zip',
        'target_relpath': Path('coauthor_cs'),
    },
    'amazon-computers': {
        'kind': 'dgl',
        'url': 'https://data.dgl.ai/dataset/amazon_co_buy_computer.zip',
        'target_relpath': Path('amazon_co_buy_computer'),
    },
    'reddit': {
        'kind': 'dgl',
        'url': 'https://data.dgl.ai/dataset/reddit.zip',
        'target_relpath': Path('reddit'),
    },
    'ogbn-arxiv': {
        'kind': 'ogb',
        'url': 'http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip',
        'download_name': 'arxiv',
        'target_relpath': Path('ogb_downloaded') / 'ogbn_arxiv',
    },
    'ogbn-products': {
        'kind': 'ogb',
        'url': 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip',
        'download_name': 'products',
        'target_relpath': Path('ogb_downloaded') / 'ogbn_products',
    },
}

DEFAULT_DATASETS = list(DATASET_SPECS.keys())


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"[skip] archive exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"          -> {destination}")

    with urllib.request.urlopen(url) as response, destination.open('wb') as f:
        shutil.copyfileobj(response, f)


def extract_dgl_archive(archive_path: Path, target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if force:
            shutil.rmtree(target_dir)
        else:
            print(f"[skip] extracted dir exists: {target_dir}")
            return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive_path} -> {target_dir}")
    with ZipFile(archive_path, 'r') as zf:
        zf.extractall(target_dir)


def extract_ogb_archive(
    archive_path: Path,
    parent_dir: Path,
    downloaded_name: str,
    target_dir: Path,
    force: bool,
) -> None:
    if target_dir.exists():
        if force:
            shutil.rmtree(target_dir)
        else:
            print(f"[skip] extracted dir exists: {target_dir}")
            return

    parent_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = parent_dir / downloaded_name

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    print(f"[extract] {archive_path} -> {parent_dir}")
    with ZipFile(archive_path, 'r') as zf:
        zf.extractall(parent_dir)

    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted directory {extracted_dir} after unpacking {archive_path}"
        )

    print(f"[rename] {extracted_dir} -> {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_dir), str(target_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Download official raw Exp1 datasets and unpack them locally'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
    )
    parser.add_argument(
        '--raw_root',
        type=Path,
        default=PROJECT_ROOT.parent / 'external_runs' / 'raw',
        help='Root directory to store downloaded raw data',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download archives and overwrite extracted folders',
    )
    parser.add_argument(
        '--remove_archives',
        action='store_true',
        help='Delete zip archives after successful extraction',
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        target_dir = args.raw_root / spec['target_relpath']
        archive_name = Path(spec['url']).name
        archive_path = args.raw_root / archive_name

        download_file(spec['url'], archive_path, force=args.force)

        if spec['kind'] == 'dgl':
            extract_dgl_archive(archive_path, target_dir, force=args.force)
        else:
            extract_ogb_archive(
                archive_path=archive_path,
                parent_dir=args.raw_root / 'ogb_downloaded',
                downloaded_name=spec['download_name'],
                target_dir=target_dir,
                force=args.force,
            )

        if args.remove_archives and archive_path.exists():
            print(f"[cleanup] removing archive {archive_path}")
            archive_path.unlink()

    print("\nDone. Upload this directory to the cloud:")
    print(args.raw_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
