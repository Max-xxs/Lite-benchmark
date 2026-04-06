# Exp1 Cloud Layout

Recommended upload layout on the cloud notebook:

```text
workspace/
  CommunityExpertCL/
  external/
    CGLB/
    DeLoMe/
```

The default one-click runner assumes exactly this structure.

## Files/Folders To Upload

Upload these directories in full:

- `CommunityExpertCL/`
- `external/CGLB/`
- `external/DeLoMe/`

The runner will create these at runtime if missing:

- `external_runs/raw/`
- `external_runs/data/`
- `external_runs/results/`

## Local Raw Download

To pre-download the official raw archives locally and unpack them into the
shared raw-data directory:

```bash
python3 scripts/download_exp1_raw_data.py
```

This fills `external_runs/raw/`, which is also the default raw root used by the
one-click runner for both `LiteExpertCL` and the official baselines.

## Main Entry

From inside `CommunityExpertCL/`:

```bash
python3 scripts/run_exp1_benchmark.py --gpu 0
```

## Dry Run

To inspect commands without launching training:

```bash
python3 scripts/run_exp1_benchmark.py --dry_run
```

## Current Default SVD

Applied only to `LiteExpertCL`:

- `cora-full -> 512`
- `coauthor-cs -> 256`
- `ogbn-arxiv -> 0`
- other datasets -> `0`

## Notes

- Official `CGLB` baselines are run with `GCN`.
- `LiteExpertCL` now defaults to `--data_protocol cglb` inside the one-click runner, so `cora-full`, `reddit`, `ogbn-arxiv`, and `ogbn-products` follow the same source family as the official benchmark loaders.
- `DeLoMe` is currently run with `GCN` in the one-click benchmark for backbone unification.
- If needed, switch it with `--delome_backbone SGC`.
- `SimGCL` is not included yet.
