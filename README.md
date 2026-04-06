# Lite-benchmark

Benchmark codebase for the LiteExpert Exp1 pipeline.

## Repository Layout

```text
CommunityExpertCL/
external/
  CGLB/
    NCGL/
  DeLoMe/
```

## Included

- `CommunityExpertCL/`: benchmark runner, LiteExpert implementation, reporting scripts, and docs
- `external/CGLB/NCGL/`: benchmark-aligned CGLB NCGL baseline code
- `external/DeLoMe/`: benchmark-aligned DeLoMe code

## Excluded

- raw datasets
- generated results
- local caches and temporary files

## Typical Run Layout

The code expects a workspace like:

```text
workspace/
  Lite-benchmark/
  external_runs/
    raw/
```

Then run from `CommunityExpertCL/`:

```bash
python scripts/run_exp1_benchmark.py --gpu 0
```

For cloud usage details, see:

- `CommunityExpertCL/docs/run_exp1_cloud.md`
- `CommunityExpertCL/docs/exp1_baseline_alignment.md`
