# Exp1 Baseline Alignment Notes

This note records which parts of the official baselines were kept intact and which parts were adapted only at the input or I/O layer.

## Scope

- Included now: `CGLB` baselines, `DeLoMe`, `LiteExpert`
- Deferred for now: `SimGCL`

## Official-Code Changes

### CGLB

Files:
- `external/CGLB/NCGL/train.py`
- `external/CGLB/NCGL/pipeline.py`

Changes:
- Added `--task_seq_file` to load an explicit class-session partition from JSON.
- Added `--task_seq_name` so result filenames encode the session schedule.
- Replaced the internal fixed-width task construction with `resolve_task_seq(...)`.

Impact assessment:
- This does **not** change model architecture, optimizer, loss, sampler, replay logic, or evaluation formula.
- This **does** change how tasks are defined, because Exp1 requires a session rule that cannot always be expressed by `n_cls_per_task`.
- Therefore the adaptation is an **input-setting change**, not an algorithmic change.

### DeLoMe

Files:
- `external/DeLoMe/train.py`
- `external/DeLoMe/pipeline.py`
- `external/DeLoMe/Backbones/utils.py`

Changes:
- Added the same `--task_seq_file` and `--task_seq_name` interface as above.
- Patched OGB data roots to use `args.ori_data_path/ogb_downloaded` instead of a hard-coded lab path.
- Added a benchmark-level option to run `DeLoMe` with a unified `GCN` backbone instead of the README's example `SGC` command when strict backbone unification is required.

Impact assessment:
- The path fix is purely environmental and does **not** change training behavior.
- The task-sequence bridge is the same kind of **input-setting change** as in CGLB.
- The `GCN` switch is a **backbone-setting change** and should be reported explicitly when used.

## What Remains Official

The following are still taken from the official implementations:

- Backbone choice and model code
- Training loops
- Replay or regularization logic
- Evaluation matrices
- CGLB-style AP/AF outputs

## Current Gaps

- `coauthor-cs` and `amazon-computers` are now supported through a dataset-loader adapter in the official codebase, so these runs are no longer "untouched default official settings".
- Official `CGLB/DeLoMe` outputs do not directly provide LLM4GCL-style joint micro metrics, so those stay unavailable unless a secondary evaluator is added.
- `SimGCL` is intentionally postponed.

## Recommended Reporting Language

When writing the paper or internal notes, describe these baselines as:

`Official implementation with explicit Exp1 task partition adapter`

This wording is precise and avoids overstating that the baselines were run in the untouched default benchmark setting.
