# GRATrack Server Migration

This migration snapshot freezes the workspace on 2026-07-10 while branch
`gratrack-scge-experiment` is checked out with uncommitted work preserved.

## Experiment State

- Existing stages 0-2, diagnostics, checkpoints, benchmark outputs, and analysis are preserved.
- The old V0-b method completed but failed the accuracy/mechanism gate.
- The redesigned SCGE experiment was approved.
- Training-integrity Task 1 is complete: six-channel grayscale handling and GIoU error propagation are implemented; four CPU/CUDA tests and independent review pass.
- Counterfactual sidecar Task 2 has not started. No partial sidecar implementation exists.
- No process from this repository was running when migration packaging began.

## Bundle Components

```text
00_metadata/             manifests, environment, Git bundle, restore helpers
01_code/                 repository, .git, dirty files, logs and outputs
02_python_env/           current .venv (same-platform fallback)
03_weights_and_runs/     OSTrack pretrained files and GRATrack/baseline runs
04_codex_memory/         project sessions and non-secret state records
10_data_lasher/          LasHeR training/testing data
11_data_rgbt234/         RGBT234
12_data_depthtrack/      DepthTrack
13_data_visevent/        VisEvent
14_data_vot22_rgbd/      VOT22-RGBD
```

Each archive is a compressed stream split into files ending in `.part-NNNN`.
Verify `SHA256SUMS` before extracting.

## Security Boundary

The Codex archive intentionally excludes authentication/account files,
installation IDs, credentials, and shell snapshots. Sign in to Codex again on
the destination server.

## Restore Order

1. Verify checksums with `sha256sum -c SHA256SUMS` in each component directory.
2. Restore code.
3. Recreate the Python environment from `00_metadata/pip-freeze.txt`; use the
   archived `.venv` only when OS, architecture, Python, CUDA, and install path match.
4. Restore weights/runs and datasets.
5. Run `restore_bundle.sh` or `relocate_paths.py` to update machine-specific paths.
6. Restore Codex records after installing and signing in to Codex.
7. Run unit tests before resuming SCGE Task 2.

The authoritative implementation plan is
`docs/superpowers/plans/2026-07-10-gratrack-scge.md`.

