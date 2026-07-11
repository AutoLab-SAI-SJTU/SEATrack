# GRATrack Server Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a self-contained, resumable, checksum-verified migration bundle for the current GRATrack workspace, useful checkpoints, required RGB-X datasets, and project-relevant Codex records.

**Architecture:** Keep components independent so code can be restored before the large datasets arrive. Archive streams use `tar | zstd -T8 -1 | split -b 10G`; each component gets source inventory, part-level SHA-256, and a restore command. Credentials and machine identities are excluded from the Codex package.

**Tech Stack:** GNU tar 1.35, zstd 1.5.7, GNU split, SHA-256, Git bundle, Bash, Python 3.12.

## Global Constraints

- Pause SCGE implementation and launch no training or evaluation process.
- Preserve branch `gratrack-scge-experiment`, `.git`, tracked modifications, and untracked files.
- Include OSTrack pretrained weights, all GRATrack stage/clean-baseline run directories, logs, manifests, and checkpoints.
- Include LasHeR, RGBT234, DepthTrack, VisEvent, and VOT22-RGBD as separate archives.
- Include only project-relevant Codex sessions plus history/state/memory databases needed for continuity.
- Exclude `~/.codex/auth.json`, `~/.codex/accounts`, installation IDs, tokens, credentials, and shell snapshots.
- Split compressed streams into 10 GiB parts so interrupted transfer can resume per part.
- Do not delete source data or existing experiment output after packaging.

---

### Task 1: Freeze and Inventory

- [ ] Confirm no process from `/home/yufan/code/SEATrack-ProbAlign-VRE` is training or evaluating.
- [ ] Record `git status`, branch, commit, binary diff, untracked list, system/CUDA information, Python freeze, source sizes, file counts, and selected Codex session paths.
- [ ] Create a full Git bundle for committed history while retaining the dirty working tree in the code archive.

### Task 2: Migration Metadata and Restore Helpers

- [ ] Write `migration/README.md`, `migration/restore_bundle.sh`, and `migration/relocate_paths.py`.
- [ ] Document the paused SCGE state: Task 1 complete/reviewed; Task 2 counterfactual sidecar not started.
- [ ] Copy metadata helpers into the bundle's `00_metadata` directory.

### Task 3: Code, Environment, Weights, and Codex Archives

- [ ] Archive the repository with `.git`, logs, outputs, knowledge base, plans, tests, and dirty files; exclude only `.venv` and pretrained weights because they have dedicated components.
- [ ] Archive `.venv` separately and include a reproducible `pip freeze`; document that recreation is preferred on different CUDA/OS versions.
- [ ] Archive both OSTrack pretrained files and six selected training-run directories.
- [ ] Archive the seven project-relevant Codex JSONL sessions, history, SQLite state/memory/goal snapshots, AGENTS/rules/hooks, and no authentication material.

### Task 4: Dataset Archives

- [ ] Archive `/mnt/tipro4t/data/LasHeR0327`.
- [ ] Archive `/mnt/tipro4t/data/RGB_T234`.
- [ ] Archive `/mnt/tipro4t/data/DepthTrack`.
- [ ] Archive `/mnt/tipro4t/data/VisEvent`.
- [ ] Archive `/mnt/tipro4t/data/VOT22-RGBD`.
- [ ] Check free space before each archive and stop before filesystem exhaustion.

### Task 5: Integrity and Delivery

- [ ] Generate SHA-256 for every archive part and metadata file.
- [ ] Reconstruct and run `zstd -t` on each component stream.
- [ ] Compare archived top-level names against the declared component manifest.
- [ ] Report bundle path, total size, part counts, any omitted item, and exact restore commands.

