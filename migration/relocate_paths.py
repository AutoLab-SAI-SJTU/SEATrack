#!/usr/bin/env python3
import argparse
from pathlib import Path


PATH_FILES = (
    "lib/train/admin/local.py",
    "lib/test/evaluation/local.py",
    "Depthtrack_workspace/trackers.ini",
    "VOT22RGBD_workspace/trackers.ini",
)


def main():
    parser = argparse.ArgumentParser(description="Relocate GRATrack machine-specific paths")
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    replacements = (
        ("/home/yufan/code/SEATrack-ProbAlign-VRE", str(repo_root)),
        ("/home/yufan/code/SEATrack", str(repo_root)),
    )

    for relative_path in PATH_FILES:
        path = repo_root / relative_path
        text = path.read_text()
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)
            print(f"updated {path}")
        else:
            print(f"unchanged {path}")


if __name__ == "__main__":
    main()

