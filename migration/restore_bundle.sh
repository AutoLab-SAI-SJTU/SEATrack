#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 BUNDLE_DIR INSTALL_PARENT DATA_ROOT RUNS_ROOT" >&2
    exit 2
fi

BUNDLE_DIR=$(realpath "$1")
INSTALL_PARENT=$(realpath -m "$2")
DATA_ROOT=$(realpath -m "$3")
RUNS_ROOT=$(realpath -m "$4")

extract_parts() {
    local component_dir=$1
    local destination=$2
    local first_part
    first_part=$(find "$component_dir" -maxdepth 1 -name '*.part-0000' -print -quit)
    if [[ -z "$first_part" ]]; then
        echo "missing part-0000 in $component_dir" >&2
        return 1
    fi
    mkdir -p "$destination"
    cat "${first_part%.part-0000}".part-* | zstd -d -c | tar -xf - -C "$destination"
}

for component in "$BUNDLE_DIR"/{01_code,02_python_env,03_weights_and_runs,04_codex_memory,10_data_lasher,11_data_rgbt234,12_data_depthtrack,13_data_visevent,14_data_vot22_rgbd}; do
    (cd "$component" && sha256sum -c SHA256SUMS)
done

extract_parts "$BUNDLE_DIR/01_code" "$INSTALL_PARENT"
REPO_ROOT="$INSTALL_PARENT/SEATrack-ProbAlign-VRE"
extract_parts "$BUNDLE_DIR/02_python_env" "$REPO_ROOT"
WEIGHTS_STAGE=$(mktemp -d)
trap 'rm -rf "$WEIGHTS_STAGE"' EXIT
extract_parts "$BUNDLE_DIR/03_weights_and_runs" "$WEIGHTS_STAGE"

mkdir -p "$REPO_ROOT/pretrained" "$RUNS_ROOT"
cp -a "$WEIGHTS_STAGE/repo/pretrained/." "$REPO_ROOT/pretrained/"
cp -a "$WEIGHTS_STAGE/runs/." "$RUNS_ROOT/"

extract_parts "$BUNDLE_DIR/10_data_lasher" "$DATA_ROOT"
extract_parts "$BUNDLE_DIR/11_data_rgbt234" "$DATA_ROOT"
extract_parts "$BUNDLE_DIR/12_data_depthtrack" "$DATA_ROOT"
extract_parts "$BUNDLE_DIR/13_data_visevent" "$DATA_ROOT"
extract_parts "$BUNDLE_DIR/14_data_vot22_rgbd" "$DATA_ROOT"

mkdir -p "$REPO_ROOT/datasets"
ln -sfn "$DATA_ROOT/LasHeR0327" "$REPO_ROOT/datasets/LasHeR"
ln -sfn "$DATA_ROOT/RGB_T234" "$REPO_ROOT/datasets/RGBT234"
ln -sfn "$DATA_ROOT/DepthTrack" "$REPO_ROOT/datasets/DepthTrack"
ln -sfn "$DATA_ROOT/VisEvent" "$REPO_ROOT/datasets/VisEvent"
ln -sfn "$DATA_ROOT/VOT22-RGBD" "$REPO_ROOT/datasets/VOT22-RGBD"

python3 "$REPO_ROOT/migration/relocate_paths.py" --repo-root "$REPO_ROOT"

echo "Code and experiment assets restored to $REPO_ROOT"
echo "Runs restored to $RUNS_ROOT"
echo "Datasets restored to $DATA_ROOT"
echo "Restore Codex memory separately after installing and authenticating Codex."
