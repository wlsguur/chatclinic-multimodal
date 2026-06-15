#!/usr/bin/env bash
# Download the MONAI 'lung_nodule_ct_detection' bundle (~80 MB, Apache-2.0) into ./bundle/.
# Public weights are NOT committed to git; fetch them once with this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/bundle"
python -m monai.bundle download lung_nodule_ct_detection --bundle_dir "$HERE/bundle"
echo "Done -> $HERE/bundle/lung_nodule_ct_detection/models/model.pt"
