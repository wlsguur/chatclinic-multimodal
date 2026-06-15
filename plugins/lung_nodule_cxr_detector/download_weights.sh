#!/usr/bin/env bash
# Clone the NODE21 detection baseline and pull model.pth (~158 MB, via git-lfs) into ./node21_repo/.
# Public weights are NOT committed to git (exceeds GitHub's 100 MB limit); fetch once with this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/node21_repo"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/node21challenge/node21_detection_baseline "$DEST"
fi
cd "$DEST"
git lfs install
git lfs pull
echo "Done -> $DEST/model.pth"
