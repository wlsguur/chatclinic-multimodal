#!/usr/bin/env bash
# Clone YOLO-OB (code + cfg) and download SUN-pretrained weights (~300 MB) into ./yolo_ob_repo/.
# Not committed to git (exceeds GitHub's 100 MB limit); fetch once with this script.
# Requires `gdown` (pip install gdown) for the Google-Drive weights.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/yolo_ob_repo"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/seanyan62/YOLO-OB "$DEST"
fi
# SUN-pretrained weights (Google-Drive id from the YOLO-OB README, sec 2.5)
gdown 'https://drive.google.com/uc?id=1PIwmFLxTXice19-ENFkqg1N25ymZ1s7I' -O "$DEST/yolo_ob_sun.pth"
# Compatibility patch for modern torch (>=2.6 flips weights_only default to True):
sed -i 's/torch\.load(\([^)]*weights_path[^)]*\))/torch.load(\1, weights_only=False)/' "$DEST/models.py" 2>/dev/null || \
  echo "NOTE: could not auto-patch models.py; if torch.load fails, add weights_only=False manually."
echo "Done -> $DEST/yolo_ob_sun.pth"
