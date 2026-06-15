# lung_nodule_cxr_detector

Lung **nodule detection in 2D frontal chest X-ray** — NODE21 baseline (Faster R-CNN ResNet50-FPN).

- **Entrypoint:** `plugins.lung_nodule_cxr_detector.logic:execute`
- **Input:** `payload["image_path"]` — raster CXR (`.png/.jpg/.tif`) or `.mha/.mhd/.nii(.gz)`;
  optional `payload["nms_iou"]` (default 0.3).
- **Output:** dict with `detections` (`slice`, `box_xyxy_mm`, `probability`), `artifacts.detection_review`,
  `studio_cards`, `draft_answer`, `provenance`.
- **source_types:** `image` · **approval_required:** false (≈15 ms/img, CPU-friendly).

## Weights (NOT committed — public, ~158 MB > GitHub 100 MB limit)
```bash
pip install -r requirements.txt
bash download_weights.sh          # git clone + git lfs pull -> ./node21_repo/model.pth
```
Override location with `$CHATCLINIC_NODE21_REPO`.

## Measured (RTX PRO 5000 Blackwell, GPU)
~13–19 ms/image, ~1.2 GB. Sanity: local inference reproduced the repo's `expected_output.json`
exactly (29/29 boxes, IoU 1.0). **License:** Apache-2.0 / MIT.

## Notes
Expects a frontal CXR; raster inputs are normalized per-image by max intensity (matching the baseline).
Studio `detection_review` renderer is a frontend follow-up.
