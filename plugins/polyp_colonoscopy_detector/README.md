# polyp_colonoscopy_detector

**Real-time** colon polyp detection (colonoscopy frames/video) — YOLO-OB (anchor-free, SUN-pretrained).
For single still frames prefer `gi_lesion_detector` (higher in-domain accuracy); use this for
video / streaming triage.

- **Entrypoint:** `plugins.polyp_colonoscopy_detector.logic:execute`
- **Input:** `payload["image_path"]` (colonoscopy frame); optional `conf` (default **0.15** —
  cross-dataset confidences run low/bimodal), `nms_iou` (default 0.3).
- **Output:** dict with `detections` (`box_xyxy`, `score`, `label="polyp"`), `artifacts.detection_review`,
  `studio_cards`, `draft_answer`, `provenance`.
- **source_types:** `image` · **approval_required:** false (67.7 FPS).

## Weights + repo code (NOT committed — ~300 MB > GitHub 100 MB limit)
```bash
pip install -r requirements.txt
bash download_weights.sh          # git clone YOLO-OB + gdown weights -> ./yolo_ob_repo/
```
`logic.py` adds `./yolo_ob_repo` to `sys.path` (it provides `models/` and `utils/`).
Override with `$CHATCLINIC_YOLOOB_REPO` / `$CHATCLINIC_YOLOOB_WEIGHTS`.

## Measured (RTX PRO 5000 Blackwell, GPU, 416px)
67.7 FPS (~15 ms/frame), <1 GB. Cross-dataset rough check (SUN→Kvasir, IoU>0.3): box recall 0.73 /
precision 0.87 (conservative lower bound). **License:** Apache-2.0.

## Notes
`download_weights.sh` applies a `torch.load(..., weights_only=False)` compatibility patch for torch ≥ 2.6.
Studio `detection_review` renderer is a frontend follow-up.
