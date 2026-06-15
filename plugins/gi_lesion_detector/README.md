# gi_lesion_detector

**Accuracy-oriented** GI/endoscopic lesion detection on **still** endoscopy frames —
YOLO11s **fine-tuned in-domain** on Kvasir-SEG (class `lesion`). This is the team's trained tool.
The weights are **committed in-repo** (`./weights/best.pt`, ~19 MB) because they are our own artifact
with no public download source.

- **Entrypoint:** `plugins.gi_lesion_detector.logic:execute`
- **Input:** `payload["image_path"]` (endoscopy still frame); optional `conf` (default 0.25).
- **Output:** dict with `detections` (`box_xyxy`, `score`, `label="lesion"`), `artifacts.detection_review`,
  `studio_cards`, `draft_answer`, `provenance`.
- **source_types:** `image` · **approval_required:** false (~1 ms/img on GPU).

## Setup
```bash
pip install -r requirements.txt   # ultralytics (AGPL-3.0)
# weights already present at ./weights/best.pt (committed)
```
Override location with `$CHATCLINIC_GI_WEIGHTS`.

## Measured (RTX PRO 5000 Blackwell)
Trained 100 epochs (~10.8 min, GPU). **Val (200 imgs/209 inst): mAP@50 0.908, mAP@50-95 0.735,
P 0.873, R 0.889.** Inference ~1 ms/img.

## Notes / follow-ups
- Currently single class `lesion` (= colon polyp, from Kvasir-SEG); extend to multi-class GI findings
  by retraining with `nc>1` (HyperKvasir bbox etc.) — same pipeline.
- **License:** Ultralytics YOLO is **AGPL-3.0**; Kvasir-SEG research/educational (cite Jha et al., MMM 2020).
- Studio `detection_review` renderer is a frontend follow-up; `execute()` runs and returns boxes today.
