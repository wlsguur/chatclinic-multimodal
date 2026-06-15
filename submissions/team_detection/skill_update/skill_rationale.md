# Skill Rationale — Detection suite

## Why these tools should exist
Detection (localizing abnormalities with bounding boxes) is a distinct, high-value clinical task from
classification or segmentation. We cover the two organ systems with the strongest public detection
benchmarks and ready models — **lung** (CT + CXR) and **lower-GI / colon** (colonoscopy) — so the
platform can answer "where is the nodule / polyp / lesion?" across the modalities users actually upload.

## Why the orchestrator should call each one (explicit, non-overlapping triggers)
The four tools are chosen by a layered decision so selection is deterministic and explainable:
1. **Source family** cleanly separates the CT tool (3D volume) from the three 2D tools — a 3D detector
   and a 2D detector are not interchangeable, so this is a hard split, not a preference.
2. **Anatomical region** separates chest-CXR from colonoscopy among raster images. We rely on the
   existing `image_review_tool` modality hint rather than guessing, which keeps the routing honest and
   reuses platform infrastructure instead of duplicating intake.
3. **Clinical context** is the interesting case: `polyp_colonoscopy_detector` and `gi_lesion_detector`
   accept the *same* colonoscopy frame, so source type cannot decide between them. We make the choice on
   **accuracy-vs-latency**: the fine-tuned `gi_lesion_detector` (in-domain val mAP@50 0.908) is the
   default for single still frames and report-grade output; the real-time `polyp_colonoscopy_detector`
   (67.7 FPS) is selected for video / streaming / many-frame triage. This teaches that tool choice can
   depend on the *task context*, not only the data type.

## Why approval differs by tool (cost-based, justified)
Approval gates **running cost**, not output trust. `lung_nodule_ct_detector` needs ~9 GB GPU and ~8 s
per 3D volume, so it requires approval and is gated on host capability (it warns before a slow CPU run).
The three 2D detectors are millisecond-to-sub-second and CPU-friendly, so they may auto-run on intake.
Independently, *every* detection output is framed as decision-support for clinician review — never a
diagnosis — regardless of the approval gate.

## Why this is more than four isolated tools
The value is the **orchestration policy**: one block defines, for every supported input, exactly which
detector runs, in what order (intake → modality hint → detector), with which approval and host rules,
and how the choice changes when the same colonoscopy frame arrives in a still-image vs video context.
Each tool also produces a *complementary, non-duplicate* artifact (3D box overlay vs 2D box overlay vs
polyp/lesion overlay) that the Studio renderer can display.

## Educational value
The suite demonstrates the core lesson that agentic AI is system design, not a single model: combining
ready-to-use public models (CT/CXR/polyp) with one in-domain **fine-tuned** model (GI lesion), and
wiring them with explicit, auditable selection logic that reasons over source type, anatomy, clinical
context, runtime budget, and approval — exactly the multi-tool-plus-routing pattern the course rewards.

## Honest limitations (so reviewers can trust the claims)
- CT/CXR/polyp "performance" figures are functional GPU verification + rough/sanity checks (plus the
  MONAI bundle's own reported mAP), not fresh official FROC/leaderboard numbers.
- `gi_lesion_detector` is currently single-class (polyp, from Kvasir-SEG); broadening to multi-class GI
  findings is a retrain with `nc>1` on the same pipeline.
- `gi_lesion_detector` uses Ultralytics YOLO (**AGPL-3.0**); flagged in its `tool.json` and README.
