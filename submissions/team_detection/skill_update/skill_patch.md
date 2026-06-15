# Skill Patch Proposal — Detection suite (lung nodules + GI lesions)

> Target section of the master `SKILL.md`: **`## Orchestration rules`** only.
> (Welcome / Help messages are instructor-maintained and untouched.)
> This patch adds ONE self-contained routing block that selects among four detection tools.

## Tools introduced
- `lung_nodule_ct_detector` — 3D chest CT pulmonary nodule detection (MONAI 3D RetinaNet)
- `lung_nodule_cxr_detector` — 2D chest X-ray lung nodule detection (NODE21 Faster R-CNN)
- `polyp_colonoscopy_detector` — real-time colonoscopy polyp detection (YOLO-OB)
- `gi_lesion_detector` — accuracy-oriented endoscopic lesion detection on still frames (fine-tuned YOLO11s)

## Proposed addition to `## Orchestration rules`

```md
### Detection: lung nodules and GI lesions

Applies when the user intent is to **detect / find / screen / localize** abnormalities on a medical
image (keywords: detect, find, screen, localize, nodule, polyp, lesion, "is there a ...").
Selection is a 5-step decision; resolve top-to-bottom and stop at the first matching route.

**Step 1 — Source family (from the auto-detected source type).**
- DICOM **or** NIfTI (`.nii`/`.nii.gz`) that is a **chest CT volume**  →  `lung_nodule_ct_detector`.
- Raster image (PNG/JPG/JPEG/TIFF/BMP/WEBP)  →  go to Step 2.
- (A 3D volume only ever routes to the CT tool; the 2D detectors never run on a volume.)

**Step 2 — Anatomical region of the raster image** (use `image_review_tool`'s modality hint first,
plus the user's wording).
- frontal **chest X-ray**  →  `lung_nodule_cxr_detector`.
- **colonoscopy / lower-GI endoscopy** frame  →  go to Step 3.
- region unsupported / unclear  →  run no detector; ask the user to confirm the region.

**Step 3 — Colonoscopy tools: choose by CLINICAL CONTEXT, not by source type** (both take the
same colonoscopy frame, so context decides):
- **single still frame**, accuracy / report-grade priority  →  `gi_lesion_detector`  (priority 78).
- **video / live stream / many frames / real-time triage**  →  `polyp_colonoscopy_detector`  (priority 70).
- tie-break: default to `gi_lesion_detector` (higher priority + higher in-domain accuracy);
  switch to `polyp_colonoscopy_detector` when the request mentions video / stream / real-time /
  frame-rate, or when latency matters more than accuracy.

**Step 4 — Host / runtime gate** (read each tool's `runtime` block).
- `lung_nodule_ct_detector` needs ~9 GB GPU and ~8 s per volume: on a GPU host run after approval;
  on a CPU-only host, warn that it will be slow (`allow_cpu_fallback=true`) and confirm before running.
- the three 2D detectors are CPU-friendly: may run on CPU and without GPU.

**Step 5 — Approval.**
- `lung_nodule_ct_detector`: **approval required** (GPU + runtime cost).
- 2D detectors: **approval not required** (fast/cheap, may auto-run on intake).
- All detection outputs are decision-support shown for clinician review — never presented as a diagnosis.

**Ordering / dependencies.** For raster images, run `image_review_tool` first (intake + modality hint),
then the selected detector. Each detector emits a `*_detection_overlay` artifact for the Studio
bounding-box renderer; the CT tool emits a 3D-box overlay (`xyzxyz` voxel coords).
```

## Per-tool summary (template fields)

| Field | ct | cxr | polyp | gi_lesion |
|---|---|---|---|---|
| Source type | dicom / nifti | image | image | image |
| Modality | medical-image | medical-image | medical-image | medical-image |
| Region | chest CT (3D) | chest CXR (2D) | colonoscopy | endoscopy (still) |
| Recommended stage | post-intake | post-intake | post-intake | post-intake |
| Depends on | none | image_review_tool | image_review_tool | image_review_tool |
| Approval | required | not required | not required | not required |
| Produces | nodule_detection_overlay_3d | nodule_detection_overlay_2d | polyp_detection_overlay | lesion_detection_overlay |
| Host | gpu preferred (cpu slow) | cpu/gpu | gpu preferred (cpu ok) | gpu preferred (cpu ok) |

## Example routing notes
- *"Screen this chest CT for nodules"* + a DICOM/NIfTI volume → Step 1 → `lung_nodule_ct_detector`
  (ask approval first; ~8 s on GPU).
- *"Any lung nodules on this X-ray?"* + a PNG → Step 1 raster → Step 2 chest CXR → `lung_nodule_cxr_detector` (auto-runs).
- *"Find polyps in this colonoscopy snapshot"* + a JPG → Step 2 colonoscopy → Step 3 still frame →
  `gi_lesion_detector`.
- *"Run real-time polyp detection on this colonoscopy video"* → Step 3 video/real-time →
  `polyp_colonoscopy_detector`.
