# lung_nodule_ct_detector

Pulmonary **nodule detection in 3D chest CT** — MONAI `lung_nodule_ct_detection` bundle
(3D RetinaNet, LUNA16). Detection (3D boxes + scores), not segmentation.

- **Entrypoint:** `plugins.lung_nodule_ct_detector.logic:execute`
- **Input:** `payload["image_path"]` (`.nii.gz` / `.mhd`+`.raw` / DICOM series dir); optional
  `payload["resampled"]=true` if already at 0.703×0.703×1.25 mm; optional `max_detections`.
- **Output:** dict with `detections` (`box_xyzxyz_voxel`, `label`, `score`), `artifacts.detection_review`,
  `studio_cards`, `draft_answer`, `provenance`.
- **source_types:** `dicom`, `nifti` · **approval_required:** true (GPU/runtime cost).

## Weights (NOT committed — public, ~80 MB)
```bash
pip install -r requirements.txt
bash download_weights.sh          # -> ./bundle/lung_nodule_ct_detection/models/model.pt
```
Override location with `$CHATCLINIC_CT_BUNDLE`.

## Measured (RTX PRO 5000 Blackwell, GPU)
~7.6 s/volume, ~8.8 GB. Sanity: high-score box centers 0.7–0.9 mm from GT on LUNA16 (3/3).
Bundle official: mAP 0.852 / mAR 0.998 (LUNA16 fold0). **License:** Apache-2.0.

## Notes / follow-ups
GPU strongly preferred (CPU fallback slow). Studio `detection_review` renderer (3D bbox overlay)
is a frontend follow-up; `execute()` itself runs and returns boxes today.
