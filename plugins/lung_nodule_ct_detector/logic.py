"""lung_nodule_ct_detector — ChatClinic plugin entrypoint.

Wraps the MONAI Model Zoo 'lung_nodule_ct_detection' bundle (3D RetinaNet, ResNet50-FPN,
LUNA16). Detects pulmonary nodules in 3D chest CT and returns 3D bounding boxes + scores
(detection, NOT segmentation).

Platform contract: `execute(payload: dict) -> dict`.
payload:
    image_path     : str  -- chest CT volume (.nii.gz / .mhd+raw / DICOM series dir)
    resampled      : bool -- optional; True if already at 0.703x0.703x1.25 mm spacing
    max_detections : int  -- optional cap on returned boxes (default 20)

Bundle weights are NOT committed (public, ~80MB). Run ./download_weights.sh first, or set
$CHATCLINIC_CT_BUNDLE to an existing bundle dir.
"""
from __future__ import annotations

import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get("CHATCLINIC_CT_BUNDLE",
                        os.path.join(HERE, "bundle", "lung_nodule_ct_detection"))

_NETWORK = None
_DETECTOR = None


def _load():
    global _NETWORK, _DETECTOR
    if _DETECTOR is not None:
        return
    import torch
    from monai.bundle import ConfigParser

    if not os.path.exists(os.path.join(BUNDLE, "models", "lung_nodule_ct_detector.pt")):
        raise FileNotFoundError(
            f"MONAI bundle weights not found under {BUNDLE}. "
            "Run plugins/lung_nodule_ct_detector/download_weights.sh or set $CHATCLINIC_CT_BUNDLE.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = ConfigParser()
    parser.read_config(os.path.join(BUNDLE, "configs", "inference.json"))
    parser["bundle_root"] = BUNDLE
    parser["device"] = device
    network = parser.get_parsed_content("network")
    detector = parser.get_parsed_content("detector")
    parser.get_parsed_content("detector_ops")
    sd = torch.load(os.path.join(BUNDLE, "models", "lung_nodule_ct_detector.pt"),
                    map_location="cpu", weights_only=True)
    network.load_state_dict(sd, strict=False)
    network.eval()
    detector.eval()
    _NETWORK, _DETECTOR = network, detector


def _preprocess(image_path, resampled):
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, EnsureTyped,
    )
    tfs = []
    if resampled:
        tfs.append(LoadImaged(keys="image"))
    else:
        tfs.append(LoadImaged(keys="image", reader="nibabelreader"))
    tfs += [EnsureChannelFirstd(keys="image"), Orientationd(keys="image", axcodes="RAS")]
    if not resampled:
        tfs.append(Spacingd(keys="image", pixdim=[0.703125, 0.703125, 1.25]))
    tfs += [
        ScaleIntensityRanged(keys="image", a_min=-1024.0, a_max=300.0,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="image"),
    ]
    return Compose(tfs)({"image": image_path})["image"]


def execute(payload: dict) -> dict:
    import contextlib
    import torch

    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    resampled = bool(payload.get("resampled", False))
    max_det = int(payload.get("max_detections", 20))

    _load()
    img = _preprocess(image_path, resampled)
    device = next(_NETWORK.parameters()).device
    img = img.to(device)
    try:
        ctx = torch.autocast(device_type="cuda", enabled=torch.cuda.is_available())
    except Exception:
        ctx = contextlib.nullcontext()
    t0 = time.time()
    with torch.no_grad(), ctx:
        preds = _DETECTOR([img], use_inferer=True)
    runtime = round(time.time() - t0, 2)

    pred = preds[0]
    bk = _DETECTOR.target_box_key
    lk = _DETECTOR.target_label_key
    sk = getattr(_DETECTOR, "pred_score_key", "label_scores")
    boxes = pred[bk].detach().cpu().tolist()
    labels = pred[lk].detach().cpu().tolist()
    scores = pred[sk].detach().cpu().tolist()
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:max_det]
    dets = [{
        "box_xyzxyz_voxel": [round(v, 1) for v in boxes[i]],
        "label": int(labels[i]),
        "score": round(float(scores[i]), 4),
    } for i in order]

    # --- save preprocessed CT + detection mask for 3D viewer overlay ---
    import nibabel as nib
    import numpy as np
    import re as _re

    def _to_numpy(t):
        if hasattr(t, 'detach'):
            t = t.detach().cpu()
        if hasattr(t, 'numpy'):
            return t.numpy()
        return np.array(t)

    try:
        # affine from MONAI MetaTensor
        raw_img = img.cpu()
        affine = _to_numpy(raw_img.meta.get('affine', np.eye(4)))
        vol_shape = raw_img.shape[1:]  # (X, Y, Z)

        # preprocessed CT
        base = _re.sub(r'(\.nii\.gz|\.nii)$', '', image_path)
        prep_path = base + '_prep.nii.gz'
        mask_path_out = base + '_nodule_mask.nii.gz'

        prep_data = _to_numpy(raw_img[0]).astype(np.float32)
        nib.save(nib.Nifti1Image(prep_data, affine), prep_path)

        # detection mask
        mask = np.zeros(vol_shape, dtype=np.uint8)
        for box in boxes[:max_det]:
            x1, y1, z1, x2, y2, z2 = [int(round(v)) for v in box]
            x1, x2 = sorted([max(0, min(x1, vol_shape[0]-1)), max(0, min(x2, vol_shape[0]-1))])
            y1, y2 = sorted([max(0, min(y1, vol_shape[1]-1)), max(0, min(y2, vol_shape[1]-1))])
            z1, z2 = sorted([max(0, min(z1, vol_shape[2]-1)), max(0, min(z2, vol_shape[2]-1))])
            mask[x1:x2+1, y1:y2+1, z1:z2+1] = 1
        nib.save(nib.Nifti1Image(mask, affine), mask_path_out)
    except Exception as _e:
        prep_path = None
        mask_path_out = None
        print(f"[lung_nodule_ct_detector] mask generation failed: {_e}")

    name = os.path.basename(image_path)
    draft = (f"Lung-nodule CT detection ready for `{name}`.\n\n"
             f"- {len(dets)} nodule candidate(s) (3D boxes, voxel coords); top score "
             f"{dets[0]['score'] if dets else 0:.3f}.\n"
             f"- Runtime {runtime}s on {device}. Decision-support only — review by a clinician.")

    return {
        "tool": "lung_nodule_ct_detector",
        "task": "lung_nodule_ct_detection",
        "summary": f"{len(dets)} lung nodule candidate(s) detected in chest CT.",
        "draft_answer": draft,
        "detections": dets,
        "artifacts": {
            "detection_review": {
                "type": "bbox3d",
                "box_mode": "xyzxyz_voxel",
                "boxes": dets,
                "mask_nifti_path": mask_path_out,
                "preprocessed_nifti_path": prep_path,
            },
            "metadata": {"file_name": name, "num_detections": len(dets)},
        },
        "studio_cards": [{"id": "detection_review", "title": "Lung Nodule Detection (CT)",
                          "subtitle": "3D bounding boxes + scores"}],
        "provenance": {
            "model": "MONAI lung_nodule_ct_detection (3D RetinaNet, LUNA16)",
            "weights": os.path.join(BUNDLE, "models", "lung_nodule_ct_detector.pt"),
            "device": str(device), "runtime_sec": runtime,
        },
        "used_tools": ["lung_nodule_ct_detector"],
    }
