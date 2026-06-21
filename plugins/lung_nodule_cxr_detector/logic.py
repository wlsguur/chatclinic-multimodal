"""lung_nodule_cxr_detector — ChatClinic plugin entrypoint.

Wraps the NODE21 detection baseline (Faster R-CNN ResNet50-FPN) for lung-nodule detection
on 2D frontal chest X-rays. Returns 2D boxes + likelihood.

Platform contract: `execute(payload: dict) -> dict`.
payload:
    image_path : str   -- frontal CXR: raster (.png/.jpg/.tif) OR .mha/.mhd/.nii(.gz)
    nms_iou    : float -- optional greedy-NMS IoU threshold (default 0.3)

Weights (lung_nodule_cxr_detector.pth, ~158MB, public) are NOT committed. Run ./download_weights.sh first, or
set $CHATCLINIC_NODE21_REPO to a checkout of node21_detection_baseline with lung_nodule_cxr_detector.pth pulled.
"""
from __future__ import annotations

import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CHATCLINIC_NODE21_REPO", os.path.join(HERE, "node21_repo"))

_MODEL = None


def _load():
    global _MODEL
    if _MODEL is not None:
        return
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = os.path.join(REPO, "lung_nodule_cxr_detector.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"NODE21 weights not found at {weights}. "
            "Run plugins/lung_nodule_cxr_detector/download_weights.sh or set $CHATCLINIC_NODE21_REPO.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)  # background + nodule
    model.load_state_dict(torch.load(weights, map_location=device), strict=True)
    model.eval().to(device)
    _MODEL = model


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    denom = areaA + areaB - inter
    return inter / float(denom) if denom > 0 else 0.0


def _read_slices(image_path):
    import numpy as np
    p = image_path.lower()
    if p.endswith((".mha", ".mhd", ".nii", ".nii.gz")):
        import SimpleITK as sitk
        im = sitk.ReadImage(image_path)
        arr = np.array(sitk.GetArrayFromImage(im)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        sp = im.GetSpacing()
        return arr, (float(sp[0]), float(sp[1]))
    from PIL import Image
    arr = np.array(Image.open(image_path).convert("L")).astype(np.float32)[None]
    return arr, (1.0, 1.0)


def execute(payload: dict) -> dict:
    import torch

    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    nms_iou = float(payload.get("nms_iou", 0.3))

    _load()
    arr, spacing = _read_slices(image_path)
    device = next(_MODEL.parameters()).device

    dets = []
    t0 = time.time()
    for j in range(arr.shape[0]):
        sl = arr[j]
        mx = float(sl.max()) or 1.0
        t = torch.from_numpy((sl / mx)[None]).to(device)
        with torch.no_grad():
            p = _MODEL([t])[0]
        scores = p["scores"].cpu().tolist()
        boxes = p["boxes"].cpu().tolist()
        keep = []
        for sc, bx in zip(scores, boxes):
            if all(_iou(bx, kb) <= nms_iou for _, kb in keep):
                keep.append((sc, bx))
        for sc, bx in keep:
            dets.append({
                "slice": j,
                "box_xyxy_mm": [round(bx[0] * spacing[0], 2), round(bx[1] * spacing[1], 2),
                                round(bx[2] * spacing[0], 2), round(bx[3] * spacing[1], 2)],
                "probability": round(float(sc), 3),
            })
    runtime = round(time.time() - t0, 3)
    dets.sort(key=lambda d: -d["probability"])

    name = os.path.basename(image_path)
    draft = (f"Lung-nodule CXR detection ready for `{name}`.\n\n"
             f"- {len(dets)} nodule candidate(s) (2D boxes); top likelihood "
             f"{dets[0]['probability'] if dets else 0:.2f}.\n"
             f"- Runtime {runtime}s on {device}. Decision-support only — review by a clinician.")

    return {
        "tool": "lung_nodule_cxr_detector",
        "task": "lung_nodule_cxr_detection",
        "summary": f"{len(dets)} lung nodule candidate(s) detected in chest X-ray.",
        "draft_answer": draft,
        "detections": dets,
        "artifacts": {
            "detection_review": {"type": "bbox", "format": "Multiple 2D bounding boxes", "boxes": dets},
            "metadata": {"file_name": name, "num_detections": len(dets)},
        },
        "studio_cards": [{"id": "detection_review", "title": "Lung Nodule Detection (CXR)",
                          "subtitle": "2D bounding boxes + likelihood"}],
        "provenance": {
            "model": "NODE21 baseline (Faster R-CNN ResNet50-FPN)",
            "weights": os.path.join(REPO, "lung_nodule_cxr_detector.pth"),
            "device": str(device), "runtime_sec": runtime,
        },
        "used_tools": ["lung_nodule_cxr_detector"],
    }
