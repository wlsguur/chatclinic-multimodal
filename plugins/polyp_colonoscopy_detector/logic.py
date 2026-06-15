"""polyp_colonoscopy_detector — ChatClinic plugin entrypoint.

Wraps YOLO-OB (anchor-free ObjectBox polyp detector, SUN-pretrained) for real-time colon
polyp detection on colonoscopy frames. Returns 2D boxes + confidence.

Platform contract: `execute(payload: dict) -> dict`.
payload:
    image_path : str   -- colonoscopy frame (.png/.jpg/...)
    conf       : float -- optional confidence threshold (default 0.15; YOLO-OB cross-dataset
                          confidences run low/bimodal — 0.15 is its validated operating point)
    nms_iou    : float -- optional NMS IoU (default 0.3)

The YOLO-OB repo (code + SUN weights, ~300MB) is NOT committed. Run ./download_weights.sh
first (clones the repo + downloads weights into ./yolo_ob_repo), or set $CHATCLINIC_YOLOOB_REPO.
"""
from __future__ import annotations

import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CHATCLINIC_YOLOOB_REPO", os.path.join(HERE, "yolo_ob_repo"))
WEIGHTS = os.environ.get("CHATCLINIC_YOLOOB_WEIGHTS", os.path.join(REPO, "yolo_ob_sun.pth"))
IMG_SIZE = 416

_MODEL = None
_TFM = None


def _load():
    global _MODEL, _TFM
    if _MODEL is not None:
        return
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(
            f"YOLO-OB weights not found at {WEIGHTS}. "
            "Run plugins/polyp_colonoscopy_detector/download_weights.sh or set "
            "$CHATCLINIC_YOLOOB_REPO / $CHATCLINIC_YOLOOB_WEIGHTS.")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        import torchvision.transforms as T
        from models import load_model
        from utils.transforms import Resize, DEFAULT_TRANSFORMS
        args = types.SimpleNamespace(cuda_idx="0")  # within CUDA_VISIBLE_DEVICES mask -> device 0
        model = load_model(os.path.join(REPO, "config", "Config.cfg"), WEIGHTS, args)
        model.eval()
        _MODEL = model
        _TFM = T.Compose([DEFAULT_TRANSFORMS, Resize(IMG_SIZE)])
    finally:
        os.chdir(cwd)


def execute(payload: dict) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    conf = float(payload.get("conf", 0.15))
    nms_iou = float(payload.get("nms_iou", 0.3))

    _load()
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from utils.utils import non_max_suppression, rescale_boxes

    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    inp = _TFM((img, np.zeros((1, 5))))[0].unsqueeze(0)
    device = next(_MODEL.parameters()).device
    inp = inp.to(device)

    t0 = time.time()
    with torch.no_grad():
        out = _MODEL(inp)
    runtime = round(time.time() - t0, 4)

    det = non_max_suppression(out, conf_thres=conf, iou_thres=nms_iou)[0]
    dets = []
    if det is not None and det.shape[0] > 0:
        dr = rescale_boxes(det.clone(), IMG_SIZE, (H, W))
        for row in dr:
            x1, y1, x2, y2, c, _cls = [float(v) for v in row]
            dets.append({"box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                         "score": round(c, 4), "label": "polyp"})

    name = os.path.basename(image_path)
    draft = (f"Real-time polyp detection ready for `{name}`.\n\n"
             f"- {len(dets)} polyp candidate(s); top conf {dets[0]['score'] if dets else 0:.3f}.\n"
             f"- Runtime {runtime}s on {device}. Decision-support only — review by a clinician.")

    return {
        "tool": "polyp_colonoscopy_detector",
        "task": "polyp_detection",
        "summary": f"{len(dets)} polyp candidate(s) detected in colonoscopy frame.",
        "draft_answer": draft,
        "detections": dets,
        "artifacts": {
            "detection_review": {"type": "bbox", "boxes": dets},
            "metadata": {"file_name": name, "num_detections": len(dets)},
        },
        "studio_cards": [{"id": "detection_review", "title": "Polyp Detection (colonoscopy)",
                          "subtitle": "real-time 2D boxes + confidence"}],
        "provenance": {
            "model": "YOLO-OB (anchor-free ObjectBox, SUN-pretrained)",
            "weights": WEIGHTS, "device": str(device), "runtime_sec": runtime,
        },
        "used_tools": ["polyp_colonoscopy_detector"],
    }
