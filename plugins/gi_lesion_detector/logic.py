"""gi_lesion_detector — ChatClinic plugin entrypoint.

Accuracy-oriented GI/endoscopic lesion detection on still endoscopy frames. YOLO11s
fine-tuned in-domain on Kvasir-SEG (single class 'lesion'). The fine-tuned weights
(gi_lesion_detector.pt, ~19MB) are committed in-repo under ./weights/ (our own artifact — no public source).

Platform contract: `execute(payload: dict) -> dict`.
payload:
    image_path : str   -- endoscopy still frame (.png/.jpg/...)
    conf       : float -- optional confidence threshold (default 0.25)

NOTE: Ultralytics YOLO is AGPL-3.0.
"""
from __future__ import annotations

import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("CHATCLINIC_GI_WEIGHTS", os.path.join(HERE, "weights", "gi_lesion_detector.pt"))
# keep Ultralytics' writable config off $HOME on shared/locked hosts (-> /tmp/Ultralytics)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(WEIGHTS):
            raise FileNotFoundError(f"Fine-tuned weights not found at {WEIGHTS}.")
        from ultralytics import YOLO
        _MODEL = YOLO(WEIGHTS)
    return _MODEL


def execute(payload: dict) -> dict:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    conf = float(payload.get("conf", 0.25))

    t0 = time.time()
    r = _model().predict(image_path, conf=conf, verbose=False)[0]
    runtime = round(time.time() - t0, 4)

    dets = []
    if r.boxes is not None:
        for xyxy, score in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            dets.append({"box_xyxy": [round(v, 2) for v in xyxy],
                         "score": round(float(score), 4), "label": "lesion"})

    name = os.path.basename(image_path)
    draft = (f"GI lesion detection ready for `{name}`.\n\n"
             f"- {len(dets)} lesion candidate(s); top conf {dets[0]['score'] if dets else 0:.3f}.\n"
             f"- Runtime {runtime}s. Decision-support only — review by a clinician.")

    return {
        "tool": "gi_lesion_detector",
        "task": "gi_lesion_detection",
        "summary": f"{len(dets)} GI lesion candidate(s) detected.",
        "draft_answer": draft,
        "detections": dets,
        "artifacts": {
            "detection_review": {"type": "bbox", "boxes": dets},
            "metadata": {"file_name": name, "num_detections": len(dets)},
        },
        "studio_cards": [{"id": "detection_review", "title": "GI Lesion Detection",
                          "subtitle": "in-domain 2D boxes + confidence"}],
        "provenance": {
            "model": "yolo11s fine-tuned on Kvasir-SEG (class 'lesion')",
            "weights": WEIGHTS, "device": str(_model().device), "runtime_sec": runtime,
        },
        "used_tools": ["gi_lesion_detector"],
    }
