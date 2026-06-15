"""CXR detection tool — model-agnostic object detection on a 2D chest X-ray.

Backend is chosen at runtime via CXR_DETECTION_BACKEND:
  fallback     -> deterministic synthetic box (runs with no weights)
  torchvision  -> local checkpoint loaded into a torchvision detection arch
  remote       -> POST the image to CXR_DETECTION_API_URL with CXR_DETECTION_API_KEY

A backend returns a list of raw detections:
  [{"label": str, "score": float, "box_xyxy": [x1, y1, x2, y2]}]   (original pixels)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from app.services import cxr_common as C

_TASK = "DETECTION"
_DEFAULT_LABELS = ["__background__", "finding"]

# module-level model cache (torchvision backend)
_TV_MODEL: Any = None
_TV_CFG: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _backend_fallback(img, width: int, height: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic synthetic detection so the pipeline runs without weights."""
    bw, bh = width * 0.30, height * 0.30
    cx, cy = width * 0.50, height * 0.55
    box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    return [{"label": "demo_finding", "score": 0.90, "box_xyxy": box, "fallback": True}]


def _load_torchvision_model(cfg: dict[str, Any]):
    """Lazy-load + cache a torchvision detection model from a local checkpoint.

    ADAPT HERE if your checkpoint is not a plain torchvision state_dict:
    replace the architecture construction / load_state_dict call to match.
    """
    global _TV_MODEL, _TV_CFG
    if _TV_MODEL is not None and _TV_CFG == cfg:
        return _TV_MODEL
    import torch
    import torchvision

    weights = cfg["weights"]
    if not weights or not Path(weights).exists():
        raise C.BackendError(
            "CXR_DETECTION_BACKEND=torchvision but CXR_DETECTION_WEIGHTS is unset or missing. "
            "Point it at a torchvision detection state_dict (.pth)."
        )
    arch = cfg["arch"] or "fasterrcnn_resnet50_fpn"
    ctor = getattr(torchvision.models.detection, arch, None)
    if ctor is None:
        raise C.BackendError(f"Unknown CXR_DETECTION_ARCH='{arch}'.")
    model = ctor(weights=None, num_classes=len(cfg["labels"]))
    state = torch.load(weights, map_location=cfg["device"])
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    model.eval().to(cfg["device"])
    _TV_MODEL, _TV_CFG = model, cfg
    return model


def _backend_torchvision(img, width: int, height: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from torchvision.transforms.functional import to_tensor

    model = _load_torchvision_model(cfg)
    labels = cfg["labels"]
    with torch.inference_mode():
        out = model([to_tensor(img).to(cfg["device"])])[0]
    dets: list[dict[str, Any]] = []
    for box, score, lab in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
        idx = int(lab)
        name = labels[idx] if 0 <= idx < len(labels) else str(idx)
        dets.append({"label": name, "score": float(score), "box_xyxy": [float(v) for v in box]})
    return dets


def _backend_remote(img, width: int, height: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """POST the image to a hosted detection API.

    Expects JSON: {"detections": [{"label","score","box_xyxy":[x1,y1,x2,y2]}]}
    ADAPT the request/response mapping to your API.
    """
    import json
    import urllib.request

    url = cfg["api_url"]
    if not url:
        raise C.BackendError("CXR_DETECTION_BACKEND=remote but CXR_DETECTION_API_URL is unset.")
    payload = {"image_base64": C.png_data_url(img).split(",", 1)[1]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    if cfg["api_key"]:
        req.add_header("Authorization", f"Bearer {cfg['api_key']}")
    with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    return [
        {"label": str(d.get("label", "finding")), "score": float(d.get("score", 0.0)),
         "box_xyxy": [float(v) for v in d["box_xyxy"]]}
        for d in data.get("detections", [])
    ]


_BACKENDS = {
    "fallback": _backend_fallback,
    "torchvision": _backend_torchvision,
    "remote": _backend_remote,
}


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def _config(load_labels: bool = True) -> dict[str, Any]:
    # Only read the labels file when a real backend needs it; the fallback path
    # must not touch the filesystem so it can never be broken by a misconfigured env.
    return {
        "weights": str(C.weights_path(_TASK) or ""),
        "labels": C.load_labels(_TASK, _DEFAULT_LABELS) if load_labels else list(_DEFAULT_LABELS),
        "arch": C.env_str("CXR_DETECTION_ARCH"),
        "device": C.select_device(_TASK),
        "api_url": C.env_str("CXR_DETECTION_API_URL"),
        "api_key": C.env_str("CXR_DETECTION_API_KEY"),
        "timeout": C.env_int("CXR_DETECTION_API_TIMEOUT", 30),
    }


def run_detection(image_path: str, score: float | None = None) -> dict[str, Any]:
    """Return the `cxr_detection` artifact dict (used by the orchestrator too)."""
    img, width, height = C.load_image(image_path)
    threshold = score if score is not None else C.env_float("CXR_DETECTION_SCORE_THRESHOLD", 0.5)
    backend_name, backend = C.resolve_backend(_TASK, _BACKENDS)
    cfg = _config(load_labels=backend_name != "fallback")

    warnings: list[str] = []
    raw = backend(img, width, height, cfg)
    if backend_name == "fallback":
        warnings.append("Detection ran in fallback mode (no real weights). Set CXR_DETECTION_BACKEND + CXR_DETECTION_WEIGHTS.")

    findings: list[dict[str, Any]] = []
    for det in raw:
        if float(det.get("score", 0.0)) < threshold:
            continue
        measured = C.box_measurements(det["box_xyxy"], width, height)
        measured.update({"label": det.get("label", "finding"), "score": round(float(det.get("score", 0.0)), 4)})
        findings.append(measured)
    findings.sort(key=lambda f: f["score"], reverse=True)

    by_label: dict[str, int] = {}
    for f in findings:
        by_label[f["label"]] = by_label.get(f["label"], 0) + 1

    preview, scale = C.build_thumbnail(img)
    artifact = {
        "backend": backend_name,
        "image_width": width,
        "image_height": height,
        "preview_data_url": preview,
        "preview_scale": scale,
        "score_threshold": threshold,
        "findings": findings,
        "summary": {
            "total": len(findings),
            "by_label": by_label,
            "top": findings[0] if findings else None,
        },
    }
    return {"artifact": artifact, "warnings": warnings, "preview": preview,
            "width": width, "height": height}


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    score = payload.get("score")
    score = float(score) if score not in (None, "") else None
    file_name = str(payload.get("file_name") or Path(image_path).name)

    res = run_detection(image_path, score)
    art = res["artifact"]
    total = art["summary"]["total"]
    top = art["summary"]["top"]
    lines = [f"CXR detection found **{total}** finding(s) at score ≥ {art['score_threshold']:.2f} (backend: `{art['backend']}`)."]
    if top:
        lines.append(f"Top finding: **{top['label']}** ({top['score']:.0%}) — {top['location_note']}, {top['area_fraction']:.1%} of the image.")
    if res["warnings"]:
        lines.append("> " + " ".join(res["warnings"]))
    draft = "\n\n".join(lines)

    analysis = C.build_analysis(
        source_image_path=image_path,
        file_name=file_name,
        width=res["width"],
        height=res["height"],
        requested_view="cxr_detection",
        artifact_key="cxr_detection",
        artifact=art,
        studio_card={"id": "cxr_detection", "title": "CXR Detection", "subtitle": "Detected findings with bounding-box overlays"},
        draft_answer=draft,
        preview_data_url=res["preview"],
        used_tools=["cxr_detection_tool"],
        warnings=res["warnings"],
    )
    return {"analysis": analysis}
