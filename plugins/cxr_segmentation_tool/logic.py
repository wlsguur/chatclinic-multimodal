"""CXR segmentation tool — model-agnostic region segmentation on a 2D chest X-ray.

Backend chosen at runtime via CXR_SEGMENTATION_BACKEND:
  fallback     -> deterministic synthetic region overlay (runs with no weights)
  torchvision  -> local checkpoint loaded into a torchvision segmentation arch
  remote       -> POST the image to CXR_SEGMENTATION_API_URL with CXR_SEGMENTATION_API_KEY

A backend returns:
  {"regions": [{"name": str, "area_fraction": float, "bbox_xyxy": [x1,y1,x2,y2]}],
   "overlay_data_url": "<base64 png, thumbnail-sized, RGBA>"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.services import cxr_common as C

_TASK = "SEGMENTATION"
_DEFAULT_TARGETS = ["lung_left", "lung_right", "heart"]
_COLORS = {
    "lung_left": (66, 135, 245, 110),
    "lung_right": (52, 199, 89, 110),
    "heart": (255, 99, 71, 120),
}
# normalized ellipse boxes (x0, y0, x1, y1) for the fallback layout
_FALLBACK_ELLIPSES = {
    "lung_left": (0.10, 0.18, 0.43, 0.78),
    "lung_right": (0.57, 0.18, 0.90, 0.78),
    "heart": (0.40, 0.45, 0.63, 0.80),
}

_TV_MODEL: Any = None
_TV_CFG: dict[str, Any] | None = None


def _backend_fallback(img, width: int, height: int, cfg: dict[str, Any]) -> dict[str, Any]:
    targets = cfg["targets"]
    thumb = img.copy()
    thumb.thumbnail((C.THUMBNAIL_MAX_PX, C.THUMBNAIL_MAX_PX), Image.LANCZOS)
    tw, th = thumb.size
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    regions: list[dict[str, Any]] = []
    for name in targets:
        nb = _FALLBACK_ELLIPSES.get(name, (0.35, 0.35, 0.65, 0.65))
        color = _COLORS.get(name, (200, 200, 0, 110))
        draw.ellipse([nb[0] * tw, nb[1] * th, nb[2] * tw, nb[3] * th], fill=color)
        # ellipse area fraction = pi/4 * w * h
        frac = 3.14159 / 4.0 * (nb[2] - nb[0]) * (nb[3] - nb[1])
        regions.append({
            "name": name,
            "area_fraction": round(frac, 4),
            "bbox_xyxy": [round(nb[0] * width, 1), round(nb[1] * height, 1),
                          round(nb[2] * width, 1), round(nb[3] * height, 1)],
        })
    return {"regions": regions, "overlay_data_url": C.png_data_url(overlay), "fallback": True}


def _load_torchvision_model(cfg: dict[str, Any]):
    """Lazy-load + cache a torchvision segmentation model. ADAPT to your checkpoint."""
    global _TV_MODEL, _TV_CFG
    if _TV_MODEL is not None and _TV_CFG == cfg:
        return _TV_MODEL
    import torch
    import torchvision

    weights = cfg["weights"]
    if not weights or not Path(weights).exists():
        raise C.BackendError(
            "CXR_SEGMENTATION_BACKEND=torchvision but CXR_SEGMENTATION_WEIGHTS is unset or missing."
        )
    arch = cfg["arch"] or "deeplabv3_resnet50"
    ctor = getattr(torchvision.models.segmentation, arch, None)
    if ctor is None:
        raise C.BackendError(f"Unknown CXR_SEGMENTATION_ARCH='{arch}'.")
    model = ctor(weights=None, num_classes=len(cfg["targets"]) + 1)
    state = torch.load(weights, map_location=cfg["device"])
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    model.eval().to(cfg["device"])
    _TV_MODEL, _TV_CFG = model, cfg
    return model


def _backend_torchvision(img, width: int, height: int, cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torchvision.transforms.functional import to_tensor, resize

    model = _load_torchvision_model(cfg)
    size = cfg["input"]
    x = to_tensor(resize(img, [size, size])).unsqueeze(0).to(cfg["device"])
    with torch.inference_mode():
        logits = model(x)["out"][0]            # (num_classes, H, W)
    pred = logits.argmax(0).cpu()              # per-pixel class index
    targets = cfg["targets"]
    thumb = img.copy()
    thumb.thumbnail((C.THUMBNAIL_MAX_PX, C.THUMBNAIL_MAX_PX), Image.LANCZOS)
    tw, th = thumb.size
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    import numpy as np

    mask = pred.numpy()
    total = mask.size
    regions: list[dict[str, Any]] = []
    for idx, name in enumerate(targets, start=1):
        region_mask = (mask == idx)
        frac = float(region_mask.sum()) / float(total) if total else 0.0
        # colorize onto the overlay (resize class mask to thumb)
        color = _COLORS.get(name, (200, 200, 0, 110))
        m_img = Image.fromarray((region_mask * 255).astype("uint8")).resize((tw, th))
        tint = Image.new("RGBA", (tw, th), color)
        overlay = Image.composite(tint, overlay, m_img)
        ys, xs = np.where(region_mask)
        if xs.size:
            sx, sy = width / mask.shape[1], height / mask.shape[0]
            bbox = [float(xs.min() * sx), float(ys.min() * sy), float(xs.max() * sx), float(ys.max() * sy)]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
        regions.append({"name": name, "area_fraction": round(frac, 4),
                        "bbox_xyxy": [round(v, 1) for v in bbox]})
    return {"regions": regions, "overlay_data_url": C.png_data_url(overlay)}


def _backend_remote(img, width: int, height: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """POST to a hosted segmentation API. Expects {"regions":[...], "overlay_base64": "..."}. ADAPT."""
    import json
    import urllib.request

    url = cfg["api_url"]
    if not url:
        raise C.BackendError("CXR_SEGMENTATION_BACKEND=remote but CXR_SEGMENTATION_API_URL is unset.")
    payload = {"image_base64": C.png_data_url(img).split(",", 1)[1], "targets": cfg["targets"]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    if cfg["api_key"]:
        req.add_header("Authorization", f"Bearer {cfg['api_key']}")
    with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    overlay = data.get("overlay_base64", "")
    return {
        "regions": data.get("regions", []),
        "overlay_data_url": f"data:image/png;base64,{overlay}" if overlay else None,
    }


_BACKENDS = {"fallback": _backend_fallback, "torchvision": _backend_torchvision, "remote": _backend_remote}


def _config(targets: list[str]) -> dict[str, Any]:
    return {
        "weights": str(C.weights_path(_TASK) or ""),
        "targets": targets,
        "arch": C.env_str("CXR_SEGMENTATION_ARCH"),
        "device": C.select_device(_TASK),
        "input": C.env_int("CXR_SEGMENTATION_INPUT", 512),
        "api_url": C.env_str("CXR_SEGMENTATION_API_URL"),
        "api_key": C.env_str("CXR_SEGMENTATION_API_KEY"),
        "timeout": C.env_int("CXR_SEGMENTATION_API_TIMEOUT", 30),
    }


def run_segmentation(image_path: str, targets: list[str] | None = None) -> dict[str, Any]:
    img, width, height = C.load_image(image_path)
    targets = targets or C.env_list("CXR_SEGMENTATION_TARGETS", _DEFAULT_TARGETS)
    backend_name, backend = C.resolve_backend(_TASK, _BACKENDS)
    cfg = _config(targets)
    warnings: list[str] = []
    raw = backend(img, width, height, cfg)
    if backend_name == "fallback":
        warnings.append("Segmentation ran in fallback mode (no real weights). Set CXR_SEGMENTATION_BACKEND + CXR_SEGMENTATION_WEIGHTS.")
    preview, scale = C.build_thumbnail(img)
    artifact = {
        "backend": backend_name,
        "image_width": width,
        "image_height": height,
        "preview_data_url": preview,
        "preview_scale": scale,
        "regions": raw.get("regions", []),
        "overlay_data_url": raw.get("overlay_data_url"),
    }
    return {"artifact": artifact, "warnings": warnings, "preview": preview, "width": width, "height": height}


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    targets_raw = payload.get("targets")
    targets = ([t.strip() for t in targets_raw.split(",") if t.strip()] if isinstance(targets_raw, str) and targets_raw
               else targets_raw if isinstance(targets_raw, list) else None)
    file_name = str(payload.get("file_name") or Path(image_path).name)

    res = run_segmentation(image_path, targets)
    art = res["artifact"]
    region_txt = ", ".join(f"{r['name']} ({r['area_fraction']:.1%})" for r in art["regions"]) or "none"
    lines = [f"CXR segmentation produced **{len(art['regions'])}** region(s) (backend: `{art['backend']}`): {region_txt}."]
    if res["warnings"]:
        lines.append("> " + " ".join(res["warnings"]))
    draft = "\n\n".join(lines)

    analysis = C.build_analysis(
        source_image_path=image_path,
        file_name=file_name,
        width=res["width"],
        height=res["height"],
        requested_view="cxr_segmentation",
        artifact_key="cxr_segmentation",
        artifact=art,
        studio_card={"id": "cxr_segmentation", "title": "CXR Segmentation", "subtitle": "Region masks with area fractions"},
        draft_answer=draft,
        preview_data_url=res["preview"],
        used_tools=["cxr_segmentation_tool"],
        warnings=res["warnings"],
    )
    return {"analysis": analysis}
