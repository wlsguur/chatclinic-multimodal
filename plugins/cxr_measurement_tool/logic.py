"""CXR measurement tool — deterministic clinical metrics. No model required.

Computes:
  - cardiothoracic ratio (CTR) from segmentation regions (preferred) or detection boxes
  - per-finding sizes (from detection)
  - lung-zone distribution (3x2 grid) of detected findings

Detection/segmentation artifacts are taken from the payload when provided,
otherwise fetched by chaining the sibling tools via run_tool().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.services import cxr_common as C
from app.services.tool_runner import run_tool


def _get_artifact(payload: dict[str, Any], key: str, alias: str, args: dict[str, Any]) -> tuple[Optional[dict], Optional[str]]:
    """Return (artifact, warning). Use a passed-in artifact or chain the tool."""
    supplied = payload.get(key)
    if isinstance(supplied, dict) and supplied:
        return supplied, None
    try:
        out = run_tool(alias, args)
        return out["analysis"]["artifacts"][key], None
    except Exception as exc:  # backend missing weights, etc.
        return None, f"{alias} unavailable for measurement: {exc}"


def _ctr_from_segmentation(seg: dict[str, Any]) -> Optional[dict[str, Any]]:
    regions = {r["name"]: r for r in seg.get("regions", [])}
    heart = regions.get("heart")
    lungs = [regions[n] for n in ("lung_left", "lung_right") if n in regions]
    if not heart or len(lungs) < 2:
        return None
    cardiac_w = heart["bbox_xyxy"][2] - heart["bbox_xyxy"][0]
    thoracic_w = max(l["bbox_xyxy"][2] for l in lungs) - min(l["bbox_xyxy"][0] for l in lungs)
    if thoracic_w <= 0:
        return None
    ctr = cardiac_w / thoracic_w
    return {"ctr": round(ctr, 3), "cardiac_width_px": round(cardiac_w, 1),
            "thoracic_width_px": round(thoracic_w, 1), "source": "segmentation"}


def _ctr_from_detection(det: dict[str, Any]) -> Optional[dict[str, Any]]:
    findings = det.get("findings", [])
    heart = next((f for f in findings if any(k in f["label"].lower() for k in ("heart", "cardi"))), None)
    thorax = next((f for f in findings if any(k in f["label"].lower() for k in ("thorax", "lung", "chest"))), None)
    if not heart or not thorax:
        return None
    cardiac_w = heart["width_px"]
    thoracic_w = thorax["width_px"]
    if thoracic_w <= 0:
        return None
    return {"ctr": round(cardiac_w / thoracic_w, 3), "cardiac_width_px": cardiac_w,
            "thoracic_width_px": thoracic_w, "source": "detection"}


def _zone_distribution(det: dict[str, Any]) -> dict[str, int]:
    zones: dict[str, int] = {}
    W = det.get("image_width", 1) or 1
    H = det.get("image_height", 1) or 1
    for f in det.get("findings", []):
        cx, cy = f["center"]
        col = "left" if cx < W / 2 else "right"
        row = "upper" if cy < H / 3 else ("mid" if cy < 2 * H / 3 else "lower")
        zones[f"{row}-{col}"] = zones.get(f"{row}-{col}", 0) + 1
    return zones


def run_measurement(image_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    _, width, height = C.load_image(image_path)
    score = payload.get("score")
    det_args = {"image_path": image_path}
    if score not in (None, ""):
        det_args["score"] = score

    warnings: list[str] = []
    det, w1 = _get_artifact(payload, "cxr_detection", "cxr_detection_tool", det_args)
    seg, w2 = _get_artifact(payload, "cxr_segmentation", "cxr_segmentation_tool", {"image_path": image_path})
    warnings += [w for w in (w1, w2) if w]

    ctr = None
    if seg:
        ctr = _ctr_from_segmentation(seg)
    if ctr is None and det:
        ctr = _ctr_from_detection(det)
    if ctr is not None:
        ctr["interpretation"] = (
            "cardiomegaly likely (CTR > 0.50 on a frontal film)" if ctr["ctr"] > 0.5
            else "within normal limits (CTR ≤ 0.50)"
        )
    else:
        warnings.append("CTR not computed: need heart + lung regions (segmentation) or heart + thorax boxes (detection).")

    finding_sizes = [
        {"label": f["label"], "score": f["score"], "area_fraction": f["area_fraction"],
         "width_px": f["width_px"], "height_px": f["height_px"], "location_note": f["location_note"]}
        for f in (det.get("findings", []) if det else [])
    ]
    zones = _zone_distribution(det) if det else {}

    preview, _ = C.build_thumbnail(C.load_image(image_path)[0])
    artifact = {
        "image_width": width, "image_height": height, "preview_data_url": preview,
        "cardiothoracic_ratio": ctr,
        "finding_sizes": finding_sizes,
        "zone_distribution": zones,
        "inputs_used": {"detection": det is not None, "segmentation": seg is not None},
    }
    return {"artifact": artifact, "warnings": warnings, "preview": preview, "width": width, "height": height}


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    file_name = str(payload.get("file_name") or Path(image_path).name)
    res = run_measurement(image_path, payload)
    art = res["artifact"]
    ctr = art["cardiothoracic_ratio"]
    lines = ["CXR measurements:"]
    if ctr:
        lines.append(f"- **Cardiothoracic ratio: {ctr['ctr']:.2f}** ({ctr['interpretation']}, from {ctr['source']}).")
    lines.append(f"- {len(art['finding_sizes'])} finding(s) sized; zone distribution: {art['zone_distribution'] or 'n/a'}.")
    if res["warnings"]:
        lines.append("> " + " ".join(res["warnings"]))
    draft = "\n".join(lines)

    analysis = C.build_analysis(
        source_image_path=image_path, file_name=file_name, width=res["width"], height=res["height"],
        requested_view="cxr_measurement", artifact_key="cxr_measurement", artifact=art,
        studio_card={"id": "cxr_measurement", "title": "CXR Measurements", "subtitle": "Cardiothoracic ratio, sizes, zone distribution"},
        draft_answer=draft, preview_data_url=res["preview"], used_tools=["cxr_measurement_tool"], warnings=res["warnings"],
    )
    return {"analysis": analysis}
