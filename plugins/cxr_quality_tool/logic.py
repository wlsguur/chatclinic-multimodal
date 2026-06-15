"""CXR quality / QC tool — deterministic image checks. No model required.

A pre-analysis gate: dimensions/aspect, exposure (intensity statistics),
grayscale check, a simple projection hint, and a 'looks like a CXR' heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.services import cxr_common as C


def run_quality(image_path: str) -> dict[str, Any]:
    img, width, height = C.load_image(image_path)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    rgb = np.asarray(img, dtype=np.float32)

    mean = float(gray.mean())
    std = float(gray.std())
    p_low = float(np.percentile(gray, 1))
    p_high = float(np.percentile(gray, 99))
    clipped_low = float((gray < 5).mean())
    clipped_high = float((gray > 250).mean())
    # near-grayscale if channels are ~equal
    chan_spread = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
    is_grayscale = chan_spread < 5.0
    aspect = round(width / height, 3) if height else 0.0

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "pass" if ok else "warn", "detail": detail})

    add("grayscale", is_grayscale, "channels near-equal" if is_grayscale else "looks colored — may not be a plain CXR")
    add("aspect_ratio", 0.7 <= aspect <= 1.4, f"{aspect} (expected ~0.8–1.2 for a frontal CXR)")
    add("min_resolution", min(width, height) >= 512, f"{width}x{height}")
    add("exposure", 40 <= mean <= 215 and std >= 25, f"mean={mean:.0f}, std={std:.0f}")
    add("clipping", clipped_low < 0.25 and clipped_high < 0.25,
        f"black={clipped_low:.0%}, white={clipped_high:.0%}")

    looks_like_cxr = is_grayscale and 0.7 <= aspect <= 1.4 and std >= 25
    projection_hint = "frontal (AP/PA) — heuristic only; cannot reliably distinguish AP vs PA from pixels"

    preview, _ = C.build_thumbnail(img)
    artifact = {
        "image_width": width, "image_height": height, "aspect_ratio": aspect,
        "preview_data_url": preview,
        "intensity": {"mean": round(mean, 1), "std": round(std, 1),
                      "p1": round(p_low, 1), "p99": round(p_high, 1),
                      "clipped_black": round(clipped_low, 4), "clipped_white": round(clipped_high, 4)},
        "is_grayscale": is_grayscale,
        "looks_like_cxr": looks_like_cxr,
        "projection_hint": projection_hint,
        "checks": checks,
        "overall": "pass" if all(c["status"] == "pass" for c in checks) else "warn",
    }
    warnings = [] if looks_like_cxr else ["Image may not be a standard frontal chest X-ray (failed grayscale/aspect/contrast heuristic)."]
    return {"artifact": artifact, "warnings": warnings, "preview": preview, "width": width, "height": height}


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    file_name = str(payload.get("file_name") or Path(image_path).name)
    res = run_quality(image_path)
    art = res["artifact"]
    warns = [c["check"] for c in art["checks"] if c["status"] == "warn"]
    lines = [f"CXR quality check: **{art['overall'].upper()}** ({art['image_width']}x{art['image_height']}, aspect {art['aspect_ratio']})."]
    lines.append(f"- Exposure mean/std: {art['intensity']['mean']}/{art['intensity']['std']}; looks like a CXR: {art['looks_like_cxr']}.")
    if warns:
        lines.append(f"- Warnings on: {', '.join(warns)}.")
    draft = "\n".join(lines)

    analysis = C.build_analysis(
        source_image_path=image_path, file_name=file_name, width=res["width"], height=res["height"],
        requested_view="cxr_quality", artifact_key="cxr_quality", artifact=art,
        studio_card={"id": "cxr_quality", "title": "CXR Quality", "subtitle": "Exposure, aspect, grayscale, projection hint"},
        draft_answer=draft, preview_data_url=res["preview"], used_tools=["cxr_quality_tool"], warnings=res["warnings"],
    )
    return {"analysis": analysis}
