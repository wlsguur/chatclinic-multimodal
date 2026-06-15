"""Shared helpers for the CXR tool suite.

This module keeps the CXR plugins (`plugins/cxr_*`) thin and, crucially,
**model-agnostic**: each model-backed tool resolves its inference backend at
runtime from environment variables via :func:`resolve_backend`, so a real
checkpoint or hosted API can be plugged in later without touching the tool
contract, the FastAPI endpoint, the orchestrator, or the frontend.

Backend selection (per task TASK in {DETECTION, SEGMENTATION}):

    CXR_<TASK>_BACKEND = fallback | torchvision | remote   (default: fallback)

- ``fallback``   : deterministic synthetic output; runs with no weights.
- ``torchvision``: lazy-load a local checkpoint (CXR_<TASK>_WEIGHTS) into a
                   torchvision architecture (CXR_<TASK>_ARCH); cached per process.
- ``remote``     : POST the image to CXR_<TASK>_API_URL with CXR_<TASK>_API_KEY.

Pure-deterministic tools (`@measure`, `@quality`) do not use a backend.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

THUMBNAIL_MAX_PX = 512


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) else default


def env_float(name: str, default: float) -> float:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    raw = env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def env_list(name: str, default: list[str]) -> list[str]:
    raw = env_str(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# --------------------------------------------------------------------------- #
# image loading / preview
# --------------------------------------------------------------------------- #
def load_image(image_path: str) -> tuple[Image.Image, int, int]:
    """Open an image as RGB and return (image, width, height)."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image source not found: {path}")
    img = Image.open(path).convert("RGB")
    width, height = img.size
    return img, width, height


def build_thumbnail(img: Image.Image, max_px: int = THUMBNAIL_MAX_PX) -> tuple[Optional[str], float]:
    """Return (base64 PNG data URL, scale) where scale = thumb_px / original_px."""
    try:
        original_w = max(img.size[0], 1)
        thumb = img.copy()
        thumb.thumbnail((max_px, max_px), Image.LANCZOS)
        scale = thumb.size[0] / original_w
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}", scale
    except Exception:
        return None, 1.0


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_metadata(path: str, width: int, height: int, file_name: str) -> dict[str, Any]:
    p = Path(path)
    size = p.stat().st_size if p.exists() else 0
    return {
        "file_name": file_name,
        "width": width,
        "height": height,
        "file_size_bytes": size,
    }


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def quadrant(cx: float, cy: float, width: int, height: int) -> str:
    vertical = "upper" if cy < height / 2 else "lower"
    horizontal = "left" if cx < width / 2 else "right"
    return f"{vertical}-{horizontal}"


def cxr_location_note(quad: str) -> str:
    # On a frontal CXR the image is mirrored: image-left = patient-right.
    side = "patient-right" if quad.endswith("left") else "patient-left"
    return f"{quad} of image ({side} lung field)"


def box_measurements(box_xyxy: list[float], width: int, height: int) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    w_px = max(x2 - x1, 0.0)
    h_px = max(y2 - y1, 0.0)
    area_px = w_px * h_px
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    quad = quadrant(cx, cy, width, height)
    return {
        "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "width_px": round(w_px, 1),
        "height_px": round(h_px, 1),
        "area_px": round(area_px, 1),
        "area_fraction": round(area_px / float(width * height), 4) if width and height else 0.0,
        "center": [round(cx, 1), round(cy, 1)],
        "quadrant": quad,
        "location_note": cxr_location_note(quad),
    }


# --------------------------------------------------------------------------- #
# backend registry
# --------------------------------------------------------------------------- #
class BackendError(RuntimeError):
    """Raised when a backend cannot be resolved or a model cannot be loaded."""


def resolve_backend(
    task: str,
    registry: dict[str, Callable[..., Any]],
    *,
    default: str = "fallback",
) -> tuple[str, Callable[..., Any]]:
    """Return (backend_name, backend_callable) for CXR_<TASK>_BACKEND."""
    name = env_str(f"CXR_{task.upper()}_BACKEND", default) or default
    fn = registry.get(name)
    if fn is None:
        raise BackendError(
            f"Unknown CXR_{task.upper()}_BACKEND='{name}'. "
            f"Available: {sorted(registry)}."
        )
    return name, fn


def weights_path(task: str) -> Optional[Path]:
    raw = env_str(f"CXR_{task.upper()}_WEIGHTS")
    return Path(raw).expanduser() if raw else None


def load_labels(task: str, fallback_labels: list[str]) -> list[str]:
    raw = env_str(f"CXR_{task.upper()}_LABELS")
    if not raw:
        return list(fallback_labels)
    path = Path(raw).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return [str(x) for x in data]
    return [line.strip() for line in text.splitlines() if line.strip()]


def select_device(task: str) -> str:
    """Resolve CXR_<TASK>_DEVICE; blank -> cuda if available else cpu.

    torch is imported lazily so non-model tools / app startup never pay for it.
    """
    forced = env_str(f"CXR_{task.upper()}_DEVICE").lower()
    if forced:
        return forced
    try:
        import torch  # noqa: WPS433 (lazy by design)

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# response assembly
# --------------------------------------------------------------------------- #
def build_analysis(
    *,
    source_image_path: str,
    file_name: str,
    width: int,
    height: int,
    requested_view: str,
    artifact_key: str,
    artifact: dict[str, Any],
    studio_card: dict[str, Any],
    draft_answer: str,
    preview_data_url: Optional[str],
    used_tools: list[str],
    warnings: Optional[list[str]] = None,
    extra_artifacts: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Assemble the platform-standard `analysis` payload (a plain dict).

    The FastAPI endpoint validates this into the typed Cxr*Response model.
    Mirrors the shape produced by `image_review_tool`.
    """
    artifacts: dict[str, dict[str, Any]] = {
        "metadata": image_metadata(source_image_path, width, height, file_name),
        artifact_key: artifact,
    }
    if extra_artifacts:
        artifacts.update(extra_artifacts)
    return {
        "analysis_id": "",
        "source_type": "image",
        "result_kind": "image_analysis",
        "requested_view": requested_view,
        "studio": {"renderer": requested_view},
        "source_image_path": source_image_path,
        "file_name": file_name,
        "width": width,
        "height": height,
        "metadata_items": [image_metadata(source_image_path, width, height, file_name)],
        "studio_cards": [studio_card],
        "artifacts": artifacts,
        "preview_data_url": preview_data_url,
        "draft_answer": draft_answer,
        "used_tools": used_tools,
        "warnings": list(warnings or []),
    }
