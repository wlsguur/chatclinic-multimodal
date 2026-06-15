"""CXR screening orchestrator — chains the CXR tools and grounds a combined summary.

This is the multi-tool orchestrator: it does NO inference itself. It calls the
other CXR plugins via run_tool() (the platform's chaining mechanism), merges
their artifacts, and writes one combined grounded answer. It degrades gracefully
when a stage's backend has no weights (records a warning, keeps going).

Default stage order: quality -> detect -> segment -> measure
(override with `stages=` or CXR_SCREENING_STAGES).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services import cxr_common as C
from app.services.tool_runner import run_tool

# stage name -> (tool alias, artifact key)
_STAGES = {
    "quality": ("cxr_quality_tool", "cxr_quality"),
    "detect": ("cxr_detection_tool", "cxr_detection"),
    "segment": ("cxr_segmentation_tool", "cxr_segmentation"),
    "measure": ("cxr_measurement_tool", "cxr_measurement"),
}
_DEFAULT_STAGES = ["quality", "detect", "segment", "measure"]


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    file_name = str(payload.get("file_name") or Path(image_path).name)
    score = payload.get("score")

    raw_stages = payload.get("stages")
    if isinstance(raw_stages, str) and raw_stages:
        stages = [s.strip() for s in raw_stages.split(",") if s.strip()]
    elif isinstance(raw_stages, list) and raw_stages:
        stages = raw_stages
    else:
        stages = C.env_list("CXR_SCREENING_STAGES", _DEFAULT_STAGES)

    _, width, height = C.load_image(image_path)
    sub_artifacts: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    ran: list[str] = []
    used_tools = ["cxr_screening_orchestrator"]

    for stage in stages:
        if stage not in _STAGES:
            warnings.append(f"Unknown screening stage '{stage}' skipped.")
            continue
        alias, key = _STAGES[stage]
        args: dict[str, Any] = {"image_path": image_path}
        if stage in ("detect", "measure") and score not in (None, ""):
            args["score"] = score
        if stage == "measure":  # reuse already-computed artifacts, avoid re-running
            if "cxr_detection" in sub_artifacts:
                args["cxr_detection"] = sub_artifacts["cxr_detection"]
            if "cxr_segmentation" in sub_artifacts:
                args["cxr_segmentation"] = sub_artifacts["cxr_segmentation"]
        try:
            out = run_tool(alias, args)
            analysis = out["analysis"]
            sub_artifacts[key] = analysis["artifacts"][key]
            warnings.extend(analysis.get("warnings", []))
            used_tools.append(alias)
            ran.append(stage)
        except Exception as exc:
            warnings.append(f"Stage '{stage}' ({alias}) failed and was skipped: {exc}")

    # combined grounded summary
    lines = [f"**CXR screening** ran stages: {', '.join(ran) or 'none'}."]
    det = sub_artifacts.get("cxr_detection")
    if det:
        lines.append(f"- Detection: {det['summary']['total']} finding(s) (backend `{det['backend']}`).")
    seg = sub_artifacts.get("cxr_segmentation")
    if seg:
        lines.append(f"- Segmentation: {len(seg['regions'])} region(s) (backend `{seg['backend']}`).")
    meas = sub_artifacts.get("cxr_measurement")
    if meas and meas.get("cardiothoracic_ratio"):
        ctr = meas["cardiothoracic_ratio"]
        lines.append(f"- CTR: {ctr['ctr']:.2f} ({ctr['interpretation']}).")
    qual = sub_artifacts.get("cxr_quality")
    if qual:
        lines.append(f"- Quality: {qual['overall'].upper()} (looks like a CXR: {qual['looks_like_cxr']}).")
    if warnings:
        lines.append("\n> Notes: " + " ".join(warnings))
    draft = "\n".join(lines)

    screening_summary = {
        "image_width": width, "image_height": height,
        "stages_requested": stages, "stages_ran": ran,
        "available": sorted(sub_artifacts.keys()),
    }
    analysis = C.build_analysis(
        source_image_path=image_path, file_name=file_name, width=width, height=height,
        requested_view="cxr_screening", artifact_key="cxr_screening", artifact=screening_summary,
        studio_card={"id": "cxr_screening", "title": "CXR Screening", "subtitle": "Combined detection / segmentation / measurement / quality"},
        draft_answer=draft, preview_data_url=det.get("preview_data_url") if det else (qual.get("preview_data_url") if qual else None),
        used_tools=used_tools, warnings=warnings, extra_artifacts=sub_artifacts,
    )
    return {"analysis": analysis}
