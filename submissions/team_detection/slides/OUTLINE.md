# Presentation deck — outline (export to PDF before submitting)

Filename to submit: `team_detection_chatclinic_tool_presentation.pdf`.
Covers the four points the course requires: (1) background model/method, (2) implementation,
(3) integration into ChatClinic, (4) how the Skill patch controls selection.

1. **Title** — Team, topic (Detection), one line: "4 detection tools + 1 orchestration patch."
2. **Problem & scope** — "Where is the abnormality?" across lung (CT, CXR) and colon (colonoscopy).
3. **Background models** (point 1) — RetinaNet / Faster R-CNN / YOLO lineage; the four concrete sources
   (MONAI bundle, NODE21 baseline, YOLO-OB, fine-tuned YOLO11). 3 ready-to-use + 1 fine-tuned.
4. **Implementation** (point 2) — the `logic.py` `run(payload)->dict` contract; per-tool I/O; the
   fine-tuning pipeline for `gi_lesion_detector` (Kvasir-SEG masks→boxes, 100 epochs).
5. **Measured results** — table: CT ~7.6 s/vol·8.8 GB·3/3 sanity; CXR ~15 ms·exact reference repro;
   polyp 67.7 FPS·recall 0.73 (cross-dataset); GI **mAP@50 0.908 / mAP@50-95 0.735** (in-domain).
   Show one annotated example per tool.
6. **ChatClinic integration** (point 3) — source-type auto-detection → `image_review_tool` →
   detector; each tool's `*_detection_overlay` artifact in Studio; `tool.json` manifest + runtime block.
7. **Orchestration / Skill patch** (point 4) — the 5-step decision tree
   (source → region → context → host → approval); highlight the still-vs-video context switch between
   the two colonoscopy tools, and the cost-based approval gate on the CT tool.
8. **Limitations & licensing** — sanity-vs-official metrics; single-class GI (extensible to multi-class);
   Ultralytics AGPL-3.0 flag.
9. **Takeaway** — agentic AI = system design: ready models + one fine-tuned model + explicit, auditable routing.
