# Detection Topic — Medical-Image Detection Suite for ChatClinic

This document describes the **detection** topic contributed to the shared **ChatClinic**
platform: a suite of four medical-image detectors, how they plug into the platform's
deterministic-tools-first architecture, how the core LLM grounds and orchestrates them,
and what the combined system can do.

> **Design principle (inherited from the platform):** deterministic tools establish facts →
> Studio cards render those facts → the LLM only *explains* grounded results. The detectors
> never ask the model to "find" anything; they produce bounding boxes, and the model narrates
> them. **All outputs are decision-support for clinician review — never a diagnosis.**

---

## 1. What this topic adds

Four self-contained detector **plugins** (no change to the platform core), one shared
**Studio renderer** (`detection_review`) that overlays their boxes, and an **orchestration
policy** (in `SKILL.md`) that routes an image to the right detector.

| Tool (`@`-invokable) | Modality / source type | Purpose | Model | Output | Weights | Approval |
|---|---|---|---|---|---|---|
| `lung_nodule_ct_detector` | 3D chest **CT** — `dicom`, `nifti` | Pulmonary nodule candidates | MONAI `lung_nodule_ct_detection` bundle — 3D RetinaNet, ResNet50-FPN (LUNA16) | 3D voxel boxes `xyzxyz` + score | `download_weights.sh` (~80 MB) | **Required** (GPU/runtime) |
| `lung_nodule_cxr_detector` | 2D frontal **chest X-ray** — `image` | Lung nodules on CXR | NODE21 baseline — Faster R-CNN ResNet50-FPN | 2D boxes + likelihood | `download_weights.sh` (~158 MB) | No |
| `gi_lesion_detector` | **Endoscopy still frame** — `image` | Accuracy-oriented GI/endoscopic lesion detection | YOLO11s fine-tuned in-domain on Kvasir-SEG (class `lesion`); **mAP@50 0.91** | 2D boxes + confidence | **committed** `weights/best.pt` (~19 MB) | No |
| `polyp_colonoscopy_detector` | **Colonoscopy frame/video** — `image` | Real-time colon polyp detection | YOLO-OB (anchor-free ObjectBox, SUN-pretrained); ~67.7 FPS | 2D boxes + confidence | `download_weights.sh` (~300 MB) | No |

All four share the platform plugin contract: `plugins/<name>/{tool.json, logic.py}` with
`execute(payload) -> dict`, auto-discovered by the registry and invokable as `@<alias>`.

---

## 2. Where it sits in the platform

ChatClinic is a three-panel workspace; the detection suite plugs into all three:

```
  Sources (left)            Chat (center)                 Studio (right)
  ┌──────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
  │ upload CXR /  │   │ @detector or routing │   │ detection_review card:    │
  │ CT / endoscopy│──▶│  via SKILL policy     │──▶│  image + bbox overlay     │
  │ (auto-detect) │   │ $studio → grounded    │   │  (2D) / 3D box table (CT) │
  └──────────────┘   │  explanation (LLM)    │   │  + scores + provenance    │
                     └──────────────────────┘   └──────────────────────────┘
```

**Core platform is untouched.** No changes to the LLM/chat layer (`app/services/chat.py`),
API routing (`app/main.py`), data models (`app/models.py`), the tool runner, or the source
registry. The topic is purely additive: new `plugins/`, an additive `detection_review`
renderer + `@`-invocation in the web UI, and additive `SKILL.md` entries.

---

## 3. End-to-end flow

1. **Upload** a source file — the platform auto-detects the type and runs its bootstrap review
   (Image Review for `image`, DICOM/NIfTI Review for volumes). This gives a preview + metadata.
2. **Select a detector** — either the user types `@<detector>` in chat, or the orchestration
   policy (§4) picks the right one for the modality/region.
3. **Execute** — the web UI calls `POST /api/v1/tools/<alias>/run` → the tool runner imports the
   plugin's `execute()` → the model runs and returns:
   ```json
   {
     "tool": "...", "draft_answer": "...grounded summary...",
     "artifacts": { "detection_review": { "type": "bbox|bbox3d", "boxes": [ { "box_xyxy": [...], "score": 0.9, "label": "..." } ] } },
     "studio_cards": [ { "id": "detection_review", "title": "..." } ],
     "provenance": { "model": "...", "device": "...", "runtime_sec": ... }
   }
   ```
4. **Render** — the `detection_review` Studio card draws the boxes **over the source image**
   (2D detectors) or as a 3D voxel-box table (CT), with scores, locations, and model provenance.
5. **Explain** — the user asks questions in chat. With a grounding trigger (`$studio`,
   `$grounded`), the LLM answers **only from the detection facts in Studio** (counts, scores,
   locations); without a trigger it answers as general knowledge.

---

## 4. How the core LLM orchestrates the tools

The platform's LLM is **OpenAI `gpt-5-mini`** (configurable via `OPENAI_MODEL`), called from
`app/services/chat.py`. It does **not** run detection itself — it plays two roles:

**(a) Routing / selection (orchestration policy).**
The orchestrator skill (`skills/chatgenome-orchestrator/SKILL.md`) encodes a deterministic
**5-step routing rule** the model follows to choose the correct detector:

1. **Source family** — DICOM/NIfTI chest **CT volume** → `lung_nodule_ct_detector`; raster image → step 2.
2. **Anatomical region** of the raster image — frontal **chest X-ray** → `lung_nodule_cxr_detector`;
   **colonoscopy/lower-GI** frame → step 3.
3. **Clinical context** for colonoscopy (same frame, context decides) — single **still / report-grade**
   → `gi_lesion_detector`; **video / real-time triage** → `polyp_colonoscopy_detector`.
4. **Host/runtime gate** — the CT tool needs ~9 GB GPU and ~8 s/volume; the three 2D detectors are CPU-friendly.
5. **Approval** — the CT tool requires approval; the 2D detectors do not. Every result is framed as decision-support.

**(b) Grounded explanation.**
Once a detector has populated the Studio `detection_review` card, the LLM is asked to interpret
it. Under a grounding trigger it explains the **tool-derived facts only** — number of findings,
confidence, location/zone — and is explicitly told these are decision-support, not diagnosis.
This keeps hallucination out of the loop: the model reasons over boxes the model itself did not invent.

So "orchestration" = **policy-guided tool selection + grounded narration**, with deterministic
plugins doing the actual detection.

---

## 5. Frontend integration (the `detection_review` renderer)

Added in `webapp/app/components/customStudioRenderers.tsx` (+ registry in `studioRenderers.tsx`,
+ `@`-invocation in `page.tsx`) — all additive:

- **2D detectors** (CXR, GI, polyp): boxes are positioned as **percentages of the original image
  size**, overlaid on the source preview from the auto Image Review — so they scale correctly with
  no per-image math. Key normalization handles the detectors' mixed schemas (`box_xyxy` /
  `box_xyxy_mm` + `probability`).
- **3D detector** (CT): voxel boxes (`box_xyzxyz_voxel`) render as a table (no slice overlay).
- Each card also shows a findings table, model/runtime **provenance**, and a "decision-support only" note.

Invocation reuses the platform's generic tool endpoint; the result is merged into the active source
analysis and the Studio view switches to `detection_review`.

---

## 6. What you can do with this platform

- **Chest CT** → upload a `.nii.gz`/DICOM volume, run `@lung_nodule_ct_detector` (with approval),
  review 3D nodule candidates with scores.
- **Chest X-ray** → upload a PNG/JPG CXR, run `@lung_nodule_cxr_detector`, see 2D nodule boxes
  overlaid, then ask `"$studio how many nodules and where?"` for a grounded summary.
- **Endoscopy still** → `@gi_lesion_detector` for accuracy-oriented lesion boxes (works out of the
  box — weights committed).
- **Colonoscopy video frames** → `@polyp_colonoscopy_detector` for real-time polyp triage.
- **Ask grounded questions** about any result via `$studio` / `$grounded` — the LLM explains the
  detected findings without inventing new ones.
- Because the platform is multimodal, detection coexists with the other teams' topics (VCF, FHIR,
  imaging review, etc.) in the same workspace.

---

## 7. Setup & running

**Environment** (Python 3.10, torch 2.5.1 / torchvision 0.20.1, CUDA 12.1):
```bash
conda env create -f environment.yml && conda activate chatclinic
cp sample.env .env          # add OPENAI_API_KEY for grounded chat (tools work without it)
```

**Per-detector dependencies** (declared in each `plugins/<name>/requirements.txt`):
- `gi_lesion_detector`, `polyp_colonoscopy_detector`: `ultralytics`
- `lung_nodule_ct_detector`: `monai`
- `lung_nodule_cxr_detector`: `torch` / `torchvision` (already in the base env)

**Weights:** `gi_lesion_detector` ships its checkpoint (committed). For the others:
```bash
bash plugins/lung_nodule_ct_detector/download_weights.sh
bash plugins/lung_nodule_cxr_detector/download_weights.sh
bash plugins/polyp_colonoscopy_detector/download_weights.sh
```
(Each tool returns a clear error, not a crash, if its deps/weights are missing.)

**Run:**
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001   # backend
cd webapp && npm install && npm run dev                       # frontend → http://localhost:3000
```

---

## 8. Files (all additive — integration map)

| Area | Files |
|---|---|
| Tools | `plugins/lung_nodule_ct_detector/`, `plugins/lung_nodule_cxr_detector/`, `plugins/gi_lesion_detector/`, `plugins/polyp_colonoscopy_detector/` |
| Studio renderer | `webapp/app/components/customStudioRenderers.tsx` (`DetectionReviewCard`, `CxrImageWithBoxes`), `webapp/app/components/studioRenderers.tsx` (renderer key) |
| Web UI invocation | `webapp/app/page.tsx` (`StudioView` + `runPreAnalysisTool` detector branch) |
| Orchestration policy | `skills/chatgenome-orchestrator/SKILL.md` (Help entries + "Medical-image detection workflows" routing) |
| Submission docs | `submissions/team_detection/` (rationale, references, slides) |

**Unchanged platform core:** `app/services/chat.py` (LLM), `app/main.py`, `app/models.py`,
`app/services/tool_runner.py`, source registry/bootstrap, and the web-app shell.

---

## 9. Safety

All detectors output **decision-support** findings for clinician review and must not be presented
as a diagnosis. The CT detector is gated behind explicit approval due to GPU/runtime cost; every
detector's grounded summary repeats the decision-support caveat.
