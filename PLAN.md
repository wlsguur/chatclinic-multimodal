# Plan: CXR Multi-Tool Suite (detection + complementary tools + orchestrator)

> **Assignment spec:** `README.md` → **"Adding a New Tool"** (6 steps) + **"Design Principle"**.
> **Bonus:** external handout's *"multiple tools → +10 pts"*. In this platform a **tool = a plugin (`tool.json` + `logic.py`) invoked as `@command`**. The bonus is earned by shipping **2+ tools that do *different things*** (different output type or pipeline stage) — not multiple models inside one tool, and never by having the LLM do the work (the Design Principle forbids that: tools create facts, the LLM only explains them via `$studio`).
>
> **Status:** scaffold implemented and runnable now via deterministic fallbacks; real models plug in by config later. See **§7 Implementation Status**.

---

## 1. Why a *suite* (and why not detect + classify)

A detector that emits **bbox + label** already does classification implicitly, so a separate `@classify` is redundant. The bonus needs tools that differ in **output type** or **pipeline stage**:

| Tool | `@cmd` | Output type | Stage | Needs a model? |
|---|---|---|---|---|
| **Detection** | `@detect` | bounding boxes + labels + scores | analyze | **yes** (1 detector) |
| **Segmentation** | `@segment` | pixel masks / region overlay (lung, heart, lesion) | analyze | **yes** (1 segmenter) |
| **Measurement** | `@measure` | clinical metrics (CTR, sizes, zone %) | quantify (post) | **no** (deterministic) |
| **Quality / QC** | `@quality` | image checks (exposure, aspect, is-CXR, projection hint) | validate (pre) | **no** (deterministic) |
| **Screening orchestrator** | `@screen` | unified grounded report combining the above | orchestrate | **no** (chains tools) |

These are non-redundant because each yields a *different kind* of artifact (box ≠ mask ≠ metric ≠ QC flag) or runs at a *different stage* (validate → detect/segment → quantify).

### Possible "multiple tools" combinations (pick any ≥2)
1. **Detection + Measurement** — 1 model + 1 deterministic analytics tool. Lightest; no second weight. CTR/size/zone from boxes.
2. **Detection + Segmentation** — 2 distinct models (box vs mask). Most impressive; needs a seg checkpoint.
3. **Quality → Detection → Measurement** — 1 model + 2 deterministic tools, real pipeline.
4. **Detection + Segmentation + Measurement** — 2 models + deterministic fusion (true CTR from masks). Richest.
5. **Any of the above + `@screen`** — orchestrator chains the chosen tools into one grounded report (cherry on top; matches `skills/chatgenome-orchestrator` chaining).

> The implementation below ships **all five plugins**, so any combination above is available immediately. Pick the subset to feature once the real models are chosen; the rest still work via fallback or stay unused.

---

## 2. Model-agnostic design (the flexibility mechanism)

Model identity is **not** hard-coded. Each model tool resolves its inference backend at runtime from env, via a tiny **backend registry** in `app/services/cxr_common.py`:

```
CXR_<TASK>_BACKEND = fallback | torchvision | remote     (default: fallback)
```

- **`fallback`** — deterministic synthetic output (boxes / mask overlay). Lets every `@command` + the orchestrator + Studio cards work **end-to-end today, with no weights**. Marks results `fallback: true` and adds a visible warning.
- **`torchvision`** — lazy-loads a local checkpoint (`CXR_<TASK>_WEIGHTS`) into a torchvision architecture (`CXR_<TASK>_ARCH`); model cached at module level. **This is the slot for a local `.pth`.**
- **`remote`** — POSTs the image to `CXR_<TASK>_API_URL` with `CXR_<TASK>_API_KEY`; expects a documented JSON shape. **This is the slot for a hosted-model API key.**

Swapping in a real model = set env (+ for a non-standard checkpoint, fill one clearly-marked adapter function). No changes to the tool's payload contract, the endpoint, the orchestrator, or the frontend. `@measure`/`@quality` are pure-deterministic (no backend).

**Inference adapter contract** (what each backend function returns to the tool):
- detection → `[{"label": str, "score": float, "box_xyxy": [x1,y1,x2,y2]}]` (original-image pixels)
- segmentation → `{"regions": [{"name": str, "area_fraction": float, "bbox_xyxy": [...]}], "overlay_data_url": "<base64 png, thumbnail-sized>"}`

The tool wraps that raw output into the platform's `artifacts` + `studio_cards` + grounded `draft_answer`.

---

## 3. The five plugins (README Steps 1–3)

Each is `plugins/<name>/{tool.json, logic.py}`, `execute(payload)->{"analysis": {...}}`, mirroring `plugins/image_review_tool`. Shared helpers in `app/services/cxr_common.py`.

- **`cxr_detection_tool`** (`@detect`): backend detection → boxes; per-finding measurement (size, `area_fraction`, quadrant, CXR `location_note`); confidence filter (`score`/`CXR_DETECTION_SCORE_THRESHOLD`); aggregate. Artifact `cxr_detection`, renderer `cxr_detection`.
- **`cxr_segmentation_tool`** (`@segment`): backend segmentation → region overlay + per-region area. Artifact `cxr_segmentation`, renderer `cxr_segmentation`.
- **`cxr_measurement_tool`** (`@measure`): deterministic. Uses passed-in `detection`/`segmentation` artifacts (or calls those tools via `run_tool`) to compute **cardiothoracic ratio**, finding sizes, lung-zone distribution. Artifact `cxr_measurement`, renderer `cxr_measurement`.
- **`cxr_quality_tool`** (`@quality`): deterministic. Image checks — dimensions/aspect, intensity stats (exposure), grayscale check, simple projection hint, "looks like a CXR?" heuristic. Artifact `cxr_quality`, renderer `cxr_quality`.
- **`cxr_screening_orchestrator`** (`@screen`, alias `@cxr`): no torch; `run_tool()`-chains a configurable stage list (default `quality → detect → segment → measure`), merges all artifacts, writes one combined grounded `draft_answer`. Degrades gracefully if a stage's backend has no weights. Artifact bag = the sub-tools' artifacts, renderer `cxr_screening` (composes the sub-cards).

## 4. Backend wiring (README Step 4)
- **`app/models.py`**: `CxrToolResponse(BaseSourceResponse)` (common: `file_name, source_image_path, width, height, metadata_items, studio_cards, artifacts, warnings, preview_data_url`) + thin per-tool subclasses; per-tool request models (`image_path` + tool-specific params).
- **`app/main.py`**: five endpoints `POST /api/v1/{detection,segmentation,measurement,quality,screening}/run`, each `_run_registered_tool_model("<alias>", req.model_dump(), <Resp>, result_key="analysis")` (mirrors `run_liftover_vcf`). Plugins also auto-work via the generic `/api/v1/tools/{alias}/run`.

## 5. Studio rendering (README Step 5)
- **`studioRenderers.tsx`**: register keys `cxr_detection`, `cxr_segmentation`, `cxr_measurement`, `cxr_quality`, `cxr_screening` (route by `requested_view`).
- **`customStudioRenderers.tsx`**: card components (boxes overlay / mask overlay / metrics table / QC table / composite) + registry entries (model after `ImageReviewCard`).
- **`page.tsx`**: `StudioView` union additions + `runPreAnalysisTool` branches that POST to the dedicated endpoints (mirroring `@liftover`/`@samtools`).

## 6. SKILL.md (README Step 6)
Add `@detect`/`@segment`/`@measure`/`@quality`/`@screen` to `## Help message` → PNG/JPG/TIFF Image, and an image/CXR entry under `## Source-specific follow-up policy` describing the screening chain + env requirements.

### Env scheme (`.env.example`)
```
# per model task TASK ∈ {DETECTION, SEGMENTATION}
CXR_<TASK>_BACKEND=fallback        # fallback | torchvision | remote
CXR_<TASK>_WEIGHTS=                 # torchvision: /abs/path/checkpoint.pth
CXR_<TASK>_LABELS=                  # /abs/path/labels.txt|.json
CXR_<TASK>_ARCH=                    # torchvision arch (e.g. fasterrcnn_resnet50_fpn / deeplabv3_resnet50)
CXR_<TASK>_DEVICE=                  # blank=auto (cuda if available) | cpu
CXR_<TASK>_API_URL=                 # remote backend endpoint
CXR_<TASK>_API_KEY=                 # remote backend key
CXR_DETECTION_SCORE_THRESHOLD=0.5
CXR_SEGMENTATION_TARGETS=lung_left,lung_right,heart
CXR_SCREENING_STAGES=quality,detect,segment,measure
```

### Verification
`py_compile` new files → resolve aliases (`@detect/@segment/@measure/@quality/@screen`) → run each via `run_tool` with `fallback` (valid artifacts, no 500) → `npm run build` → upload a CXR PNG and exercise each `@command` + `@screen`.

---

## 7. Implementation Status

### ✅ Implemented and verified (backend, runs now via fallback)
All five plugins + the framework are in the repo and pass `py_compile`. A fallback run of every `@command` and the orchestrator chain succeeds (boxes, mask overlay, CTR=computed, combined summary), and all five typed responses validate.

| File | What |
|---|---|
| `app/services/cxr_common.py` | image load / thumbnail, geometry, env helpers, **backend registry** (`resolve_backend`), response assembly |
| `plugins/cxr_detection_tool/{tool.json,logic.py}` | `@detect` — backends: fallback / torchvision / remote |
| `plugins/cxr_segmentation_tool/{tool.json,logic.py}` | `@segment` — backends: fallback / torchvision / remote |
| `plugins/cxr_measurement_tool/{tool.json,logic.py}` | `@measure` — deterministic CTR / sizes / zones (chains detect+segment) |
| `plugins/cxr_quality_tool/{tool.json,logic.py}` | `@quality` — deterministic QC gate |
| `plugins/cxr_screening_orchestrator/{tool.json,logic.py}` | `@screen` / `@cxr` — chains the four via `run_tool` |
| `app/models.py` | `Cxr*Request` / `Cxr*Response` (after `ImageChatResponse`) |
| `app/main.py` | imports + `POST /api/v1/{detection,segmentation,measurement,quality,screening}/run` |
| `.env.example` | `CXR_*` config block |

### ✅ Implemented and verified (frontend — `npm run build` passes, types valid)
Built in the real env (Next.js 15.5.12, node 24): `✓ Compiled successfully`, type-check clean, 4/4 pages generated.
- `webapp/app/components/studioRenderers.tsx` — 5 renderer-metadata keys.
- `webapp/app/components/customStudioRenderers.tsx` — `CxrImageWithBoxes` + 5 cards (`CxrDetection/Segmentation/Measurement/Quality/Screening`) + registry entries. **Box overlays use percentage-of-original-size positioning, so scaling needs no ref/onLoad math.**
- `webapp/app/page.tsx` — `StudioView` union + a CXR branch in `runPreAnalysisTool` (POSTs to the dedicated endpoints, reuses `imageAnalysis` state).

### ✅ Verified in the `chatclinic` conda env
- `app.main` imports (pysam present); torch 2.5.1 / torchvision 0.20.1, CUDA available.
- All 5 HTTP endpoints return 200 with correct `requested_view`/artifacts (FastAPI TestClient).
- `torchvision` backend with no weights → clean **400** with an actionable message; `@screen` **degrades gracefully** (runs the other stages, warns about the skipped one).
- **Env fixes:** removed unused `vllm`/`transformers` from `environment.yml`/`requirements.txt` (vllm forced an uninstallable torch 2.10); rewrote the `.env.example` CXR block to **full-line comments only** — the repo's `.env` loader (`main.py:_load_local_env`) does **not** strip inline `# comments`, so `KEY=  # note` would store the comment as the value. Leave a var empty by ending the line right after `=`.

### ⏳ Remaining work
1. **Frontend build check:** `cd webapp && npm install && npm run build`; fix any type nits (the new cards are typed `any` like the existing ones, so risk is low).
2. ~~SKILL.md (README Step 6)~~ ✅ **done** — `@detect/@segment/@measure/@quality/@screen` added to `## Help message` and a "Chest X-ray (image) workflows" entry under `## Source-specific follow-up policy`.
3. **Real models** — see the how-to below. Pick which tools to feature for the bonus once weights are in hand (detection alone + measurement already clears "multiple tools"; add segmentation when its checkpoint exists).

---

## 8. How to plug in the real models (do this once weights/APIs are chosen)

**The contract never changes** — only env (and, for a non-standard checkpoint, one clearly-marked adapter function). Nothing in endpoints / orchestrator / frontend needs editing.

### A. Local checkpoint (a `.pth` you have)
1. Put the file anywhere readable, e.g. `models/cxr/detector.pth`, `models/cxr/segmenter.pth`.
2. In `.env`:
   ```
   CXR_DETECTION_BACKEND=torchvision
   CXR_DETECTION_WEIGHTS=/abs/path/models/cxr/detector.pth
   CXR_DETECTION_LABELS=/abs/path/models/cxr/labels.txt   # line 0 = __background__
   CXR_DETECTION_ARCH=fasterrcnn_resnet50_fpn             # whatever your checkpoint is
   ```
   (Segmentation mirror: `CXR_SEGMENTATION_BACKEND=torchvision`, `_WEIGHTS`, `_ARCH=deeplabv3_resnet50`.)
3. If your checkpoint is **not** a plain torchvision `state_dict`, edit the single marked function:
   - detection: `_load_torchvision_model` in `plugins/cxr_detection_tool/logic.py` (build your arch + map keys).
   - segmentation: `_load_torchvision_model` in `plugins/cxr_segmentation_tool/logic.py`.
   These are the **only** places that know the model internals.

### B. Hosted model behind an API key
1. In `.env`:
   ```
   CXR_DETECTION_BACKEND=remote
   CXR_DETECTION_API_URL=https://your-host/predict
   CXR_DETECTION_API_KEY=sk-...
   ```
2. Map request/response in `_backend_remote` (same files). Current expectation:
   - **send:** `{"image_base64": "<png bytes, no data: prefix>"}` (+ `Authorization: Bearer <key>`)
   - **detection expects back:** `{"detections":[{"label","score","box_xyxy":[x1,y1,x2,y2]}]}`
   - **segmentation expects back:** `{"regions":[{"name","area_fraction","bbox_xyxy"}], "overlay_base64":"<png>"}`
   Adjust those two lines to your API's actual shape.

### C. Output wiring (where a model's numbers go → screen)
```
backend fn returns raw  ─►  tool execute() measures/aggregates  ─►  artifacts["cxr_<task>"]
   (detection list /          (cxr_common.box_measurements,          │
    segmentation regions)      build_analysis)                       ▼
                                                     FastAPI /api/v1/<task>/run (typed Cxr*Response)
                                                                     │
                                          page.tsx runPreAnalysisTool ▼ setImageAnalysis + activateStudioFromPayload
                                                     Studio renderer "cxr_<task>" (Cxr*Card reads artifacts.cxr_<task>)
```
- Detection box coords must be **original-image pixels** `[x1,y1,x2,y2]` (the card scales by %).
- Segmentation `overlay_data_url` should be a PNG the **same aspect** as the preview (drawn at 100%×100%).
- `num_classes` / label-map length must match the checkpoint or `load_state_dict` fails — the loader raises a named error.

### Verify after wiring
```
# backend (no UI needed):
python -c "from app.services.tool_runner import run_tool, manifest_for_alias as m; \
print(run_tool(m('screen')['name'], {'image_path':'<your_cxr.png>'})['analysis']['draft_answer'])"
# full app:
uvicorn app.main:app --port 8001    # (needs the chatclinic env: pysam etc.)
cd webapp && npm install && npm run dev   # then upload a CXR PNG and try @detect / @screen
```
