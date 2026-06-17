# Demo guide — recording the detection tools (branch `team-detection`)

How to launch ChatClinic on this machine and record a demo of all four detection tools.

---

## 0. Pre-flight (one-time setup)

```bash
cd /home/cvlab21/project/hk/repo/chatclinic-class/chatclinic-multimodal
git checkout team-detection        # the merged branch with the 4 tools + frontend

# --- Python env (backend + all plugins share ONE env) ---
conda env create -f environment.yml      # creates env 'chatclinic' (py3.10, FastAPI, ...)
conda activate chatclinic

# ⚠️ Blackwell (RTX PRO 5000, sm_120) needs CUDA 12.8 torch — environment.yml ships cu121,
#    which will NOT run on these GPUs. Reinstall torch from the cu128 index:
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128

# plugin deps (MONAI, ultralytics, SimpleITK, gdown, ...)
pip install -r plugins/lung_nodule_ct_detector/requirements.txt
pip install -r plugins/lung_nodule_cxr_detector/requirements.txt
pip install -r plugins/polyp_colonoscopy_detector/requirements.txt
pip install -r plugins/gi_lesion_detector/requirements.txt
# re-pin in case a dep pulled a different torch:
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # expect 2.x+cu128  True

# --- model weights (3 public; GI is already committed) ---
bash plugins/lung_nodule_ct_detector/download_weights.sh      # ~80 MB
bash plugins/lung_nodule_cxr_detector/download_weights.sh     # ~158 MB (git-lfs)
bash plugins/polyp_colonoscopy_detector/download_weights.sh   # ~300 MB (gdown) + clones YOLO-OB code
# gi_lesion best.pt already at plugins/gi_lesion_detector/weights/best.pt (committed)

# --- env file ---
cp .env.example .env
# Detection tool runs do NOT need an API key. For grounded chat that *explains* the boxes,
# set OPENAI_API_KEY=... in .env (model defaults to gpt-5-mini).

# --- frontend deps ---
cd webapp && npm install && cd ..
```

> **Avoid the env headache?** The benchmark env at `/mnt/nvme1n1/hk/chatclinic_detection/envs/` already has
> torch cu128 + plugin deps, but it lacks FastAPI/app deps — the backend needs `chatclinic`. Keep them separate.

---

## 1. Start the stack (two terminals)

**Terminal A — backend (use GPU 2 for inference per our convention; never 0/1):**
```bash
cd /home/cvlab21/project/hk/repo/chatclinic-class/chatclinic-multimodal
conda activate chatclinic
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

**Terminal B — frontend:**
```bash
cd /home/cvlab21/project/hk/repo/chatclinic-class/chatclinic-multimodal/webapp
npm run dev          # http://localhost:3000  (backend hardcoded to 127.0.0.1:8001)
```

Open **http://localhost:3000** → 3-column workspace (Sources · Chat · Studio).

---

## 2. Verify each tool BEFORE recording (optional but recommended)

Run the model once via the backend so the first (slow) model-load happens off-camera and you confirm
detections fire. The tool-run endpoint takes the file path directly:

```bash
D=/mnt/nvme1n1/hk/chatclinic_detection/demo_inputs
check(){ curl -s -X POST http://127.0.0.1:8001/api/v1/tools/$1/run \
  -H 'Content-Type: application/json' -d "{\"payload\":{\"image_path\":\"$2\"}}" \
  | python -c 'import sys,json;d=json.load(sys.stdin);r=d.get("result",d);print(r.get("tool"),"->",len(r.get("detections",[])),"detections")'; }

check gi_lesion_detector        "$D/gi_lesion_endoscopy.jpg"
check polyp_colonoscopy_detector "$D/polyp_colonoscopy.jpg"
check lung_nodule_cxr_detector   "$D/cxr_lung_nodule.png"
check lung_nodule_ct_detector    "$D/ct_lung_nodule.nii.gz"
```
Expect ≥1 detection for GI / CXR / CT and for the provided polyp frame.

---

## 3. Turnkey demo inputs

Pre-staged at **`/mnt/nvme1n1/hk/chatclinic_detection/demo_inputs/`** (pick these in the UI's Attach dialog):

| Tool | Upload file | Detected source | Chat command |
|---|---|---|---|
| GI lesion (fine-tuned) | `gi_lesion_endoscopy.jpg` | image | `@gi_lesion` |
| Polyp (real-time) | `polyp_colonoscopy.jpg` | image | `@polyp` |
| Lung nodule — CXR | `cxr_lung_nodule.png` | image | `@lung_nodule_cxr` |
| Lung nodule — CT (3D) | `ct_lung_nodule.nii.gz` | nifti | `@lung_nodule_ct` |

(Aliases also accept the full tool name, e.g. `@gi_lesion_detector`.)

---

## 4. Recording script (suggested order — tells the orchestration story)

For each tool: **Attach the file → wait for the auto "review" card → type the `@command` in chat →
the `detection_review` Studio card appears with boxes.**

1. **GI lesion (our fine-tuned model).** Attach `gi_lesion_endoscopy.jpg` → image review card →
   type `@gi_lesion` → red box on the polyp + confidence. Say: "in-domain fine-tuned YOLO11s, mAP@50 0.91."
2. **Polyp, same modality, different context.** Attach `polyp_colonoscopy.jpg` → `@polyp` → boxes.
   Narrate the routing rule: *same colonoscopy frame, but a still image → accuracy tool (gi_lesion),
   while video/real-time → polyp tool (YOLO-OB, 67 FPS).* This is the orchestration highlight.
3. **Lung nodule on X-ray.** Attach `cxr_lung_nodule.png` → `@lung_nodule_cxr` → 2D nodule box.
4. **Lung nodule on CT (3D + approval).** Attach `ct_lung_nodule.nii.gz` → NIfTI review (Niivue 3D viewer)
   → `@lung_nodule_ct` → the detection card lists 3D voxel boxes + scores. Note it's the GPU/approval tool.
5. *(If `OPENAI_API_KEY` is set)* Ask a grounded question: `$studio explain the detected lesions` →
   the chat grounds its answer in the detection card.

Each card ends with "decision-support only — review by a clinician, not a diagnosis."

---

## 5. Gotchas / notes for a clean take

- **GPU:** keep `CUDA_VISIBLE_DEVICES=2` on the backend. CT uses ~9 GB and ~8 s/volume — do the off-camera
  warm-up in step 2 so the on-camera run is the only (already-cached) one.
- **CXR must be a PNG/JPG**, not `.mha` (the platform only auto-detects raster image suffixes). The provided
  `cxr_lung_nodule.png` was converted from the NODE21 stack for exactly this reason.
- **CT via the UI** sends only `image_path` (no `resampled` flag); on the pre-resampled demo volume the
  resample step is ≈ identity, so detection still fires. Verified in step 2.
- **Detection runs without an LLM key**; only the grounded *explanation* chat needs `OPENAI_API_KEY`.
- **Polyp** is SUN-pretrained (cross-dataset) — confidences are low/bimodal; the provided frame is one it
  fires on. If you pick another colonoscopy frame it may not detect at the default threshold.
- First call to each model loads weights (slow); subsequent calls are fast (warm up in step 2).

---

## 6. Architecture deck

Open `submissions/team_detection/slides/architecture.html` in a browser (self-contained; arrow keys / on-screen
arrows to navigate) for the system overview to show alongside the demo.
