# Demo Video Scenarios — ChatClinic Detection Tools

Record **one video per tool** (approx. 1–2 min each). Each video follows the same rhythm:

1. Upload the demo file via the Sources panel "Attach" button
2. Watch the automatic source review card appear in Studio
3. Type `@command natural-language description` in the Chat input
4. Wait for the detection card to render with results

All demo inputs are in `demo/data/`.

---

## Video 1 — GI Lesion Detection (Fine-tuned YOLO11s)

**File:** `gi_lesion_endoscopy.jpg`
**Command:** `@gi_lesion please detect any GI lesions in this endoscopy image`

**Steps:**
1. Open ChatClinic at `http://<server>:3000` (3-column view: Sources · Chat · Studio)
2. In the Sources panel, click **Attach** → select `gi_lesion_endoscopy.jpg`
3. Studio automatically shows the **Image Review** card with the raw image
4. In the Chat input type:
   ```
   @gi_lesion please detect any GI lesions in this endoscopy image
   ```
5. Detection fires; Studio switches to the **Detection Review** card
6. Point out: red bounding box with confidence score drawn over the lesion region

**Narration points:**
- "This is our fine-tuned YOLO11s model trained on Kvasir-SEG — mAP@50 of 0.91 in-domain."
- "A single `@gi_lesion` command routes the image directly to the GI lesion detector — no extra UI steps."
- "The red box marks the detected lesion with a confidence score in the corner."
- "Output is decision-support only — clinical review required."

---

## Video 2 — Polyp Detection (Real-time, SUN-pretrained YOLO-OB)

**File:** `polyp_colonoscopy.jpg`
**Command:** `@polyp please detect polyps in this colonoscopy frame`

**Steps:**
1. Attach `polyp_colonoscopy.jpg` in Sources → Image Review card appears
2. In Chat type:
   ```
   @polyp please detect polyps in this colonoscopy frame
   ```
3. Detection fires; Detection Review card shows bounding boxes

**Narration points:**
- "This is the YOLO-OB (Oriented Bounding Box) model pretrained on SUN-SEG — built for real-time video frames at 67 FPS."
- "Same colonoscopy modality as the GI lesion tool — the explicit `@polyp` command lets the user choose the right tool for their task: `@gi_lesion` for static still-frame accuracy, `@polyp` for real-time speed."
- "This is how ChatClinic routes between tools: the `@` prefix is the explicit execution trigger per the orchestration skill spec."
- Show the confidence scores and compare to `@gi_lesion` if time allows.

---

## Video 3 — Lung Nodule Detection on Chest X-Ray (NODE21 Faster R-CNN)

**File:** `cxr_lung_nodule.png`
**Command:** `@lung_nodule_cxr please detect lung nodules on this chest X-ray`

**Steps:**
1. Attach `cxr_lung_nodule.png` → Image Review card shows the CXR
2. In Chat type:
   ```
   @lung_nodule_cxr please detect lung nodules on this chest X-ray
   ```
3. Detection card shows 2D bounding boxes over the nodule regions

**Narration points:**
- "Faster R-CNN fine-tuned on NODE21 — detects pulmonary nodules in 2D chest X-rays."
- "The CXR is a PNG raster image, so the platform routes it as an image source — the `@lung_nodule_cxr` command targets this detector specifically."
- "Each box comes with a confidence score; number of detections is shown in the card header."
- "For clinical follow-up, this flags regions that warrant CT confirmation."

---

## Video 4 — Lung Nodule Detection on CT (3D, MONAI RetinaNet)

**File:** `ct_lung_nodule.nii.gz`
**Command:** `@lung_nodule_ct please detect lung nodules in this 3D CT scan`

**Steps:**
1. Attach `ct_lung_nodule.nii.gz` → NIfTI Review card appears with the **Niivue 3D interactive viewer**
   - Toggle slice views (Axial / Coronal / Sagittal / Multi-planar / 3D Render) to show the viewer
2. In Chat type:
   ```
   @lung_nodule_ct please detect lung nodules in this 3D CT scan
   ```
3. Inference runs (~8–10 s; model already warm from pre-flight step)
4. Studio reloads the NIfTI viewer with:
   - **Preprocessed CT volume** as the base (RAS-oriented, isotropic 0.703 × 0.703 × 1.25 mm)
   - **Red semi-transparent overlay** marking all detected nodule bounding-box regions
5. Scroll through the detected nodules table below the viewer

**Narration points:**
- "Unlike the 2D tools, this is a full 3D RetinaNet (ResNet50-FPN backbone) from the MONAI Model Zoo, trained on LUNA16."
- "The NIfTI is loaded directly into the Niivue WebGL 3D viewer — you can rotate, slice, and inspect the volume interactively."
- "After `@lung_nodule_ct` runs, the detected bounding boxes are converted into a binary mask NIfTI and loaded as a red semi-transparent overlay — the same rendering pattern as the segmentation team's tool."
- "The table below lists each nodule with its confidence score and 3D voxel coordinates."
- "This is the 'approval-gated' heavy tool — 9 GB GPU, 8 s/volume — suited for batch workflows or specialist review, not real-time."

---

## Recording tips

- **Warm up all four models before recording** (see `demo/guide.md` § 2) so inference is fast on camera
- **GPU:** `CUDA_VISIBLE_DEVICES=2`, never 0 or 1
- **Backend startup:**
  ```bash
  CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
  ```
- **Frontend startup:**
  ```bash
  cd webapp && npm run dev -- --hostname 0.0.0.0
  ```
- Access from recording machine: `http://<server-ip>:3000`
- Zoom in on the Studio panel when showing detection boxes
- For Video 4: switch to "3D Render" slice type before running the detector, then show the overlay appearing after detection fires
