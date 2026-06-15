# team_detection — Detection tool suite (submission)

Integration of four **detection** tools into ChatClinic, plus the orchestration proposal.
Runnable plugins live in the platform's `plugins/` directory; this folder holds the submission
artifacts (orchestration patch, rationale, references, slides outline).

## Tools (in `plugins/`)

| Plugin | Model | source_types | weights | License |
|---|---|---|---|---|
| `lung_nodule_ct_detector` | MONAI 3D RetinaNet (LUNA16) | dicom, nifti | `download_weights.sh` (~80 MB) | Apache-2.0 |
| `lung_nodule_cxr_detector` | NODE21 Faster R-CNN | image | `download_weights.sh` (~158 MB) | Apache-2.0 |
| `polyp_colonoscopy_detector` | YOLO-OB (SUN) | image | `download_weights.sh` (~300 MB) | Apache-2.0 |
| `gi_lesion_detector` | YOLO11s fine-tuned (Kvasir-SEG) | image | **committed** `weights/best.pt` (~19 MB) | AGPL-3.0 |

**Weights policy:** the three public pretrained models are fetched on setup via each plugin's
`download_weights.sh` (large/public assets are not committed, matching the repo's `.gitignore`
convention; two exceed GitHub's 100 MB file limit). The fine-tuned `gi_lesion_detector` weight is our
own artifact with no public source, so it is committed directly (19 MB, well under the limit).

## Orchestration
- `skill_update/skill_patch.md` — explicit 5-step routing (source → region → context → host → approval).
- `skill_update/skill_rationale.md` — why each tool, approval policy, educational value.
- Integrated into `skills/chatgenome-orchestrator/SKILL.md` (Help tool list + a
  "Medical-image detection workflows" subsection under Source-specific follow-up policy).

## Verification (GPU, measured)
CT ~7.6 s/vol (3/3 nodule sanity hits); CXR ~15 ms/img (reproduces NODE21 reference output exactly);
polyp 67.7 FPS (cross-dataset recall ~0.73); GI fine-tuned **mAP@50 0.908 / mAP@50-95 0.735** in-domain.
See `references/background_papers.md` and `slides/OUTLINE.md`.

## Follow-ups
- Studio `detection_review` renderer (2D/3D bbox overlay) is a frontend task; `execute()` runs and
  returns boxes today.
- `gi_lesion_detector` is single-class (`lesion`); extend to multi-class GI findings by retraining `nc>1`.
- Rename `team_detection` to the real team name before final submission.
