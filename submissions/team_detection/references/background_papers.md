# Background papers & model sources

Representative references for each tool, with how our tool relates to the baseline. (Per course policy,
these are representative — we cite the exact model/weights we actually use.)

## Detection architectures (building blocks, not deployable medical models)
- Joseph Redmon et al. *You Only Look Once (YOLO): Unified, Real-Time Object Detection.* CVPR 2016.
  → lineage behind `polyp_colonoscopy_detector` (YOLO-OB) and `gi_lesion_detector` (YOLO11).
- Tsung-Yi Lin et al. *Focal Loss for Dense Object Detection (RetinaNet).* ICCV 2017.
  → the detector architecture instantiated by `lung_nodule_ct_detector` (3D RetinaNet).
- Shaoqing Ren et al. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS 2015.
  → architecture of `lung_nodule_cxr_detector` (NODE21 baseline).
- Xin Yi et al. *Deep Learning for Pulmonary Nodule Detection: A Survey.* (survey — context only, no weights.)

## lung_nodule_ct_detector (chest CT)
- MONAI Model Zoo bundle **lung_nodule_ct_detection** — pretrained 3D RetinaNet (Apache-2.0).
  https://huggingface.co/MONAI/lung_nodule_ct_detection
- LUNA16 / LIDC-IDRI benchmark: Setio et al., *Validation, comparison, and combination of algorithms for
  automatic detection of pulmonary nodules in CT (the LUNA16 challenge),* Medical Image Analysis, 2017.
- **Relation:** we use the bundle's shipped weights as-is (no training); it implements RetinaNet (Lin 2017)
  in 3D on a ResNet50-FPN backbone.

## lung_nodule_cxr_detector (chest X-ray)
- NODE21 challenge + detection baseline (Faster R-CNN). https://github.com/node21challenge/node21_detection_baseline ;
  https://node21.grand-challenge.org/ ; Sogancioglu et al., NODE21 (IEEE TMI / arXiv:2401.02192).
- **Relation:** we load the baseline's shipped `model.pth` as-is (no training); it is Faster R-CNN
  (Ren 2015) with a ResNet50-FPN backbone, num_classes = 2.

## polyp_colonoscopy_detector (colonoscopy, real-time)
- YOLO-OB: *An improved anchor-free real-time multiscale colon polyp detector in colonoscopy.*
  arXiv:2312.08628 ; https://github.com/seanyan62/YOLO-OB (Apache-2.0). SUN colonoscopy database (pretraining).
- **Relation:** we use the SUN-pretrained weights as-is (no training); anchor-free ObjectBox head on a
  YOLO-style backbone.

## gi_lesion_detector (endoscopy, fine-tuned)
- Ultralytics **YOLO11** (AGPL-3.0). https://github.com/ultralytics/ultralytics
- Kvasir-SEG: Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset,* MMM 2020. https://datasets.simula.no/kvasir-seg/
- **Relation:** this is the one tool we **fine-tune** — COCO-pretrained YOLO11s fine-tuned on Kvasir-SEG
  (segmentation masks converted to detection boxes), 100 epochs, in-domain val mAP@50 0.908 / mAP@50-95 0.735.

## Reference policy note
Other appropriate papers may be substituted as long as they are cited, justified, and the tool's relation
to the baseline is explained (course reference policy).
