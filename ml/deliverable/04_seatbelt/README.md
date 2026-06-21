# Module 04 — Seatbelt Detection (§3.5)

## Summary
Two-stage pipeline: YOLOv11n detects windshields → MobileNetV3-large classifies seatbelt worn/not-worn from the windshield crop. Both models are fine-tuned on the seat_belt-and-mobile OBB dataset (779 train / 337 valid). End-to-end F1=0.8082 — beats the GT-crop classifier baseline by +13 percentage points.

## Models
### Stage 1 — Windshield Detector
| Item | Detail |
|---|---|
| Architecture | YOLOv11n (Ultralytics, AGPL/benchmark) |
| Task | Single-class object detection: "windshield" |
| Dataset | seatbelt-OBB train 778 images, valid 337 images |
| Training | 80 epochs, CUDA, 6.69 min |
| **mAP@0.5** | **0.995** |
| mAP@0.5:0.95 | 0.8576 |
| Checkpoint | `checkpoints/windshield/v1/weights/best.pt` |
| Confidence threshold (inference) | 0.30 |

### Stage 2 — Belt Classifier
| Item | Detail |
|---|---|
| Architecture | MobileNetV3-large (torchvision, pretrained ImageNet) |
| Input | 224×224 windshield crop (from Stage-1 box or GT) |
| Classes | seatbelt (0), no_seatbelt (1) |
| Dataset | 780 crops (469 seatbelt / 311 no_seatbelt) |
| Training | 120 epochs, CosineAnnealingLR, CUDA, 1.33 min |
| **Best val no_seatbelt F1** | **0.8048** |
| Checkpoint | `checkpoints/seatbelt/v4/model.pt` |

## End-to-End Results (Stage1 boxes → Stage2 classification)
| Metric | GT-crop baseline | **Two-stage e2e** |
|---|---|---|
| no_seatbelt F1 | 0.678 | **0.8082** |
| Accuracy | — | 0.8605 |
| Precision | — | 0.8534 |
| Recall | — | 0.7674 |
| GT windshields | 338 | 338 |
| Detector boxes | — | 338 |
| IoU-matched | — | 337/338 |

## Training Runs (seatbelt classifier evolution)
| Version | Backbone | Epochs | Scheduler | val F1 | e2e F1 |
|---|---|---|---|---|---|
| v2 | MobileNetV3-small | 20 | step | 0.678 | — (GT crop baseline) |
| v3 | MobileNetV3-large | 80 | step | 0.8016 | 0.7866 |
| **v4** | **MobileNetV3-large** | **120** | **cosine** | **0.8048** | **0.8082** ← BEST |

## Configuration
```
# Stage 1
src/train/train_windshield_detector.py
# Stage 2
src/train/train_seatbelt.py --backbone large --epochs 120 --input 224 --scheduler cosine --version v4
# Eval
src/eval/eval_seatbelt_e2e.py --clf checkpoints/seatbelt/v4/model.pt
```

## Files in This Folder
- `README.md` — this file
- `windshield_train_log.json` — YOLOv11n windshield detector training (run 20260620-165700-d53a)
- `eval_log.json` — end-to-end seatbelt eval (run 20260620-173851-18d0)
- `model_location.txt` — checkpoint paths for both stages
