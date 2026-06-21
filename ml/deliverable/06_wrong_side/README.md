# Module 06 — Wrong-Side / Wrong-Way Driving (§3.7)

## Summary
MobileNetV3-small fine-tuned on Wrong-Way OBB dataset. Crops vehicle instances from OBB annotations, trains a binary classifier (correct-side / wrong-side). Heading is the signal — horizontal flip is explicitly disabled during augmentation to preserve left/right direction. Trained locally on CPU in 3.8 min, F1=0.9551 on held-out test.

## Model
| Item | Detail |
|---|---|
| Architecture | MobileNetV3-small (torchvision, ImageNet pretrained) |
| Input | 224×224 vehicle crop |
| Classes | correct_side (0), wrong_side (1) |
| Class weights | 7:1 (wrong_side:correct_side) — handles imbalance |
| H-flip augmentation | **DISABLED** — heading is the signal |
| License | BSD-3 (torchvision) |

## Dataset
| Split | Images | Annotation |
|---|---|---|
| Train | 426 | OBB (9-token format), 2 classes |
| Val | 91 | OBB |
| **Test** | **91** | OBB (held-out) |

## Training
| Item | Detail |
|---|---|
| Epochs | 15 |
| Device | CPU |
| Time | 3.77 min |
| Optimizer | Adam |
| Checkpoint | `checkpoints/wrongside/v1/model.pt` |

## Results (Held-out Test Set)
| Metric | Value |
|---|---|
| **F1 (wrong-side)** | **0.9551** |
| Accuracy | 0.9889 |
| Precision | 0.977 |
| Recall | 0.934 |

## Configuration
```
src/train/train_wrongside.py — python -m src.train.train_wrongside --epochs 15 --version v1
src/eval/eval_wrongside.py   — python -m src.eval.eval_wrongside --weights checkpoints/wrongside/v1/model.pt
```

## Files in This Folder
- `README.md` — this file
- `train_log.json` — training run (wrongside_train_20260619-190850-2ee9)
- `eval_log.json` — test evaluation (wrongside_eval_20260619-191336-e45a)
- `model_location.txt` — checkpoint path
