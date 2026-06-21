# Module 07 — Red-Light Running (Trajectory / Event-Level) (§3.8)

## Summary
LSTM trajectory classifier trained on RunningRedlight dataset. Takes per-vehicle trajectory features (cx, cy, w, h + velocity) resampled to T=32 frames, classifies cross/no-cross at the stop line. Split by video (21 held-out clips) to prevent temporal leakage. Test F1=0.900 from baseline not-testable.

## Model
| Item | Detail |
|---|---|
| Architecture | LSTM (6-dim input → hidden=48, 1 layer) + Linear head |
| Input features | (cx, cy, w, h, vx, vy) — 6 dims per frame |
| Sequence length | T=32 frames (resampled) |
| Classes | no_cross (0), cross (1) |
| License | Pure PyTorch (BSD) |

## Dataset — RunningRedlight
| Split | Clips | Videos |
|---|---|---|
| Train | 1071 | 21 held-out videos excluded |
| **Test** | **243** | **21 videos (leakage-safe split-by-video)** |
| Total | 1331 clips | 15,839 frames |

## Training
| Item | Detail |
|---|---|
| Epochs | 30 |
| Device | CPU |
| Time | 3.6 sec |
| Checkpoint | `checkpoints/redlight/v1/model.pt` |

## Results (Held-out Test, Split-by-Video)
| Metric | Value |
|---|---|
| **F1 (cross)** | **0.9000** |
| Accuracy | 0.9218 |
| Precision | 0.914 |
| Recall | 0.885 |

## Configuration
```
src/train/train_redlight.py
src/eval/eval_redlight_sequence.py
```

## Files in This Folder
- `README.md` — this file
- `train_log.json` — training run (redlight_train_20260619-194422-dd0e)
- `model_location.txt` — checkpoint path
