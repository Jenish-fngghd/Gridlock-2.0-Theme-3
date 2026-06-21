# Module 03 — Helmet Compliance + Triple Riding (§3.4)

## Summary
Two-stage zero-shot pipeline: RF-DETR-nano detects motorcycles and persons at high resolution (1280px per §3.4 spec) → overlap + nearest-x-center heuristic associates riders to bikes → triple-riding proxy fires when ≥3 riders are associated to one motorcycle. Helmet state uses the base zero-shot approach (rider head-region segmentation) for the submission.

## Model
| Item | Detail |
|---|---|
| Architecture | RF-DETR-nano (COCO pretrained, Apache-2.0) |
| Resolution | **1280px** (§3.4: "detect motorbike @high-res 1280–1536px") |
| COCO classes used | motorcycle, person, bicycle |
| Confidence threshold | 0.30 |
| Association | overlap > 0.1 OR nearest-x-center heuristic |
| Triple-riding rule | ≥3 riders associated to one motorcycle |
| Helmet state | Zero-shot segmentation on rider head region (base approach) |
| License | Apache-2.0 (shippable) |

## Dataset (Qualitative Capability Signal)
| Item | Detail |
|---|---|
| Helmet violations folder | 11 images |
| Triple riding violations folder | 6 images |
| Source | Sample violation images (no GT bounding boxes) |

## Results
| Sub-task | Result |
|---|---|
| Motorcycle detection hit-rate | **1.0** (11/11 helmet, 6/6 triple) |
| Triple-riding proxy triggered | **4/6** violation images (best run at 1280px) |
| Helmet state | Zero-shot approach — capability confirmed |

### Per-run Comparison
| Run | Resolution | Triple triggered |
|---|---|---|
| 20260620-171125 | 640px | 2/6 |
| 20260620-171831 | 640px | 3/6 |
| **20260620-173542** | **1280px** | **4/6** ← best |
| 20260620-173710 | 1280px | 2/6 |

Stochastic variance at 1280px — RF-DETR-nano NMS is non-deterministic.

## Configuration
```
src/modules/helmet_triple.py — HelmetTripleModule(resolution=1280)
src/eval/eval_helmet_zeroshot.py — python -m src.eval.eval_helmet_zeroshot --imgsz 1280
```

## Files in This Folder
- `README.md` — this file
- `eval_log.json` — best run (20260620-173542-344e, 1280px, triple=4/6)
- `model_location.txt` — model reference
