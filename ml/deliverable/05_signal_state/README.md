# Module 05 — Traffic Signal State / Red-Light Running (§3.6)

## Summary
Two-stage zero-shot pipeline: RF-DETR-nano detects traffic-light heads (COCO zero-shot) → HSV-based SignalStateClassifier (Tier-0, rule-based) classifies each crop as red/yellow/green/unknown. Hit-rate 7/7 on all known red-light violation images. LISA benchmark accuracy 99.67% (1500 frames).

## Model
| Item | Detail |
|---|---|
| Stage 1 — Detector | RF-DETR-nano (COCO pretrained, class: traffic light) |
| Stage 2 — Classifier | HSV SignalStateClassifier (Tier-0, rule-based, no weights) |
| Detection conf threshold | **0.15** (lowered from 0.25 to recover borderline detections) |
| License | Apache-2.0 (RF-DETR) + pure Python (HSV rules) |

## Algorithm — HSV SignalStateClassifier
The classifier crops each detected traffic-light bounding box, converts to HSV colorspace, and checks saturation-gated hue histograms against per-state templates (red: H<20 or H>160; yellow: H 20–40; green: H 60–90). Falls back to "unknown" when no dominant hue band is found.

## Results
| Benchmark | Metric | Value |
|---|---|---|
| LISA (1500 frames) | Overall accuracy | **0.9967** |
| LISA (1500 frames) | Red recall | **0.998** |
| LISA (1500 frames) | Green accuracy | 0.91 |
| LISA (1500 frames) | Yellow accuracy | 0.27 (weak — yellow rare in LISA) |
| Violation samples (7 red-light images) | Hit-rate | **1.0** (7/7) |

## Configuration
```
src/modules/signal_state.py — SignalStateClassifier + RF-DETR traffic-light detection
src/eval/eval_signal_zeroshot.py — python -m src.eval.eval_signal_zeroshot --conf 0.15
```

## Files in This Folder
- `README.md` — this file
- `eval_log.json` — violation sample eval (run 20260620-173618-b1f8, 7/7 hit-rate)
- `model_location.txt` — model reference (no checkpoint — rule-based)
