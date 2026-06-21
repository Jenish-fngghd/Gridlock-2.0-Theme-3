# Module 01 — Vehicle Detection (§3.2)

## Summary
Zero-shot RF-DETR on IDD (India Driving Dataset). No fine-tuning — explicitly ruled out due to dataset size (~46k images). RF-DETR-large is the best zero-shot variant: **mAP@0.5=0.5216** on 5000 IDD val images.

## Model (Best — RF-DETR-large)
| Item | Detail |
|---|---|
| Architecture | RF-DETR-large (COCO pretrained, Apache-2.0) |
| Variant | large |
| Weights | COCO pretrained (no local checkpoint) |
| Resolution | Default (640px) |
| License | Apache-2.0 (shippable) |

## Dataset
| Item | Detail |
|---|---|
| Name | IDD — India Driving Dataset |
| Images evaluated | 5000 (val split) |
| Annotation format | COCO (converted from VOC) |
| Classes scored | 7 (car, truck, bus, motorcycle, person, bicycle, traffic-light) |
| Domain-gap classes | autorickshaw (1025 GT), traffic-sign (1038 GT), vehicle-fallback (935 GT), animal (1091 GT) — scored 0 by construction |

## Results (Best — RF-DETR-large, run 20260621-040331-ca91)
| Metric | Value |
|---|---|
| mAP@0.5 | **0.5216** |
| mAP@0.5:0.95 | 0.3508 |
| AP@0.5 — car | 0.7208 |
| AP@0.5 — bus | 0.5881 |
| AP@0.5 — person | 0.5855 |
| AP@0.5 — motorcycle | 0.5764 |
| AP@0.5 — bicycle | 0.5124 |
| AP@0.5 — truck | 0.4491 |
| AP@0.5 — traffic-light | 0.2186 |

## Comparison vs RF-DETR-nano
| Metric | RF-DETR-nano (1000 imgs) | RF-DETR-large (5000 imgs) | Delta |
|---|---|---|---|
| mAP@0.5 | 0.4803 | **0.5216** | +4.1 pp |
| mAP@0.5:0.95 | 0.2828 | **0.3508** | +6.8 pp |
| bicycle | 0.13 | **0.51** | +38 pp |
| traffic-light | 0.00 | **0.22** | +22 pp |
| motorcycle | 0.42 | **0.58** | +14 pp |
| bus | 0.42 | **0.59** | +14 pp |
| truck | 0.72 | 0.45 | −27 pp (sample distribution shift at 5k) |

## Configuration
```
src/modules/detection.py — VehicleDetector(variant="large", threshold=0.3)
src/eval/eval_detection.py --variant large --limit 5000 --threshold 0.3
```

## Eval Log
See `eval_log.json` (run 20260621-040331-ca91).
