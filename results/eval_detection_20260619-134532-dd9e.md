# Detection eval — IDD (run 20260619-134532-dd9e)

- Model: RF-DETR-nano (zero-shot, COCO weights) @ thr=0.3
- Images: 100 · GT scored: 313 · Detections: 162

## Quantitative (mAP, COCO-mappable classes)

| Metric | Value | §7 reference |
|---|---|---|
| mAP@0.5 | **0.0023** | DriveIndia ~0.787 (fine-tuned target) |
| mAP@0.5:0.95 | **0.0003** | — |

| Class | AP@0.5 |
|---|---|
| person | 0.0 |
| bicycle | 0.0135 |
| car | 0.0 |
| motorcycle | 0.0004 |
| bus | 0.0 |
| truck | 0.0 |
| traffic light | 0.0 |

## Structural domain gap (NOT scored — no COCO class)

| IDD class | GT instances (undetectable zero-shot) |
|---|---|
| vehicle fallback | 41 |
| autorickshaw | 21 |
| traffic sign | 7 |
| animal | 3 |

> These IDD classes have NO COCO equivalent, so zero-shot RF-DETR cannot detect them (~0 recall). Closing this needs the data-engine + fine-tune (§3.10/§8). NOT included in the mAP above.