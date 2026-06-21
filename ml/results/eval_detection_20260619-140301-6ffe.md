# Detection eval — IDD (run 20260619-140301-6ffe)

- Model: RF-DETR-nano (zero-shot, COCO weights) @ thr=0.3
- Images: 100 · GT scored: 313 · Detections: 297

## Quantitative (mAP, COCO-mappable classes)

| Metric | Value | §7 reference |
|---|---|---|
| mAP@0.5 | **0.4043** | DriveIndia ~0.787 (fine-tuned target) |
| mAP@0.5:0.95 | **0.2272** | — |

| Class | AP@0.5 |
|---|---|
| person | 0.4853 |
| bicycle | 0.0 |
| car | 0.6319 |
| motorcycle | 0.2929 |
| bus | 0.3126 |
| truck | 0.7034 |
| traffic light | 0.0 |

## Structural domain gap (NOT scored — no COCO class)

| IDD class | GT instances (undetectable zero-shot) |
|---|---|
| vehicle fallback | 41 |
| autorickshaw | 21 |
| traffic sign | 7 |
| animal | 3 |

> These IDD classes have NO COCO equivalent, so zero-shot RF-DETR cannot detect them (~0 recall). Closing this needs the data-engine + fine-tune (§3.10/§8). NOT included in the mAP above.