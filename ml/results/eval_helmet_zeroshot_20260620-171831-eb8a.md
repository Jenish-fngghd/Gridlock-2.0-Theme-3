# Helmet + triple-riding zero-shot — §3.4 documented model (run 20260620-171831-eb8a)

- Stage-1: RF-DETR-nano (COCO-pretrained, Apache-2.0 — documented backbone §3.4)
- Dataset: datasets/Helmet & Triple Riding/sample images of violations/

## Hit-rate (capability signal — no per-image GT boxes)

| Folder | Images | Motorcycle hit | Hit-rate | Triple-riding triggered |
|---|---|---|---|---|
| helmet violations | 11 | 11 | 1.0 | 4 |
| triple riding violations | 6 | 6 | 1.0 | 3 |

## Sub-task status

| Sub-task | Status |
|---|---|
| Motorcycle detection (RF-DETR COCO zero-shot) | helmet=1.0 · triple=1.0 |
| Triple-riding proxy (≥3 riders) | 3/6 triggered |
| Helmet state (7-class AICC) | NOT TESTABLE — not_testable — requires AICC Track-5 7-class fine-tune (Tier 1) |

> Hit-rate = fraction of known-violation images where Stage-1 fires on a motorcycle. Helmet STATE is not_testable until AICC fine-tune. Triple-riding proxy fires when ≥3 riders are associated to one motorcycle.