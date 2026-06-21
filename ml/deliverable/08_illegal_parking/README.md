# Module 08 — Illegal Parking (§3.9)

## Summary
**BLOCKED — no ground-truth labels available.**

The ISLab-PVD dataset (16 .mp4 videos) is on disk but contains zero event-level GT annotations. Quantitative evaluation is impossible without parking event labels. The geometry-dwell rule engine is implemented and ready (`src/modules/geometry_engine.py`) but cannot be benchmarked.

## Status
| Item | Status |
|---|---|
| Dataset available | ISLab-PVD — 16 videos, 0 GT annotation files |
| Rule engine | Implemented (`geometry_engine.py` — dwell-time + no-parking zone rule) |
| Quantitative eval | **NOT TESTABLE** — no GT event labels |
| Unblock path | Obtain annotated GT event labels for ISLab-PVD, or use an alternate parking dataset |

## Implementation
The dwell-time rule is functional:
- Vehicle tracked across frames via `src/modules/tracking.py`
- If vehicle bounding box overlaps a configured no-parking zone for > T seconds → parking violation
- Zone geometry configured per-camera via Scene Context Model (`configs/scene_context.yaml`)

## Geometric Rule Logic
```
For each track t:
    if overlap(t.bbox, no_parking_zone) AND t.dwell_time > threshold_seconds:
        flag_parking_violation(t)
```

## Configuration
```
src/modules/geometry_engine.py — ParkingRule
src/eval/eval_illegal_parking.py — harness ready, blocked on GT
```
